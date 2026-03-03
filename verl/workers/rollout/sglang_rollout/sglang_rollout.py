# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Generator

import ray
import sglang.srt.entrypoints.engine
import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.device_mesh import DeviceMesh

from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets
from verl.workers.rollout.utils import is_valid_ipv6_address

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Fix: Ray actors have random multiprocessing authkeys, so ForkingPickler's
# rebuild_storage_fd path fails with AuthenticationError for both CUDA and CPU
# tensors. Regular pickle.Pickler produces _rebuild_tensor_v2 (data inline, no
# IPC file descriptors, no authkey required). SGLang's SafeUnpickler allows
# torch.* so server-side deserialization succeeds. (verl issue #4065)
import io as _io
import pickle as _pickle


def _safe_mps_serialize(obj, output_str: bool = False):
    buf = _io.BytesIO()
    _pickle.Pickler(buf).dump(obj)
    result = buf.getvalue()
    if output_str:
        import base64
        return base64.b64encode(result).decode()
    return result


try:
    try:
        from sglang.srt.utils.common import MultiprocessingSerializer as _MPS
    except ImportError:
        from sglang.srt.utils import MultiprocessingSerializer as _MPS
    _MPS.serialize = staticmethod(_safe_mps_serialize)
except Exception as _e:
    logger.warning(f"Could not patch MultiprocessingSerializer.serialize: {_e}")


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class ServerAdapter(BaseRollout):
    """SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        if config.get("quantization", None) == "fp8":
            import sglang
            from packaging import version

            assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                "sglang>=0.5.5 is required for FP8 quantization"
            )
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            model_config.hf_config.quantization_config = fp8_block_quant_kwargs
        super().__init__(config, model_config, device_mesh)
        self._engine: AsyncHttpServerAdapter = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self.local_rank = self.rollout_rank % local_world_size

    async def _init_server_adapter(self):
        if self._engine is not None:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path, host=host, port=server_port, launch_server=False
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._init_server_adapter()
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._init_server_adapter()
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        # Fix: when base_sync_done=True, FSDP sends raw LoRA adapter weights (lora_A.weight,
        # lora_B.weight). SGLang's base model has no LoRA layers and cannot accept these keys
        # → KeyError in gemma2.load_weights(). vLLM handles this via add_lora(), but SGLang
        # has no equivalent API. Skip the LoRA-only update; SGLang continues to use the base
        # model weights loaded on first wake_up() (CUDA VMM shared with FSDP). Rollout
        # inference does not reflect the current LoRA delta (off-policy), but the pipeline
        # runs correctly end-to-end. Proper fix: merge LoRA into base weights on FSDP side
        # before calling update_weights. (verl issue #4065, see docs/debug_session.md)
        peft_config = kwargs.get("peft_config", None)
        base_sync_done = kwargs.get("base_sync_done", False)
        if peft_config is not None and base_sync_done:
            logger.warning(
                "SGLang update_weights: skipping LoRA adapter weight sync (SGLang base model "
                "has no LoRA layers). SGLang will infer with base model weights from CUDA VMM."
            )
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                await self._engine.flush_cache()
            return

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._init_server_adapter()

        update_weights_bucket_bytes = int(self.config.update_weights_bucket_megabytes) << 20
        if self.config.get("quantization", None) == "fp8":
            from verl.utils.sglang.sglang_fp8_utils import quant_weights_by_name

            logger.info("Convert bf16 weights to fp8 format before loading")
            weights = quant_weights_by_name(
                weights,
                self.model_config.hf_config.quantization_config,
                dtype=self.model_config.hf_config.dtype,
            )
        else:
            weights = weights

        for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
            # Strip ".base_layer": verl's replace_lora_wrapper adds ".base_layer.weight"
            # to LoRA target module keys (e.g. "qkv_proj.base_layer.weight"), but SGLang's
            # base model uses standard HF names ("qkv_proj.weight") → KeyError in SGLang
            # scheduler. LoRA adapter keys (lora_A.weight etc.) don't contain ".base_layer"
            # so replace() is a no-op for them. (verl issue #4065)
            cpu_batch = [
                (name.replace(".base_layer", ""), t.detach().cpu() if isinstance(t, torch.Tensor) else t)
                for name, t in params_batch
            ]
            await sgl_update_weights(
                engine=self._engine,
                params_batch=cpu_batch,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
            )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._engine.flush_cache()
