# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper

from vllm.attention.backends.abstract import (AttentionLayer, AttentionType,
                                              is_quantized_kv_cache)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonImpl,
                                                   MLACommonMetadata)

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


class FlashInferMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl


g_fi_workspace = torch.zeros(
    FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
    dtype=torch.uint8,
    device="cuda",
)


class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashInferMLA V1 with FP8 KV cache not yet supported")

        self._workspace_buffer = g_fi_workspace
        self._decode_wrapper: Optional[BatchMLAPagedAttentionWrapper] = None

    def _get_decode_wrapper(self) -> BatchMLAPagedAttentionWrapper:
        if self._decode_wrapper is None:
            self._decode_wrapper = BatchMLAPagedAttentionWrapper(
                self._workspace_buffer,
                backend="auto",
            )
        return self._decode_wrapper

    @staticmethod
    def _build_decode_page_indices(
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = seq_lens.device
        seq_lens = seq_lens.to(dtype=torch.int32)
        block_table = block_table.to(device=device, dtype=torch.int32)

        blocks_per_req = torch.div(
            seq_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        )
        page_offsets = torch.arange(
            block_table.size(1),
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0)
        page_mask = page_offsets < blocks_per_req.unsqueeze(1)
        kv_indices = block_table[page_mask].contiguous()
        kv_indptr = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            blocks_per_req.cumsum(dim=0, dtype=torch.int32),
        ])
        return kv_indptr, kv_indices, seq_lens

    def _forward_decode(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
        else:
            q_nope, q_pe = q.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        if q_nope.shape[0] != attn_metadata.decode.seq_lens.numel():
            raise NotImplementedError(
                "FlashInfer MLA BatchMLAPagedAttentionWrapper decode path "
                "expects exactly one decode query token per request.")

        page_size = kv_c_and_k_pe_cache.size(1)
        kv_indptr, kv_indices, kv_len_arr = self._build_decode_page_indices(
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            page_size,
        )
        qo_indptr = torch.arange(
            0,
            q_nope.shape[0] + 1,
            dtype=torch.int32,
            device=q_nope.device,
        )

        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank:]

        decode_wrapper = self._get_decode_wrapper()
        decode_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            page_size,
            False,
            self.scale,
            q_nope.dtype,
            kv_c_cache.dtype,
        )

        o = decode_wrapper.run(
            q_nope,
            q_pe,
            kv_c_cache,
            k_pe_cache,
            return_lse=False,
        )

        return o, None
