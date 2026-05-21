# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode import eagle
from vllm.v1.spec_decode.eagle import (EagleProposer, _get_max_num_tokens,
                                       _resolve_target_num_tokens)


def _make_common_metadata() -> CommonAttentionMetadata:
    # Request query lengths are [4, 1, 5]. After rejecting [2, 0, 4],
    # draft-side query lengths become [2, 1, 1], so the decode-like requests
    # must be moved before the longer request.
    query_start_loc_cpu = torch.tensor([0, 4, 5, 10], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([4, 1, 5], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.clone(),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_cpu.clone(),
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=torch.tensor([0, 0, 0], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=10,
        max_query_len=5,
        max_seq_len=5,
        block_table_tensor=torch.tensor(
            [
                [0, 1],
                [2, 3],
                [4, 5],
            ],
            dtype=torch.int32,
        ),
        slot_mapping=torch.arange(10, dtype=torch.int64),
    )


def test_eagle_max_num_tokens_reserves_draft_headroom() -> None:
    assert _get_max_num_tokens(
        max_num_batched_tokens=16,
        max_num_seqs=3,
        num_speculative_tokens=2,
    ) == 22


def test_resolve_target_num_tokens_uses_actual_token_contract() -> None:
    metadata = _make_common_metadata()
    metadata.num_actual_tokens = 4

    token_ids = torch.arange(8, dtype=torch.int64)
    positions = torch.arange(8, dtype=torch.int64)
    hidden_states = torch.zeros((8, 16), dtype=torch.float32)

    assert _resolve_target_num_tokens(
        token_ids,
        positions,
        hidden_states,
        metadata,
    ) == 4


def test_resolve_target_num_tokens_rejects_short_tensors() -> None:
    metadata = _make_common_metadata()
    metadata.num_actual_tokens = 4

    with pytest.raises(ValueError, match="shorter"):
        _resolve_target_num_tokens(
            torch.arange(3, dtype=torch.int64),
            torch.arange(4, dtype=torch.int64),
            torch.zeros((4, 16), dtype=torch.float32),
            metadata,
        )


def test_eagle_prepare_inputs_returns_draft_request_order(monkeypatch) -> None:
    monkeypatch.setattr(eagle, "is_pin_memory_available", lambda: False)
    proposer = SimpleNamespace(
        runner=SimpleNamespace(reorder_batch_threshold=1),
        token_arange_np=torch.arange(32, dtype=torch.int64).numpy(),
    )
    metadata = _make_common_metadata()
    num_rejected_tokens = torch.tensor([2, 0, 4], dtype=torch.int32)

    updated, token_indices, request_order = EagleProposer.prepare_inputs(
        proposer,
        metadata,
        num_rejected_tokens,
    )

    assert torch.equal(request_order, torch.tensor([1, 2, 0]))
    assert torch.equal(
        updated.query_start_loc_cpu,
        torch.tensor([0, 1, 2, 4], dtype=torch.int32),
    )
    assert torch.equal(token_indices, torch.tensor([4, 5, 0, 1]))
    assert torch.equal(updated.block_table_tensor, metadata.block_table_tensor[
        request_order])
    assert torch.equal(updated.slot_mapping, metadata.slot_mapping[
        token_indices])
    assert updated.max_query_len == 2
