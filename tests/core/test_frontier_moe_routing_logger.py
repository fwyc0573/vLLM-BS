import json

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from vllm.v1.utils import (FrontierMoeRoutingLogger,
                           frontier_moe_routing_context,
                           log_frontier_moe_routing_from_context,
                           log_frontier_moe_routing)


def test_frontier_moe_routing_logger_writes_expected_fields(tmp_path):
    log_path = tmp_path / "frontier_moe_routing.jsonl"
    logger = FrontierMoeRoutingLogger(str(log_path))

    logger.start_batch(
        batch_id=0,
        batch_size=2,
        batch_num_tokens=4,
        batch_num_prefill_tokens=4,
        batch_num_decode_tokens=0,
        request_ids=["req-0", "req-1"],
        request_num_computed_tokens=[0, 5],
        request_num_scheduled_tokens=[1, 1],
        pp_rank=0,
    )

    topk_ids = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)
    with logger.activate():
        log_frontier_moe_routing(
            layer_name="layers.0.block_sparse_moe",
            topk_ids=topk_ids,
            num_tokens=2,
            router_topk=2,
            global_num_experts=4,
            local_num_experts=4,
            ep_rank=0,
            ep_size=1,
            expert_map=None,
            dp_rank=0,
            dispatched_across_dp=False,
        )

    logger.finish_batch()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["batch_id"] == 0
    assert record["batch_size"] == 2
    assert record["batch_num_tokens"] == 4
    assert record["layer_name"] == "layers.0.block_sparse_moe"
    assert record["total_routed_tokens"] == 4
    assert record["per_expert_tokens"] == {"0": 1, "1": 3}
    assert record["pp_rank"] == 0 and record["dp_rank"] == 0
    assert record["requests"] == [
        {"request_id": "req-0", "num_computed_tokens": 0,
         "num_scheduled_tokens": 1, "expert_tokens": {"0": 1, "1": 1}},
        {"request_id": "req-1", "num_computed_tokens": 5,
         "num_scheduled_tokens": 1, "expert_tokens": {"1": 2}},
    ]


class _DummyMoEMethod:

    def __init__(self) -> None:
        self.fused_experts = self._dummy_fused_experts
        self.has_bias = False
        self.topk_indices_dtype = torch.int64
        self.rocm_aiter_moe_enabled = False

    @staticmethod
    def _dummy_fused_experts(**kwargs):
        return kwargs["hidden_states"]

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group=None,
        num_expert_group=None,
        global_num_experts: int = -1,
        expert_map=None,
        custom_routing_function=None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias=None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view=None,
        logical_to_physical_map=None,
        logical_replica_count=None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )
        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            dp_rank=0,
            dispatched_across_dp=False,
        )


class _DummyMoELayer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer_name = "layers.0.block_sparse_moe"
        self.local_num_experts = 4
        self.global_num_experts = 4
        self.ep_rank = 0
        self.ep_size = 1
        self.w13_weight = torch.zeros((1, 1, 1))
        self.w2_weight = torch.zeros((1, 1, 1))


def test_frontier_moe_routing_logger_uses_local_expected_total_when_expert_map_present(tmp_path):
    log_path = tmp_path / "frontier_moe_routing_local_expected.jsonl"
    logger = FrontierMoeRoutingLogger(str(log_path))

    logger.start_batch(
        batch_id=0,
        batch_size=2,
        batch_num_tokens=4,
        batch_num_prefill_tokens=4,
        batch_num_decode_tokens=0,
        request_ids=["req-0", "req-1"],
        request_num_computed_tokens=[0, 5],
        request_num_scheduled_tokens=[1, 1],
        pp_rank=0,
    )

    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int64)

    with logger.activate():
        log_frontier_moe_routing(
            layer_name="layers.0.block_sparse_moe",
            topk_ids=topk_ids,
            num_tokens=2,
            router_topk=2,
            global_num_experts=4,
            local_num_experts=2,
            ep_rank=0,
            ep_size=2,
            expert_map=expert_map,
            dp_rank=0,
            dispatched_across_dp=False,
        )

    logger.finish_batch()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["num_experts_per_device"] == 2
    assert record["total_routed_tokens"] == 2
    assert record["expected_total_routed_tokens"] == 2
    assert record["per_expert_tokens"] == {"0": 1, "1": 1}

def test_fused_moe_forward_cuda_logs_routing(tmp_path):
    log_path = tmp_path / "frontier_moe_forward.jsonl"
    logger = FrontierMoeRoutingLogger(str(log_path))

    logger.start_batch(
        batch_id=0,
        batch_size=2,
        batch_num_tokens=4,
        batch_num_prefill_tokens=4,
        batch_num_decode_tokens=0,
        request_ids=["req-0", "req-1"],
        request_num_computed_tokens=[0, 5],
        request_num_scheduled_tokens=[1, 1],
        pp_rank=0,
    )

    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 4)

    def custom_routing_function(hidden_states, gating_output, topk, renormalize):
        topk_ids = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=hidden_states.dtype)
        return topk_weights, topk_ids

    method = _DummyMoEMethod()
    layer = _DummyMoELayer()

    with logger.activate():
        with frontier_moe_routing_context(
                layer_name=layer.layer_name,
                num_tokens=hidden_states.shape[0],
                router_topk=2,
                global_num_experts=layer.global_num_experts,
                local_num_experts=layer.local_num_experts,
                ep_rank=layer.ep_rank,
                ep_size=layer.ep_size,
                expert_map=None,
            dp_rank=0,
            dispatched_across_dp=False,
        ):
            method.forward_cuda(
                layer=layer,
                x=hidden_states,
                use_grouped_topk=False,
                top_k=2,
                router_logits=router_logits,
                renormalize=False,
                global_num_experts=layer.global_num_experts,
                expert_map=None,
                custom_routing_function=custom_routing_function,
                scoring_func="softmax",
                routed_scaling_factor=1.0,
                e_score_correction_bias=None,
                apply_router_weight_on_input=False,
                activation="silu",
                enable_eplb=False,
                expert_load_view=None,
                logical_to_physical_map=None,
                logical_replica_count=None,
            )

    logger.finish_batch()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["layer_name"] == layer.layer_name
    assert record["total_routed_tokens"] == 4


def test_moe_routing_context_required_for_logging(tmp_path):
    log_path = tmp_path / "frontier_moe_context_required.jsonl"
    logger = FrontierMoeRoutingLogger(str(log_path))

    logger.start_batch(
        batch_id=0,
        batch_size=1,
        batch_num_tokens=2,
        batch_num_prefill_tokens=2,
        batch_num_decode_tokens=0,
        request_ids=["req-0"],
        request_num_computed_tokens=[0],
        request_num_scheduled_tokens=[1],
        pp_rank=0,
    )

    topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    with logger.activate():
        with pytest.raises(RuntimeError, match="routing context"):
            log_frontier_moe_routing_from_context(topk_ids)

    logger.finish_batch()


class _DispatchedForwardContext:

    class dp_metadata:
        # DP rank 0 scheduled 3 tokens and DP rank 1 scheduled 2.
        cu_tokens_across_dp_cpu = torch.tensor([3, 5])


def test_frontier_moe_routing_logger_splits_its_dp_slice_by_request(
        tmp_path, monkeypatch):
    import vllm.forward_context
    monkeypatch.setattr(vllm.forward_context, "get_forward_context",
                        lambda: _DispatchedForwardContext)
    log_path = tmp_path / "frontier_moe_routing_dp_slice.jsonl"
    logger = FrontierMoeRoutingLogger(str(log_path))
    logger.start_batch(
        batch_id=3,
        batch_size=2,
        batch_num_tokens=2,
        batch_num_prefill_tokens=0,
        batch_num_decode_tokens=2,
        request_ids=["req-a", "req-b"],
        request_num_computed_tokens=[16, 40],
        request_num_scheduled_tokens=[1, 1],
        pp_rank=1,
    )
    # Rows 0-2 are DP rank 0's tokens; rows 3 and 4 are this rank's requests.
    topk_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [2, 3], [1, 3]])
    expert_map = torch.tensor([-1, -1, 0, 1])
    with logger.activate():
        log_frontier_moe_routing(
            layer_name="layers.0.mlp.experts",
            topk_ids=topk_ids,
            num_tokens=5,
            router_topk=2,
            global_num_experts=4,
            local_num_experts=2,
            ep_rank=1,
            ep_size=2,
            expert_map=expert_map,
            dp_rank=1,
            dispatched_across_dp=True,
        )
    logger.finish_batch()

    record = json.loads(log_path.read_text())
    # The local load still covers every routed token of both DP ranks.
    assert record["per_expert_tokens"] == {"0": 1, "1": 2}
    assert (record["pp_rank"], record["dp_rank"], record["ep_rank"]) == (1, 1, 1)
    assert record["requests"] == [
        {"request_id": "req-a", "num_computed_tokens": 16,
         "num_scheduled_tokens": 1, "expert_tokens": {"2": 1, "3": 1}},
        {"request_id": "req-b", "num_computed_tokens": 40,
         "num_scheduled_tokens": 1, "expert_tokens": {"1": 1, "3": 1}},
    ]


def test_frontier_moe_routing_logger_rejects_rows_that_are_not_the_batch(
        tmp_path):
    logger = FrontierMoeRoutingLogger(str(tmp_path / "rows.jsonl"))
    logger.start_batch(
        batch_id=0,
        batch_size=1,
        batch_num_tokens=3,
        batch_num_prefill_tokens=3,
        batch_num_decode_tokens=0,
        request_ids=["req-0"],
        request_num_computed_tokens=[0],
        request_num_scheduled_tokens=[3],
        pp_rank=0,
    )
    # A chunk of the batch, as the chunked forward path routes it.
    with logger.activate(), pytest.raises(RuntimeError, match="scheduled tokens"):
        log_frontier_moe_routing(
            layer_name="layers.0.mlp.experts",
            topk_ids=torch.tensor([[0, 1], [1, 2]]),
            num_tokens=2,
            router_topk=2,
            global_num_experts=4,
            local_num_experts=4,
            ep_rank=0,
            ep_size=1,
            expert_map=None,
            dp_rank=0,
            dispatched_across_dp=False,
        )
