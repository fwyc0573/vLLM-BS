# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import torch

from transformers import Qwen3MoeConfig

from vllm.config import SpeculativeConfig
from vllm.model_executor.models import registry as model_registry


def test_hf_config_override_promotes_qwen3_moe_to_qwen3_moe_mtp() -> None:
    config = Qwen3MoeConfig()
    config.num_nextn_predict_layers = 1
    config.architectures = ["Qwen3MoeForCausalLM"]

    overridden = SpeculativeConfig.hf_config_override(config)

    assert overridden.model_type == "qwen3_moe_mtp"
    assert overridden.architectures == ["Qwen3MoeMTP"]
    assert overridden.n_predict == 1


def test_model_registry_exposes_qwen3_moe_mtp_architecture() -> None:
    assert model_registry._TEXT_GENERATION_MODELS["Qwen3MoeMTP"] == (
        "qwen3_moe_mtp",
        "Qwen3MoeMTP",
    )


def test_qwen3_moe_mtp_module_is_importable() -> None:
    module = importlib.import_module("vllm.model_executor.models.qwen3_moe_mtp")

    assert hasattr(module, "Qwen3MoeMTP")


def test_qwen3_moe_mtp_load_weights_maps_expert_weights(monkeypatch) -> None:
    module = importlib.import_module("vllm.model_executor.models.qwen3_moe_mtp")

    recorded = {}

    class DummyParam:

        def weight_loader(self,
                          param,
                          loaded_weight,
                          name,
                          *,
                          shard_id,
                          expert_id):
            recorded["shape"] = tuple(loaded_weight.shape)
            recorded["name"] = name
            recorded["shard_id"] = shard_id
            recorded["expert_id"] = expert_id

    dummy_param = DummyParam()
    predictor = SimpleNamespace(
        config=SimpleNamespace(num_experts=1),
        named_parameters=lambda: [(
            "model.layers.36.mlp.experts.0.gate_up_proj.weight",
            dummy_param,
        )],
    )

    monkeypatch.setattr(module, "is_pp_missing_parameter", lambda *_: False)
    monkeypatch.setattr(
        module.FusedMoE,
        "make_expert_params_mapping",
        lambda **_: [("gate_up_proj", "gate_proj", 0, 0)],
    )

    loaded = module.Qwen3MoeMultiTokenPredictor.load_weights(
        predictor,
        [("model.layers.36.mlp.experts.0.gate_proj.weight", torch.ones(1))],
    )

    assert "model.layers.36.mlp.experts.0.gate_up_proj.weight" in loaded
    assert recorded == {
        "shape": (1,),
        "name": "model.layers.36.mlp.experts.0.gate_up_proj.weight",
        "shard_id": 0,
        "expert_id": 0,
    }
