# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys

import torch

MODULE_NAME = "vllm.model_executor.layers.vocab_parallel_embedding"
CUSTOM_OP_NAMES = ("vocab_parallel_embedding", "parallel_lm_head")


def _load_vocab_parallel_embedding(monkeypatch, *,
                                   compile_disable: str | None):
    compile_calls = []

    def fake_compile(*compile_args, **compile_kwargs):
        if compile_args and callable(compile_args[0]) and len(compile_args) == 1:
            fn = compile_args[0]
            compile_calls.append({
                "args": (),
                "kwargs": compile_kwargs,
                "fn_name": fn.__name__,
            })
            return fn

        def decorator(fn):
            compile_calls.append({
                "args": compile_args,
                "kwargs": compile_kwargs,
                "fn_name": fn.__name__,
            })
            return fn

        return decorator

    monkeypatch.setattr(torch, "compile", fake_compile)
    if compile_disable is None:
        monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    else:
        monkeypatch.setenv("TORCH_COMPILE_DISABLE", compile_disable)

    from vllm.model_executor.custom_op import CustomOp
    for custom_op_name in CUSTOM_OP_NAMES:
        CustomOp.op_registry.pop(custom_op_name, None)
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME), compile_calls


def test_vocab_parallel_embedding_skips_import_time_compile_when_disabled(
        monkeypatch):
    _, compile_calls = _load_vocab_parallel_embedding(monkeypatch,
                                                      compile_disable="1")

    assert compile_calls == []


def test_vocab_parallel_embedding_uses_import_time_compile_by_default(
        monkeypatch):
    module, compile_calls = _load_vocab_parallel_embedding(monkeypatch,
                                                           compile_disable=None)

    assert module.get_masked_input_and_mask.__name__ == \
        "get_masked_input_and_mask"
    assert len(compile_calls) == 1
    compile_call = compile_calls[0]
    assert compile_call["fn_name"] == "get_masked_input_and_mask"
    assert compile_call["kwargs"]["dynamic"] is True
    assert compile_call["kwargs"]["backend"] == \
        module._get_current_platform().simple_compile_backend
