# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.frontier_request_metrics import \
    FrontierRequestMetricsJSONLLogger
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest,
                                              RequestResponseMetadata)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.metrics.stats import FrontierRequestMetrics

MODEL_NAME = "test-model"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


@dataclass
class MockHFConfig:
    model_type: str = "llama"


@dataclass
class MockModelConfig:
    max_model_len: int = 4096
    generation_config: str = "auto"
    diff_sampling_param: dict[str, Any] = field(default_factory=dict)
    model: str = MODEL_NAME
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)

    def get_diff_sampling_param(self):
        return self.diff_sampling_param


def _build_models() -> OpenAIServingModels:
    return OpenAIServingModels(
        engine_client=MagicMock(),
        model_config=MockModelConfig(),
        base_model_paths=BASE_MODEL_PATHS,
    )


def _build_completion_serving(monkeypatch: pytest.MonkeyPatch, log_path: str
                              ) -> OpenAIServingCompletion:
    monkeypatch.setenv("VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH", log_path)
    return OpenAIServingCompletion(
        engine_client=MagicMock(),
        model_config=MockModelConfig(),
        models=_build_models(),
        request_logger=None,
    )


def _build_chat_serving(monkeypatch: pytest.MonkeyPatch,
                        log_path: str) -> OpenAIServingChat:
    monkeypatch.setenv("VLLM_FRONTIER_REQUEST_METRICS_LOG_PATH", log_path)
    return OpenAIServingChat(
        engine_client=MagicMock(),
        model_config=MockModelConfig(),
        models=_build_models(),
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


def _make_metrics(request_id: str,
                  *,
                  arrival_time: float = 100.0,
                  completion_time: float = 112.5) -> FrontierRequestMetrics:
    return FrontierRequestMetrics(
        request_id=request_id,
        request_e2e_time=12.5,
        ttft=4.5,
        tpot=1.25,
        request_model_execution_time=8.0,
        request_num_prefill_tokens=3,
        request_num_decode_tokens=2,
        arrival_time=arrival_time,
        completion_time=completion_time,
    )


def _make_request_output(request_id: str,
                         *,
                         finished: bool = True,
                         metrics: FrontierRequestMetrics | None = None,
                         text: str = "hello") -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[4, 5],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop" if finished else None,
                stop_reason=None,
            )
        ],
        finished=finished,
        metrics=metrics,
    )


async def _completion_stream_results(*results: RequestOutput):
    for result in results:
        yield 0, result


async def _chat_results(*results: RequestOutput):
    for result in results:
        yield result


def test_frontier_request_metrics_logger_deduplicates_finished_requests(
        tmp_path):
    log_path = tmp_path / "frontier_metrics.jsonl"
    logger = FrontierRequestMetricsJSONLLogger(str(log_path))

    logger.log(_make_request_output("req-1", metrics=_make_metrics("req-1")))
    logger.log(_make_request_output("req-1", metrics=_make_metrics("req-1")))
    logger.log(
        _make_request_output(
            "req-2",
            finished=False,
            metrics=_make_metrics("req-2"),
        ))
    logger.log(_make_request_output("req-3", finished=True, metrics=None))

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["request_id"] == "req-1"
    assert payload["ttft"] == pytest.approx(4.5)




def test_frontier_request_metrics_logger_keeps_same_request_id_across_replays(
        tmp_path):
    log_path = tmp_path / "frontier_metrics_replays.jsonl"
    logger = FrontierRequestMetricsJSONLLogger(str(log_path))

    logger.log(
        _make_request_output(
            "req-1",
            metrics=_make_metrics(
                "req-1", arrival_time=100.0, completion_time=112.5)))
    logger.log(
        _make_request_output(
            "req-1",
            metrics=_make_metrics(
                "req-1", arrival_time=200.0, completion_time=212.5)))

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert [payload["completion_time"] for payload in payloads] == [112.5, 212.5]


def test_completion_response_logs_finished_metrics(tmp_path,
                                                   monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "completion_response.jsonl"
    serving = _build_completion_serving(monkeypatch, str(log_path))
    request_output = _make_request_output("completion-response",
                                          metrics=_make_metrics(
                                              "completion-response"))

    response = serving.request_output_to_completion_response(
        final_res_batch=[request_output],
        request=CompletionRequest(model=MODEL_NAME,
                                  prompt="prompt",
                                  max_tokens=2),
        request_id="cmpl-parent",
        created_time=123,
        model_name=MODEL_NAME,
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="cmpl-parent"),
    )

    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["request_id"] == "completion-response"


def test_completion_stream_logs_finished_metrics(tmp_path,
                                                 monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "completion_stream.jsonl"
    serving = _build_completion_serving(monkeypatch, str(log_path))
    request_output = _make_request_output("completion-stream",
                                          metrics=_make_metrics(
                                              "completion-stream"))

    async def _collect_chunks() -> list[str]:
        return [
            chunk async for chunk in serving.completion_stream_generator(
                request=CompletionRequest(model=MODEL_NAME,
                                          prompt="prompt",
                                          max_tokens=2,
                                          stream=True),
                engine_prompts=[{"prompt": "prompt"}],
                result_generator=_completion_stream_results(request_output),
                request_id="cmpl-stream-parent",
                created_time=123,
                model_name=MODEL_NAME,
                num_prompts=1,
                tokenizer=MagicMock(),
                request_metadata=RequestResponseMetadata(
                    request_id="cmpl-stream-parent"),
                enable_force_include_usage=False,
            )
        ]

    chunks = asyncio.run(_collect_chunks())

    assert chunks[-1] == "data: [DONE]\n\n"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["request_id"] == "completion-stream"


def test_chat_full_generator_logs_finished_metrics(tmp_path,
                                                   monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "chat_full.jsonl"
    serving = _build_chat_serving(monkeypatch, str(log_path))
    request_output = _make_request_output("chat-full",
                                          metrics=_make_metrics("chat-full"))

    response = asyncio.run(
        serving.chat_completion_full_generator(
            request=ChatCompletionRequest(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=2,
            ),
            result_generator=_chat_results(request_output),
            request_id="chat-full-parent",
            model_name=MODEL_NAME,
            conversation=[{"role": "user", "content": "hello"}],
            tokenizer=MagicMock(),
            request_metadata=RequestResponseMetadata(
                request_id="chat-full-parent"),
        ))

    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["request_id"] == "chat-full"


def test_chat_stream_logs_finished_metrics(tmp_path,
                                           monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "chat_stream.jsonl"
    serving = _build_chat_serving(monkeypatch, str(log_path))
    request_output = _make_request_output("chat-stream",
                                          metrics=_make_metrics("chat-stream"))

    async def _collect_chunks() -> list[str]:
        return [
            chunk async for chunk in serving.chat_completion_stream_generator(
                request=ChatCompletionRequest(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=2,
                    stream=True,
                ),
                result_generator=_chat_results(request_output),
                request_id="chat-stream-parent",
                model_name=MODEL_NAME,
                conversation=[{"role": "user", "content": "hello"}],
                tokenizer=MagicMock(),
                request_metadata=RequestResponseMetadata(
                    request_id="chat-stream-parent"),
                enable_force_include_usage=False,
            )
        ]

    chunks = asyncio.run(_collect_chunks())

    assert chunks[-1] == "data: [DONE]\n\n"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["request_id"] == "chat-stream"
