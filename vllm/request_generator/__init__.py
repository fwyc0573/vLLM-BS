# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Request Generator module.

This module provides utilities to generate workload requests for vLLM inference,
integrating with Frontier's request generation framework.
"""

from vllm.request_generator.config import (
    RequestGeneratorConfig,
    FixedLengthConfig,
    UniformLengthConfig,
    ZipfLengthConfig,
)
from vllm.request_generator.prompt_generator import PromptGenerator
from vllm.request_generator.vllm_request_generator import (
    VLLMRequestGenerator,
    VLLMRequest,
)
from vllm.request_generator.kv_sync import (
    KVTransferSync,
    create_kv_sync,
)

__all__ = [
    "RequestGeneratorConfig",
    "FixedLengthConfig",
    "UniformLengthConfig",
    "ZipfLengthConfig",
    "PromptGenerator",
    "VLLMRequestGenerator",
    "VLLMRequest",
    "KVTransferSync",
    "create_kv_sync",
]
