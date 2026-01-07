# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Request Generator - Main Module.

This module generates workload requests compatible with vLLM's inference API,
using Frontier's request generation framework for workload specification.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from vllm.sampling_params import SamplingParams
from vllm.request_generator.config import (
    RequestGeneratorConfig,
    FixedLengthConfig,
    UniformLengthConfig,
    ZipfLengthConfig,
    LengthGeneratorType,
)
from vllm.request_generator.prompt_generator import PromptGenerator


# Default Frontier path - dynamically resolve from vllm location
# Can be overridden by:
# 1. Setting config.frontier_path in RequestGeneratorConfig
# 2. Setting FRONTIER_PATH environment variable
# Example: export FRONTIER_PATH=/path/to/frontier
# Path structure: frontier/sota-infer-engine/vllm/vllm/request_generator/vllm_request_generator.py
_VLLM_REQUEST_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FRONTIER_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_VLLM_REQUEST_GEN_DIR)))))


def _resolve_frontier_path(config_path: Optional[str] = None) -> str:
    """Resolve the Frontier project path.
    
    Priority order:
    1. config_path (if provided)
    2. FRONTIER_PATH environment variable
    3. DEFAULT_FRONTIER_PATH
    
    Args:
        config_path: Path from RequestGeneratorConfig.frontier_path
        
    Returns:
        Resolved and normalized path to Frontier project.
        
    Raises:
        ValueError: If the resolved path doesn't exist.
    """
    # Determine path using priority order
    if config_path:
        path = config_path
        source = "config.frontier_path"
    elif "FRONTIER_PATH" in os.environ:
        path = os.environ["FRONTIER_PATH"]
        source = "FRONTIER_PATH environment variable"
    else:
        path = DEFAULT_FRONTIER_PATH
        source = "default path"
    
    # Normalize path: expand user (~), resolve symlinks, remove trailing slashes
    path = os.path.normpath(os.path.expanduser(path))
    
    # Validate path exists
    if not os.path.exists(path):
        raise ValueError(
            f"Frontier path does not exist: {path}\n"
            f"  Source: {source}\n"
            f"  To fix, either:\n"
            f"    - Set FRONTIER_PATH environment variable to the correct path\n"
            f"    - Pass frontier_path in RequestGeneratorConfig\n"
            f"    - Ensure the default path exists: {DEFAULT_FRONTIER_PATH}"
        )
    
    # Validate it looks like a Frontier project
    frontier_module = os.path.join(path, "frontier")
    if not os.path.isdir(frontier_module):
        raise ValueError(
            f"Invalid Frontier path: {path}\n"
            f"  Expected 'frontier' module directory not found.\n"
            f"  Ensure the path points to the Frontier project root."
        )
    
    return path


@dataclass
class VLLMRequest:
    """A request object compatible with vLLM's generate() API.
    
    Attributes:
        prompt: Either a text string or a dict with prompt_token_ids.
        sampling_params: vLLM SamplingParams for this request.
        request_id: Optional identifier for tracking.
        prefill_tokens: Number of input tokens (from Frontier metadata).
        decode_tokens: Number of output tokens to generate.
        arrived_at: Arrival time (0 for offline mode).
    """
    prompt: Union[str, dict[str, Any]]
    sampling_params: SamplingParams
    request_id: Optional[str] = None
    prefill_tokens: int = 0
    decode_tokens: int = 0
    arrived_at: float = 0.0


class VLLMRequestGenerator:
    """Generate vLLM-compatible requests using Frontier's request generator.
    
    This class bridges Frontier's request generation framework with vLLM's
    inference API, enabling workload testing with controllable characteristics.
    
    Example:
        config = RequestGeneratorConfig(
            num_requests=100,
            length_config=FixedLengthConfig(prefill_tokens=2048, decode_tokens=256),
        )
        generator = VLLMRequestGenerator(config)
        requests = generator.generate()
        
        # Use with vLLM
        prompts = [r.prompt for r in requests]
        sampling_params = [r.sampling_params for r in requests]
        llm.generate(prompts, sampling_params)
    """
    
    def __init__(
        self,
        config: RequestGeneratorConfig,
        tokenizer=None,
    ):
        """Initialize the request generator.
        
        Args:
            config: Request generation configuration.
            tokenizer: Optional tokenizer for text-based prompt generation.
                      Required if config.use_token_ids=False.
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # Validate: tokenizer required for text mode
        if not config.use_token_ids and tokenizer is None:
            raise ValueError(
                "tokenizer is required when use_token_ids=False. "
                "Either provide a tokenizer or set use_token_ids=True for precise length control."
            )
        
        # Initialize prompt generator with config settings
        self.prompt_generator = PromptGenerator(
            tokenizer=tokenizer,
            vocab_size=config.vocab_size,
            seed=config.seed,
            use_token_ids=config.use_token_ids,  # Use config setting
            min_token_id=config.min_token_id,
            max_token_id=config.max_token_id,
        )
        
        # Setup Frontier import path
        self._setup_frontier_path()
        
        # Import Frontier modules (lazy)
        self._frontier_imported = False
        self._frontier_generator = None
        
    def _setup_frontier_path(self):
        """Setup Python path for Frontier imports.
        
        Resolves the Frontier path using priority order:
        1. config.frontier_path (if set)
        2. FRONTIER_PATH environment variable
        3. DEFAULT_FRONTIER_PATH constant
        
        The resolved path is validated to ensure:
        - It exists on the filesystem
        - It contains a 'frontier' module directory
        """
        frontier_path = _resolve_frontier_path(self.config.frontier_path)
        self._resolved_frontier_path = frontier_path
        
        if frontier_path not in sys.path:
            sys.path.insert(0, frontier_path)
            
    def _import_frontier(self):
        """Lazy import of Frontier modules."""
        if self._frontier_imported:
            return
            
        try:
            from frontier.request_generator.synthetic_request_generator import (
                SyntheticRequestGenerator,
            )
            from frontier.config import (
                SyntheticRequestGeneratorConfig,
                FixedRequestLengthGeneratorConfig,
                UniformRequestLengthGeneratorConfig,
                ZipfRequestLengthGeneratorConfig,
                StaticRequestIntervalGeneratorConfig,
            )
            
            self._SyntheticRequestGenerator = SyntheticRequestGenerator
            self._SyntheticRequestGeneratorConfig = SyntheticRequestGeneratorConfig
            self._FixedRequestLengthGeneratorConfig = FixedRequestLengthGeneratorConfig
            self._UniformRequestLengthGeneratorConfig = UniformRequestLengthGeneratorConfig
            self._ZipfRequestLengthGeneratorConfig = ZipfRequestLengthGeneratorConfig
            self._StaticRequestIntervalGeneratorConfig = StaticRequestIntervalGeneratorConfig
            
            self._frontier_imported = True
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import Frontier modules from: {self._resolved_frontier_path}\\n"
                f"  Original error: {e}\\n"
                f"  To fix, ensure Frontier is correctly installed:\\n"
                f"    - Set FRONTIER_PATH environment variable\\n"
                f"    - Or pass frontier_path in RequestGeneratorConfig"
            )
    
    def _create_frontier_length_config(self):
        """Create Frontier length generator config from our config."""
        self._import_frontier()
        
        length_type = self.config.get_length_generator_type()
        length_cfg = self.config.length_config
        
        if length_type == LengthGeneratorType.FIXED:
            cfg = length_cfg  # type: FixedLengthConfig
            return self._FixedRequestLengthGeneratorConfig(
                prefill_tokens=cfg.prefill_tokens,
                decode_tokens=cfg.decode_tokens,
                seed=self.config.seed,
            )
        elif length_type == LengthGeneratorType.UNIFORM:
            cfg = length_cfg  # type: UniformLengthConfig
            return self._UniformRequestLengthGeneratorConfig(
                min_tokens=cfg.min_tokens,
                max_tokens=cfg.max_tokens,
                prefill_to_decode_ratio=cfg.prefill_to_decode_ratio,
                seed=self.config.seed,
            )
        elif length_type == LengthGeneratorType.ZIPF:
            cfg = length_cfg  # type: ZipfLengthConfig
            return self._ZipfRequestLengthGeneratorConfig(
                theta=cfg.theta,
                min_tokens=cfg.min_tokens,
                max_tokens=cfg.max_tokens,
                prefill_to_decode_ratio=cfg.prefill_to_decode_ratio,
                scramble=cfg.scramble,
                seed=self.config.seed,
            )
        else:
            raise ValueError(f"Unknown length generator type: {length_type}")
    
    def _create_frontier_generator(self):
        """Create the Frontier SyntheticRequestGenerator."""
        self._import_frontier()
        
        length_config = self._create_frontier_length_config()
        
        # Use static interval for offline mode (all arrive at time 0)
        interval_config = self._StaticRequestIntervalGeneratorConfig(
            seed=self.config.seed,
        )
        
        generator_config = self._SyntheticRequestGeneratorConfig(
            length_generator_config=length_config,
            interval_generator_config=interval_config,
            num_requests=self.config.num_requests,
            seed=self.config.seed,
        )
        
        return self._SyntheticRequestGenerator(generator_config)
    
    def generate_frontier_requests(self) -> list:
        """Generate raw Frontier request objects.
        
        Returns:
            List of Frontier Request objects with metadata.
        """
        if self._frontier_generator is None:
            self._frontier_generator = self._create_frontier_generator()
            
        return self._frontier_generator.generate()
    
    def _create_sampling_params(
        self,
        decode_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> SamplingParams:
        """Create SamplingParams for a request.
        
        Args:
            decode_tokens: Target output length.
            temperature: Sampling temperature (0 = greedy).
            top_p: Top-P sampling parameter.
            
        Returns:
            Configured SamplingParams object.
            
        Note:
            When force_exact_output_length=True (default), the following
            settings are applied to guarantee exact output length:
            - ignore_eos=True: Model won't stop at EOS token
            - stop=None: No stop strings
            - stop_token_ids=None: No stop token IDs
        """
        if self.config.force_exact_output_length:
            # Force exact output length by ignoring all stop conditions
            return SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=decode_tokens,
                ignore_eos=True,      # Force generation past EOS
                stop=None,            # No stop strings
                stop_token_ids=None,  # No stop tokens
            )
        else:
            # Allow early stopping at EOS or stop tokens
            return SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=decode_tokens,
                # Default behavior: stop at EOS
            )
    
    def generate(
        self,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> list[VLLMRequest]:
        """Generate vLLM-compatible requests.
        
        Args:
            temperature: Sampling temperature for all requests.
            top_p: Top-P sampling for all requests.
            
        Returns:
            List of VLLMRequest objects ready for vLLM.generate().
        """
        # Generate Frontier request metadata
        frontier_requests = self.generate_frontier_requests()
        
        # Convert to vLLM requests
        vllm_requests = []
        
        for idx, freq in enumerate(frontier_requests):
            prefill_tokens = freq.num_prefill_tokens
            decode_tokens = freq.num_decode_tokens
            arrived_at = freq.arrived_at
            
            # Generate prompt of specified length
            # When use_token_ids=True: generates exact length token ID list
            # When use_token_ids=False: generates approximate length text
            prompt_content = self.prompt_generator.generate(prefill_tokens)
            
            # Create prompt in appropriate format
            if self.config.use_token_ids:
                # TokensPrompt format for precise length control
                prompt = {"prompt_token_ids": prompt_content}
            else:
                # Text string format (length may vary after tokenization)
                prompt = prompt_content
            
            # Create sampling params
            sampling_params = self._create_sampling_params(
                decode_tokens=decode_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            vllm_request = VLLMRequest(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=f"req_{idx}",
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                arrived_at=arrived_at,
            )
            
            vllm_requests.append(vllm_request)
        
        return vllm_requests
    
    def get_prompts_and_params(
        self,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> tuple[list[dict], list[SamplingParams]]:
        """Generate prompts and sampling params for vLLM.generate().
        
        Convenience method that returns separate lists for direct use with LLM.
        
        Args:
            temperature: Sampling temperature.
            top_p: Top-P sampling parameter.
            
        Returns:
            Tuple of (prompts, sampling_params) lists.
        """
        requests = self.generate(temperature=temperature, top_p=top_p)
        
        prompts = [r.prompt for r in requests]
        sampling_params = [r.sampling_params for r in requests]
        
        return prompts, sampling_params
    
    def get_workload_summary(self) -> dict:
        """Get summary statistics of the generated workload.
        
        Returns:
            Dict with workload statistics.
        """
        requests = self.generate()
        
        prefill_lens = [r.prefill_tokens for r in requests]
        decode_lens = [r.decode_tokens for r in requests]
        total_lens = [r.prefill_tokens + r.decode_tokens for r in requests]
        
        return {
            "num_requests": len(requests),
            "prefill_tokens": {
                "min": min(prefill_lens),
                "max": max(prefill_lens),
                "mean": sum(prefill_lens) / len(prefill_lens),
                "total": sum(prefill_lens),
            },
            "decode_tokens": {
                "min": min(decode_lens),
                "max": max(decode_lens),
                "mean": sum(decode_lens) / len(decode_lens),
                "total": sum(decode_lens),
            },
            "total_tokens": {
                "min": min(total_lens),
                "max": max(total_lens),
                "mean": sum(total_lens) / len(total_lens),
                "total": sum(total_lens),
            },
            "config": {
                "length_type": self.config.get_length_generator_type().value,
                "seed": self.config.seed,
            },
        }


def create_generator(
    num_requests: int = 10,
    prefill_tokens: int = 1024,
    decode_tokens: int = 128,
    seed: int = 42,
) -> VLLMRequestGenerator:
    """Convenience function to create a fixed-length request generator.
    
    Args:
        num_requests: Number of requests to generate.
        prefill_tokens: Input length for each request.
        decode_tokens: Output length for each request.
        seed: Random seed.
        
    Returns:
        Configured VLLMRequestGenerator.
    """
    config = RequestGeneratorConfig(
        num_requests=num_requests,
        length_config=FixedLengthConfig(
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        ),
        seed=seed,
    )
    return VLLMRequestGenerator(config)
