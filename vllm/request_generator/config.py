# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration classes for vLLM Request Generator.

These configs wrap Frontier's request generator configuration system
and provide a simpler interface for common use cases.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LengthGeneratorType(str, Enum):
    """Type of request length generator."""
    FIXED = "fixed"
    UNIFORM = "uniform"
    ZIPF = "zipf"


@dataclass
class FixedLengthConfig:
    """Configuration for fixed-length request generation.
    
    All requests will have the same input (prefill) and output (decode) lengths.
    """
    prefill_tokens: int = 1024
    """Number of input tokens for each request."""
    
    decode_tokens: int = 128
    """Number of output tokens for each request."""
    
    def __post_init__(self):
        if self.prefill_tokens < 1:
            raise ValueError(f"prefill_tokens must be >= 1, got {self.prefill_tokens}")
        if self.decode_tokens < 1:
            raise ValueError(f"decode_tokens must be >= 1, got {self.decode_tokens}")


@dataclass
class UniformLengthConfig:
    """Configuration for uniform-distribution length generation.
    
    Total tokens are sampled uniformly from [min_tokens, max_tokens],
    then split according to prefill_to_decode_ratio.
    """
    min_tokens: int = 512
    """Minimum total tokens (prefill + decode)."""
    
    max_tokens: int = 4096
    """Maximum total tokens (prefill + decode)."""
    
    prefill_to_decode_ratio: float = 8.0
    """Ratio of prefill tokens to decode tokens. E.g., 8.0 means ~8:1."""
    
    def __post_init__(self):
        if self.min_tokens < 2:
            raise ValueError(f"min_tokens must be >= 2, got {self.min_tokens}")
        if self.max_tokens < self.min_tokens:
            raise ValueError(f"max_tokens ({self.max_tokens}) must be >= min_tokens ({self.min_tokens})")
        if self.prefill_to_decode_ratio <= 0:
            raise ValueError(f"prefill_to_decode_ratio must be > 0, got {self.prefill_to_decode_ratio}")


@dataclass
class ZipfLengthConfig:
    """Configuration for Zipf-distribution length generation.
    
    Request lengths follow a Zipf distribution, commonly seen in real workloads.
    """
    theta: float = 0.6
    """Zipf distribution parameter (skewness)."""
    
    min_tokens: int = 512
    """Minimum total tokens."""
    
    max_tokens: int = 4096
    """Maximum total tokens."""
    
    prefill_to_decode_ratio: float = 8.0
    """Ratio of prefill tokens to decode tokens."""
    
    scramble: bool = False
    """Whether to scramble the Zipf distribution."""
    
    def __post_init__(self):
        if self.theta <= 0 or self.theta >= 1:
            raise ValueError(f"theta must be in (0, 1), got {self.theta}")
        if self.min_tokens < 2:
            raise ValueError(f"min_tokens must be >= 2, got {self.min_tokens}")
        if self.max_tokens < self.min_tokens:
            raise ValueError(f"max_tokens ({self.max_tokens}) must be >= min_tokens ({self.min_tokens})")


@dataclass
class RequestGeneratorConfig:
    """Main configuration for the vLLM Request Generator.
    
    Example usage:
        # Fixed length requests with token IDs (precise control)
        config = RequestGeneratorConfig(
            num_requests=100,
            length_config=FixedLengthConfig(prefill_tokens=2048, decode_tokens=256),
            use_token_ids=True,  # Default: precise length control
        )
        
        # Uniform distribution with text prompts
        config = RequestGeneratorConfig(
            num_requests=50,
            length_config=UniformLengthConfig(min_tokens=512, max_tokens=8192),
            use_token_ids=False,  # Use text prompts (length may vary)
        )
    """
    
    num_requests: int = 10
    """Number of requests to generate."""
    
    length_config: FixedLengthConfig | UniformLengthConfig | ZipfLengthConfig = field(
        default_factory=FixedLengthConfig
    )
    """Length distribution configuration."""
    
    seed: int = 42
    """Random seed for reproducibility."""
    
    # Prompt generation mode
    use_token_ids: bool = True
    """If True, generate token ID lists for precise length control (recommended).
    If False, generate text prompts (actual token length may vary after tokenization)."""
    
    # Token ID generation parameters (only used when use_token_ids=True)
    vocab_size: int = 128256
    """Model vocabulary size for token ID generation. Default is Llama-3 vocab size."""
    
    min_token_id: int = 1000
    """Minimum token ID to use (to avoid special tokens like BOS/EOS/PAD)."""
    
    max_token_id: Optional[int] = None
    """Maximum token ID to use. Defaults to vocab_size - 1."""
    
    # Output length control
    force_exact_output_length: bool = True
    """If True, force model to generate exact decode_tokens length by:
    - Setting ignore_eos=True (ignore EOS token)
    - Setting stop=None (no stop strings)
    - Setting stop_token_ids=None (no stop token IDs)
    This ensures the model generates the full specified output length.
    If False, model may stop early at EOS or stop tokens."""
    
    # Frontier project path (can be overridden via environment variable)
    frontier_path: Optional[str] = None
    """Path to Frontier project root. Defaults to env var FRONTIER_PATH or hardcoded path."""
    
    def __post_init__(self):
        if self.num_requests < 1:
            raise ValueError(f"num_requests must be >= 1, got {self.num_requests}")
    
    def get_length_generator_type(self) -> LengthGeneratorType:
        """Return the type of length generator based on length_config."""
        if isinstance(self.length_config, FixedLengthConfig):
            return LengthGeneratorType.FIXED
        elif isinstance(self.length_config, UniformLengthConfig):
            return LengthGeneratorType.UNIFORM
        elif isinstance(self.length_config, ZipfLengthConfig):
            return LengthGeneratorType.ZIPF
        else:
            raise ValueError(f"Unknown length config type: {type(self.length_config)}")
