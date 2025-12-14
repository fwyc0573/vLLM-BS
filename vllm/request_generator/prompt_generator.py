# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prompt Generator for vLLM Request Generator.

Generates prompts of specific token lengths for testing purposes.
"""

from typing import Optional, Union
import random


class PromptGenerator:
    """Generates prompts with precise token lengths.
    
    Two strategies are supported:
    1. Token IDs: Generate random token IDs for exact length control
    2. Filler text: Repeat simple tokens to approximate target length
    
    For precise control, use token IDs mode which bypasses tokenization.
    """
    
    # Common tokens with predictable lengths (for filler mode)
    FILLER_TOKENS = ["Hi ", "Hey ", "The ", "And ", "But ", "For ", "So "]
    
    def __init__(
        self,
        tokenizer=None,
        vocab_size: int = 128256,  # Llama-3 vocab size
        seed: int = 42,
        use_token_ids: bool = True,
        # Token ID range for generation (avoid special tokens)
        min_token_id: int = 1000,
        max_token_id: Optional[int] = None,
    ):
        """Initialize the prompt generator.
        
        Args:
            tokenizer: Optional tokenizer for text-based generation.
            vocab_size: Model vocabulary size for token ID generation.
            seed: Random seed for reproducibility.
            use_token_ids: If True, generate token IDs; if False, generate text.
            min_token_id: Minimum token ID to use (to avoid special tokens).
            max_token_id: Maximum token ID to use. Defaults to vocab_size - 1.
        """
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.seed = seed
        self.use_token_ids = use_token_ids
        self.min_token_id = min_token_id
        self.max_token_id = max_token_id or (vocab_size - 1)
        
        self._rng = random.Random(seed)
        
    def generate_prompt_token_ids(self, target_length: int) -> list[int]:
        """Generate a list of token IDs with exact target length.
        
        Args:
            target_length: Desired number of tokens.
            
        Returns:
            List of token IDs with exactly target_length elements.
        """
        if target_length < 1:
            raise ValueError(f"target_length must be >= 1, got {target_length}")
            
        return [
            self._rng.randint(self.min_token_id, self.max_token_id)
            for _ in range(target_length)
        ]
    
    def generate_prompt_text(self, target_length: int) -> str:
        """Generate text prompt aiming for approximately target_length tokens.
        
        Note: Actual tokenized length may vary. For precise control, use token IDs.
        
        Args:
            target_length: Approximate desired token count.
            
        Returns:
            Generated text string.
        """
        if self.tokenizer is None:
            # Fallback: use filler tokens, roughly 1 token per word
            filler = self._rng.choice(self.FILLER_TOKENS)
            return filler * target_length
        
        # With tokenizer, iteratively build prompt
        prompt = ""
        filler = self._rng.choice(self.FILLER_TOKENS)
        
        # Start with estimated repetitions
        estimated_reps = target_length
        prompt = filler * estimated_reps
        
        # Refine
        current_len = len(self.tokenizer.encode(prompt))
        while current_len < target_length:
            prompt += filler
            current_len = len(self.tokenizer.encode(prompt))
        
        # Truncate if too long
        while current_len > target_length:
            prompt = prompt[:-len(filler)]
            current_len = len(self.tokenizer.encode(prompt))
        
        return prompt
    
    def generate(
        self, 
        target_length: int,
    ) -> Union[list[int], str]:
        """Generate a prompt with target token length.
        
        Args:
            target_length: Desired number of tokens.
            
        Returns:
            Token IDs list (if use_token_ids=True) or text string.
        """
        if self.use_token_ids:
            return self.generate_prompt_token_ids(target_length)
        else:
            return self.generate_prompt_text(target_length)
    
    def reset_seed(self, seed: Optional[int] = None):
        """Reset the random number generator with a new seed.
        
        Args:
            seed: New seed. If None, uses the original seed.
        """
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
