#!/usr/bin/env python3
"""
Test configuration for profiling overhead validation experiments.

This module defines all test configurations, model parameters, and
experimental settings for the overhead validation study.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ScopeConfig(Enum):
    """Scope instrumentation configurations for Phase 3 experiments."""
    FULL = "full"           # All scopes enabled
    MODEL_ONLY = "model"    # Only model layer scopes (no runner/backend)
    ATTN_ONLY = "attn"      # Only attention-related scopes
    MLP_ONLY = "mlp"        # Only MLP-related scopes
    NONE = "none"           # No custom scopes (baseline)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str
    hf_id: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    estimated_vram_gb: float
    
    @property
    def estimated_scopes_full(self) -> int:
        """Estimate total scope count for full instrumentation."""
        # 5 runner scopes + 10 per-layer model scopes + 2 per-layer backend scopes
        return 5 + self.num_layers * 12
    
    @property
    def estimated_scopes_model_only(self) -> int:
        """Estimate scope count for model-only instrumentation."""
        return self.num_layers * 10
    
    @property
    def estimated_scopes_attn_only(self) -> int:
        """Estimate scope count for attention-only instrumentation."""
        # attn_pre_proj, attn_rope, attn, attn_post_proj, input_layernorm
        return self.num_layers * 5
    
    @property
    def estimated_scopes_mlp_only(self) -> int:
        """Estimate scope count for MLP-only instrumentation."""
        # mlp_up_proj, mlp_act, mlp_down_proj
        return self.num_layers * 3


@dataclass
class TestCase:
    """Single test case configuration."""
    test_id: str
    model: str
    batch_size: int
    seq_len: int
    profiling_enabled: bool
    scope_config: ScopeConfig = ScopeConfig.FULL
    num_runs: int = 5
    warmup_runs: int = 1
    priority: str = "MEDIUM"
    phase: int = 1


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# NOTE: Model configurations updated to use locally cached models.
# If a model fails to load, check HF cache or authentication.

MODELS = {
    # Llama-3.2-1B-Instruct (16 layers)
    # Cached at: /local/ytyang/chenyuetao/cache/hub/
    "1B": ModelConfig(
        name="Llama-3.2-1B",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        num_layers=16,
        hidden_size=2048,
        num_heads=32,
        num_kv_heads=8,
        intermediate_size=8192,
        estimated_vram_gb=2.0,
    ),
    # Llama-2-7B (32 layers) - Use instead of 3B for model size variation
    # This model is locally cached and doesn't require authentication
    "7B": ModelConfig(
        name="Llama-2-7B",
        hf_id="meta-llama/Llama-2-7b-hf",
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=32,  # Llama-2 uses MHA not GQA
        intermediate_size=11008,
        estimated_vram_gb=14.0,
    ),
    # Meta-Llama-3-8B (32 layers) - Base model, locally cached
    "8B": ModelConfig(
        name="Meta-Llama-3-8B",
        hf_id="meta-llama/Meta-Llama-3-8B",
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        estimated_vram_gb=16.0,
    ),
}


# ============================================================================
# BATCH/SEQUENCE CONFIGURATIONS
# ============================================================================

BATCH_SEQ_CONFIGS = [
    {"batch": 1, "seq_len": 512},
    {"batch": 1, "seq_len": 1024},
    {"batch": 1, "seq_len": 2048},
    {"batch": 4, "seq_len": 512},
    {"batch": 4, "seq_len": 1024},
    {"batch": 4, "seq_len": 2048},
    {"batch": 8, "seq_len": 512},
    {"batch": 8, "seq_len": 1024},
]


# ============================================================================
# EXPERIMENTAL SETTINGS
# ============================================================================

EXPERIMENT_SETTINGS = {
    "runs_per_config": 5,
    "warmup_runs": 1,
    "gpu_cooldown_seconds": 5,
    "max_tokens_generated": 1,  # Prefill-focused
    "gpu_memory_utilization": 0.8,
    "enforce_eager": True,
}


# ============================================================================
# PHASE 1: MODEL SIZE VARIATION
# ============================================================================

def generate_phase1_tests() -> list[TestCase]:
    """Generate Phase 1 test cases (model size variation)."""
    tests = []
    test_idx = 1
    
    for model_key in ["1B", "3B", "8B"]:
        # With profiling
        tests.append(TestCase(
            test_id=f"P1-{test_idx}A",
            model=model_key,
            batch_size=4,
            seq_len=1024,
            profiling_enabled=True,
            scope_config=ScopeConfig.FULL,
            priority="HIGH",
            phase=1,
        ))
        # Without profiling
        tests.append(TestCase(
            test_id=f"P1-{test_idx}B",
            model=model_key,
            batch_size=4,
            seq_len=1024,
            profiling_enabled=False,
            scope_config=ScopeConfig.NONE,
            priority="HIGH",
            phase=1,
        ))
        test_idx += 1
    
    return tests


# ============================================================================
# PHASE 2: BATCH/SEQUENCE VARIATION
# ============================================================================

def generate_phase2_tests() -> list[TestCase]:
    """Generate Phase 2 test cases (batch/sequence variation)."""
    tests = []
    test_idx = 1
    
    # Use 1B model for batch/seq variation (fastest iteration)
    selected_configs = [
        {"batch": 1, "seq_len": 512},
        {"batch": 1, "seq_len": 2048},
        {"batch": 8, "seq_len": 512},
        {"batch": 8, "seq_len": 1024},
    ]
    
    for config in selected_configs:
        # With profiling
        tests.append(TestCase(
            test_id=f"P2-{test_idx}A",
            model="1B",
            batch_size=config["batch"],
            seq_len=config["seq_len"],
            profiling_enabled=True,
            scope_config=ScopeConfig.FULL,
            priority="MEDIUM",
            phase=2,
        ))
        # Without profiling
        tests.append(TestCase(
            test_id=f"P2-{test_idx}B",
            model="1B",
            batch_size=config["batch"],
            seq_len=config["seq_len"],
            profiling_enabled=False,
            scope_config=ScopeConfig.NONE,
            priority="MEDIUM",
            phase=2,
        ))
        test_idx += 1
    
    return tests


# ============================================================================
# PHASE 3: SCOPE COUNT ISOLATION
# ============================================================================

def generate_phase3_tests() -> list[TestCase]:
    """Generate Phase 3 test cases (scope count isolation)."""
    tests = []
    
    scope_configs = [
        (ScopeConfig.FULL, "A"),
        (ScopeConfig.MODEL_ONLY, "B"),
        (ScopeConfig.ATTN_ONLY, "C"),
        (ScopeConfig.MLP_ONLY, "D"),
        (ScopeConfig.NONE, "E"),
    ]
    
    for scope_config, suffix in scope_configs:
        profiling = scope_config != ScopeConfig.NONE
        tests.append(TestCase(
            test_id=f"P3-1{suffix}",
            model="1B",
            batch_size=4,
            seq_len=1024,
            profiling_enabled=profiling,
            scope_config=scope_config,
            priority="HIGH",
            phase=3,
        ))
    
    return tests


# ============================================================================
# ALL TESTS
# ============================================================================

def get_all_tests() -> list[TestCase]:
    """Get all test cases for all phases."""
    return generate_phase1_tests() + generate_phase2_tests() + generate_phase3_tests()


def get_tests_by_phase(phase: int) -> list[TestCase]:
    """Get test cases for a specific phase."""
    return [t for t in get_all_tests() if t.phase == phase]


def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key."""
    return MODELS[model_key]


# ============================================================================
# SCOPE COUNT ESTIMATION
# ============================================================================

def estimate_scope_count(model_key: str, scope_config: ScopeConfig) -> int:
    """Estimate the number of scope calls for a configuration."""
    model = MODELS[model_key]
    
    if scope_config == ScopeConfig.FULL:
        return model.estimated_scopes_full
    elif scope_config == ScopeConfig.MODEL_ONLY:
        return model.estimated_scopes_model_only
    elif scope_config == ScopeConfig.ATTN_ONLY:
        return model.estimated_scopes_attn_only
    elif scope_config == ScopeConfig.MLP_ONLY:
        return model.estimated_scopes_mlp_only
    else:  # NONE
        return 0


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PROFILING OVERHEAD VALIDATION - TEST CONFIGURATION")
    print("=" * 80)
    
    print("\n## Model Configurations ##")
    for key, model in MODELS.items():
        print(f"\n{key}: {model.name}")
        print(f"  - Layers: {model.num_layers}")
        print(f"  - Hidden: {model.hidden_size}")
        print(f"  - Est. VRAM: {model.estimated_vram_gb} GB")
        print(f"  - Est. Scopes (Full): {model.estimated_scopes_full}")
    
    print("\n" + "=" * 80)
    print("## Test Cases Summary ##")
    
    for phase in [1, 2, 3]:
        tests = get_tests_by_phase(phase)
        print(f"\nPhase {phase}: {len(tests)} tests")
        for t in tests:
            model = MODELS[t.model]
            scopes = estimate_scope_count(t.model, t.scope_config)
            print(f"  {t.test_id}: {model.name}, BS={t.batch_size}, Seq={t.seq_len}, "
                  f"Prof={'ON' if t.profiling_enabled else 'OFF'}, Scopes~{scopes}")
    
    total_tests = len(get_all_tests())
    total_runs = sum(t.num_runs for t in get_all_tests())
    print(f"\n## Total: {total_tests} test cases, {total_runs} runs ##")
