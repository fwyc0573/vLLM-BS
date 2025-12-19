# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
import torch
from vllm import LLM, SamplingParams
from vllm.request_generator import (FixedLengthConfig, RequestGeneratorConfig,
                                    VLLMRequestGenerator)

_DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline monolithic profiling using request generator.")
    parser.add_argument("--num-requests",
                        type=int,
                        default=128,
                        help="Number of requests to issue.")
    parser.add_argument("--prefill-tokens",
                        type=int,
                        default=512,
                        help="Prefill tokens per request.")
    parser.add_argument("--decode-tokens",
                        type=int,
                        default=2,
                        help="Decode tokens per request.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Base seed for deterministic request generation.")
    parser.add_argument("--warmup-iters",
                        type=int,
                        default=3,
                        help="Number of warm-up iterations to run.")
    parser.add_argument("--model",
                        type=str,
                        default=_DEFAULT_MODEL,
                        help="Model name or path.")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.7,
                        help="GPU memory utilization ratio.")
    parser.add_argument("--profile",
                        action="store_true",
                        help="Enable torch profiler (requires env dir set).")
    parser.add_argument(
        "--profile-max-decode-tokens",
        type=int,
        default=256,
        help=("Cap decode tokens during profiling to bound trace size. "
              "Must be >= decode_tokens to capture full decode."),
    )
    return parser.parse_args()


def _build_prompts(num_requests: int, prefill_tokens: int, decode_tokens: int,
                   seed: int, tokenizer):
    cfg = RequestGeneratorConfig(
        num_requests=num_requests,
        length_config=FixedLengthConfig(prefill_tokens=prefill_tokens,
                                        decode_tokens=decode_tokens),
        seed=seed,
        # IMPORTANT: Do not generate random token IDs with a hard-coded vocab.
        # Token-ID mode defaults to a Llama-3 sized vocab (128256) and will
        # produce out-of-vocabulary token IDs for models with smaller vocabs.
        # Instead, generate text prompts and let vLLM tokenize with the
        # model's real tokenizer to guarantee validity across models.
        use_token_ids=False,
    )
    generator = VLLMRequestGenerator(cfg, tokenizer=tokenizer)
    requests = generator.generate()
    return [req.prompt for req in requests]


def _validate_profile_env(enable_profile: bool) -> None:
    if enable_profile and not os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        raise RuntimeError(
            "Profiling requested but VLLM_TORCH_PROFILER_DIR is not set. "
            "Please set it before running to avoid dropping traces.")


def main():
    args = _parse_args()
    _validate_profile_env(args.profile)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=min(args.decode_tokens, args.profile_max_decode_tokens
                       if args.profile else args.decode_tokens),
        ignore_eos=True,
    )

    # Explicitly increase max_model_len to accommodate prefill + decode.
    # This script generates fixed-length prompts for profiling; if prefill
    # consumes the original max_model_len, vLLM will fail during decode.
    required_max_model_len = args.prefill_tokens + sampling_params.max_tokens

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=required_max_model_len,
    )

    # Validate that the override took effect.
    try:
        max_model_len = llm.llm_engine.model_config.max_model_len  # type: ignore[attr-defined]
    except AttributeError as e:
        raise RuntimeError(
            "Failed to read max_model_len from the vLLM engine. "
            "Cannot validate the max_model_len override."
        ) from e
    if max_model_len < required_max_model_len:
        raise RuntimeError(
            "max_model_len override did not take effect. "
            f"required_max_model_len={required_max_model_len}, "
            f"engine_max_model_len={max_model_len}."
        )

    tokenizer = llm.get_tokenizer()
    warmup_prompts = _build_prompts(args.num_requests, args.prefill_tokens,
                                    args.decode_tokens, args.seed + 1,
                                    tokenizer)
    prompts = _build_prompts(args.num_requests, args.prefill_tokens,
                             args.decode_tokens, args.seed, tokenizer)

    for _ in range(args.warmup_iters):
        llm.generate(warmup_prompts, sampling_params)

    if args.profile:
        llm.start_profile()

    wall_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    if args.profile:
        llm.stop_profile()

    print("-" * 50)
    print(f"Total generation wall-clock time: {wall_end - wall_start:.6f} s")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Allow profiler writes to flush when multiprocessing is enabled elsewhere.
    time.sleep(10)


if __name__ == "__main__":
    main()
