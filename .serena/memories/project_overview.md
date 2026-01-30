# vLLM Project Overview

## Purpose
vLLM is a fast, easy-to-use library for LLM inference and serving, with optimized attention (PagedAttention), efficient batching, and support for multiple model families and parallelism modes.

## Tech Stack
- Python (PyTorch)
- CUDA/HIP custom ops (csrc/)
- FastAPI-based API servers (entrypoints)

## Codebase Structure
- `vllm/`: core Python implementation (engine, model executor, entrypoints)
- `csrc/`: CUDA/C++ kernels and custom ops
- `tests/`: test suites
- `docs/`: documentation
- `benchmarks/`, `examples/`, `tools/`

## Entrypoints
- `vllm/entrypoints/openai/api_server.py`: OpenAI-compatible server
- `vllm/entrypoints/api_server.py`: demo API server (not for production)
- `vllm/entrypoints/cli/`: CLI entrypoints