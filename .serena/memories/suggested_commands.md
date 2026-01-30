# Suggested Commands (vLLM)

## Lint/Format
- `pip install -r requirements/lint.txt`
- `pre-commit install`
- `pre-commit run --all-files`

## Tests
- `pytest tests`

## Entrypoints
- `python -m vllm.entrypoints.openai.api_server --help`
- `python -m vllm.entrypoints.api_server --help`