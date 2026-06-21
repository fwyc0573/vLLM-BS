"""Load the installed vLLM MoE extension for source-tree runs."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


_PACKAGE_ENV = "VLLM_FRONTIER_COMPILED_PACKAGE"


def _required_extension_path() -> Path:
    package_dir = os.environ.get(_PACKAGE_ENV)
    if not package_dir:
        raise ImportError(f"{_PACKAGE_ENV} must point to the installed vLLM package")
    extension_path = Path(package_dir) / "_moe_C.abi3.so"
    if not extension_path.is_file():
        raise ImportError(f"Required vLLM MoE extension not found: {extension_path}")
    return extension_path


_extension_path = _required_extension_path()
_spec = importlib.util.spec_from_file_location(__name__, _extension_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load vLLM MoE extension: {_extension_path}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
