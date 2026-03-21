# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group
from vllm.v1.utils import (record_frontier_op_meta,
                           record_function_or_nullcontext,
                           should_record_frontier_op_meta)


def tensor_model_parallel_all_reduce(
        input_: torch.Tensor,
        record_scope_name: Optional[str] = "tensor_parallel_allreduce",
) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    tp_group = get_tp_group()
    if record_scope_name and tp_group.world_size > 1:
        with record_function_or_nullcontext(record_scope_name):
            if should_record_frontier_op_meta(record_scope_name):
                record_frontier_op_meta(
                    record_scope_name,
                    {
                        "collective_base_op_name": record_scope_name,
                        "collective_domain": "TP",
                        "collective_group_unique_name": tp_group.unique_name,
                        "collective_rank_in_group": tp_group.rank_in_group,
                        "collective_world_size": tp_group.world_size,
                    },
                )
            return tp_group.all_reduce(input_)
    return tp_group.all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(input_: torch.Tensor,
                                         dim: int = -1) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
