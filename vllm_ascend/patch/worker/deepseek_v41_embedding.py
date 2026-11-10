# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-sharded INT8_SYM Engram tables with E8M0 block scales.

The W8A8 checkpoint explicitly identifies these tables as INT8_SYM. They are
not the FP8 tables in the unquantized official inference example. Keep the
quantized rows resident and dequantize only the requested embeddings.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ShardedEngramEmbedding(nn.Module):
    SCALE_BLOCK_SIZE = 32
    E8M0_BIAS = 127
    LOAD_CHUNK_ROWS = 65536

    def __init__(self, num_embeddings, embedding_dim, *, rank, world_size, device, quant_type):
        super().__init__()
        if quant_type != "INT8_SYM":
            raise ValueError(f"Unsupported Engram table quantization: {quant_type}")
        if num_embeddings < 1 or embedding_dim < 1 or embedding_dim % self.SCALE_BLOCK_SIZE:
            raise ValueError("Engram dimensions require rows and complete 32-channel scale blocks")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid Engram partition rank or world size")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.world_size = world_size
        self.rows_per_rank = (num_embeddings + world_size - 1) // world_size
        self.row_start = rank * self.rows_per_rank
        self.row_end = min(self.row_start + self.rows_per_rank, num_embeddings)
        self.register_buffer("weight", torch.zeros(self.rows_per_rank, embedding_dim, dtype=torch.int8, device=device))
        self.register_buffer(
            "scale",
            torch.full(
                (self.rows_per_rank, embedding_dim // self.SCALE_BLOCK_SIZE),
                self.E8M0_BIAS,
                dtype=torch.uint8,
                device=device,
            ),
        )

    def load_rows(self, weights, scales, *, chunk_rows=LOAD_CHUNK_ROWS):
        """Load only this rank's rows from safetensors slices (or CPU tensors).

        Do not pass a fully materialized giant table to the ordinary weight
        iterator. Slicing here bounds host staging memory independently of the
        checkpoint size. Dtype/NaN checks run on the host during loading only.
        """
        weight_shape = weights.get_shape() if hasattr(weights, "get_shape") else weights.shape
        scale_shape = scales.get_shape() if hasattr(scales, "get_shape") else scales.shape
        if tuple(weight_shape) != (self.num_embeddings, self.embedding_dim):
            raise ValueError("Engram weight shape does not match the table")
        if tuple(scale_shape) != (self.num_embeddings, self.embedding_dim // self.SCALE_BLOCK_SIZE):
            raise ValueError("Engram scale shape does not match the table")
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        for start in range(self.row_start, self.row_end, chunk_rows):
            stop = min(start + chunk_rows, self.row_end)
            weight = weights[start:stop]
            scale = scales[start:stop]
            if weight.device.type != "cpu" or scale.device.type != "cpu":
                raise ValueError("Engram checkpoint slices must be staged on CPU")
            if weight.dtype != torch.int8:
                raise ValueError("INT8_SYM Engram weights must have int8 storage")
            if scale.dtype == getattr(torch, "float8_e8m0fnu", None):
                scale = scale.view(torch.uint8)
            elif scale.dtype != torch.uint8:
                raise ValueError("Engram scales must contain E8M0 bytes")
            if torch.any(scale == 255):
                raise ValueError("Engram scale contains E8M0 NaN")
            destination = slice(start - self.row_start, stop - self.row_start)
            self.weight[destination].copy_(weight)
            self.scale[destination].copy_(scale)

    def local_lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """This partition's contribution; off-rank and padded IDs return zero."""
        owned = (indices >= self.row_start) & (indices < self.row_end)
        local_ids = torch.where(owned, indices - self.row_start, 0)
        rows = F.embedding(local_ids, self.weight).float()
        exponents = F.embedding(local_ids, self.scale).float() - self.E8M0_BIAS
        values = rows.unflatten(-1, (-1, self.SCALE_BLOCK_SIZE)) * torch.exp2(exponents).unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        return values.masked_fill(~owned.unsqueeze(-1), 0)

    def forward(self, indices: torch.Tensor, group=None) -> torch.Tensor:
        """Lookup differing DP requests using the same EP-wide padded shape.

        Callers pad the leading token dimension to the common DP/graph bucket
        with -1 IDs. The collective ordering is identical on idle ranks too.
        GroupCoordinator gathers rank-major queries and reduces/scatters the
        matching row contributions back to their originating rank.
        """
        if self.world_size == 1:
            return self.local_lookup(indices)
        if group is None or group.world_size != self.world_size or group.rank_in_group != self.rank:
            raise ValueError("Engram lookup requires the same group used to partition the table")
        # Selecting one layer from [tokens, layers, hash_columns] leaves a
        # strided view. HCCL all_gather requires contiguous input storage.
        gathered = group.all_gather(indices.contiguous(), dim=0)
        values = self.local_lookup(gathered)
        return group.reduce_scatter(values, dim=0)
