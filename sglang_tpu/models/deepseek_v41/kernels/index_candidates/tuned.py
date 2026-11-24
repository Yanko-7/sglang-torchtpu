# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU v7 tiles and pipeline depths for DeepSeek-V4.1 candidate selection."""

from sglang_tpu.kernels.index_selection.fp8.tuned import PagedMQAConfig

PACKED_FP8_PREPARED_MQA_CONFIG = PagedMQAConfig(2, 32768, 12, 6)
# Query pairs per gathered-scorer iteration; they share one MXU issue window.
PACKED_FP8_PREPARED_PAIRS = 2

# v7 H32/D128: fully static prefixes avoid dynamic per-fragment addresses.
# A 4K step bounds inactive computation while retaining a long scheduling window.
PACKED_FP8_PREPARED_PREFIX_TOKENS = 4096


# Rolling SparseCore gather: three buffers overlap reads and writes for 8-token blocks.
# 64 pages per tile divides tokens * 2 * 2048 candidate blocks over 32 subcores, so the
# gather output needs no trailing slice; 96 left a sliced 8.25 GiB copy at 2048 tokens.
PAGED_MQA_GATHER_CONFIG = {8: (64, 3)}

# (generation, queries, token-score width, block size): (rows, output blocks).
PUBLISH_BLOCK_MAX_CONFIG = {(7, 96, 98816, 8): (96, 1024)}
