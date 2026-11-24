# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 index-key cache geometry, shared by the kernels and KV pools.

A cache is uint8 [pages, KEY_FIELDS, WORD_BYTES, page_size]: byte b of field f
of the key in slot t is cache[page, f, b, t]. Fields 0-31 hold channels
4f..4f+3 as E4M3; field 32 holds the key's FP32 scale, little-endian. The TPU
tiles each (4, 128) byte group as one int32 row, so a kernel views the cache as
int32 [pages * page_rows, PAGE_RECORDS] without a copy; field f of 128-slot
part h of a page is its row f * (page_size // PAGE_RECORDS) + h.

Pure Python, so KV pools can size their buffers without importing JAX.
"""

KEY_CHANNELS = 128
WORD_BYTES = 4
# 32 words of four E4M3 channels, then the FP32 scale word.
KEY_FIELDS = KEY_CHANNELS // WORD_BYTES + 1
# Slots per int32 row; a page holds a whole number of rows per field.
PAGE_RECORDS = 128


def cache_shape(pages: int, page_size: int = PAGE_RECORDS) -> tuple[int, ...]:
    """uint8 shape of a cache with `pages` pages of `page_size` slots."""
    return _checked((pages, KEY_FIELDS, WORD_BYTES, page_size))


def cache_geometry(shape) -> tuple[int, int]:
    """(pages, page_size) of a cache shape; rejects any other shape."""
    pages, _, _, page_size = _checked(tuple(shape))
    return pages, page_size


def page_rows(page_size: int) -> int:
    """int32 rows of one page in the kernels' word view."""
    return KEY_FIELDS * page_size // PAGE_RECORDS


def _checked(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != 4 or shape[1:3] != (KEY_FIELDS, WORD_BYTES) or shape[0] < 1:
        raise ValueError(
            f"Packed FP8 cache must be uint8 [pages, {KEY_FIELDS}, {WORD_BYTES}, "
            f"page_size], got {shape}"
        )
    page_size = shape[3]
    if page_size < PAGE_RECORDS or page_size % PAGE_RECORDS:
        raise ValueError(
            f"Packed FP8 page size must be a multiple of {PAGE_RECORDS}, got {page_size}"
        )
    # Kernels address words, and SparseCore 8-word fragments, as int32.
    if shape[0] * page_rows(page_size) * PAGE_RECORDS >= 1 << 31:
        raise ValueError("Packed FP8 cache word addresses exceed int32")
    return shape
