# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 index-key page geometry, shared by the kernels and KV pools.

Pure Python, so KV pools can size their buffers without importing JAX.
"""

from dataclasses import dataclass

PAGE_RECORDS = 128
PAGE_ROWS = 33
# int32 arrays are stored in tiles of 8 rows. A [P, rows, 128] cache stays
# page-major only when rows is a multiple of 8; with 33 rows XLA lays the pages
# out across tiles and copies the whole cache before a kernel can read a page.
ROW_TILE = 8


def padded_page_rows(page_size: int) -> int:
    """int32 rows one page occupies: its fields rounded up to whole row tiles."""
    rows = (PAGE_ROWS * page_size + PAGE_RECORDS - 1) // PAGE_RECORDS
    return -(-rows // ROW_TILE) * ROW_TILE


@dataclass(frozen=True, init=False)
class PackedFP8Layout:
    """Location of this layer's pages inside a shared word-row cache."""

    pages: int
    page_stride: int
    page_offset: int
    page_size: int

    def __init__(
        self,
        pages: int,
        page_stride: int | None = None,
        page_offset: int = 0,
        page_size: int = PAGE_RECORDS,
    ) -> None:
        stride = padded_page_rows(page_size)
        for name, value in (
            ("pages", pages),
            ("page_stride", stride if page_stride is None else page_stride),
            ("page_offset", page_offset),
            ("page_size", page_size),
        ):
            object.__setattr__(self, name, value)

    @property
    def page_rows(self) -> int:
        return (PAGE_ROWS * self.page_size + PAGE_RECORDS - 1) // PAGE_RECORDS

    @property
    def shape(self) -> tuple[int, int, int]:
        """int32 shape of an owner that holds exactly these pages from row 0."""
        return self.pages, self.page_stride, PAGE_RECORDS

    def validate(self, rows: int) -> None:
        if self.page_size < 1 or self.page_size & (self.page_size - 1):
            raise ValueError("Packed FP8 page capacity must be a power of two")
        if self.pages < 1 or self.page_stride < self.page_rows or self.page_offset < 0:
            raise ValueError("Packed FP8 pages require nonoverlapping page slots")
        end = self.page_offset + (self.pages - 1) * self.page_stride + self.page_rows
        if end > rows or rows >= 1 << 29:
            raise ValueError("Packed FP8 page addresses exceed the cache")
