# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v41/index_selection/fp8/cache.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 index keys: int32 key pages and a scale row per page.

A page holds 128 keys. Keys are int32 [pages, 32, 128]: for token t and
feature d, byte d%4 of word [d//4, t] contains K[t, d]. Scales are int32
[pages, 128] holding FP32 bits, scales[p, t] for token t of page p. Both are
int32 rows of 128 lanes, whose tiled TPU layout is row-major, so SparseCore
reads them in place and no kernel converts them.
Page 0 is the null page, never written or read. Unwritten keys are masked by
sequence lengths.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

PAGE_RECORDS = 128
# A key's fields: 32 words of four FP8 features, then its FP32 scale.
KEY_ROWS = 32
PAGE_ROWS = KEY_ROWS + 1

# Stage up to 128 touched pages, stopping at the sorted invalid-record tail.
PACKED_FP8_WRITE_BATCH = 128


class FP8Cache(NamedTuple):
    """One layer's packed FP8 index keys."""

    keys: jax.Array  # int32 [pages, 32, 128]
    scales: jax.Array  # int32 [pages, 128], FP32 bits


def quantize_index_rows(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return E4M3 values, FP32 scales and E8M0 scale bytes for D128 vectors."""
    # One power-of-two scale covers each post-RoPE vector, independent of layout.
    if values.shape[-1] != 128:
        raise ValueError("V4.1 FP8 index vectors must have 128 channels")
    values = values.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(values).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scales = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    quantized = (values / scales[..., None]).astype(jnp.float8_e4m3fn)
    return quantized, scales, scale_bytes


@dataclass(frozen=True)
class PackedFP8Layout:
    """Pages of a packed FP8 cache; a page holds 128 keys."""

    pages: int
    page_size = PAGE_RECORDS
    # Rows of a page's gathered fields: the key rows, then the scales.
    page_rows = PAGE_ROWS

    @property
    def shapes(self) -> tuple[tuple[int, int, int], tuple[int, int]]:
        """The (keys, scales) shapes."""
        return (self.pages, KEY_ROWS, PAGE_RECORDS), (self.pages, PAGE_RECORDS)

    def zeros(self) -> FP8Cache:
        keys, scales = self.shapes
        return FP8Cache(jnp.zeros(keys, jnp.int32), jnp.zeros(scales, jnp.int32))

    @classmethod
    def of(cls, cache: FP8Cache) -> "PackedFP8Layout":
        """The layout of int32 keys [pages, 32, 128] and scales [pages, 128]."""
        keys, scales = cache
        layout = cls(keys.shape[0])
        if (
            keys.dtype != jnp.int32
            or scales.dtype != jnp.int32
            or (tuple(keys.shape), tuple(scales.shape)) != layout.shapes
            or layout.pages < 1
        ):
            raise ValueError(
                "Packed FP8 cache must be int32 keys [pages, 32, 128] and int32 "
                f"scales [pages, 128], got {keys.dtype} {tuple(keys.shape)} and "
                f"{scales.dtype} {tuple(scales.shape)}"
            )
        return layout


def pack_fp8_words(quantized: jax.Array, scales: jax.Array) -> jax.Array:
    """Int32 [N, 33] page fields of E4M3 [N, 128] keys and their FP32 scales."""
    bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(-1, KEY_ROWS, 4)
    words = bits[:, :, 0] | (bits[:, :, 1] << 8)
    words |= (bits[:, :, 2] << 16) | (bits[:, :, 3] << 24)
    scales = jax.lax.bitcast_convert_type(scales.reshape(-1, 1), jnp.int32)
    return jnp.concatenate((words.astype(jnp.int32), scales), axis=1)


def _record_addresses(records: jax.Array, layout: PackedFP8Layout):
    # Page 0 is the null page: its records are never written or read.
    valid = (records >= layout.page_size) & (records < layout.pages * layout.page_size)
    safe = jnp.clip(records, 0, layout.pages * layout.page_size - 1)
    return safe // layout.page_size, safe % layout.page_size, valid


def write_fp8_records(
    cache: FP8Cache, records: jax.Array, keys: jax.Array
) -> FP8Cache:
    """Quantize only new keys and update their columns; valid slots are unique."""
    if keys.shape != (records.size, PAGE_RECORDS) or records.ndim != 1:
        raise ValueError("Expected one logical record slot per D128 key")
    layout = PackedFP8Layout.of(cache)
    quantized, scales, _ = quantize_index_rows(keys)
    values = pack_fp8_words(quantized, scales)
    page, column, valid = _record_addresses(records, layout)
    if jax.default_backend() == "tpu":
        return write_packed_pages(cache, jnp.where(valid, page, 0), column, values)
    page = jnp.where(valid, page, layout.pages)
    rows = jnp.arange(KEY_ROWS)[None, :]
    return FP8Cache(
        cache.keys.at[page[:, None], rows, column[:, None]].set(
            values[:, :KEY_ROWS], mode="drop"
        ),
        cache.scales.at[page, column].set(values[:, KEY_ROWS], mode="drop"),
    )


def read_fp8_records(cache: FP8Cache, records: jax.Array) -> jax.Array:
    """Decode selected records to BF16 for eager consumers; invalid slots are zero."""
    layout = PackedFP8Layout.of(cache)
    shape = records.shape
    page, column, valid = _record_addresses(records.reshape(-1), layout)
    words = cache.keys[page[:, None], jnp.arange(KEY_ROWS)[None, :], column[:, None]]
    words = jnp.where(valid[:, None], words, 0)
    scales = jax.lax.bitcast_convert_type(cache.scales[page, column], jnp.float32)
    scales = jnp.where(valid, scales, 0)
    bits = (words[:, :, None].astype(jnp.uint32) >> (jnp.arange(4) * 8)) & 255
    values = jax.lax.bitcast_convert_type(
        bits.astype(jnp.uint8).reshape(-1, PAGE_RECORDS), jnp.float8_e4m3fn
    )
    result = (values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], result, 0).reshape(*shape, PAGE_RECORDS)


def write_packed_pages(
    cache: FP8Cache,
    pages: jax.Array,
    columns: jax.Array,
    values: jax.Array,
    *,
    interpret: bool = False,
) -> FP8Cache:
    """Write int32 page fields ``values`` [N, 33] at unique token slots.

    Row i goes to token ``columns[i]`` of page ``pages[i]``: fields 0-31 to the
    key rows, field 32 (FP32 bits) to the scales. Page 0 (the null page) and
    pages past the cache are skipped.
    """
    count, fields = values.shape
    if fields != PAGE_ROWS:
        raise ValueError(f"Expected {PAGE_ROWS} page fields, got {fields}")
    if not count:
        return cache
    layout = PackedFP8Layout.of(cache)
    lanes, end = PAGE_RECORDS, layout.pages
    batch = min(count, PACKED_FP8_WRITE_BATCH)
    valid = (pages > 0) & (pages < end)
    pages, order = jax.lax.sort_key_val(
        jnp.where(valid, pages, end), jnp.arange(count, dtype=jnp.int32)
    )
    columns, values = columns[order], values[order]
    # The final sentinel terminates the last page's merge without an end scan.
    pages = jnp.pad(pages, (0, -count % batch + 1), constant_values=end)
    columns = jnp.pad(columns, (0, -count % batch))
    values = jnp.pad(values.T, ((0, 0), (0, -count % lanes)))

    def kernel(page_ref, column_ref, value_ref, _keys, _scales, *refs):
        keys_ref, scales_ref, staged_keys, staged_scales, sem = refs

        def first(record):
            page = page_ref[record]
            return (page < end) & (
                (record == 0) | (page != page_ref[jnp.maximum(record - 1, 0)])
            )

        def each_valid(action):
            def body(slot):
                action(slot, 0)
                return slot + 1

            # Sorting puts invalid records last; decode writes only compressed
            # token boundaries, so most input rows need no page work.
            jax.lax.while_loop(
                lambda slot: (
                    (slot < batch) & (page_ref[pl.program_id(0) * batch + slot] < end)
                ),
                body,
                jnp.int32(0),
            )

        def copies(*, write, wait):
            def body(slot, carry):
                record = pl.program_id(0) * batch + slot

                @pl.when(first(record))
                def _():
                    page = page_ref[record]
                    for hbm, vmem in (
                        (keys_ref.at[page], staged_keys.at[slot]),
                        (scales_ref.at[pl.ds(page, 1)], staged_scales.at[slot]),
                    ):
                        src, dst = (vmem, hbm) if write else (hbm, vmem)
                        copy = pltpu.make_async_copy(src, dst, sem)
                        if wait:
                            copy.wait()
                        else:
                            copy.start()

                return carry

            each_valid(body)

        copies(write=False, wait=False)
        copies(write=False, wait=True)

        def merge(slot, carry):
            start = pl.program_id(0) * batch + slot

            @pl.when(first(start))
            def _():
                def update(record):
                    word = jnp.take_along_axis(
                        value_ref[:, pl.ds(record // lanes * lanes, lanes)],
                        jnp.full((PAGE_ROWS, lanes), record % lanes),
                        axis=1,
                        mode="fill",
                    )
                    hit = (
                        jax.lax.broadcasted_iota(jnp.int32, (PAGE_ROWS, lanes), 1)
                        == column_ref[record]
                    )
                    staged_keys[slot] = jnp.where(
                        hit[:KEY_ROWS], word[:KEY_ROWS], staged_keys[slot]
                    )
                    staged_scales[slot] = jnp.where(
                        hit[KEY_ROWS:], word[KEY_ROWS:], staged_scales[slot]
                    )
                    return record + 1

                jax.lax.while_loop(
                    lambda record: page_ref[record] == page_ref[start], update, start
                )

            return carry

        each_valid(merge)
        copies(write=True, wait=False)
        copies(write=True, wait=True)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    keys, scales = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(cache.keys.shape, cache.keys.dtype),
            jax.ShapeDtypeStruct(cache.scales.shape, cache.scales.dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(pl.cdiv(count, batch),),
            in_specs=(pl.BlockSpec(memory_space=pltpu.VMEM), hbm, hbm),
            out_specs=(hbm, hbm),
            scratch_shapes=(
                pltpu.VMEM((batch, KEY_ROWS, lanes), jnp.int32),
                pltpu.VMEM((batch, 1, lanes), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={3: 0, 4: 1},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="write_packed_fp8_pages",
    )(pages, columns, values, cache.keys, cache.scales)
    return FP8Cache(keys, scales)
