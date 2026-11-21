# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v41/index_selection/fp8/cache.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-major FP8 pages with an in-page FP32 scale per key.

For token t and feature d, byte d%4 of word [d//4,t] contains K[t,d].
Field 32 holds scales[t] as FP32 bits. Each [33,page_size] page is rounded
to whole 128-word rows. A cache is uint8 [pages, page_rows, 4, 128]: byte b of
word [r, t] is element [r, b, t], the bytes of int32 [pages * page_rows, 128].
Page 0 is the null page, never written or read. Unwritten keys are masked by
sequence lengths.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

PAGE_RECORDS = 128
PAGE_ROWS = 33

# Stage up to 128 touched pages, stopping at the sorted invalid-record tail.
PACKED_FP8_WRITE_BATCH = 128


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
    """Pages of a packed FP8 cache."""

    pages: int
    page_size: int = PAGE_RECORDS

    @property
    def page_rows(self) -> int:
        return (PAGE_ROWS * self.page_size + PAGE_RECORDS - 1) // PAGE_RECORDS

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """The cache shape, uint8 [pages, page_rows, 4, 128]."""
        return (self.pages, self.page_rows, 4, PAGE_RECORDS)

    @classmethod
    def of(cls, cache) -> "PackedFP8Layout":
        """The layout of a uint8 [pages, page_rows, 4, 128] cache."""
        if (
            cache.dtype != jnp.uint8
            or len(cache.shape) != 4
            or tuple(cache.shape[2:]) != (4, PAGE_RECORDS)
        ):
            raise ValueError(
                f"Packed FP8 cache must be uint8 [pages, rows, 4, {PAGE_RECORDS}], "
                f"got {cache.dtype} {tuple(cache.shape)}"
            )
        pages, rows = cache.shape[:2]
        # Pages hold a power of two of keys, so the row count names one size.
        page_size = 1 << max(0, (rows * PAGE_RECORDS // PAGE_ROWS).bit_length() - 1)
        layout = cls(pages, page_size)
        if pages < 1 or layout.page_rows != rows:
            raise ValueError(f"{rows} rows is not a packed FP8 page")
        # The dense scorer's page table holds row << 1 in an int32.
        if pages * rows >= 1 << 30:
            raise ValueError("Packed FP8 page rows exceed the page table encoding")
        return layout


def cache_words(cache: jax.Array) -> jax.Array:
    """The cache as int32 words [pages * page_rows, 128]."""
    pages, rows, _, lanes = cache.shape
    words = jax.lax.bitcast_convert_type(jnp.swapaxes(cache, 2, 3), jnp.int32)
    return words.reshape(pages * rows, lanes)


def cache_from_words(words: jax.Array, layout: PackedFP8Layout) -> jax.Array:
    """The inverse of ``cache_words``."""
    pages, rows, _, lanes = layout.shape
    cache = jax.lax.bitcast_convert_type(words.reshape(pages, rows, lanes), jnp.uint8)
    return jnp.swapaxes(cache, 2, 3)


def cache_words_ref(cache_ref):
    """``cache_words`` of a kernel ref, in place."""
    pages, rows, packing, lanes = cache_ref.shape
    words = cache_ref.reshape(pages * rows, packing, lanes).bitcast(jnp.int32)
    return words.reshape(pages * rows, lanes)


def pack_fp8_words(quantized: jax.Array, scales: jax.Array) -> jax.Array:
    """Int32 [N, 33] page fields of E4M3 [N, 128] keys and their FP32 scales."""
    bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(-1, PAGE_ROWS - 1, 4)
    words = bits[:, :, 0] | (bits[:, :, 1] << 8)
    words |= (bits[:, :, 2] << 16) | (bits[:, :, 3] << 24)
    scales = jax.lax.bitcast_convert_type(scales.reshape(-1, 1), jnp.int32)
    return jnp.concatenate((words.astype(jnp.int32), scales), axis=1)


def _record_addresses(records: jax.Array, layout: PackedFP8Layout):
    # Page 0 is the null page: its records are never written or read.
    valid = (records >= layout.page_size) & (records < layout.pages * layout.page_size)
    safe = jnp.clip(records, 0, layout.pages * layout.page_size - 1)
    return safe // layout.page_size, safe % layout.page_size, valid


def _word_addresses(page: jax.Array, column: jax.Array, layout: PackedFP8Layout):
    """Flat int32 word addresses [N, 33] of each record's fields."""
    return (
        (page * layout.page_rows * PAGE_RECORDS)[:, None]
        + jnp.arange(PAGE_ROWS)[None, :] * layout.page_size
        + column[:, None]
    )


def write_fp8_records(
    cache: jax.Array, records: jax.Array, keys: jax.Array
) -> jax.Array:
    """Quantize only new keys and update their columns; valid slots are unique."""
    if keys.shape != (records.size, PAGE_RECORDS) or records.ndim != 1:
        raise ValueError("Expected one logical record slot per D128 key")
    layout = PackedFP8Layout.of(cache)
    quantized, scales, _ = quantize_index_rows(keys)
    values = pack_fp8_words(quantized, scales)
    page, column, valid = _record_addresses(records, layout)
    if jax.default_backend() == "tpu":
        return write_packed_pages(cache, jnp.where(valid, page, 0), column, values)
    words = cache_words(cache)
    addresses = _word_addresses(page, column, layout)
    rows = jnp.where(valid[:, None], addresses // PAGE_RECORDS, words.shape[0])
    words = words.at[rows, addresses % PAGE_RECORDS].set(values, mode="drop")
    return cache_from_words(words, layout)


def read_fp8_records(cache: jax.Array, records: jax.Array) -> jax.Array:
    """Decode selected records to BF16 for eager consumers; invalid slots are zero."""
    layout = PackedFP8Layout.of(cache)
    shape = records.shape
    page, column, valid = _record_addresses(records.reshape(-1), layout)
    addresses = _word_addresses(page, column, layout)
    words = cache_words(cache)[addresses // PAGE_RECORDS, addresses % PAGE_RECORDS]
    words = jnp.where(valid[:, None], words, 0)
    bits = (words[:, :32, None].astype(jnp.uint32) >> (jnp.arange(4) * 8)) & 255
    values = jax.lax.bitcast_convert_type(
        bits.astype(jnp.uint8).reshape(-1, PAGE_RECORDS), jnp.float8_e4m3fn
    )
    scales = jax.lax.bitcast_convert_type(words[:, 32], jnp.float32)
    result = (values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], result, 0).reshape(*shape, PAGE_RECORDS)


def write_packed_pages(
    cache: jax.Array,
    pages: jax.Array,
    columns: jax.Array,
    values: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Write int32 page fields ``values`` [N, fields] at unique token slots.

    Row i goes to column ``columns[i]`` of page ``pages[i]``. Page 0 (the null
    page) and pages past the cache are skipped.
    """
    count, fields = values.shape
    if not count:
        return cache
    layout = PackedFP8Layout.of(cache)
    page_tokens, page_size, end = PAGE_RECORDS, layout.page_size, layout.pages
    page_rows = pl.cdiv(fields * page_size, page_tokens)
    padded_fields = page_rows * page_tokens // page_size
    batch = min(count, PACKED_FP8_WRITE_BATCH)
    valid = (pages > 0) & (pages < end)
    pages, order = jax.lax.sort_key_val(
        jnp.where(valid, pages, end), jnp.arange(count, dtype=jnp.int32)
    )
    columns, values = columns[order], values[order]
    # The final sentinel terminates the last page's merge without an end scan.
    pages = jnp.pad(pages, (0, -count % batch + 1), constant_values=end)
    columns = jnp.pad(columns, (0, -count % batch))
    values = jnp.pad(values.T, ((0, padded_fields - fields), (0, -count % page_tokens)))

    def kernel(page_ref, column_ref, value_ref, _, out_ref, staged, sem):
        words = cache_words_ref(out_ref)

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
                    start = page_ref[record] * layout.page_rows
                    hbm = words.at[pl.ds(start, page_rows), :]
                    vmem = staged.at[slot]
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
                        value_ref[
                            :, pl.ds(record // page_tokens * page_tokens, page_tokens)
                        ],
                        jnp.full((padded_fields, page_tokens), record % page_tokens),
                        axis=1,
                        mode="fill",
                    )
                    if page_size < page_tokens:
                        word = word[:, :page_size].reshape(page_rows, page_tokens)
                    elif page_size > page_tokens:
                        word = jnp.tile(word, (1, page_size // page_tokens)).reshape(
                            page_rows, page_tokens
                        )
                    positions = jnp.arange(page_rows * page_tokens).reshape(
                        page_rows, page_tokens
                    )
                    staged[slot] = jnp.where(
                        (positions % page_size == column_ref[record])
                        & (positions < fields * page_size),
                        word,
                        staged[slot],
                    )
                    return record + 1

                jax.lax.while_loop(
                    lambda record: page_ref[record] == page_ref[start], update, start
                )

            return carry

        each_valid(merge)
        copies(write=True, wait=False)
        copies(write=True, wait=True)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(pl.cdiv(count, batch),),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((batch, page_rows, page_tokens), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={3: 0},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="write_packed_fp8_pages",
    )(pages, columns, values, cache)
