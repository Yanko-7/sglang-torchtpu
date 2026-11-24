# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write and read packed FP8 index keys with an in-page FP32 scale per key.

The cache layout is described in pages.py. Unwritten keys are masked by
sequence lengths.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels.index_selection.fp8.pages import (
    KEY_CHANNELS,
    KEY_FIELDS,
    PAGE_RECORDS,
    WORD_BYTES,
    cache_geometry,
    page_rows,
)

# Stage up to 128 touched pages, stopping at the sorted invalid-record tail.
PACKED_FP8_WRITE_BATCH = 128


def quantize_index_rows(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return E4M3 values, FP32 scales and E8M0 scale bytes for D128 vectors."""
    # One power-of-two scale covers each post-RoPE vector, independent of layout.
    if values.shape[-1] != KEY_CHANNELS:
        raise ValueError("FP8 index vectors must have 128 channels")
    values = values.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(values).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scales = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    quantized = (values / scales[..., None]).astype(jnp.float8_e4m3fn)
    return quantized, scales, scale_bytes


def _record_slots(records: jax.Array, pages: int, page_size: int):
    valid = (records >= 0) & (records < pages * page_size)
    safe = jnp.clip(records, 0, pages * page_size - 1)
    return safe // page_size, safe % page_size, valid


def write_fp8_records(
    cache: jax.Array, records: jax.Array, keys: jax.Array
) -> jax.Array:
    """Quantize only new keys and update their slots; valid slots are unique."""
    pages, page_size = cache_geometry(cache.shape)
    if keys.shape != (records.size, KEY_CHANNELS) or records.ndim != 1:
        raise ValueError("Expected one logical record slot per D128 key")
    quantized, scales, _ = quantize_index_rows(keys)
    bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(-1, KEY_FIELDS - 1, WORD_BYTES)
    words = bits[:, :, 0] | (bits[:, :, 1] << 8)
    words |= (bits[:, :, 2] << 16) | (bits[:, :, 3] << 24)
    values = jnp.concatenate(
        (
            words.astype(jnp.int32),
            jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None],
        ),
        axis=1,
    )
    page, column, valid = _record_slots(records, pages, page_size)
    return write_packed_pages(cache, jnp.where(valid, page, -1), column, values)


def read_fp8_records(cache: jax.Array, records: jax.Array) -> jax.Array:
    """Decode selected records to BF16 for eager consumers; invalid slots are zero."""
    pages, page_size = cache_geometry(cache.shape)
    page, column, valid = _record_slots(records.reshape(-1), pages, page_size)
    raw = cache[page, :, :, column]  # [N, KEY_FIELDS, WORD_BYTES]
    values = jax.lax.bitcast_convert_type(
        raw[:, :-1].reshape(-1, KEY_CHANNELS), jnp.float8_e4m3fn
    )
    scale = raw[:, -1].astype(jnp.uint32)
    scale = scale[:, 0] | (scale[:, 1] << 8) | (scale[:, 2] << 16) | (scale[:, 3] << 24)
    scales = jax.lax.bitcast_convert_type(scale, jnp.float32)
    result = (values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], result, 0).reshape(*records.shape, KEY_CHANNELS)


def write_packed_pages(
    cache: jax.Array,
    pages: jax.Array,
    columns: jax.Array,
    values: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Write int32 `values` [N, KEY_FIELDS] at unique (page, column) slots; -1 skips."""
    count, fields = values.shape
    if not count:
        return cache
    num_pages, _ = cache_geometry(cache.shape)
    if fields != KEY_FIELDS:
        raise ValueError(f"Expected {KEY_FIELDS} int32 fields per key, got {fields}")
    batch = min(count, PACKED_FP8_WRITE_BATCH)
    valid = (pages >= 0) & (pages < num_pages)
    pages, order = jax.lax.sort_key_val(
        jnp.where(valid, pages, num_pages), jnp.arange(count, dtype=jnp.int32)
    )
    columns, values = columns[order], values[order]
    # The final sentinel terminates the last page's merge without an end scan.
    pages = jnp.pad(pages, (0, -count % batch + 1), constant_values=num_pages)
    columns = jnp.pad(columns, (0, -count % batch))
    values = jnp.pad(values.T, ((0, 0), (0, -count % PAGE_RECORDS)))

    return _write_sorted_packed_pages(
        cache, pages, columns, values, count=count, interpret=interpret
    )


def _write_sorted_packed_pages(cache, pages, columns, values, *, count, interpret):
    num_pages, page_size = cache_geometry(cache.shape)
    rows, parts = page_rows(page_size), page_size // PAGE_RECORDS
    batch = min(count, PACKED_FP8_WRITE_BATCH)

    def kernel(page_ref, column_ref, value_ref, _, output, staged, sem):
        words = output.bitcast(jnp.int32).reshape(num_pages * rows, PAGE_RECORDS)

        def first(record):
            page = page_ref[record]
            return (page < num_pages) & (
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
                    (slot < batch)
                    & (page_ref[pl.program_id(0) * batch + slot] < num_pages)
                ),
                body,
                jnp.int32(0),
            )

        def copies(*, write, wait):
            def body(slot, carry):
                record = pl.program_id(0) * batch + slot

                @pl.when(first(record))
                def _():
                    hbm = words.at[pl.ds(page_ref[record] * rows, rows), :]
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
                            :,
                            pl.ds(record // PAGE_RECORDS * PAGE_RECORDS, PAGE_RECORDS),
                        ],
                        jnp.full((KEY_FIELDS, PAGE_RECORDS), record % PAGE_RECORDS),
                        axis=1,
                        mode="fill",
                    )
                    # Row f * parts + h holds field f of part h.
                    if parts > 1:
                        word = jnp.tile(word, (1, parts)).reshape(rows, PAGE_RECORDS)
                    positions = jnp.arange(rows * PAGE_RECORDS).reshape(
                        rows, PAGE_RECORDS
                    )
                    staged[slot] = jnp.where(
                        positions % page_size == column_ref[record],
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
                pltpu.VMEM((batch, rows, PAGE_RECORDS), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={3: 0},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="write_packed_fp8_pages",
    )(pages, columns, values, cache)
