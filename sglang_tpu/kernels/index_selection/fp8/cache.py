# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-major FP8 pages with an in-page FP32 scale per key.

For token t and feature d, byte d%4 of word [d//4,t] contains K[t,d].
Field 32 holds scales[t] as FP32 bits. Each [33,page_size] page is rounded
to whole owner rows, then padded to whole 8-row tiles (pages.py). Unwritten
keys are masked by sequence lengths.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels.index_selection.fp8.pages import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
)

# Stage up to 128 touched pages, stopping at the sorted invalid-record tail.
PACKED_FP8_WRITE_BATCH = 128


def quantize_index_rows(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return E4M3 values, FP32 scales and E8M0 scale bytes for D128 vectors."""
    # One power-of-two scale covers each post-RoPE vector, independent of layout.
    if values.shape[-1] != 128:
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


def _record_addresses(records: jax.Array, layout: PackedFP8Layout):
    valid = (records >= 0) & (records < layout.pages * layout.page_size)
    safe = jnp.clip(records, 0, layout.pages * layout.page_size - 1)
    base = layout.page_offset + (safe // layout.page_size) * layout.page_stride
    return base, safe % layout.page_size, valid


def write_fp8_records(
    cache: jax.Array,
    records: jax.Array,
    keys: jax.Array,
    layout: PackedFP8Layout,
) -> jax.Array:
    """Quantize only new keys and update their columns; valid slots are unique."""
    layout.validate(cache.shape[0])
    if keys.shape != (records.size, PAGE_RECORDS) or records.ndim != 1:
        raise ValueError("Expected one logical record slot per D128 key")
    quantized, scales, _ = quantize_index_rows(keys)
    bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(-1, 32, 4)
    words = bits[:, :, 0] | (bits[:, :, 1] << 8)
    words |= (bits[:, :, 2] << 16) | (bits[:, :, 3] << 24)
    values = jnp.concatenate(
        (
            words.astype(jnp.int32),
            jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None],
        ),
        axis=1,
    )
    base, columns, valid = _record_addresses(records, layout)
    return write_packed_pages(
        cache,
        jnp.where(valid, base, -1),
        columns,
        values,
        page_size=layout.page_size,
    )


def read_fp8_records(
    cache: jax.Array, records: jax.Array, layout: PackedFP8Layout
) -> jax.Array:
    """Decode selected records to BF16 for eager consumers; invalid slots are zero."""
    layout.validate(cache.shape[0])
    shape = records.shape
    base, columns, valid = _record_addresses(records.reshape(-1), layout)
    addresses = (
        base[:, None] * PAGE_RECORDS
        + jnp.arange(PAGE_ROWS)[None, :] * layout.page_size
        + columns[:, None]
    )
    words = cache[addresses // PAGE_RECORDS, addresses % PAGE_RECORDS]
    bits = (words[:, :32, None].astype(jnp.uint32) >> (jnp.arange(4) * 8)) & 255
    values = jax.lax.bitcast_convert_type(
        bits.astype(jnp.uint8).reshape(-1, PAGE_RECORDS), jnp.float8_e4m3fn
    )
    scales = jax.lax.bitcast_convert_type(words[:, 32], jnp.float32)
    result = (values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], result, 0).reshape(*shape, PAGE_RECORDS)


def write_packed_pages(
    cache: jax.Array,
    bases: jax.Array,
    columns: jax.Array,
    values: jax.Array,
    *,
    page_size: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Update unique token slots at their page row bases."""
    count, fields = values.shape
    if not count:
        return cache
    rows, page_tokens = cache.shape
    page_size = page_tokens if page_size is None else page_size
    page_rows = pl.cdiv(fields * page_size, page_tokens)
    padded_fields = page_rows * page_tokens // page_size
    batch = min(count, PACKED_FP8_WRITE_BATCH)
    valid = (bases >= 0) & (bases + page_rows <= rows)
    bases, order = jax.lax.sort_key_val(
        jnp.where(valid, bases, rows), jnp.arange(count, dtype=jnp.int32)
    )
    columns, values = columns[order], values[order]
    # The final sentinel terminates the last page's merge without an end scan.
    bases = jnp.pad(bases, (0, -count % batch + 1), constant_values=rows)
    columns = jnp.pad(columns, (0, -count % batch))
    values = jnp.pad(values.T, ((0, padded_fields - fields), (0, -count % page_tokens)))

    return _write_sorted_packed_pages(
        cache,
        bases,
        columns,
        values,
        count=count,
        fields=fields,
        page_size=page_size,
        interpret=interpret,
    )


def _write_sorted_packed_pages(
    cache,
    bases,
    columns,
    values,
    *,
    count: int,
    fields: int,
    page_size: int,
    interpret: bool,
):
    rows, page_tokens = cache.shape
    page_rows = pl.cdiv(fields * page_size, page_tokens)
    padded_fields = page_rows * page_tokens // page_size
    batch = min(count, PACKED_FP8_WRITE_BATCH)

    def kernel(base_ref, column_ref, value_ref, _, output, pages, sem):

        def first(record):
            base = base_ref[record]
            return (base < rows) & (
                (record == 0) | (base != base_ref[jnp.maximum(record - 1, 0)])
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
                    & (base_ref[pl.program_id(0) * batch + slot] < rows)
                ),
                body,
                jnp.int32(0),
            )

        def copies(*, write, wait):
            def body(slot, carry):
                record = pl.program_id(0) * batch + slot
                base = base_ref[record]

                @pl.when(first(record))
                def _():
                    hbm = output.at[pl.ds(base, page_rows), :]
                    vmem = pages.at[slot]
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
                    pages[slot] = jnp.where(
                        (positions % page_size == column_ref[record])
                        & (positions < fields * page_size),
                        word,
                        pages[slot],
                    )
                    return record + 1

                jax.lax.while_loop(
                    lambda record: base_ref[record] == base_ref[start], update, start
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
    )(bases, columns, values, cache)
