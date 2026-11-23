# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Sources (epic/dsv41):
#   src/vllm_torchtpu/kernels/deepseek_v41/indexer.py (score/top-k pipelines)
#   src/vllm_torchtpu/models/vllm/deepseek_v41/v2/indexer_select.py
#   src/vllm_torchtpu/models/vllm/deepseek_v41/v2/indexer_fp8.py (JAX parts)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 index selection: dense scoring with top-k.

Layers score the paged cache with ``fp8_paged_mqa_logits``; prefill bounds the
score buffer by query chunks pipelined against streaming top-k. ``TileBlocks``
also returns each row's highest-scoring key blocks, which model-specific
candidate selection builds on. Torch op registration stays with the model layer.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import xla_metadata

from sglang_tpu.kernels.index_selection import (
    block_topk,
    streaming_topk,
)
from sglang_tpu.kernels.index_selection.fp8.mqa_logits.dense import (
    fp8_paged_mqa_logits,
)
from sglang_tpu.kernels.index_selection.fp8.pages import PackedFP8Layout
from sglang_tpu.kernels.index_selection.fp8.tuned import (
    PACKED_FP8_MQA_CONFIG,
    indexer_query_rows,
    paged_mqa_key_tile,
    topk_reconstruction_threshold,
)


def pipeline_score_topk_unrolled(
    num_tiles: int,
    prepare: Callable[[jax.Array], Any],
    score: Callable[[Any, int | None], jax.Array],
    select: Callable[[jax.Array, jax.Array, int | None], Any],
    write: Callable[[jax.Array, Any, Any], Any],
    output: Any,
) -> Any:
    """Leave independent score/selection tiles available to the scheduler.

    Use for tuned small-query batches; score lifetimes are not bounded here.
    """
    if num_tiles < 1:
        raise ValueError("The score/top-k pipeline requires at least one tile")
    for tile in range(num_tiles):
        index = jnp.int32(tile)
        scores = score(prepare(index), None)
        selected = select(index, scores, None)
        output = write(index, selected, output)
    return output


def pipeline_score_topk_bounded(
    num_tiles: int,
    prepare: Callable[[jax.Array], Any],
    score: Callable[[Any, int | None], jax.Array],
    select: Callable[[jax.Array, jax.Array, int | None], Any],
    write: Callable[[jax.Array, Any, Any], Any],
    output: Any,
) -> Any:
    """Bound score lifetime and pair next-tile scoring with current selection.

    Callbacks accept an optional scheduling group after their data arguments.
    Each pair gets a program-unique group so the compiler keeps TensorCore
    scoring between SparseCore selection start and completion.
    """
    if num_tiles < 1:
        raise ValueError("The score/top-k pipeline requires at least one tile")
    if num_tiles > 16:
        return pipeline_score_topk(
            num_tiles,
            prepare,
            lambda prepared: score(prepared, None),
            lambda tile, scores: select(tile, scores, None),
            write,
            output,
        )

    # Share V4's process-wide counter so the two pipelines never reuse an ID.
    # Imported here: the dsv4 package pulls in its attention backend.
    from sglang_tpu.dsv4.streamindex_topk import config

    group_base = (
        config.reserve_scheduling_group_ids(num_tiles - 1) if num_tiles > 1 else 0
    )
    # Gate current scores and next preparation on the previous selection
    # so unrolling does not extend every score buffer's lifetime.
    with jax.named_scope("DSV41ScoreTopKBoundedPipeline"):
        current = score(prepare(jnp.int32(0)), None)
        previous = None
        for tile in range(num_tiles - 1):
            prepared = prepare(jnp.int32(tile + 1))
            if previous is not None:
                previous, current, prepared = lax.optimization_barrier(
                    (previous, current, prepared)
                )
                output = write(jnp.int32(tile - 1), previous, output)
            group = group_base + tile
            following = score(prepared, group)
            previous = select(jnp.int32(tile), current, group)
            current = following
        if previous is not None:
            previous, current = lax.optimization_barrier((previous, current))
            output = write(jnp.int32(num_tiles - 2), previous, output)
        last = jnp.int32(num_tiles - 1)
        return write(last, select(last, current, None), output)


def pipeline_score_topk(num_tiles, prepare, score, select, write, output):
    """Order TensorCore scoring and SparseCore selection across two query tiles.

    The barrier gates score-buffer reuse on selection completion, before
    output scatter. Physical execution overlap depends on the compiler.
    """
    first = score(prepare(jnp.int32(0)))

    def advance(tile, state):
        current, result = state
        following = score(prepare(tile + 1))
        return following, write(tile, select(tile, current), result)

    def body(pair, state):
        tile = 2 * pair
        current, result = state
        following = score(prepare(tile + 1))
        selected = select(tile, current)
        prepared, selected = lax.optimization_barrier((prepare(tile + 2), selected))
        replacement = score(prepared)
        following_selected = select(tile + 1, following)
        result = write(tile, selected, result)
        result = write(tile + 1, following_selected, result)
        return replacement, result

    pairs = (num_tiles - 1) // 2
    with jax.named_scope("DSV41ScoreTopKPipeline"):
        state = lax.fori_loop(0, pairs, body, (first, output))
        if 2 * pairs < num_tiles - 1:
            state = advance(jnp.int32(2 * pairs), state)
        last, result = state
        return write(
            jnp.int32(num_tiles - 1), select(jnp.int32(num_tiles - 1), last), result
        )


_INDEX_UNTILED_SCORE_ELEMENTS = 1 << 27
# Keys per block maximum that block top-k reduces on SparseCore for layers
# without candidate publication: one maximum per half 16-lane vector.
_BLOCK_TOPK_BLOCK = 8


@dataclass(frozen=True)
class TileBlocks:
    """Also return each query row's highest-scoring key blocks with its top-k.

    ``rows(scores, lengths, scheduling_group_id=None)`` returns one score tile's
    int32 [rows, count] block IDs. Decode instead takes them from
    ``block_topk``'s single SparseCore pass over the same block maxima, where
    ``force_last_block`` ranks each row's last visible block first; ``rows``
    must apply the same rule.
    """

    count: int
    block_size: int
    force_last_block: bool
    rows: Callable[..., jax.Array]


def _compact_selected(records: jax.Array) -> jax.Array:
    """Sort valid (>= 0) indices ascending and move -1 to the tail."""
    limit = jnp.iinfo(jnp.int32).max
    records = jnp.sort(jnp.where(records >= 0, records, limit), axis=-1)
    return jnp.where(records == limit, -1, records)


def _prefix_count(mask: jax.Array, block: int = 128) -> jax.Array:
    """Inclusive ``cumsum(mask, -1)`` as exact blockwise 0/1 matmuls."""
    rows, width = mask.shape
    blocks = -(-width // block)
    x = jnp.pad(mask, ((0, 0), (0, blocks * block - width))).astype(jnp.bfloat16)
    x = x.reshape(rows, blocks, block)
    upper = jnp.triu(jnp.ones((block, block), jnp.bfloat16))
    inner = jnp.einsum("nbi,ij->nbj", x, upper, preferred_element_type=jnp.float32)
    totals = inner[..., -1].astype(jnp.bfloat16)  # at most `block`: exact
    strict = jnp.triu(jnp.ones((blocks, blocks), jnp.bfloat16), 1)
    before = jnp.einsum("nb,bc->nc", totals, strict, preferred_element_type=jnp.float32)
    return (inner + before[..., None]).reshape(rows, -1)[:, :width].astype(jnp.int32)


def _threshold_top_k(scores: jax.Array, k: int) -> jax.Array:
    """Ascending indices of ``jax.lax.top_k(scores, k)``, no sort.

    Finds the k-th largest value by bisecting an order-preserving integer key,
    breaks ties at it by lowest index (as top_k does), then compacts.
    """
    rows = scores.shape[0]
    # No -0.0 -> +0.0 folding: top_k ranks +0.0 above -0.0 (TPU and CPU), as
    # does the order-preserving key below.
    bits = jax.lax.bitcast_convert_type(scores, jnp.int32)
    key = jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits) ^ jnp.int32(-(2**31))
    key = jax.lax.bitcast_convert_type(key, jnp.uint32)

    def bisect(bit, threshold):
        candidate = threshold | (jnp.uint32(1) << (31 - bit).astype(jnp.uint32))
        enough = (key >= candidate[:, None]).sum(-1) >= k
        return jnp.where(enough, candidate, threshold)

    threshold = jax.lax.fori_loop(0, 32, bisect, jnp.zeros((rows,), jnp.uint32))
    above = key > threshold[:, None]
    tied = key == threshold[:, None]
    remaining = k - above.sum(-1)
    chosen = above | (tied & (_prefix_count(tied) <= remaining[:, None]))
    ranks = _prefix_count(chosen)
    targets = jnp.broadcast_to(jnp.arange(1, k + 1, dtype=jnp.int32), (rows, k))
    return jax.vmap(functools.partial(jnp.searchsorted, side="left"))(
        ranks, targets
    ).astype(jnp.int32)


def _sparsecore_top_k(
    scores: jax.Array,
    k: int,
    lengths: jax.Array,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """V4's SparseCore top-k (`sglang_tpu/dsv4/sparsecore_topk.py`).

    Same set as ``jax.lax.top_k`` over the finite scores (ties at the k-th
    value by lowest index), unordered, ``-1`` suffix-padded; ``-inf`` and
    columns ``>= lengths`` are never selected.
    """
    from sglang_tpu.dsv4.sparsecore_topk import LANES, sparsecore_topk

    if not (0 < k <= 4096 and k % LANES == 0):
        masked = jnp.where(
            jnp.arange(scores.shape[1]) < lengths[:, None], scores, -jnp.inf
        )
        values, ids = jax.lax.top_k(masked, k)
        return jnp.where(values > -jnp.inf, ids, -1)
    # The SGLang ops run unpartitioned on each rank, so no shard_map is needed
    # where the source wraps this call for its SPMD model mesh.
    return sparsecore_topk(
        scores, k=k, row_lengths=lengths, scheduling_group_id=scheduling_group_id
    )


def _visible_top_k(
    scores: jax.Array,
    k: int,
    bound: jax.Array,
    lengths: jax.Array | None = None,
    unordered: bool = False,
) -> jax.Array:
    """Ascending ``jax.lax.top_k(scores, k)`` indices when columns ``>= bound``
    are ``-inf``.

    Selects only within a static power-of-two bucket of columns covering
    ``bound`` (the full width otherwise). Exact: buckets hold at least ``k``
    columns, so ``-inf`` ties still resolve to the lowest indices. Large
    selections use `_threshold_top_k`; small ones (decode) sort.

    ``unordered`` callers (which re-sort and drop ``-1``) accept the finite
    top-k set in any order, ``-1`` padded instead of ``-inf`` columns; large
    selections then run on SparseCore over the full width, bounded by
    ``lengths`` instead of buckets (no SparseCore op inside `lax.switch`).
    Small decode query batches stay unpadded to avoid extra subcore waves.
    """
    width = scores.shape[1]
    if (
        unordered
        and lengths is not None
        and width >= 2048
        and width % 16 == 0
        and k % 16 == 0
    ):
        return _sparsecore_top_k(scores, k, jnp.minimum(lengths, width))

    def select(columns, scores):
        scores = scores[:, :columns]
        if scores.shape[0] >= 16 and columns >= 2048:
            return _threshold_top_k(scores, k)
        return jnp.sort(jax.lax.top_k(scores, k)[1], axis=-1)

    buckets = []
    bucket = max(k, 1024)
    while bucket < width:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(width)
    if len(buckets) == 1:
        return select(width, scores)
    branch = jnp.searchsorted(jnp.asarray(buckets[:-1]), bound, side="left")
    return jax.lax.switch(
        branch, [functools.partial(select, b) for b in buckets], scores
    )


def select_indices_jax(
    scores: jax.Array,
    lengths: jax.Array,
    topk: int,
    unordered: bool = False,
    blocks: TileBlocks | None = None,
):
    """Top-k visible key IDs per row, and each row's top blocks with ``blocks``.

    ``unordered``: see `_visible_top_k` (the selection may come back in any
    order with ``-1`` anywhere; FP8 attention re-sorts via `compact_records`).
    """
    count, width = scores.shape
    if topk < 1 or width < 1:
        raise ValueError("Positive top-k and score width required")
    lengths = jnp.clip(lengths.reshape(count), 0, width)
    logical = jnp.arange(width)[None, :]
    visible = logical < lengths[:, None]
    scores = jnp.where(visible, scores, -jnp.inf)
    published = None if blocks is None else blocks.rows(scores, lengths)
    # Ascending: preserves V4.1's logical, hence attention accumulation, order.
    k = min(topk, width)
    indices = _visible_top_k(scores, k, lengths.max(), lengths, unordered)
    if unordered and width >= 2048 and width % 16 == 0 and k % 16 == 0:
        selected = _compact_selected(indices.astype(jnp.int32))
    else:
        values = jnp.take_along_axis(scores, jnp.maximum(indices, 0), axis=1)
        valid = (indices >= 0) & (values > -jnp.inf)
        selected = jnp.where(valid, indices, -1).astype(jnp.int32)
        if unordered:
            selected = _compact_selected(selected)
    return selected if blocks is None else (selected, published)


def _query_start_loc(requests: jax.Array, num_requests: int) -> jax.Array:
    """[num_requests + 1] row boundaries; padding (-1) rows join the last request."""
    last = num_requests - 1
    owners = jnp.where(requests >= 0, jnp.minimum(requests, last), last)
    bounds = jnp.arange(num_requests + 1, dtype=owners.dtype)
    return jnp.sum(owners[:, None] < bounds, axis=0, dtype=jnp.int32)


def _select_fp8_indexer_jax(
    q: jax.Array,
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    unordered: bool,
    score_width: int,
    blocks: TileBlocks | None = None,
    *,
    query_start_loc: jax.Array | None = None,
    prefill_chunk_rows: int | None = None,
):
    layout = PackedFP8Layout(*layout_args)
    tokens, width = q.shape[0], score_width
    if tokens == 0:
        selected = jnp.empty((0, min(topk, width)), jnp.int32)
        if blocks is None:
            return selected
        return selected, jnp.empty((0, blocks.count), jnp.int32)
    if q.shape[1:] != (32, 128) or width < 1:
        raise ValueError(
            "Packed FP8 model scorer requires H32/D128 and nonempty context"
        )
    key_tile = paged_mqa_key_tile(
        width, layout.page_size, PACKED_FP8_MQA_CONFIG.key_tile
    )
    query_tile = PACKED_FP8_MQA_CONFIG.query_tile
    capacity = (width + key_tile - 1) // key_tile * key_tile
    pages = page_table[:, : (width + layout.page_size - 1) // layout.page_size]
    pages = jnp.pad(
        pages,
        ((0, 0), (0, capacity // layout.page_size - pages.shape[1])),
        constant_values=-1,
    )

    rows = prefill_chunk_rows or min(
        max(tokens, query_tile),
        max(query_tile, _INDEX_UNTILED_SCORE_ELEMENTS // capacity),
    )
    padded = (tokens + rows - 1) // rows * rows
    # Callers group each request's queries contiguously in ascending request
    # order; padding requests (-1) trail them. Clamping padding to the last
    # request keeps that order, which the pipelined path relies on.
    safe_requests = jnp.where(
        requests >= 0,
        jnp.minimum(requests, page_table.shape[0] - 1),
        page_table.shape[0] - 1,
    )
    valid_requests = (requests >= 0) & (requests < page_table.shape[0])
    lengths = jnp.where(valid_requests, jnp.clip(lengths, 0, width), 0)
    inputs = (q, weights, lengths, safe_requests)
    inputs = tuple(
        jnp.pad(x, ((0, padded - tokens),) + ((0, 0),) * (x.ndim - 1)) for x in inputs
    )
    # Derived boundaries only count rows per request; they are valid only for
    # grouped rows, so the pipelined path below checks grouping at run time.
    derived_offsets = query_start_loc is None
    if derived_offsets:
        query_start_loc = _query_start_loc(requests, page_table.shape[0])
    if query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("query_start_loc needs one entry per page-table row, plus one")
    # Trailing padding rows belong to the last request, as in safe_requests.
    query_offsets = jnp.minimum(query_start_loc.astype(jnp.int32), tokens)
    query_offsets = query_offsets.at[-1].set(tokens)
    if padded > tokens:
        # Tile padding rows own request 0, after its rows once sorted.
        query_offsets = query_offsets.at[1:].add(padded - tokens)
    stream_select = unordered and streaming_topk.is_supported(
        width,
        topk,
        reconstruct_threshold=topk_reconstruction_threshold(topk),
    )
    publish_rows = blocks is not None
    select_topk = functools.partial(
        streaming_topk.streaming_topk,
        k=topk,
        exclude_neg_inf=True,
        reconstruct_threshold=topk_reconstruction_threshold(topk),
        # Attention re-sorts the records; skip ID tie-breaks and output sorting.
        unordered=unordered,
    )
    chunk_rows = prefill_chunk_rows or indexer_query_rows(tokens, query_tile)
    # Decode top-k scans only the blocks that can hold a winner. Callers asking
    # for top blocks rank the same SparseCore block maxima for them.
    select_block = blocks.block_size if publish_rows else _BLOCK_TOPK_BLOCK
    block_select = (
        stream_select
        and prefill_chunk_rows is None
        and chunk_rows is not None
        and block_topk.is_supported(chunk_rows, capacity, topk, select_block)
    )
    pool = banks[0] if len(banks) == 1 else banks

    def score(start, rows, inputs, *, pipelined, group=None):
        """Scores of `rows` queries from `start`; decode tiles share the inputs."""
        shared = pipelined and prefill_chunk_rows is None
        queries, row_weights, row_lengths = (
            inputs[:3]
            if shared
            else (jax.lax.dynamic_slice_in_dim(x, start, rows) for x in inputs[:3])
        )
        # Pipelined tiles inline inside the outer model jit so scheduling
        # metadata reaches the Pallas call instead of the inner jit boundary.
        scorer = fp8_paged_mqa_logits.__wrapped__ if pipelined else fp8_paged_mqa_logits
        scores = scorer(
            queries,
            pool,
            row_weights.astype(jnp.float32),
            row_lengths,
            pages,
            query_start_loc=(
                query_offsets if shared else jnp.clip(query_offsets - start, 0, rows)
            ),
            max_model_len=capacity if stream_select else width,
            q_scale=jnp.ones_like(row_weights, dtype=jnp.float32),
            query_size=query_tile,
            cache_layout=layout,
            query_start=start if shared else 0,
            query_count=rows,
            reuse_keys=prefill_chunk_rows is not None and capacity == key_tile,
        )
        if group is None:
            return scores
        tag = functools.partial(
            xla_metadata.set_xla_metadata, _scheduling_group_id=group
        )
        return jax.tree.map(tag, scores)

    def select_serial():
        order = jnp.argsort(inputs[-1], stable=True)
        sorted_inputs = tuple(x[order] for x in inputs)
        output = jnp.full((padded, min(topk, width)), -1, jnp.int32)
        if publish_rows:
            output = (output, jnp.full((padded, blocks.count), -1, jnp.int32))
        # Unroll SparseCore calls to avoid XLA loop lowering failures;
        # barriers bound score-buffer lifetimes.
        for index in range(padded // rows):
            start, output = jax.lax.optimization_barrier(
                (jnp.int32(index * rows), output)
            )
            scores = score(start, rows, sorted_inputs, pipelined=False)
            row_lengths = jax.lax.dynamic_slice_in_dim(sorted_inputs[2], start, rows)
            if not stream_select:
                selected = select_indices_jax(
                    scores,
                    row_lengths,
                    topk,
                    unordered,
                    blocks if publish_rows else None,
                )
            else:
                selected = select_topk(scores, valid_lengths=row_lengths)
                if publish_rows:
                    selected = (selected, blocks.rows(scores, row_lengths))
            output = jax.tree.map(
                lambda dst, src: jax.lax.dynamic_update_slice_in_dim(
                    dst, src, start, axis=0
                ),
                output,
                selected,
            )
        inverse = jnp.argsort(order)
        return jax.tree.map(lambda x: x[inverse][:tokens], output)

    if not stream_select or chunk_rows is None or chunk_rows > rows or padded != tokens:
        return select_serial()

    def select(tile, scores, group):
        row_lengths = jax.lax.dynamic_slice_in_dim(
            inputs[2], tile * chunk_rows, chunk_rows
        )
        if block_select:
            # One SparseCore pass ranks the block maxima for both outputs.
            return block_topk.block_topk(
                scores,
                row_lengths,
                k=topk,
                block_size=select_block,
                top_blocks=blocks.count if publish_rows else 0,
                force_last_block=publish_rows and blocks.force_last_block,
                scheduling_group_id=group,
            )
        selected = select_topk(
            scores, valid_lengths=row_lengths, scheduling_group_id=group
        )
        if publish_rows:
            return selected, blocks.rows(scores, row_lengths, scheduling_group_id=group)
        return selected

    def write(tile, selected, output):
        update = functools.partial(
            jax.lax.dynamic_update_slice_in_dim, start_index=tile * chunk_rows, axis=0
        )
        return jax.tree.map(update, output, selected)

    def select_pipelined():
        # Grouped requests need neither sorting nor output reordering, so no
        # runtime branch to the serial path: a branch costs an overlay load at its
        # entry. Score tiles may split a request; the kernel resumes it mid-group.
        output = jnp.full((tokens, topk), -1, jnp.int32)
        if publish_rows:
            output = (output, jnp.full((tokens, blocks.count), -1, jnp.int32))
        pipeline = (
            pipeline_score_topk_bounded
            if prefill_chunk_rows is not None or publish_rows
            else pipeline_score_topk_unrolled
        )
        selected = pipeline(
            padded // chunk_rows,
            lambda tile: tile * chunk_rows,
            lambda start, group: score(
                start, chunk_rows, inputs, pipelined=True, group=group
            ),
            select,
            write,
            output,
        )
        if publish_rows:
            selected, published = selected
            published = jnp.where(valid_requests[:, None], published, -1)
        # Batch-wide short-context bypass: every visible key, as in the reference.
        # A select rather than a cond keeps the scoring path free of branches.
        short = jnp.minimum(batch_max.max(), width) <= topk
        ids = jnp.arange(topk, dtype=jnp.int32)
        visible = jnp.where(ids < lengths[:tokens, None], ids, -1)
        selected = jnp.where(short, visible, selected)
        return (selected, published) if publish_rows else selected

    # A caller-supplied query_start_loc asserts grouped rows, as upstream: no
    # runtime branch. Derived boundaries fall back to the sorting serial path.
    if not derived_offsets:
        return select_pipelined()
    grouped = jnp.all(inputs[-1][1:] >= inputs[-1][:-1])
    return jax.lax.cond(grouped, select_pipelined, select_serial)


def select_fp8_indexer_jax(
    q: jax.Array,
    banks: tuple[jax.Array, ...],
    page_table: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    unordered: bool,
    score_width: int,
    blocks: TileBlocks | None = None,
    *,
    query_start_loc: jax.Array | None = None,
):
    """Select exact FP8 scores with a bounded, overlapped prefill working set.

    Returns int32 [tokens, min(topk, score_width)] key IDs, -1 padded; with
    ``blocks``, also each row's int32 [tokens, blocks.count] top block IDs.
    """
    arguments = (
        q,
        banks,
        page_table,
        weights,
        lengths,
        requests,
        batch_max,
        layout_args,
        topk,
        unordered,
    )
    layout = PackedFP8Layout(*layout_args)
    width = paged_mqa_key_tile(
        score_width, layout.page_size, PACKED_FP8_MQA_CONFIG.key_tile
    )
    capacity = (score_width + width - 1) // width * width
    chunk = max(
        PACKED_FP8_MQA_CONFIG.query_tile,
        _INDEX_UNTILED_SCORE_ELEMENTS // max(capacity, 1),
    )
    if (
        width >= score_width
        or chunk >= q.shape[0]
        or q.shape[0] % chunk
        or blocks is not None
        or not unordered
        or topk > width
        or jax.default_backend() != "tpu"
    ):
        return _select_fp8_indexer_jax(
            *arguments, score_width, blocks, query_start_loc=query_start_loc
        )

    # Preserve score-buffer row boundaries for both widths. Only contexts
    # covered by one K tile can narrow the scores and retain their keys.
    live = (requests >= 0) & (requests < page_table.shape[0])
    longest = jnp.where(live, lengths, 0).max()
    grouped = jnp.all(requests[1:] >= requests[:-1])
    return jax.lax.cond(
        grouped & (longest <= width),
        lambda: _select_fp8_indexer_jax(
            *arguments,
            width,
            query_start_loc=query_start_loc,
            prefill_chunk_rows=chunk,
        ),
        lambda: _select_fp8_indexer_jax(
            *arguments,
            score_width,
            query_start_loc=query_start_loc,
            prefill_chunk_rows=chunk,
        ),
    )
