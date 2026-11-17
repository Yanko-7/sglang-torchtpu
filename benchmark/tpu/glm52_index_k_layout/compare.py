"""GLM5.2 DSA indexer step: 256-byte records (page 64) vs packed FP8 pages (page 128).

Both layouts get the same keys, queries and head weights and run the whole
``dsa_indexer_step`` (quantize, store this step's keys, score, top-k), as the
serving op does. Per case this records:

* correctness: every sampled row picks min(visible, topk) distinct causal keys,
  none scoring more than 2% below the k-th best reference score;
* the overlap of the two layouts' picks;
* whether the compiled step copies the index-key pool;
* an xprof capture of five calls for device time.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import sglang_tpu.attention.pallas.dsa_index_topk as dit
from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--contexts", default="53248,90000",
                    help="comma-separated sequence lengths; the page table stays WIDTH wide")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

HEADS, DIM, TOPK, WIDTH = 32, 128, 2048, 92160
WEIGHT_SCALE = float(DIM**-0.5 * HEADS**-0.5)
SLACK = 0.02
FILL = 8192
LAYOUTS = {  # name: (page_size, cache shape for num_pages)
    "records256_page64": (64, lambda p: dit.index_key_cache_shape(p, 64, DIM)),
    "packed_page128": (128, lambda p: (p, 40, 128)),
}
device = jax.devices()[0]


def emit(record):
    with (args.out / "metrics.jsonl").open("a") as out:
        out.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def store(cache, key, kv_lens, pages, cu, dist):
    if cache.ndim == 3:
        return dit.store_packed_keys(cache, key, kv_lens, pages, cu, dist)
    k_fp8, _, k_byte = quantize_fp8_ue8m0(key)
    return dit._store_keys(cache, k_fp8, k_byte, kv_lens, pages, cu, dist)


fill = jax.jit(store, donate_argnums=0)


def make_keys(seq, context):
    """bf16 keys of one sequence, a power-of-two magnitude per token."""
    k1, k2 = jax.random.split(jax.random.key(1000 + seq))
    keys = jax.random.normal(k1, (context, DIM), jnp.float32)
    keys *= jnp.exp2(jax.random.randint(k2, (context, 1), -3, 4).astype(jnp.float32))
    return keys.astype(jnp.bfloat16)


def build_cache(layout, keys, q_lens):
    """Pool, flat page table and its prefix keys stored (all but the new tokens)."""
    page_size, shape = LAYOUTS[layout]
    per_seq = WIDTH // page_size
    needed = [-(-k.shape[0] // page_size) for k in keys]
    num_pages = 1 + sum(needed) + 3
    pool = 1 + np.random.default_rng(7).permutation(num_pages - 1)
    table = np.zeros((len(keys), per_seq), np.int32)
    used = 0
    for seq, n in enumerate(needed):
        table[seq, :n] = pool[used:used + n]
        used += n
    cache = jnp.zeros(shape(num_pages), jnp.int32 if layout.startswith("packed") else jnp.uint8)
    for seq, (k, q) in enumerate(zip(keys, q_lens)):
        prefix = k.shape[0] - q
        row = jnp.asarray(table[seq])
        for start in range(0, prefix, FILL):
            count = min(FILL, prefix - start)
            chunk = jnp.zeros((FILL, DIM), jnp.bfloat16).at[:count].set(k[start:start + count])
            cache = fill(cache, chunk, jnp.asarray([start + count], jnp.int32), row,
                         jnp.asarray([0, count], jnp.int32), jnp.asarray([0, 0, 1], jnp.int32))
    return cache, jnp.asarray(table.reshape(-1))


def reference_scores(q, weights, keys, seq_rows):
    """f32 scores of sampled rows: (row, sequence, visible length) -> scores[:visible]."""
    q8, qs, _ = quantize_fp8_ue8m0(q)
    w = weights * qs[..., 0] * WEIGHT_SCALE
    out = {}
    for row, seq, visible in seq_rows:
        k8, ks, _ = quantize_fp8_ue8m0(keys[seq][:visible])
        k = k8.astype(jnp.float32) * ks
        dots = jnp.einsum("hd,kd->hk", q8[row].astype(jnp.float32), k,
                          precision=jax.lax.Precision.HIGHEST)
        out[row] = np.asarray((jnp.maximum(dots, 0) * w[row][:, None]).sum(0))
    return out


def check(rows, refs, seq_rows):
    worst = 1.0
    for row, _, visible in seq_rows:
        picks = rows[row][rows[row] >= 0]
        expected = min(visible, TOPK)
        assert np.all(rows[row][expected:] == -1), (row, "trailing")
        assert len(np.unique(picks)) == len(picks) == expected, (row, len(picks), expected)
        assert picks.max() < visible, (row, "causal")
        if visible > TOPK:
            kth = np.sort(refs[row])[::-1][TOPK - 1]
            ratio = float(refs[row][picks].min() / kth)
            worst = min(worst, ratio)
            assert ratio >= 1 - SLACK, (row, ratio)
    return worst


def pool_copies(compiled, cache):
    """HLO copies whose element count is the whole pool's."""
    elements = int(np.prod(cache.shape))
    count = 0
    for line in compiled.as_text().splitlines():
        if " copy(" not in line and "copy-start" not in line:
            continue
        lhs = line.split("=", 1)[-1].strip()
        dims = lhs[lhs.find("[") + 1:lhs.find("]")]
        if dims and all(d.strip().isdigit() for d in dims.split(",")):
            if int(np.prod([int(d) for d in dims.split(",")])) == elements:
                count += 1
    return count


def run_case(context, queries, sequences):
    name = f"c{context}-q{queries}-s{sequences}"
    q_lens = [queries] * sequences
    tokens = sum(q_lens)
    decode = queries <= 6
    keys = [make_keys(seq, context) for seq in range(sequences)]
    new_keys = jnp.concatenate([k[-q:] for k, q in zip(keys, q_lens)])
    rk = jax.random.split(jax.random.key(5 + context + 31 * queries + sequences), 3)
    query = jax.random.normal(rk[0], (tokens, HEADS, DIM), jnp.float32)
    query *= jnp.exp2(jax.random.randint(rk[1], (tokens, HEADS, 1), -3, 4).astype(jnp.float32))
    query = query.astype(jnp.bfloat16)
    weights = jax.random.uniform(rk[2], (tokens, HEADS), jnp.float32, 0.25, 1.75)
    kv_lens = jnp.full((sequences,), context, jnp.int32)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    dist = jnp.asarray([sequences] * 3 if decode else [0, 0, sequences], jnp.int32)
    sample = sorted({0, tokens - 1, *np.linspace(0, tokens - 1, 8, dtype=int).tolist()})
    seq_rows = []
    for row in sample:
        seq = int(np.searchsorted(np.asarray(cu)[1:], row, side="right"))
        seq_rows.append((row, seq, context - q_lens[seq] + (row - int(cu[seq])) + 1))
    refs = reference_scores(query, weights, keys, seq_rows)
    emit(dict(kind="case", case=name, context=context, queries=queries, sequences=sequences,
              tokens=tokens, topk=TOPK, width=WIDTH))
    picks = {}
    for layout in LAYOUTS:
        cache, pages = build_cache(layout, keys, q_lens)
        key_store = "pallas" if layout.startswith("records") and tokens <= 16 else "xla"
        step = jax.jit(
            lambda q, k, w, c, kv, p, cu, d: dit.dsa_indexer_step(
                q, k, w, c, kv, p, cu, d, weight_scale=WEIGHT_SCALE, topk=TOPK,
                decode_query_size=queries if decode else 1,
                segments="decode" if decode else "mixed", key_store=key_store),
            donate_argnums=3)
        operands = (query, new_keys, weights, cache, kv_lens, pages, cu, dist)
        start = time.perf_counter()
        compiled = step.lower(*operands).compile()
        compile_s = time.perf_counter() - start
        copies = pool_copies(compiled, cache)
        rows, cache = compiled(*operands)
        rows = np.asarray(rows)
        try:
            worst = check(rows, refs, seq_rows)
            emit(dict(kind="correctness", case=name, layout=layout, passed=True,
                      sampled_rows=len(seq_rows), minimum_selected_cutoff_ratio=worst))
        except AssertionError as error:
            emit(dict(kind="correctness", case=name, layout=layout, passed=False, error=repr(error)))
        picks[layout] = rows
        # Rewrites of the same keys leave the cache unchanged, so calls repeat.
        walls = []
        for i in range(25):
            t = time.perf_counter_ns()
            out, cache = compiled(query, new_keys, weights, cache, kv_lens, pages, cu, dist)
            jax.block_until_ready(out)
            if i >= 5:
                walls.append((time.perf_counter_ns() - t) / 1e3)
        emit(dict(kind="timing", case=name, layout=layout, compile_s=compile_s,
                  pool_shape=list(cache.shape), pool_dtype=str(cache.dtype),
                  pool_bytes=int(cache.nbytes), whole_pool_copies=copies,
                  wall_us_median=statistics.median(walls), wall_us_min=min(walls)))
        options = jax.profiler.ProfileOptions()
        options.advanced_configuration = {
            "tpu_num_sparse_cores_to_trace": 2, "tpu_num_sparse_core_tiles_to_trace": 1}
        with jax.profiler.trace(str(args.out / "xprof" / name / layout), profiler_options=options):
            for i in range(5):
                with jax.profiler.StepTraceAnnotation(f"{name}_{layout}", step_num=i):
                    out, cache = compiled(query, new_keys, weights, cache, kv_lens, pages, cu, dist)
                    jax.block_until_ready(out)
        del cache, compiled
        jax.clear_caches()
    a, b = (picks[layout] for layout in LAYOUTS)
    overlap = [len(np.intersect1d(a[r][a[r] >= 0], b[r][b[r] >= 0])) / max(1, (a[r] >= 0).sum())
               for r in range(tokens)]
    emit(dict(kind="agreement", case=name, mean_overlap=float(np.mean(overlap)),
              min_overlap=float(np.min(overlap))))


cases = [(90000, 6, 4)] if args.smoke else [
    (context, q, s)
    for context in (int(c) for c in args.contexts.split(","))
    for q, s in ((1, 4), (6, 1), (6, 2), (6, 3), (6, 4), (2048, 1))
]
for case in cases:
    run_case(*case)
emit(dict(kind="pass", marker="GLM52_INDEX_K_LAYOUT_PASS"))
