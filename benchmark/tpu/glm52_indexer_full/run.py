"""Complete score/TopK/physical-slot benchmark. No model or attention compute.

All inputs are resident: query quantization and key writes are outside this
explicit score-module boundary. The FP8 candidate reads the same packed cache
bytes as production; it does not convert the full cache before timing.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("bench", "profile"), required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--smoke", action="store_true")
# Profile only the decode/MTP cases at both contexts for the named arms.
parser.add_argument("--decode-profile", action="store_true")
parser.add_argument("--arms", nargs="+")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

# Import the original fixture before JAX, as its raw-TPU contract requires.
spec = importlib.util.spec_from_file_location(
    "glm52_fixture", ROOT / "test/manual/kernels/benchmark_glm52_radix_topk.py"
)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

from sglang_tpu.attention.pallas.mla_sparse.kernel import prepare_gather_indices
from sglang_tpu.dsv4.streamindex_topk.streamindex_topk import streamindex_topk
from glm52_fp8.kernels.dsa.indexer import compute_scores_and_select_topk_indices as fp8_full
from glm52_jax.kernels.dsa.indexer import compute_scores_and_select_topk_indices as native_full

# vllm-torchtpu epic/dsv41 packed FP8 indexer: byte-identical upstream files
# (dsv41_upstream/SOURCE.json). Three package __init__ files would import vLLM
# and the vllm-torchtpu runtime (env overrides, LIBTPU_INIT_ARGS), so those
# packages are registered as empty namespaces and only kernel modules execute.
DSV41 = "dsv41_upstream_fp8_full"
DSV41_ROOT = Path(__file__).resolve().parent / "dsv41_upstream"
sys.path.insert(0, str(DSV41_ROOT))
for package in ("vllm_torchtpu", "vllm_torchtpu.models.vllm",
                "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"):
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(DSV41_ROOT / package.replace(".", "/"))]
    sys.modules[package] = namespace
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_RECORDS as DSV41_PAGE, PAGE_ROWS as DSV41_PAGE_ROWS, PackedFP8Layout, read_fp8_records)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_fp8 import select_fp8_indexer_jax as dsv41_select

# The same selector with one change (dsv41-single-block.patch): a batch whose
# scores fit the budget is one block, so one dense call and one top-k call,
# instead of gcd(tokens, 96) blocks. Larger batches keep upstream's behaviour.
# The vendored upstream files stay unmodified.
DSV41_SINGLE = "dsv41_single_block_fp8_full"
# Loaded as a sibling of the upstream module so its relative imports resolve.
single_spec = importlib.util.spec_from_file_location(
    "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_fp8_single_block",
    Path(__file__).resolve().parent / "dsv41_single_block" / "indexer_fp8.py")
dsv41_single = importlib.util.module_from_spec(single_spec)
sys.modules[single_spec.name] = dsv41_single
single_spec.loader.exec_module(dsv41_single)
DSV41_ARMS = (DSV41, DSV41_SINGLE)

device = jax.devices()[0]
runtime = dict(
    executable=sys.executable,
    python=sys.version,
    versions={n: metadata.version(n) for n in ("jax", "jaxlib", "libtpu", "torch", "torch_tpu")},
    imports={n: str(__import__(n).__file__) for n in ("jax", "jaxlib", "torch", "torch_tpu")},
    source=os.environ["SOURCE_COMMIT"],
    visible_devices=[str(d) for d in jax.devices()],
    actual_device=str(device),
    allocation_physical_chips=4,
    participating_jax_devices=1,
    flags={n: os.environ.get(n) for n in ("JAX_PLATFORMS", "XLA_FLAGS", "LIBTPU_INIT_ARGS", "TPU_LIBRARY_PATH")},
    mapped_libtpu=sorted(set(line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libtpu.so" in line)),
)
runtime["dsv41"] = dict(source=json.loads((DSV41_ROOT / "SOURCE.json").read_text()),
                        module=sys.modules[dsv41_select.__module__].__file__,
                        issue_depth_env=os.environ.get("TPU_DEEPSEEK_V41_INDEXER_ISSUE_DEPTH"))
assert runtime["versions"]["jax"] == runtime["versions"]["jaxlib"] == "0.11.2"
assert runtime["versions"]["libtpu"] == "0.0.50"
(args.out / "runtime.json").write_text(json.dumps(runtime, indent=2))
artifact_root = Path(os.environ["ARTIFACT_LOCAL_DIR"])
(artifact_root / "manifest.json").write_text(json.dumps(dict(
    schema_version=1, workflow="operator-optimization", operator_family="indexer",
    operator_name="glm52_score_topk_slot_mapping", source=runtime["source"],
    dimensions=dict(page_size=64, heads=32, head_dim=128, k=2048, capacity=92160,
                    participating_jax_devices=1, allocation_physical_chips=4),
), indent=2))


def emit(record):
    with (args.out / "metrics.jsonl").open("a") as out:
        out.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def dsv41_inputs(step, rows):
    """The same FP8 key bytes and scales in vllm-torchtpu's 128-record packed pages.

    Logical 128-token blocks get shuffled pages (page 0 unused, as in the fixture).
    The selector returns sequence-local positions; the GLM page table maps them.
    """
    glm = step.expected_cache.reshape(-1, 256)
    table = step.page_indices.reshape(len(step.kv_lens), -1)
    width = table.shape[1] * 64
    blocks = [-(-int(n) // DSV41_PAGE) for n in step.kv_lens]
    num_pages = 1 + sum(blocks) + 3
    pool = 1 + np.random.default_rng(1).permutation(num_pages - 1)
    # Unused entries are -1, as the kernel's own padding: page 0 would be read as
    # a valid page, serializing every masked DMA on one address.
    pages = np.full((len(blocks), width // DSV41_PAGE), -1, np.int32)
    bank = np.zeros((num_pages, DSV41_PAGE_ROWS, DSV41_PAGE), np.uint32)
    used = 0
    for seq, count in enumerate(blocks):
        pages[seq, :count] = pool[used:used + count]
        used += count
        positions = np.arange(count * DSV41_PAGE)
        live = positions < step.kv_lens[seq]
        slots = table[seq, positions // 64] * 64 + positions % 64
        records = np.where(live[:, None], glm[slots], 0).reshape(count, DSV41_PAGE, 256)
        # Word f of token t holds features 4f..4f+3, byte d % 4; field 32 is the FP32 scale.
        words = np.ascontiguousarray(records[..., :128]).view("<u4")
        bank[pages[seq, :count], :32] = words.transpose(0, 2, 1)
        bank[pages[seq, :count], 32] = records[..., 128].astype(np.uint32) << 23
    requests = np.full(rows, -1, np.int32)
    lengths = np.zeros(rows, np.int32)
    for seq in range(len(step.kv_lens)):
        start, end = map(int, step.cu_q_lens[seq:seq+2])
        requests[start:end] = seq
        lengths[start:end] = step.kv_lens[seq] - (end - start) + np.arange(1, end - start + 1)
    layout = PackedFP8Layout(num_pages, DSV41_PAGE_ROWS, 0, DSV41_PAGE)
    return bank.view(np.int32).reshape(-1, DSV41_PAGE), pages, requests, lengths, layout


def check_dsv41_cache(bank, pages, step, decoded, layout):
    """Upstream's reader must decode the repacked cache to the GLM keys exactly."""
    table = step.page_indices.reshape(len(step.kv_lens), -1)
    records, slots = [], []
    for seq, kv_len in enumerate(step.kv_lens):
        positions = np.unique(np.linspace(0, int(kv_len) - 1, 1024, dtype=np.int64))
        records.append(pages[seq, positions // DSV41_PAGE] * DSV41_PAGE + positions % DSV41_PAGE)
        slots.append(table[seq, positions // 64] * 64 + positions % 64)
    records, slots = np.concatenate(records).astype(np.int32), np.concatenate(slots)
    got = np.asarray(read_fp8_records((bank,), jnp.asarray(records), layout)).astype(np.float32)
    want = decoded.reshape(-1, 128)[slots].astype(np.float32)
    assert np.array_equal(got, want), ("dsv41 cache repack", int((got != want).any(-1).sum()))
    return records.size


def make_case(context, queries, sequences, *, padded=0, ragged=False):
    lens = [(queries, context)] * sequences
    if ragged:
        lens = [(6, context), (3, 53248)]
    rows = sum(q for q, _ in lens) + padded
    step = helper.build_benchmark_step(lens=lens, token_bucket=rows, is_decode=queries <= 6 and not ragged)
    query = step.query.astype(ml_dtypes.bfloat16).astype(np.float32)
    gates = step.head_weights.astype(ml_dtypes.bfloat16).astype(np.float32)
    q8, qs, _ = helper.fixture["_quantize"](query)
    weights = gates * qs[..., 0] * np.float32(helper.fixture["_WEIGHT_SCALE"])
    packed = step.expected_cache.reshape(-1, 64, 256)
    decoded = packed[..., :128].copy().view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    decoded *= np.exp2(packed[..., 128].astype(np.int32) - 127).astype(np.float32)[..., None]
    decoded = decoded.astype(ml_dtypes.bfloat16)
    # Source uses ragged packed page offsets; our production table is fixed stride.
    cu_kv = np.arange(len(lens) + 1, dtype=np.int32) * 92160
    values = (q8, weights, step.expected_cache, packed, decoded,
              step.kv_lens, step.page_indices, step.cu_q_lens, cu_kv, step.distribution)
    values = tuple(jax.device_put(x, device) for x in values)
    d_bank, d_pages, d_requests, d_lengths, d_layout = dsv41_inputs(step, rows)
    d_values = tuple(jax.device_put(x, device) for x in (d_bank, d_pages, d_requests, d_lengths))
    glm_pages = step.expected_cache.shape[0]
    metadata_case = dict(context=context, queries_per_sequence=queries, sequences=len(lens), rows=rows,
                         page_size=64, capacity=92160, k=2048, heads=32, head_dim=128,
                         packed_cache_bytes=packed.nbytes, native_cache_bytes=decoded.nbytes,
                         dsv41_cache_bytes=d_bank.nbytes, dsv41_page_size=DSV41_PAGE,
                         input_sha256=hashlib.sha256(query.tobytes()+gates.tobytes()+packed.tobytes()+step.page_indices.tobytes()).hexdigest(),
                         boundary="resident prepared query + resident updated cache -> physical slots; includes TC score, SC selection, mapping; excludes query quantization/cache writer")

    def torch_path(selector):
        def call(q, w, cache, packed_cache, native_cache, lengths, pages, cu_q, cu_kv, distribution):
            ids = streamindex_topk(
                q, w, cache, lengths, pages, cu_q, distribution,
                k=2048, compression_ratio=1, num_kv_pages_per_block=(16, 16, 2),
                num_queries_per_block=(1, 128, 128), decode_query_size=queries if queries <= 6 and not ragged else 1,
                segments="decode" if queries <= 6 and not ragged else "mixed",
                selector=selector, enable_early_exit=False,
            )
            _, slots = prepare_gather_indices(ids, pages, cu_q, page_size=64, total_num_pages=cache.shape[0])
            return jnp.where(ids >= 0, slots, -1)
        return call

    def jax_path(packed_input):
        fn = fp8_full if packed_input else native_full
        def call(q, w, cache, packed_cache, native_cache, lengths, pages, cu_q, cu_kv, distribution):
            return fn(
                q.astype(jnp.bfloat16), w, packed_cache if packed_input else native_cache,
                lengths, pages, cu_q, cu_kv, distribution,
                k=2048, pages_per_seq=1440, one_token_per_seq=queries == 1,
                topk_impl="radix", output_physical_slots=True,
                score_query_block_size=6 if queries <= 6 else 32,
                max_queries_per_seq=queries if not ragged else 6,
            )
        return call

    def dsv41_path(select=dsv41_select):
        # An unfiltered DSV4.1 layer: no candidate publication or sparse logits,
        # unordered selection as its attention consumes it. Mapping as torch_*.
        layout_args = (d_layout.pages, d_layout.page_stride, d_layout.page_offset, d_layout.page_size)
        def call(q, w, bank, d_table, requests, d_lens, cu_q, pages):
            ids, _ = select(
                q, (bank,), d_table, w, d_lens, jnp.full((q.shape[0], 1), -1, jnp.int32),
                requests, d_lens.max().reshape(1), cu_q, layout_args,
                2048, 8, False, False, True, d_table.shape[1] * DSV41_PAGE,
            )
            _, slots = prepare_gather_indices(ids, pages, cu_q, page_size=64, total_num_pages=glm_pages)
            return jnp.where(ids >= 0, slots, -1)
        return call

    functions = {"torch_baseline_full": torch_path("sparsecore"),
                 "torch_selector_only_full": torch_path("radix_v2"),
                 "jax_fp8_pipeline_full": jax_path(True),
                 "jax_native_bf16_reference": jax_path(False)}
    inputs = {name: values for name in functions}
    try:
        checked = check_dsv41_cache(d_values[0], d_pages, step, decoded, d_layout)
        emit(dict(kind="dsv41_cache_check", context=context, queries_per_sequence=queries,
                  sequences=len(lens), records_checked=checked, exact=True))
        functions[DSV41] = dsv41_path()
        functions[DSV41_SINGLE] = dsv41_path(dsv41_single.select_fp8_indexer_jax)
        for name in DSV41_ARMS:
            inputs[name] = (values[0], values[1], *d_values, values[7], values[6])
    except Exception as error:
        emit(dict(kind="error", variant=DSV41, stage="cache_repack", context=context,
                  queries_per_sequence=queries, sequences=len(lens), error=repr(error)[-4000:]))
    return step, q8, weights, decoded, inputs, functions, metadata_case


def check_slots(slots, step, q8, weights, decoded, case, variant):
    slots = np.asarray(slots)
    table = step.page_indices.reshape(len(step.kv_lens), -1)
    active = step.num_active_tokens
    # All rows: bounds, causal membership, cardinality and uniqueness.
    sample_rows = set(np.linspace(0, active - 1, min(active, 8), dtype=int).tolist())
    minimum_ratio = 1.0
    checked = []
    for seq in range(len(step.kv_lens)):
        start, end = map(int, step.cu_q_lens[seq:seq+2])
        for row in range(start, end):
            valid = int(step.kv_lens[seq]) - (end - start) + row - start + 1
            pages = table[seq, : (valid + 63) // 64]
            inverse = {int(p): i for i, p in enumerate(pages)}
            output = slots[row]
            picked = output[output >= 0]
            assert picked.size == min(valid, 2048), (case, variant, row, "cardinality", picked.size)
            assert np.unique(picked).size == picked.size, (case, variant, row, "duplicate")
            logical = np.array([inverse.get(int(s // 64), -1000000) * 64 + s % 64 for s in picked])
            assert np.all((logical >= 0) & (logical < valid)), (case, variant, row, "causality/mapping")
            if row in sample_rows:
                keys = decoded[pages].astype(np.float32).reshape(-1, 128)[:valid]
                scores = np.maximum(q8[row].astype(np.float32) @ keys.T, 0)
                scores = (scores * weights[row, :, None]).sum(axis=0)
                cutoff = np.partition(scores, -min(valid, 2048))[-min(valid, 2048)]
                lowest = scores[logical].min()
                # Matches the existing hardware test's rounding tolerance.
                ratio = float(lowest / cutoff) if cutoff > 0 else 1.0
                assert lowest >= cutoff - 0.02 * max(abs(float(cutoff)), 1e-6), (case, variant, row, lowest, cutoff)
                minimum_ratio = min(minimum_ratio, ratio)
                checked.append(row)
    assert np.all(slots[active:] == -1), (case, variant, "padding")
    emit(dict(kind="correctness", case=case, variant=variant, passed=True, all_rows_mapping_causal_unique=True,
              numpy_score_sample_rows=checked, minimum_selected_cutoff_ratio=minimum_ratio))


def run_case(context, queries, sequences, *, validation_only=False, padded=0, ragged=False):
    case = f"c{context}-q{queries}-s{sequences}" + ("-ragged" if ragged else "")
    step, q8, weights, decoded, values, functions, info = make_case(context, queries, sequences, padded=padded, ragged=ragged)
    if args.arms:
        functions = {name: fn for name, fn in functions.items() if name in args.arms}
    emit(dict(kind="case", case=case, **info))
    compiled = {}
    for name, fn in functions.items():
        start = time.perf_counter()
        # The upstream DSV4.1 arm runs off its tuned shapes (K2048, uncompressed
        # 92160 width); its failure is recorded without stopping the other arms.
        try:
            lower = jax.jit(fn).lower(*values[name])
            (args.out / f"{case}-{name}.stablehlo.txt").write_text(lower.as_text())
            compiled[name] = lower.compile()
            output = compiled[name](*values[name])
            jax.block_until_ready(output)
        except Exception as error:
            if name not in DSV41_ARMS:
                raise
            compiled.pop(name, None)
            emit(dict(kind="error", case=case, variant=name, stage="compile_or_first_call",
                      error=repr(error)[-4000:]))
            continue
        emit(dict(kind="compile", case=case, variant=name, compile_first_s=time.perf_counter()-start))
        if args.mode == "bench" or args.decode_profile:
            try:
                check_slots(output, step, q8, weights, decoded, case, name)
            except AssertionError as error:
                if name not in DSV41_ARMS:
                    raise
                emit(dict(kind="correctness", case=case, variant=name, passed=False, error=repr(error)[:2000]))
        for _ in range(5):
            jax.block_until_ready(compiled[name](*values[name]))
    if not validation_only:
        if args.mode == "bench":
            samples = {name: [] for name in compiled}
            names = list(compiled)
            for iteration in range(30):
                for name in names if iteration % 2 == 0 else names[::-1]:
                    start = time.perf_counter_ns()
                    jax.block_until_ready(compiled[name](*values[name]))
                    samples[name].append((time.perf_counter_ns()-start)/1000)
            for name, values_us in samples.items():
                emit(dict(kind="bench", case=case, variant=name, latency_ms=statistics.median(values_us)/1000,
                          median_us=statistics.median(values_us), samples_us=values_us,
                          metric="unprofiled host synchronized whole module", **info))
        else:
            options = jax.profiler.ProfileOptions()
            options.host_tracer_level = 1
            options.python_tracer_level = 0
            options.advanced_configuration = {"tpu_num_chips_to_profile_per_task": 1,
                "tpu_num_sparse_cores_to_trace": 2, "tpu_num_sparse_core_tiles_to_trace": 1}
            for name, fn in compiled.items():
                if name == "jax_native_bf16_reference":
                    continue
                dest = args.out / "xprof" / case / name
                with jax.profiler.trace(str(dest), profiler_options=options):
                    for i in range(5):
                        with jax.profiler.StepTraceAnnotation(case+"_"+name, step_num=i):
                            jax.block_until_ready(fn(*values[name]))
                assert list(dest.glob("plugins/profile/**/*.xplane.pb")), (case, name, "missing trace")
                emit(dict(kind="capture", case=case, variant=name))
    del compiled, values, functions
    jax.clear_caches()


if args.smoke:
    run_case(90000, 6, 1)
elif args.decode_profile:
    for context in (53248, 90000):
        for queries, sequences in ((1, 4), (6, 2), (6, 3), (6, 4)):
            run_case(context, queries, sequences)
elif args.mode == "bench":
    for context in (53248, 90000):
        for queries, sequences in ((1, 4), (6, 1), (6, 2), (6, 3), (6, 4), (2048, 1)):
            run_case(context, queries, sequences)
    run_case(90000, 6, 2, padded=3, ragged=True, validation_only=True)
else:
    for queries, sequences in ((6, 1), (6, 4), (2048, 1)):
        run_case(90000, queries, sequences)
emit(dict(kind="pass", mode=args.mode, marker="GLM52_FULL_INDEXER_PASS"))
