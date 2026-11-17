#!/usr/bin/env bash
# Decode/MTP xprof captures at contexts 53248 and 90000; no bench phase.
# The repo kernel variants (dsv41_repo_kernel/{a,b}) first run the repo port's
# unit test; a variant joins the capture only if its test passes. A hang in a
# variant ends at its timeout instead of the job's.
set -euo pipefail
ulimit -c 0
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
export PYTHONPATH="$PWD/python:$PWD" JAX_PLATFORMS=tpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
BENCH=benchmark/tpu/glm52_indexer_full
OUT="$ARTIFACT_LOCAL_DIR/rank-0/benchmark"
mkdir -p "$OUT" "$ARTIFACT_LOCAL_DIR/profiling" "$ARTIFACT_LOCAL_DIR/rank-0/compiler/llo"
"$PY" -m pip freeze > "$ARTIFACT_LOCAL_DIR/profiling/python-packages.txt"
env | sort > "$ARTIFACT_LOCAL_DIR/profiling/env.txt"

ARMS="torch_baseline_full dsv41_upstream_fp8_full dsv41_single_block_fp8_full"
KERNEL=sglang_tpu/kernels/deepseek_v41/index_selection/fp8/dense_logits.py
cp "$KERNEL" "$OUT/dense_logits.orig.py"
for variant in a b; do
  cp "$BENCH/dsv41_repo_kernel/$variant/dense_logits.py" "$KERNEL"
  if timeout --signal=KILL 900 "$PY" -m pytest -v -p no:cacheprovider \
      "$BENCH/dsv41_repo_kernel/test_dsv41_fp8_paged_mqa_logits.py" 2>&1 | tee "$OUT/unit-$variant.log"; then
    echo "UNIT_PASS $variant"
    ARMS="$ARMS dsv41_skip_pages_${variant}_fp8_full"
  else
    echo "UNIT_FAIL $variant rc=$?"
  fi
done
cp "$OUT/dense_logits.orig.py" "$KERNEL"

export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_xprof_register_llo_debug_info=true --xla_mosaic_dump_to=$ARTIFACT_LOCAL_DIR/rank-0/compiler/llo"
# shellcheck disable=SC2086
"$PY" -u "$BENCH/run.py" --mode profile --decode-profile \
  --arms $ARMS --out "$OUT/profiled" 2>&1 | tee "$OUT/profile.log"
cp "$OUT/profiled/metrics.jsonl" "$OUT/metrics.jsonl"
