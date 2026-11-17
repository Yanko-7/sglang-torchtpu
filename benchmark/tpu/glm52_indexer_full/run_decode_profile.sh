#!/usr/bin/env bash
# Decode/MTP xprof captures at contexts 53248 and 90000 for torch_baseline_full
# and dsv41_upstream_fp8_full only; no bench phase.
set -euo pipefail
ulimit -c 0
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
export PYTHONPATH="$PWD/python:$PWD" JAX_PLATFORMS=tpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
OUT="$ARTIFACT_LOCAL_DIR/rank-0/benchmark"
mkdir -p "$OUT" "$ARTIFACT_LOCAL_DIR/profiling" "$ARTIFACT_LOCAL_DIR/rank-0/compiler/llo"
"$PY" -m pip freeze > "$ARTIFACT_LOCAL_DIR/profiling/python-packages.txt"
env | sort > "$ARTIFACT_LOCAL_DIR/profiling/env.txt"
export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_xprof_register_llo_debug_info=true --xla_mosaic_dump_to=$ARTIFACT_LOCAL_DIR/rank-0/compiler/llo"
"$PY" -u benchmark/tpu/glm52_indexer_full/run.py --mode profile --decode-profile \
  --arms torch_baseline_full dsv41_upstream_fp8_full --out "$OUT/profiled" 2>&1 | tee "$OUT/profile.log"
cp "$OUT/profiled/metrics.jsonl" "$OUT/metrics.jsonl"
