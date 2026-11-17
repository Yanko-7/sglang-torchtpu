#!/usr/bin/env bash
# GLM5.2 index-key layout comparison: the unit tests the change touches, then
# compare.py (whole dsa_indexer_step, 256-byte records vs packed FP8 pages).
set -euo pipefail
ulimit -c 0
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
export PYTHONPATH="$PWD/python:$PWD" JAX_PLATFORMS=tpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
OUT="$ARTIFACT_LOCAL_DIR/rank-0/benchmark"
mkdir -p "$OUT" "$ARTIFACT_LOCAL_DIR/profiling"
"$PY" -m pip freeze > "$ARTIFACT_LOCAL_DIR/profiling/python-packages.txt"
"$PY" -m pip install -q pytest 2>&1 | tee "$OUT/pip-pytest.log"

for test in \
    test/registered/tpu/unit/test_dsv41_fp8_paged_mqa_logits.py \
    test/registered/tpu/unit/test_dsa_index_topk.py \
    test/registered/unit/hardware_backend/tpu/test_mla_fp8_kv.py; do
  log="$OUT/unit-$(basename "$test" .py).log"
  if timeout --signal=KILL 1200 "$PY" -m pytest -v -p no:cacheprovider "$test" > "$log" 2>&1; then
    echo "UNIT_PASS $test"
  else
    echo "UNIT_FAIL $test rc=$?"; tail -n 60 "$log"
  fi
done

"$PY" -u benchmark/tpu/glm52_index_k_layout/compare.py --out "$OUT/compare" \
    --contexts "${CONTEXTS:-53248,90000}" 2>&1 | tee "$OUT/compare.log"
cp "$OUT/compare/metrics.jsonl" "$OUT/metrics.jsonl"
