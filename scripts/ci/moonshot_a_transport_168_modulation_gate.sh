#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="${SOUNIO_MOONSHOT_A_TRANSPORT_168_DIR:-$(mktemp -d /tmp/moonshot-a-transport-168.XXXXXX)}"
SCRIPT="scripts/research/transport_168_modulation.py"
mkdir -p "$OUT_DIR"
python3 -m py_compile "$SCRIPT"
PY_RUNNER=(python3)
if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  PY_RUNNER=(uv run --with numpy python)
fi
"${PY_RUNNER[@]}" "$SCRIPT" --synthetic --classes 168 --mode subspace --covariance-mode both --out "$OUT_DIR/synthetic_subspace.json"
"${PY_RUNNER[@]}" "$SCRIPT" --synthetic --classes 168 --mode separable --covariance-mode both --out "$OUT_DIR/synthetic_separable.json"
REAL_STATUS=skipped_no_local_abide
if find artifacts/research/abide -maxdepth 1 -name '*_rois_cc200.1D' -print -quit 2>/dev/null | grep -q .; then
  "${PY_RUNNER[@]}" "$SCRIPT" --classes 168 --mode subspace --covariance-mode both --out "$OUT_DIR/real_abide_subspace.json"
  REAL_STATUS=emitted
fi
python3 - "$OUT_DIR/synthetic_subspace.json" "$OUT_DIR/synthetic_separable.json" "$REAL_STATUS" <<'PY'
import json, sys
sub=json.load(open(sys.argv[1])); sep=json.load(open(sys.argv[2]))
assert sub["classes_evaluated"] == 168 and sep["classes_evaluated"] == 168
assert sub["zd_census"]["valid_primitives"] == 84
assert sub["zd_census"]["runtime_slots_evaluated"] == 168
sub_shifts=[c["kappa_mean"]-sub["baseline"]["kappa_mean"] for c in sub["classes"] if not c.get("is_padding", False)]
sep_shifts=[c["kappa_mean"]-sep["baseline"]["kappa_mean"] for c in sep["classes"] if not c.get("is_padding", False)]
sub_span=max(abs(min(sub_shifts)),abs(max(sub_shifts)))
sep_span=max(abs(min(sep_shifts)),abs(max(sep_shifts)))
assert sub_span > 1e-5
assert sep_span < sub_span * 0.02
print(f"moonshot_a_transport_168_modulation_gate: PASS subspace_span={sub_span:.6g} separable_span={sep_span:.6g} real_status={sys.argv[3]}")
PY
echo "moonshot_a_transport_168_modulation_gate: out=$OUT_DIR"
