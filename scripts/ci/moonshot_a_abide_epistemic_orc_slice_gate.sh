#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="${SOUNIO_MOONSHOT_A_ABIDE_EPI_SLICE_DIR:-$(mktemp -d /tmp/moonshot-a-abide-epi-slice.XXXXXX)}"
SCRIPT="scripts/research/abide_epistemic_orc_slice.py"
mkdir -p "$OUT_DIR"
python3 -m py_compile "$SCRIPT"
PY_RUNNER=(python3)
if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  PY_RUNNER=(uv run --with numpy python)
fi
"${PY_RUNNER[@]}" "$SCRIPT" --synthetic --limit-edges 2 --covariance-mode both --out "$OUT_DIR/synthetic.json"
REAL_STATUS=skipped_no_local_abide
if find artifacts/research/abide -maxdepth 1 -name '*_rois_cc200.1D' -print -quit 2>/dev/null | grep -q .; then
  "${PY_RUNNER[@]}" "$SCRIPT" --limit-edges 1 --covariance-mode both --out "$OUT_DIR/real_abide.json"
  REAL_STATUS=emitted
fi
python3 - "$OUT_DIR/synthetic.json" "$OUT_DIR/real_abide.json" "$REAL_STATUS" <<'PY'
import json, math, sys
for p in [sys.argv[1]] + ([sys.argv[2]] if sys.argv[3] == "emitted" else []):
    d=json.load(open(p))
    assert d["schema"] == "sounio.moonshot_a.abide_epistemic_orc_slice.v1"
    assert d["covariance_mode"] == "both"
    assert d["edges"]
    e=d["edges"][0]
    assert "diagonal" in e["bands"] and "jacobian" in e["bands"]
    assert math.isfinite(float(e["kappa_mean"]))
print(f"moonshot_a_abide_epistemic_orc_slice_gate: PASS synthetic={sys.argv[1]} real_status={sys.argv[3]}")
PY
echo "moonshot_a_abide_epistemic_orc_slice_gate: out=$OUT_DIR"
