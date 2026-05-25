#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="${SOUNIO_MOONSHOT_A_TRANSPORT_168_LINEARITY_DIR:-$(mktemp -d /tmp/moonshot-a-transport-168-linearity.XXXXXX)}"
SCRIPT="scripts/research/transport_168_modulation.py"
mkdir -p "$OUT_DIR"
PY_RUNNER=(python3)
if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  PY_RUNNER=(uv run --with numpy python)
fi
for label in synthetic real; do
  flags=""
  [[ "$label" == synthetic ]] && flags="--synthetic"
  [[ "$label" == real ]] && ! find artifacts/research/abide -maxdepth 1 -name '*_rois_cc200.1D' -print -quit 2>/dev/null | grep -q . && continue
  for s in 0 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.046875 0.0625; do
    "${PY_RUNNER[@]}" "$SCRIPT" $flags --classes 168 --mode subspace --covariance-mode diagonal --strength "$s" --out "$OUT_DIR/${label}_${s}.json" >/dev/null
  done
done
"${PY_RUNNER[@]}" - "$OUT_DIR" <<'PY'
import json, sys, pathlib, numpy as np
base=pathlib.Path(sys.argv[1]); datasets=[]
for label in ("synthetic","real"):
    paths=sorted(base.glob(f"{label}_*.json"), key=lambda p: float(p.stem.split("_")[1]))
    if not paths: continue
    x=[]; y=[]
    for p in paths:
        s=float(p.stem.split("_")[1]); d=json.load(open(p)); x.append(s); y.append([c["kappa_mean"]-d["baseline"]["kappa_mean"] for c in d["classes"] if not c.get("is_padding", False)])
    x=np.array(x); y=np.array(y).T; slope=(y@x)/(x@x); pred=slope[:,None]*x[None,:]
    ssr=np.sum((y-pred)**2,axis=1); sst=np.sum((y-y.mean(axis=1,keepdims=True))**2,axis=1); r2=1-ssr/sst
    datasets.append({"label":label,"strengths":x.tolist(),"significant_classes":int((np.max(abs(y),axis=1)>=1e-5).sum()),"min_r2":float(np.nanmin(r2)),"max_abs_prediction_error":float(np.max(abs(y-pred))),"argmin_class_index":int(np.argmin(y[:,-1])),"argmax_class_index":int(np.argmax(y[:,-1]))})
summary={"schema":"sounio.moonshot_a.transport_168_linearity_gate.v1","covariance_mode":"diagonal","modulation_mode":"subspace","real_status":"emitted" if any(d["label"]=="real" for d in datasets) else "skipped_no_local_abide","thresholds":{"min_r2":0.995},"datasets":datasets}
(base/"transport_168_linearity_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
assert datasets, "no synthetic or real datasets were produced"
for d in datasets:
    assert d["min_r2"] >= summary["thresholds"]["min_r2"], d
print("moonshot_a_transport_168_linearity_gate: PASS " + " ".join(f"{d['label']}_min_r2={d['min_r2']:.6g}" for d in datasets) + f" real_status={summary['real_status']}")
PY
echo "moonshot_a_transport_168_linearity_gate: summary=$OUT_DIR/transport_168_linearity_summary.json"
