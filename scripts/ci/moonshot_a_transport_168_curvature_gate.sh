#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="${SOUNIO_MOONSHOT_A_TRANSPORT_168_CURVATURE_DIR:-$(mktemp -d /tmp/moonshot-a-transport-168-curvature.XXXXXX)}"
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
  flags=""; [[ "$label" == synthetic ]] && flags="--synthetic"
  [[ "$label" == real ]] && ! find artifacts/research/abide -maxdepth 1 -name '*_rois_cc200.1D' -print -quit 2>/dev/null | grep -q . && continue
  for s in 0 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.046875 0.0625; do "${PY_RUNNER[@]}" "$SCRIPT" $flags --classes 168 --mode subspace --covariance-mode diagonal --strength "$s" --out "$OUT_DIR/${label}_${s}.json" >/dev/null; done
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
    x=np.array(x); y=np.array(y).T; X=np.vstack([x,x*x]).T; coef=np.linalg.lstsq(X,y.T,rcond=None)[0].T
    pred=(X @ coef.T).T
    lin=abs(coef[:,0]*x.max()); quad=abs(coef[:,1]*x.max()*x.max()); ratio=np.where(lin>1e-12,quad/lin,0)
    max_abs_y=float(np.max(np.abs(y)))
    max_res=float(np.max(np.abs(y-pred)))
    datasets.append({"label":label,"strengths":x.tolist(),"significant_classes":int((np.max(abs(y),axis=1)>=1e-5).sum()),"argmin_class_index":int(np.argmin(y[:,-1])),"argmax_class_index":int(np.argmax(y[:,-1])),"max_quadratic_to_linear_at_max_strength":float(ratio.max()),"p95_quadratic_to_linear_at_max_strength":float(np.quantile(ratio,.95)),"max_quadratic_fit_residual":max_res,"max_quadratic_fit_relative_residual":float(max_res / max(max_abs_y, 1e-30)),"quadratic_fit_design_condition_number":float(np.linalg.cond(X))})
summary={"schema":"sounio.moonshot_a.transport_168_curvature_gate.v1","covariance_mode":"diagonal","modulation_mode":"subspace","real_status":"emitted" if any(d["label"]=="real" for d in datasets) else "skipped_no_local_abide","thresholds":{"max_quadratic_to_linear_at_max_strength":0.2,"max_quadratic_fit_residual":1e-8,"max_quadratic_fit_relative_residual":0.001,"max_design_condition_number":100.0},"datasets":datasets}
(base/"transport_168_curvature_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
assert datasets, "no synthetic or real datasets were produced"
for d in datasets:
    assert d["max_quadratic_to_linear_at_max_strength"] <= summary["thresholds"]["max_quadratic_to_linear_at_max_strength"], d
    assert d["max_quadratic_fit_residual"] <= summary["thresholds"]["max_quadratic_fit_residual"], d
    assert d["max_quadratic_fit_relative_residual"] <= summary["thresholds"]["max_quadratic_fit_relative_residual"], d
    assert d["quadratic_fit_design_condition_number"] <= summary["thresholds"]["max_design_condition_number"], d
print("moonshot_a_transport_168_curvature_gate: PASS " + " ".join(f"{d['label']}_max_quad_ratio={d['max_quadratic_to_linear_at_max_strength']:.6g}" for d in datasets) + f" real_status={summary['real_status']}")
PY
echo "moonshot_a_transport_168_curvature_gate: summary=$OUT_DIR/transport_168_curvature_summary.json"
