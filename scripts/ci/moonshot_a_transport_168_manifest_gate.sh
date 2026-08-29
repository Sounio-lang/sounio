#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MOONSHOT_A_TRANSPORT_168_MANIFEST_DIR:-$(mktemp -d /tmp/moonshot-a-transport-168-manifest.XXXXXX)}"
mkdir -p "$OUT_DIR"

MOD_DIR="$OUT_DIR/modulation"
LIN_DIR="$OUT_DIR/linearity"
CURV_DIR="$OUT_DIR/curvature"

SOUNIO_MOONSHOT_A_TRANSPORT_168_DIR="$MOD_DIR" bash scripts/ci/moonshot_a_transport_168_modulation_gate.sh >/tmp/moonshot_a_transport_168_modulation_gate.out
SOUNIO_MOONSHOT_A_TRANSPORT_168_LINEARITY_DIR="$LIN_DIR" bash scripts/ci/moonshot_a_transport_168_linearity_gate.sh >/tmp/moonshot_a_transport_168_linearity_gate.out
SOUNIO_MOONSHOT_A_TRANSPORT_168_CURVATURE_DIR="$CURV_DIR" bash scripts/ci/moonshot_a_transport_168_curvature_gate.sh >/tmp/moonshot_a_transport_168_curvature_gate.out

python3 - "$OUT_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])

def digest(path: pathlib.Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

items = {
    "synthetic_subspace": base / "modulation" / "synthetic_subspace.json",
    "synthetic_separable": base / "modulation" / "synthetic_separable.json",
    "linearity": base / "linearity" / "transport_168_linearity_summary.json",
    "curvature": base / "curvature" / "transport_168_curvature_summary.json",
}
real = base / "modulation" / "real_abide_subspace.json"
if real.exists():
    items["real_abide_subspace"] = real

artifacts = {k: digest(v) for k, v in items.items()}
mod = json.loads(items["synthetic_subspace"].read_text())
lin = json.loads(items["linearity"].read_text())
curv = json.loads(items["curvature"].read_text())

manifest = {
    "schema": "sounio.moonshot_a.transport_168_manifest.v1",
    "status": "pass",
    "covariance_mode": "both",
    "modulation_mode": "subspace",
    "artifacts": artifacts,
    "real_status": "emitted" if "real_abide_subspace" in artifacts else "skipped_no_local_abide",
    "summary": {
        "synthetic_shift_summary": mod["shift_summary"],
        "linearity": lin,
        "curvature": curv,
    },
    "boundaries": [
        "hash_binds_modulation_linearity_curvature_artifacts",
        "local_python_evidence_only",
        "does_not_claim_cuda_runtime_acceptance",
        "does_not_claim_full_abide_cohort_revalidation",
        "does_not_claim_final_zd_surgery_semantics",
    ],
}
out = base / "moonshot_a_transport_168_manifest.v1.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"moonshot_a_transport_168_manifest_gate: PASS manifest={out}")
PY

echo "moonshot_a_transport_168_manifest_gate: out=$OUT_DIR"
