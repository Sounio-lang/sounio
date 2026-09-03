#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
SUPPORT_COUNT="${SOUNIO_SEMANTIC_ORC_SUPPORT_COUNT:-4}"
SUPPORT_STRIDE="${SOUNIO_SEMANTIC_ORC_SUPPORT_STRIDE:-16}"
REGIMES="${SOUNIO_SEMANTIC_ORC_REGIMES:-normative anxious ruminative psychotic}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_MULTISUPPORT_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-multisupport.XXXXXX)}"
MANIFEST="$OUT_DIR/swow16_multisupport_manifest.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-multisupport] input=$INPUT_DIR"
echo "[semantic-orc-multisupport] regimes=$REGIMES"
echo "[semantic-orc-multisupport] support_count=$SUPPORT_COUNT"
echo "[semantic-orc-multisupport] support_stride=$SUPPORT_STRIDE"
echo "[semantic-orc-multisupport] out=$OUT_DIR"

manifest_paths=()
pack_paths=()

for regime in $REGIMES; do
  support_index=0
  while [[ "$support_index" -lt "$SUPPORT_COUNT" ]]; do
    support_dir="$OUT_DIR/${regime}/support_${support_index}"
    mkdir -p "$support_dir"
    SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
    SOUNIO_SEMANTIC_ORC_REGIME="$regime" \
    SOUNIO_SEMANTIC_ORC_SUPPORT_INDEX="$support_index" \
    SOUNIO_SEMANTIC_ORC_SUPPORT_STRIDE="$SUPPORT_STRIDE" \
    SOUNIO_SEMANTIC_ORC_KAXI_PACK_DIR="$support_dir" \
      bash scripts/ci/semantic_orc_swow16_kaxi_pack_gate.sh
    manifest_paths+=("$support_dir/swow16_fixture_manifest.json")
    pack_paths+=("$support_dir/swow16_kaxi_pack.json")
    support_index=$((support_index + 1))
  done
done

python3 - "$MANIFEST" "$SUPPORT_COUNT" "$SUPPORT_STRIDE" "${manifest_paths[@]}" -- "${pack_paths[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys
from collections import defaultdict

out_path = pathlib.Path(sys.argv[1])
support_count = int(sys.argv[2])
support_stride = int(sys.argv[3])
sep = sys.argv.index("--")
manifest_paths = [pathlib.Path(value) for value in sys.argv[4:sep]]
pack_paths = [pathlib.Path(value) for value in sys.argv[sep + 1 :]]

EXPECTED_REGIMES = ("normative", "anxious", "ruminative", "psychotic")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if len(manifest_paths) != len(pack_paths):
    raise SystemExit("manifest/pack path count mismatch")
expected_total = len(EXPECTED_REGIMES) * support_count
if len(manifest_paths) != expected_total:
    raise SystemExit(f"expected {expected_total} support packs, got {len(manifest_paths)}")

records = []
nodes_by_regime: dict[str, set[int]] = defaultdict(set)
seen_keys: set[tuple[str, int]] = set()

for manifest_path, pack_path in zip(manifest_paths, pack_paths):
    manifest = json.loads(manifest_path.read_text())
    pack = json.loads(pack_path.read_text())
    if manifest.get("schema") != "sounio.semantic_orc.swow16_fixture_manifest.v1":
        raise SystemExit(f"unexpected fixture manifest schema in {manifest_path}")
    if pack.get("schema") != "sounio.semantic_orc.swow16_kaxi_sinkhorn16_pack.v1":
        raise SystemExit(f"unexpected K-AXI pack schema in {pack_path}")
    if pack.get("status") != "pass":
        raise SystemExit(f"non-pass pack in {pack_path}")
    regime = str(manifest.get("regime", ""))
    support_index = int(manifest.get("support_index", -1))
    key = (regime, support_index)
    if key in seen_keys:
        raise SystemExit(f"duplicate regime/support: {regime}/{support_index}")
    seen_keys.add(key)
    if regime != pack.get("regime"):
        raise SystemExit(f"regime mismatch in {pack_path}")
    if support_index != int(pack.get("support_index", -2)):
        raise SystemExit(f"support_index mismatch in {pack_path}")
    if support_stride != int(manifest.get("support_stride", -1)):
        raise SystemExit(f"support_stride mismatch in {manifest_path}")
    if support_stride != int(pack.get("support_stride", -1)):
        raise SystemExit(f"support_stride mismatch in {pack_path}")
    selected = [int(value) for value in manifest.get("selected_node_ids", [])]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise SystemExit(f"support must contain 16 unique nodes in {manifest_path}")
    overlap = nodes_by_regime[regime].intersection(selected)
    if overlap:
        raise SystemExit(f"support overlap in {regime} support {support_index}: {sorted(overlap)}")
    nodes_by_regime[regime].update(selected)
    oracle = pack.get("oracle") or {}
    if int(oracle.get("iterations", 0)) != 16:
        raise SystemExit(f"K-AXI oracle iteration mismatch in {pack_path}")
    mass = float(oracle.get("mass", 0.0))
    row_err = float(oracle.get("max_row_err", 1.0))
    col_err = float(oracle.get("max_col_err", 1.0))
    if abs(mass - 1.0) >= 0.02 or row_err >= 0.015 or col_err >= 0.015:
        raise SystemExit(f"K-AXI oracle tolerance failure in {pack_path}")
    records.append(
        {
            "regime": regime,
            "support_index": support_index,
            "support_offset": int(manifest.get("support_offset", -1)),
            "selected_node_ids": selected,
            "selected_node_names": manifest.get("selected_node_names", []),
            "fixture_manifest": str(manifest_path),
            "fixture_manifest_sha256": sha256(manifest_path),
            "kaxi_pack": str(pack_path),
            "kaxi_pack_sha256": sha256(pack_path),
            "kaxi_oracle": {
                "mass": mass,
                "max_row_err": row_err,
                "max_col_err": col_err,
                "max_fixed_err": float(oracle.get("max_fixed_err", 1.0)),
                "max_fixed_self_consistency_residual": float(
                    oracle.get("max_fixed_self_consistency_residual", oracle.get("max_fixed_err", 1.0))
                ),
            },
        }
    )

missing = [
    (regime, support_index)
    for regime in EXPECTED_REGIMES
    for support_index in range(support_count)
    if (regime, support_index) not in seen_keys
]
if missing:
    raise SystemExit(f"missing regime/support records: {missing}")

records.sort(key=lambda item: (item["regime"], item["support_index"]))
out_path.write_text(
    json.dumps(
        {
            "schema": "sounio.semantic_orc.swow16_multisupport_manifest.v1",
            "status": "pass",
            "support_count_per_regime": support_count,
            "support_stride": support_stride,
            "fixture_count": len(records),
            "records": records,
            "boundaries": [
                "cpu_pack_and_generated_sounio_runtime_only",
                "no_gpu_runtime_claim",
                "entropic_regularized_transport_only",
                "clinical_claims_not_enforced",
                "no_convergence_theorem",
                "not_a_depression_biomarker_validation",
            ],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"semantic_orc_swow16_multisupport_gate: PASS artifact={out_path}")
PY
