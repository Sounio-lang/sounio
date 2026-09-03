#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-bin/souc}"
INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
EDGE_FILE="${SOUNIO_SEMANTIC_ORC_EDGE_FILE:-/workspace/hyperbolic-semantic-networks/data/processed/english_edges_FINAL.csv}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_TILE_MATRIX_REDUCER_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-graph-edge-tile-matrix-reducer.XXXXXX)}"
MATRIX_DIR="$OUT_DIR/matrix"
MATRIX_MANIFEST="$MATRIX_DIR/swow16_graph_edge_tile_matrix_manifest.json"
SOURCE="$OUT_DIR/swow16_graph_edge_tile_matrix_reducer.sio"
MANIFEST="$OUT_DIR/swow16_graph_edge_tile_matrix_reducer_manifest.json"
RUN_OUT="$OUT_DIR/swow16_graph_edge_tile_matrix_reducer.out"
EDGE_COUNT="${SOUNIO_SEMANTIC_ORC_MATRIX_EDGE_COUNT:-4}"
TILE_COUNT="${SOUNIO_SEMANTIC_ORC_TILE_COUNT:-8}"
TOTAL_EDGE_TILE_FIXTURES=$((EDGE_COUNT * TILE_COUNT))

mkdir -p "$OUT_DIR"

echo "[semantic-orc-graph-edge-tile-matrix-reducer] input=$INPUT_DIR"
echo "[semantic-orc-graph-edge-tile-matrix-reducer] edge_file=$EDGE_FILE"
echo "[semantic-orc-graph-edge-tile-matrix-reducer] out=$OUT_DIR"

SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
SOUNIO_SEMANTIC_ORC_EDGE_FILE="$EDGE_FILE" \
SOUNIO_SEMANTIC_ORC_MATRIX_EDGE_COUNT="$EDGE_COUNT" \
SOUNIO_SEMANTIC_ORC_TILE_COUNT="$TILE_COUNT" \
SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_TILE_MATRIX_DIR="$MATRIX_DIR" \
  bash scripts/ci/semantic_orc_swow16_graph_edge_tile_matrix_gate.sh

python3 scripts/research/semantic_orc_swow16_matrix_reducer_fixture.py \
  --matrix-manifest "$MATRIX_MANIFEST" \
  --sounio-out "$SOURCE" \
  --manifest "$MANIFEST"

"$SOUC" check "$SOURCE" >/dev/null
"$SOUC" run "$SOURCE" >"$RUN_OUT"

required_tokens=(
  "SOUNIO_SWOW16_GRAPH_EDGE_TILE_MATRIX_REDUCER_PASS"
  "claim_level=swow16_graph_edge_tile_matrix_reducer_fixture"
  "fixture_source=swow_english_final_graph_edge_tile_matrix_manifest"
  "edge_count=$EDGE_COUNT"
  "tile_count_per_edge=$TILE_COUNT"
  "total_edge_tile_fixtures=$TOTAL_EDGE_TILE_FIXTURES"
  "diagnostic_values_only=true"
  "fixed_iteration_sinkhorn16_only=true"
  "entropic_regularized_transport_only=true"
  "not_exact_wasserstein_distance=true"
  "not_exact_ollivier_ricci_curvature=true"
  "not_an_orc_estimator=true"
  "clinical_claims_not_enforced=true"
  "biomarker_claims_not_enforced=true"
  "statistical_inference_claims_not_enforced=true"
  "gpu_runtime_claims_not_enforced=true"
  "population_level_claims_not_enforced=true"
  "no_convergence_theorem=true"
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fqx "$token" "$RUN_OUT"; then
    echo "[semantic-orc-graph-edge-tile-matrix-reducer] FAIL: missing token: $token" >&2
    cat "$RUN_OUT" >&2
    exit 1
  fi
done

python3 - "$MANIFEST" "$MATRIX_MANIFEST" "$SOURCE" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path, matrix_path, source_path = [pathlib.Path(value) for value in sys.argv[1:4]]
manifest = json.loads(manifest_path.read_text())
matrix = json.loads(matrix_path.read_text())


def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if manifest.get("schema") != "sounio.semantic_orc.swow16_graph_edge_tile_matrix_reducer_manifest.v1":
    raise SystemExit("unexpected reducer manifest schema")
if manifest.get("status") != "pass":
    raise SystemExit("reducer manifest must be pass")
if matrix.get("schema") != "sounio.semantic_orc.swow16_graph_edge_tile_matrix_manifest.v1":
    raise SystemExit("unexpected source matrix manifest schema")
if int(manifest.get("total_edge_tile_fixtures", -1)) != int(matrix.get("total_edge_tile_fixtures", -2)):
    raise SystemExit("reducer/matrix fixture-count mismatch")
if manifest.get("numerical_values_are_engineering_diagnostics_only") is not True:
    raise SystemExit("reducer manifest must keep diagnostic-only sentinel")
if sha256(matrix_path) != manifest.get("source_matrix_manifest_sha256"):
    raise SystemExit("source matrix manifest digest mismatch")
if sha256(source_path) != manifest.get("generated_sounio_source_sha256"):
    raise SystemExit("generated Sounio source digest mismatch")
boundaries = set(manifest.get("boundaries", []))
required = {
    "sounio_reducer_fixture_only",
    "manifest_invariant_check_only",
    "does_not_recompute_transport",
    "not_exact_wasserstein_distance",
    "not_exact_ollivier_ricci_curvature",
    "not_an_orc_estimator",
    "not_statistical_inference",
}
missing = sorted(required - boundaries)
if missing:
    raise SystemExit(f"missing reducer boundaries: {missing}")
PY

echo "[semantic-orc-graph-edge-tile-matrix-reducer] PASS artifact=$MANIFEST"
