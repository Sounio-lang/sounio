#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
EDGE_FILE="${SOUNIO_SEMANTIC_ORC_EDGE_FILE:-/workspace/hyperbolic-semantic-networks/data/processed/english_edges_FINAL.csv}"
EDGE_COUNT="${SOUNIO_SEMANTIC_ORC_EDGE_COUNT:-8}"
EDGE_STRIDE="${SOUNIO_SEMANTIC_ORC_EDGE_STRIDE:-13}"
SHUFFLE_COUNT="${SOUNIO_SEMANTIC_ORC_SHUFFLE_COUNT:-16}"
SWAP_ATTEMPTS="${SOUNIO_SEMANTIC_ORC_SWAP_ATTEMPTS:-256}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_GRAPH_SHUFFLE_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-graph-shuffle.XXXXXX)}"
GRAPH_EDGE_DIR="$OUT_DIR/graph_edge"
MANIFEST="$OUT_DIR/swow16_graph_degree_shuffle_fixture_manifest.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-graph-shuffle] input=$INPUT_DIR"
echo "[semantic-orc-graph-shuffle] edge_file=$EDGE_FILE"
echo "[semantic-orc-graph-shuffle] edge_count=$EDGE_COUNT"
echo "[semantic-orc-graph-shuffle] edge_stride=$EDGE_STRIDE"
echo "[semantic-orc-graph-shuffle] shuffle_count=$SHUFFLE_COUNT"
echo "[semantic-orc-graph-shuffle] swap_attempts=$SWAP_ATTEMPTS"
echo "[semantic-orc-graph-shuffle] out=$OUT_DIR"

SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
SOUNIO_SEMANTIC_ORC_EDGE_FILE="$EDGE_FILE" \
SOUNIO_SEMANTIC_ORC_EDGE_COUNT="$EDGE_COUNT" \
SOUNIO_SEMANTIC_ORC_EDGE_STRIDE="$EDGE_STRIDE" \
SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_MULTI_DIR="$GRAPH_EDGE_DIR" \
  bash scripts/ci/semantic_orc_swow16_graph_edge_multifixture_gate.sh

python3 scripts/research/semantic_orc_swow16_graph_shuffle_fixture.py \
  --multifixture-manifest "$GRAPH_EDGE_DIR/swow16_graph_edge_multifixture_manifest.json" \
  --shuffle-count "$SHUFFLE_COUNT" \
  --swap-attempts "$SWAP_ATTEMPTS" \
  --manifest "$MANIFEST"

python3 - "$MANIFEST" "$EDGE_COUNT" "$SHUFFLE_COUNT" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
edge_count = int(sys.argv[2])
shuffle_count = int(sys.argv[3])
manifest = json.loads(manifest_path.read_text())

if manifest.get("schema") != "sounio.semantic_orc.swow16_graph_degree_shuffle_fixture_manifest.v1":
    raise SystemExit("unexpected degree-shuffle manifest schema")
if manifest.get("status") != "pass":
    raise SystemExit("degree-shuffle manifest did not pass")
if int(manifest.get("edge_fixture_count", -1)) != edge_count:
    raise SystemExit("edge_fixture_count mismatch")
if int(manifest.get("shuffle_count_per_edge_fixture", -1)) != shuffle_count:
    raise SystemExit("shuffle_count_per_edge_fixture mismatch")
if not manifest.get("degree_sequence_preserved_for_all_shuffles", False):
    raise SystemExit("degree sequence was not preserved for every shuffle")
if "not_a_statistical_null_model" not in manifest.get("boundaries", []):
    raise SystemExit("degree-shuffle manifest must reject statistical-null claim")
if "statistical_inference_claims_not_enforced" not in manifest.get("boundaries", []):
    raise SystemExit("degree-shuffle manifest must reject statistical-inference claim")
if "not_an_orc_estimator" not in manifest.get("boundaries", []):
    raise SystemExit("degree-shuffle manifest must reject ORC-estimator claim")
if "induced 16-node support" not in manifest.get("degree_preservation_scope", ""):
    raise SystemExit("degree preservation scope must be induced-support-only")
records = manifest.get("records", [])
if len(records) != edge_count:
    raise SystemExit("record count mismatch")
for record in records:
    shuffles = record.get("shuffle_records", [])
    if len(shuffles) != shuffle_count:
        raise SystemExit("shuffle record count mismatch")
    observed_degrees = record.get("observed_degree_sequence", [])
    if len(observed_degrees) != 16:
        raise SystemExit("observed degree sequence must have 16 entries")
    for shuffle in shuffles:
        if shuffle.get("degree_sequence") != observed_degrees:
            raise SystemExit("shuffle degree sequence mismatch")
        if abs(float(shuffle.get("oracle_mass", 0.0)) - 1.0) >= 0.02:
            raise SystemExit("shuffle oracle mass outside tolerance")
        if float(shuffle.get("oracle_max_row_err", 1.0)) >= 0.015:
            raise SystemExit("shuffle oracle row error outside tolerance")
        if float(shuffle.get("oracle_max_col_err", 1.0)) >= 0.015:
            raise SystemExit("shuffle oracle col error outside tolerance")
PY

echo "[semantic-orc-graph-shuffle] PASS artifact=$MANIFEST"
