#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
EDGE_FILE="${SOUNIO_SEMANTIC_ORC_EDGE_FILE:-/workspace/hyperbolic-semantic-networks/data/processed/english_edges_FINAL.csv}"
EDGE_STRIDE="${SOUNIO_SEMANTIC_ORC_EDGE_STRIDE:-13}"
TILE_STRIDE="${SOUNIO_SEMANTIC_ORC_TILE_STRIDE:-14}"
SWEEP_POINTS="${SOUNIO_SEMANTIC_ORC_PARAMETER_SWEEP_POINTS:-2x4,4x4,4x8}"
GREEN_DEFAULT_POINTS="2x4,4x4,4x8"
ALLOW_EXPERIMENTAL_POINTS="${SOUNIO_SEMANTIC_ORC_ALLOW_EXPERIMENTAL_POINTS:-0}"
PACK_ORACLE_ROW_COL_TOLERANCE="0.015"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_TILE_MATRIX_PARAMETER_SWEEP_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-graph-edge-tile-matrix-parameter-sweep.XXXXXX)}"
MANIFEST="$OUT_DIR/swow16_graph_edge_tile_matrix_parameter_sweep_manifest.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] input=$INPUT_DIR"
echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] edge_file=$EDGE_FILE"
echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] edge_stride=$EDGE_STRIDE"
echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] tile_stride=$TILE_STRIDE"
echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] points=$SWEEP_POINTS"
echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] green_default_points=$GREEN_DEFAULT_POINTS"
echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] allow_experimental_points=$ALLOW_EXPERIMENTAL_POINTS"
echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] pack_oracle_row_col_tolerance=$PACK_ORACLE_ROW_COL_TOLERANCE"
echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] out=$OUT_DIR"

IFS=',' read -r -a point_specs <<<"$SWEEP_POINTS"
point_dirs=()
for raw_spec in "${point_specs[@]}"; do
  spec="${raw_spec//[[:space:]]/}"
  if [[ ! "$spec" =~ ^[0-9]+x[0-9]+$ ]]; then
    echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] FAIL: invalid point spec: $raw_spec" >&2
    exit 1
  fi
  edge_count="${spec%x*}"
  tile_count="${spec#*x}"
  if [[ "$edge_count" -le 0 || "$tile_count" -le 0 ]]; then
    echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] FAIL: counts must be positive: $spec" >&2
    exit 1
  fi
  if [[ ",$GREEN_DEFAULT_POINTS," != *",$spec,"* && "$ALLOW_EXPERIMENTAL_POINTS" != "1" ]]; then
    echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] FAIL: point $spec is outside the green default set $GREEN_DEFAULT_POINTS" >&2
    echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] set SOUNIO_SEMANTIC_ORC_ALLOW_EXPERIMENTAL_POINTS=1 to run an exploratory non-green point" >&2
    exit 1
  fi
  point_dir="$OUT_DIR/edge_${edge_count}_tile_${tile_count}"
  mkdir -p "$point_dir"
  echo "[semantic-orc-graph-edge-tile-matrix-parameter-sweep] point=${edge_count}x${tile_count}"
  SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
  SOUNIO_SEMANTIC_ORC_EDGE_FILE="$EDGE_FILE" \
  SOUNIO_SEMANTIC_ORC_EDGE_STRIDE="$EDGE_STRIDE" \
  SOUNIO_SEMANTIC_ORC_TILE_STRIDE="$TILE_STRIDE" \
  SOUNIO_SEMANTIC_ORC_MATRIX_EDGE_COUNT="$edge_count" \
  SOUNIO_SEMANTIC_ORC_TILE_COUNT="$tile_count" \
  SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_TILE_MATRIX_REDUCER_DIR="$point_dir" \
    bash scripts/ci/semantic_orc_swow16_graph_edge_tile_matrix_reducer_gate.sh
  point_dirs+=("$point_dir")
done

python3 - "$MANIFEST" "$SWEEP_POINTS" "$EDGE_STRIDE" "$TILE_STRIDE" "$GREEN_DEFAULT_POINTS" "$ALLOW_EXPERIMENTAL_POINTS" "$PACK_ORACLE_ROW_COL_TOLERANCE" "${point_dirs[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
sweep_points = sys.argv[2]
edge_stride = int(sys.argv[3])
tile_stride = int(sys.argv[4])
green_default_points = sys.argv[5]
allow_experimental_points = sys.argv[6] == "1"
pack_oracle_row_col_tolerance = float(sys.argv[7])
point_dirs = [pathlib.Path(value) for value in sys.argv[8:]]

if not point_dirs:
    raise SystemExit("parameter sweep must contain at least one point")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


records = []
seen_points = set()
total_fixtures = 0
total_unique_endpoint_free_nodes = 0
total_endpoint_free_positions = 0
worst_row = 0.0
worst_col = 0.0

for point_dir in point_dirs:
    reducer_path = point_dir / "swow16_graph_edge_tile_matrix_reducer_manifest.json"
    matrix_path = point_dir / "matrix" / "swow16_graph_edge_tile_matrix_manifest.json"
    source_path = point_dir / "swow16_graph_edge_tile_matrix_reducer.sio"
    run_out_path = point_dir / "swow16_graph_edge_tile_matrix_reducer.out"
    reducer = json.loads(reducer_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if reducer.get("schema") != "sounio.semantic_orc.swow16_graph_edge_tile_matrix_reducer_manifest.v1":
        raise SystemExit(f"unexpected reducer schema in {reducer_path}")
    if matrix.get("schema") != "sounio.semantic_orc.swow16_graph_edge_tile_matrix_manifest.v1":
        raise SystemExit(f"unexpected matrix schema in {matrix_path}")
    if reducer.get("status") != "pass" or matrix.get("status") != "pass":
        raise SystemExit(f"non-pass point in {point_dir}")
    if sha256(matrix_path) != reducer.get("source_matrix_manifest_sha256"):
        raise SystemExit(f"matrix digest mismatch in {point_dir}")
    if sha256(source_path) != reducer.get("generated_sounio_source_sha256"):
        raise SystemExit(f"reducer source digest mismatch in {point_dir}")
    if "SOUNIO_SWOW16_GRAPH_EDGE_TILE_MATRIX_REDUCER_PASS" not in run_out_path.read_text():
        raise SystemExit(f"missing reducer PASS token in {point_dir}")
    if matrix.get("numerical_values_are_engineering_diagnostics_only") is not True:
        raise SystemExit(f"matrix diagnostics-only sentinel missing in {point_dir}")
    if reducer.get("numerical_values_are_engineering_diagnostics_only") is not True:
        raise SystemExit(f"reducer diagnostics-only sentinel missing in {point_dir}")
    boundaries = set(reducer.get("boundaries", []))
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
        raise SystemExit(f"missing reducer boundaries in {point_dir}: {missing}")
    edge_count = int(reducer["edge_count"])
    tile_count = int(reducer["tile_count_per_edge"])
    total = int(reducer["total_edge_tile_fixtures"])
    point = f"{edge_count}x{tile_count}"
    if point_dir.name != f"edge_{edge_count}_tile_{tile_count}":
        raise SystemExit(f"point directory/manifest mismatch in {point_dir}")
    if point in seen_points:
        raise SystemExit(f"duplicate parameter point: {point}")
    seen_points.add(point)
    if total != edge_count * tile_count:
        raise SystemExit(f"fixture-count mismatch in {point_dir}")
    if int(reducer.get("endpoint_free_positions_per_fixture", -1)) != 14:
        raise SystemExit(f"endpoint-free fixture width mismatch in {point_dir}")
    if int(reducer.get("matrix_record_count", -1)) != edge_count:
        raise SystemExit(f"matrix record count mismatch in {point_dir}")
    if int(reducer.get("matrix_tile_record_total", -1)) != total:
        raise SystemExit(f"matrix tile record total mismatch in {point_dir}")
    if int(matrix.get("edge_stride", -1)) != edge_stride:
        raise SystemExit(f"edge_stride mismatch in {point_dir}")
    if int(matrix.get("tile_stride", -1)) != tile_stride:
        raise SystemExit(f"tile_stride mismatch in {point_dir}")
    if int(matrix.get("edge_count", -1)) != edge_count:
        raise SystemExit(f"matrix edge_count mismatch in {point_dir}")
    if int(matrix.get("tile_count_per_edge", -1)) != tile_count:
        raise SystemExit(f"matrix tile_count_per_edge mismatch in {point_dir}")
    if float(reducer["worst_oracle_max_row_err"]) >= pack_oracle_row_col_tolerance:
        raise SystemExit(f"row diagnostic exceeds pack-oracle tolerance in {point_dir}")
    if float(reducer["worst_oracle_max_col_err"]) >= pack_oracle_row_col_tolerance:
        raise SystemExit(f"col diagnostic exceeds pack-oracle tolerance in {point_dir}")
    total_fixtures += total
    total_unique_endpoint_free_nodes += int(reducer["matrix_unique_endpoint_free_node_count"])
    total_endpoint_free_positions += int(reducer["matrix_endpoint_free_tile_positions"])
    worst_row = max(worst_row, float(reducer["worst_oracle_max_row_err"]))
    worst_col = max(worst_col, float(reducer["worst_oracle_max_col_err"]))
    records.append(
        {
            "point": point,
            "edge_count": edge_count,
            "tile_count_per_edge": tile_count,
            "total_edge_tile_fixtures": total,
            "matrix_unique_endpoint_free_node_count": int(reducer["matrix_unique_endpoint_free_node_count"]),
            "matrix_endpoint_free_tile_positions": int(reducer["matrix_endpoint_free_tile_positions"]),
            "endpoint_free_positions_per_fixture": int(reducer["endpoint_free_positions_per_fixture"]),
            "matrix_record_count": int(reducer["matrix_record_count"]),
            "matrix_tile_record_total": int(reducer["matrix_tile_record_total"]),
            "worst_oracle_max_row_err": float(reducer["worst_oracle_max_row_err"]),
            "worst_oracle_max_col_err": float(reducer["worst_oracle_max_col_err"]),
            "matrix_manifest": str(matrix_path),
            "matrix_manifest_sha256": sha256(matrix_path),
            "reducer_manifest": str(reducer_path),
            "reducer_manifest_sha256": sha256(reducer_path),
            "generated_sounio_source": str(source_path),
            "generated_sounio_source_sha256": sha256(source_path),
        }
    )

records.sort(key=lambda item: (item["edge_count"], item["tile_count_per_edge"]))
out_path.write_text(
    json.dumps(
        {
            "schema": "sounio.semantic_orc.swow16_graph_edge_tile_matrix_parameter_sweep_manifest.v1",
            "status": "pass",
            "sweep_points": sweep_points,
            "green_default_points": green_default_points,
            "experimental_points_enabled": allow_experimental_points,
            "upstream_pack_oracle_row_col_tolerance": pack_oracle_row_col_tolerance,
            "manual_sinkhorn_lse_orc_tolerances_applied": False,
            "edge_stride": edge_stride,
            "tile_stride": tile_stride,
            "point_count": len(records),
            "total_edge_tile_fixtures": total_fixtures,
            "sum_matrix_unique_endpoint_free_node_counts": total_unique_endpoint_free_nodes,
            "sum_matrix_endpoint_free_tile_positions": total_endpoint_free_positions,
            "worst_oracle_max_row_err": worst_row,
            "worst_oracle_max_col_err": worst_col,
            "runtime_result": f"{len(records)}/{len(records)} explicit parameter points passed generated Sounio reducer gates",
            "parameter_space_claim": "this manifest proves only the exact edge_count x tile_count points recorded in records; it is not interpolation, extrapolation, sampling, coverage, representativeness, or a full parameter sweep over the SWOW graph",
            "point_selection_claim": "default execution is restricted to the recorded green points; non-default exploratory points require SOUNIO_SEMANTIC_ORC_ALLOW_EXPERIMENTAL_POINTS=1 and are still exact-point evidence only",
            "numerical_values_are_engineering_diagnostics_only": True,
            "boundaries": [
                "explicit_parameter_points_only",
                "parameter_sweep_fixture_only",
                "generated_sounio_reducer_gate",
                "does_not_recompute_transport_at_sweep_level",
                "no_gpu_runtime_claim",
                "entropic_regularized_transport_only",
                "not_exact_wasserstein_distance",
                "not_exact_ollivier_ricci_curvature",
                "not_an_orc_estimator",
                "not_a_biomarker",
                "not_a_clinical_claim",
                "not_statistical_inference",
                "not_population_level_evidence",
                "not_generalizability_evidence",
            ],
            "records": records,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"semantic_orc_swow16_graph_edge_tile_matrix_parameter_sweep_gate: PASS artifact={out_path}")
PY
