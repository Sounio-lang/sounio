#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
EDGE_FILE="${SOUNIO_SEMANTIC_ORC_EDGE_FILE:-/workspace/hyperbolic-semantic-networks/data/processed/english_edges_FINAL.csv}"
EDGE_COUNT="${SOUNIO_SEMANTIC_ORC_MATRIX_EDGE_COUNT:-4}"
EDGE_STRIDE="${SOUNIO_SEMANTIC_ORC_EDGE_STRIDE:-13}"
TILE_COUNT="${SOUNIO_SEMANTIC_ORC_TILE_COUNT:-8}"
TILE_STRIDE="${SOUNIO_SEMANTIC_ORC_TILE_STRIDE:-14}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_TILE_MATRIX_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-graph-edge-tile-matrix.XXXXXX)}"
MANIFEST="$OUT_DIR/swow16_graph_edge_tile_matrix_manifest.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-graph-edge-tile-matrix] input=$INPUT_DIR"
echo "[semantic-orc-graph-edge-tile-matrix] edge_file=$EDGE_FILE"
echo "[semantic-orc-graph-edge-tile-matrix] edge_count=$EDGE_COUNT"
echo "[semantic-orc-graph-edge-tile-matrix] edge_stride=$EDGE_STRIDE"
echo "[semantic-orc-graph-edge-tile-matrix] tile_count=$TILE_COUNT"
echo "[semantic-orc-graph-edge-tile-matrix] tile_stride=$TILE_STRIDE"
echo "[semantic-orc-graph-edge-tile-matrix] out=$OUT_DIR"

tile_manifest_paths=()

edge_index=0
while [[ "$edge_index" -lt "$EDGE_COUNT" ]]; do
  edge_dir="$OUT_DIR/edge_${edge_index}"
  mkdir -p "$edge_dir"
  SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
  SOUNIO_SEMANTIC_ORC_EDGE_FILE="$EDGE_FILE" \
  SOUNIO_SEMANTIC_ORC_EDGE_INDEX="$edge_index" \
  SOUNIO_SEMANTIC_ORC_EDGE_STRIDE="$EDGE_STRIDE" \
  SOUNIO_SEMANTIC_ORC_TILE_COUNT="$TILE_COUNT" \
  SOUNIO_SEMANTIC_ORC_TILE_STRIDE="$TILE_STRIDE" \
  SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_TILE_DIR="$edge_dir" \
    bash scripts/ci/semantic_orc_swow16_graph_edge_tile_gate.sh
  tile_manifest_paths+=("$edge_dir/swow16_graph_edge_tile_manifest.json")
  edge_index=$((edge_index + 1))
done

python3 - "$MANIFEST" "$EDGE_COUNT" "$EDGE_STRIDE" "$TILE_COUNT" "$TILE_STRIDE" "${tile_manifest_paths[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
edge_count = int(sys.argv[2])
edge_stride = int(sys.argv[3])
tile_count = int(sys.argv[4])
tile_stride = int(sys.argv[5])
manifest_paths = [pathlib.Path(value) for value in sys.argv[6:]]

if edge_count <= 0:
    raise SystemExit("EDGE_COUNT must be positive")
if tile_count <= 0:
    raise SystemExit("TILE_COUNT must be positive")
if len(manifest_paths) != edge_count:
    raise SystemExit("tile manifest path count mismatch")


def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


records = []
seen_offsets = set()
all_endpoint_free_nodes = set()
all_tile_records = 0
for manifest_path in manifest_paths:
    manifest = json.loads(manifest_path.read_text())
    manifest_edge_index = manifest.get("edge_index", "unknown")
    if manifest.get("schema") != "sounio.semantic_orc.swow16_graph_edge_tile_manifest.v1":
        raise SystemExit(f"unexpected tile manifest schema in {manifest_path}")
    if manifest.get("status") != "pass":
        raise SystemExit(
            f"non-pass tile manifest for edge_index={manifest_edge_index} "
            f"status={manifest.get('status')} path={manifest_path}"
        )
    if int(manifest.get("edge_stride", -1)) != edge_stride:
        raise SystemExit("edge_stride mismatch")
    if int(manifest.get("tile_count", -1)) != tile_count:
        raise SystemExit("tile_count mismatch")
    if int(manifest.get("tile_stride", -1)) != tile_stride:
        raise SystemExit("tile_stride mismatch")
    edge_offset = int(manifest.get("edge_offset", -1))
    if edge_offset in seen_offsets:
        raise SystemExit(f"duplicate edge_offset: {edge_offset}")
    seen_offsets.add(edge_offset)
    records_in_manifest = manifest.get("records", [])
    if len(records_in_manifest) != tile_count:
        raise SystemExit("record count must match tile_count")
    for item in records_in_manifest:
        if int(item.get("tile_index", -1)) < 0:
            raise SystemExit("tile record must carry tile_index")
        selected = item.get("selected_node_ids", [])
        endpoint_free = item.get("endpoint_free_selected_node_ids", [])
        if len(selected) != 16 or len(set(selected)) != 16:
            raise SystemExit("tile record selected_node_ids must contain 16 unique nodes")
        if len(endpoint_free) != 14 or len(set(endpoint_free)) != 14:
            raise SystemExit("tile record endpoint-free support must contain 14 unique nodes")
        all_endpoint_free_nodes.update(int(value) for value in endpoint_free)
        all_tile_records += 1
    records.append(
        {
            "edge_index": int(manifest["edge_index"]),
            "edge_offset": edge_offset,
            "edge_source_node_id": int(manifest["edge_source_node_id"]),
            "edge_target_node_id": int(manifest["edge_target_node_id"]),
            "edge_source_node_name": manifest["edge_source_node_name"],
            "edge_target_node_name": manifest["edge_target_node_name"],
            "tile_count": int(manifest["tile_count"]),
            "tile_manifest": str(manifest_path),
            "tile_manifest_sha256": sha256(manifest_path),
            "unique_endpoint_free_node_count": int(manifest["unique_endpoint_free_node_count"]),
            "endpoint_free_capacity": int(manifest["endpoint_free_capacity"]),
            "pairwise_endpoint_free_tile_jaccard_range": manifest["pairwise_endpoint_free_tile_jaccard_range"],
            "measure_floor_range": manifest["measure_floor_range"],
            "measure_floor_mass_fraction_mu_range": manifest["measure_floor_mass_fraction_mu_range"],
            "measure_floor_mass_fraction_nu_range": manifest["measure_floor_mass_fraction_nu_range"],
            "worst_oracle_max_row_err": float(manifest["worst_oracle_max_row_err"]),
            "worst_oracle_max_col_err": float(manifest["worst_oracle_max_col_err"]),
            "transport_cost_range": manifest["transport_cost_range"],
            "one_minus_transport_over_endpoint_cost_probe_range": manifest["one_minus_transport_over_endpoint_cost_probe_range"],
        }
    )

records.sort(key=lambda item: item["edge_index"])
expected_tiles = edge_count * tile_count
if all_tile_records != expected_tiles:
    raise SystemExit(f"expected {expected_tiles} tile records, found {all_tile_records}")

out_path.write_text(
    json.dumps(
        {
            "schema": "sounio.semantic_orc.swow16_graph_edge_tile_matrix_manifest.v1",
            "status": "pass",
            "edge_count": edge_count,
            "edge_stride": edge_stride,
            "tile_count_per_edge": tile_count,
            "tile_stride": tile_stride,
            "total_edge_tile_fixtures": expected_tiles,
            "fixture_kind": "graph_edge_neighborhood_lazy_walk_edge_by_tile_matrix",
            "runtime_result": f"{expected_tiles}/{expected_tiles} graph-edge support tiles passed generated Sounio check/run and K-AXI pack validation across {edge_count} edges",
            "edge_selection_role": "deterministic arithmetic stride over the sorted usable edge list; engineering smoke-scale only, not representative sampling",
            "edge_sort_key": "usable graph edges are sorted by source_node_id, target_node_id, weight in the graph-edge fixture generator",
            "parameter_space_claim": "this manifest proves only the exact edge_count, edge_stride, tile_count_per_edge, and tile_stride values recorded here; it is not a parameter sweep",
            "numerical_values_are_engineering_diagnostics_only": True,
            "unique_edge_offsets": sorted(seen_offsets),
            "matrix_unique_endpoint_free_node_count": len(all_endpoint_free_nodes),
            "matrix_endpoint_free_tile_positions": expected_tiles * 14,
            "worst_oracle_max_row_err": max(item["worst_oracle_max_row_err"] for item in records),
            "worst_oracle_max_col_err": max(item["worst_oracle_max_col_err"] for item in records),
            "transport_cost_range": [
                min(float(item["transport_cost_range"][0]) for item in records),
                max(float(item["transport_cost_range"][1]) for item in records),
            ],
            "one_minus_transport_over_endpoint_cost_probe_range": [
                min(float(item["one_minus_transport_over_endpoint_cost_probe_range"][0]) for item in records),
                max(float(item["one_minus_transport_over_endpoint_cost_probe_range"][1]) for item in records),
            ],
            "boundaries": [
                "edge_by_tile_matrix_fixture_only",
                "generated_sounio_runtime_gate",
                "kaxi_pack_launch_input_only",
                "no_gpu_runtime_claim",
                "entropic_regularized_transport_only",
                "not_exact_wasserstein_distance",
                "no_sinkhorn_convergence_or_error_bound_certificate",
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
print(f"semantic_orc_swow16_graph_edge_tile_matrix_gate: PASS artifact={out_path}")
PY
