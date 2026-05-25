#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
EDGE_FILE="${SOUNIO_SEMANTIC_ORC_EDGE_FILE:-/workspace/hyperbolic-semantic-networks/data/processed/english_edges_FINAL.csv}"
EDGE_INDEX="${SOUNIO_SEMANTIC_ORC_EDGE_INDEX:-0}"
EDGE_STRIDE="${SOUNIO_SEMANTIC_ORC_EDGE_STRIDE:-1}"
TILE_COUNT="${SOUNIO_SEMANTIC_ORC_TILE_COUNT:-8}"
TILE_STRIDE="${SOUNIO_SEMANTIC_ORC_TILE_STRIDE:-14}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_TILE_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-graph-edge-tile.XXXXXX)}"
MANIFEST="$OUT_DIR/swow16_graph_edge_tile_manifest.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-graph-edge-tile] input=$INPUT_DIR"
echo "[semantic-orc-graph-edge-tile] edge_file=$EDGE_FILE"
echo "[semantic-orc-graph-edge-tile] edge_index=$EDGE_INDEX"
echo "[semantic-orc-graph-edge-tile] edge_stride=$EDGE_STRIDE"
echo "[semantic-orc-graph-edge-tile] tile_count=$TILE_COUNT"
echo "[semantic-orc-graph-edge-tile] tile_stride=$TILE_STRIDE"
echo "[semantic-orc-graph-edge-tile] out=$OUT_DIR"

manifest_paths=()
pack_paths=()

tile_index=0
while [[ "$tile_index" -lt "$TILE_COUNT" ]]; do
  tile_dir="$OUT_DIR/tile_${tile_index}"
  mkdir -p "$tile_dir"
  SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
  SOUNIO_SEMANTIC_ORC_EDGE_FILE="$EDGE_FILE" \
  SOUNIO_SEMANTIC_ORC_EDGE_INDEX="$EDGE_INDEX" \
  SOUNIO_SEMANTIC_ORC_EDGE_STRIDE="$EDGE_STRIDE" \
  SOUNIO_SEMANTIC_ORC_TILE_INDEX="$tile_index" \
  SOUNIO_SEMANTIC_ORC_TILE_STRIDE="$TILE_STRIDE" \
  SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_PACK_DIR="$tile_dir" \
    bash scripts/ci/semantic_orc_swow16_graph_edge_kaxi_pack_gate.sh
  manifest_paths+=("$tile_dir/swow16_graph_edge_fixture_manifest.json")
  pack_paths+=("$tile_dir/swow16_graph_edge_kaxi_pack.json")
  tile_index=$((tile_index + 1))
done

python3 - "$MANIFEST" "$EDGE_INDEX" "$EDGE_STRIDE" "$TILE_COUNT" "$TILE_STRIDE" "${manifest_paths[@]}" -- "${pack_paths[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
edge_index = int(sys.argv[2])
edge_stride = int(sys.argv[3])
tile_count = int(sys.argv[4])
tile_stride = int(sys.argv[5])
sep = sys.argv.index("--")
manifest_paths = [pathlib.Path(value) for value in sys.argv[6:sep]]
pack_paths = [pathlib.Path(value) for value in sys.argv[sep + 1 :]]

if tile_count <= 0:
    raise SystemExit("TILE_COUNT must be positive")
if tile_stride <= 0:
    raise SystemExit("TILE_STRIDE must be positive")
if len(manifest_paths) != tile_count or len(pack_paths) != tile_count:
    raise SystemExit("tile fixture path count mismatch")


def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jaccard(left, right):
    left = set(left)
    right = set(right)
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


records = []
seen_tile_indices = set()
edge_identity = None
all_endpoint_free_nodes = set()
pairwise_jaccards = []
for manifest_path, pack_path in zip(manifest_paths, pack_paths):
    manifest = json.loads(manifest_path.read_text())
    pack = json.loads(pack_path.read_text())
    if manifest.get("schema") != "sounio.semantic_orc.swow16_graph_edge_fixture_manifest.v1":
        raise SystemExit(f"unexpected graph-edge manifest schema in {manifest_path}")
    if pack.get("schema") != "sounio.semantic_orc.swow16_kaxi_sinkhorn16_pack.v1":
        raise SystemExit(f"unexpected K-AXI pack schema in {pack_path}")
    if manifest.get("fixture_kind") != "graph_edge_neighborhood_lazy_walk":
        raise SystemExit("fixture_kind mismatch")
    if int(manifest.get("edge_index", -1)) != edge_index:
        raise SystemExit("edge_index mismatch")
    if int(manifest.get("edge_stride", -1)) != edge_stride:
        raise SystemExit("edge_stride mismatch")
    if int(manifest.get("tile_stride", -1)) != tile_stride:
        raise SystemExit("tile_stride mismatch")
    tile_index = int(manifest.get("tile_index", -1))
    if tile_index in seen_tile_indices:
        raise SystemExit(f"duplicate tile_index: {tile_index}")
    seen_tile_indices.add(tile_index)
    if int(pack.get("tile_index", -2)) != tile_index:
        raise SystemExit("manifest/pack tile_index mismatch")
    if int(pack.get("tile_stride", -2)) != tile_stride:
        raise SystemExit("manifest/pack tile_stride mismatch")
    source = int(manifest["edge_source_node_id"])
    target = int(manifest["edge_target_node_id"])
    identity = (
        int(manifest["edge_offset"]),
        source,
        target,
        manifest["edge_source_node_name"],
        manifest["edge_target_node_name"],
    )
    if edge_identity is None:
        edge_identity = identity
    elif identity != edge_identity:
        raise SystemExit("all tiles must preserve the same selected graph edge")
    selected = [int(value) for value in manifest.get("selected_node_ids", [])]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise SystemExit("selected graph support must contain 16 unique nodes")
    if source not in selected or target not in selected:
        raise SystemExit("selected support must include graph edge endpoints")
    support_stats = manifest.get("support_stats", {})
    if int(support_stats.get("tile_extra_candidate_count", 0)) != 14:
        raise SystemExit("each tile must contribute 14 non-endpoint nodes")
    if int(support_stats.get("source_incident_support_neighbors", 0)) <= 0:
        raise SystemExit("source endpoint has no support neighbor")
    if int(support_stats.get("target_incident_support_neighbors", 0)) <= 0:
        raise SystemExit("target endpoint has no support neighbor")
    oracle = pack.get("oracle", {})
    if oracle.get("iterations") != 16:
        raise SystemExit("pack oracle must use 16 iterations")
    if abs(float(oracle.get("mass", 0.0)) - 1.0) >= 0.02:
        raise SystemExit("pack oracle mass outside tolerance")
    if float(oracle.get("max_row_err", 1.0)) >= 0.015:
        raise SystemExit("pack oracle row error outside tolerance")
    if float(oracle.get("max_col_err", 1.0)) >= 0.015:
        raise SystemExit("pack oracle col error outside tolerance")
    endpoint_free_nodes = [node_id for node_id in selected if node_id not in {source, target}]
    for previous in records:
        pairwise_jaccards.append(jaccard(endpoint_free_nodes, previous["endpoint_free_selected_node_ids"]))
    all_endpoint_free_nodes.update(endpoint_free_nodes)
    probe = manifest.get("edge_transport_probe", {})
    records.append(
        {
            "tile_index": tile_index,
            "tile_extra_candidate_offset": int(support_stats["tile_extra_candidate_offset"]),
            "selected_node_ids": selected,
            "endpoint_free_selected_node_ids": endpoint_free_nodes,
            "fixture_manifest": str(manifest_path),
            "fixture_manifest_sha256": sha256(manifest_path),
            "kaxi_pack": str(pack_path),
            "kaxi_pack_sha256": sha256(pack_path),
            "induced_undirected_edges": int(support_stats["induced_undirected_edges"]),
            "source_incident_support_neighbors": int(support_stats["source_incident_support_neighbors"]),
            "target_incident_support_neighbors": int(support_stats["target_incident_support_neighbors"]),
            "measure_floor": float(manifest["measure_mu_stats"]["floor"]),
            "measure_floor_mass_fraction_mu": float(manifest["measure_mu_stats"]["floor_mass_fraction"]),
            "measure_floor_mass_fraction_nu": float(manifest["measure_nu_stats"]["floor_mass_fraction"]),
            "transport_cost": float(probe["transport_cost"]),
            "one_minus_transport_over_endpoint_cost_probe": float(probe["one_minus_transport_over_endpoint_cost_probe"]),
            "oracle_mass": float(oracle["mass"]),
            "oracle_max_row_err": float(oracle["max_row_err"]),
            "oracle_max_col_err": float(oracle["max_col_err"]),
        }
    )

expected_tile_indices = set(range(tile_count))
if seen_tile_indices != expected_tile_indices:
    raise SystemExit(f"tile_index set mismatch: got {sorted(seen_tile_indices)}")

records.sort(key=lambda item: item["tile_index"])
edge_offset, source, target, source_name, target_name = edge_identity
out_path.write_text(
    json.dumps(
        {
            "schema": "sounio.semantic_orc.swow16_graph_edge_tile_manifest.v1",
            "status": "pass",
            "edge_index": edge_index,
            "edge_stride": edge_stride,
            "edge_offset": edge_offset,
            "edge_source_node_id": source,
            "edge_target_node_id": target,
            "edge_source_node_name": source_name,
            "edge_target_node_name": target_name,
            "tile_count": tile_count,
            "tile_stride": tile_stride,
            "fixture_kind": "graph_edge_neighborhood_lazy_walk_tiled_supports",
            "runtime_result": f"{tile_count}/{tile_count} graph-edge support tiles passed generated Sounio check/run and K-AXI pack validation",
            "unique_endpoint_free_node_count": len(all_endpoint_free_nodes),
            "endpoint_free_capacity": tile_count * 14,
            "pairwise_endpoint_free_tile_jaccard_range": [
                min(pairwise_jaccards) if pairwise_jaccards else 1.0,
                max(pairwise_jaccards) if pairwise_jaccards else 1.0,
            ],
            "measure_floor_range": [
                min(item["measure_floor"] for item in records),
                max(item["measure_floor"] for item in records),
            ],
            "measure_floor_mass_fraction_mu_range": [
                min(item["measure_floor_mass_fraction_mu"] for item in records),
                max(item["measure_floor_mass_fraction_mu"] for item in records),
            ],
            "measure_floor_mass_fraction_nu_range": [
                min(item["measure_floor_mass_fraction_nu"] for item in records),
                max(item["measure_floor_mass_fraction_nu"] for item in records),
            ],
            "worst_oracle_max_row_err": max(item["oracle_max_row_err"] for item in records),
            "worst_oracle_max_col_err": max(item["oracle_max_col_err"] for item in records),
            "transport_cost_range": [
                min(item["transport_cost"] for item in records),
                max(item["transport_cost"] for item in records),
            ],
            "one_minus_transport_over_endpoint_cost_probe_range": [
                min(item["one_minus_transport_over_endpoint_cost_probe"] for item in records),
                max(item["one_minus_transport_over_endpoint_cost_probe"] for item in records),
            ],
            "boundaries": [
                "tiled_support_fixture_only",
                "same_real_graph_edge_across_tiles",
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
print(f"semantic_orc_swow16_graph_edge_tile_gate: PASS artifact={out_path}")
PY
