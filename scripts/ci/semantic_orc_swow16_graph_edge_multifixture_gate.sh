#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
EDGE_FILE="${SOUNIO_SEMANTIC_ORC_EDGE_FILE:-/workspace/hyperbolic-semantic-networks/data/processed/english_edges_FINAL.csv}"
EDGE_COUNT="${SOUNIO_SEMANTIC_ORC_EDGE_COUNT:-8}"
EDGE_STRIDE="${SOUNIO_SEMANTIC_ORC_EDGE_STRIDE:-13}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_MULTI_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-graph-edge-multi.XXXXXX)}"
MANIFEST="$OUT_DIR/swow16_graph_edge_multifixture_manifest.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-graph-edge-multi] input=$INPUT_DIR"
echo "[semantic-orc-graph-edge-multi] edge_file=$EDGE_FILE"
echo "[semantic-orc-graph-edge-multi] edge_count=$EDGE_COUNT"
echo "[semantic-orc-graph-edge-multi] edge_stride=$EDGE_STRIDE"
echo "[semantic-orc-graph-edge-multi] out=$OUT_DIR"

manifest_paths=()
pack_paths=()

edge_index=0
while [[ "$edge_index" -lt "$EDGE_COUNT" ]]; do
  edge_dir="$OUT_DIR/edge_${edge_index}"
  mkdir -p "$edge_dir"
  SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
  SOUNIO_SEMANTIC_ORC_EDGE_FILE="$EDGE_FILE" \
  SOUNIO_SEMANTIC_ORC_EDGE_INDEX="$edge_index" \
  SOUNIO_SEMANTIC_ORC_EDGE_STRIDE="$EDGE_STRIDE" \
  SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_PACK_DIR="$edge_dir" \
    bash scripts/ci/semantic_orc_swow16_graph_edge_kaxi_pack_gate.sh
  manifest_paths+=("$edge_dir/swow16_graph_edge_fixture_manifest.json")
  pack_paths+=("$edge_dir/swow16_graph_edge_kaxi_pack.json")
  edge_index=$((edge_index + 1))
done

python3 - "$MANIFEST" "$EDGE_COUNT" "$EDGE_STRIDE" "${manifest_paths[@]}" -- "${pack_paths[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
edge_count = int(sys.argv[2])
edge_stride = int(sys.argv[3])
sep = sys.argv.index("--")
manifest_paths = [pathlib.Path(value) for value in sys.argv[4:sep]]
pack_paths = [pathlib.Path(value) for value in sys.argv[sep + 1 :]]

if edge_count <= 0:
    raise SystemExit("EDGE_COUNT must be positive")
if len(manifest_paths) != edge_count or len(pack_paths) != edge_count:
    raise SystemExit("edge fixture path count mismatch")


def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


records = []
seen_offsets = set()
for manifest_path, pack_path in zip(manifest_paths, pack_paths):
    manifest = json.loads(manifest_path.read_text())
    pack = json.loads(pack_path.read_text())
    if manifest.get("schema") != "sounio.semantic_orc.swow16_graph_edge_fixture_manifest.v1":
        raise SystemExit(f"unexpected graph-edge manifest schema in {manifest_path}")
    if pack.get("schema") != "sounio.semantic_orc.swow16_kaxi_sinkhorn16_pack.v1":
        raise SystemExit(f"unexpected K-AXI pack schema in {pack_path}")
    if manifest.get("fixture_kind") != "graph_edge_neighborhood_lazy_walk":
        raise SystemExit("fixture_kind mismatch")
    if int(manifest.get("edge_stride", -1)) != edge_stride:
        raise SystemExit("edge_stride mismatch")
    edge_offset = int(manifest.get("edge_offset", -1))
    if edge_offset in seen_offsets:
        raise SystemExit(f"duplicate edge_offset: {edge_offset}")
    seen_offsets.add(edge_offset)
    selected = manifest.get("selected_node_ids", [])
    if len(selected) != 16 or len(set(selected)) != 16:
        raise SystemExit("selected graph support must contain 16 unique nodes")
    support_stats = manifest.get("support_stats", {})
    if int(support_stats.get("induced_undirected_edges", 0)) <= 0:
        raise SystemExit("graph support must include induced graph edges")
    oracle = pack.get("oracle", {})
    if oracle.get("iterations") != 16:
        raise SystemExit("pack oracle must use 16 iterations")
    if abs(float(oracle.get("mass", 0.0)) - 1.0) >= 0.02:
        raise SystemExit("pack oracle mass outside tolerance")
    if float(oracle.get("max_row_err", 1.0)) >= 0.015:
        raise SystemExit("pack oracle row error outside tolerance")
    if float(oracle.get("max_col_err", 1.0)) >= 0.015:
        raise SystemExit("pack oracle col error outside tolerance")
    probe = manifest.get("edge_transport_probe", {})
    records.append(
        {
            "edge_index": int(manifest["edge_index"]),
            "edge_offset": edge_offset,
            "edge_source_node_name": manifest["edge_source_node_name"],
            "edge_target_node_name": manifest["edge_target_node_name"],
            "edge_weight": float(manifest["edge_weight"]),
            "selected_node_ids": selected,
            "fixture_manifest": str(manifest_path),
            "fixture_manifest_sha256": sha256(manifest_path),
            "kaxi_pack": str(pack_path),
            "kaxi_pack_sha256": sha256(pack_path),
            "induced_undirected_edges": int(support_stats["induced_undirected_edges"]),
            "source_incident_support_neighbors": int(support_stats["source_incident_support_neighbors"]),
            "target_incident_support_neighbors": int(support_stats["target_incident_support_neighbors"]),
            "transport_cost": float(probe["transport_cost"]),
            "one_minus_transport_over_endpoint_cost_probe": float(probe["one_minus_transport_over_endpoint_cost_probe"]),
            "oracle_mass": float(oracle["mass"]),
            "oracle_max_row_err": float(oracle["max_row_err"]),
            "oracle_max_col_err": float(oracle["max_col_err"]),
        }
    )

records.sort(key=lambda item: item["edge_index"])
out_path.write_text(
    json.dumps(
        {
            "schema": "sounio.semantic_orc.swow16_graph_edge_multifixture_manifest.v1",
            "status": "pass",
            "edge_count": edge_count,
            "edge_stride": edge_stride,
            "fixture_kind": "graph_edge_neighborhood_lazy_walk",
            "runtime_result": f"{edge_count}/{edge_count} graph-edge fixtures passed generated Sounio check/run and K-AXI pack validation",
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
                "graph_edge_fixture_only",
                "generated_sounio_runtime_gate",
                "kaxi_pack_launch_input_only",
                "no_gpu_runtime_claim",
                "entropic_regularized_transport_only",
                "not_exact_ollivier_ricci_curvature",
                "not_a_biomarker",
                "not_a_clinical_claim",
            ],
            "records": records,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"semantic_orc_swow16_graph_edge_multifixture_gate: PASS artifact={out_path}")
PY
