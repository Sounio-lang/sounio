#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-bin/souc}"
INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
EDGE_FILE="${SOUNIO_SEMANTIC_ORC_EDGE_FILE:-/workspace/hyperbolic-semantic-networks/data/processed/english_edges_FINAL.csv}"
EDGE_INDEX="${SOUNIO_SEMANTIC_ORC_EDGE_INDEX:-0}"
EDGE_STRIDE="${SOUNIO_SEMANTIC_ORC_EDGE_STRIDE:-1}"
TILE_INDEX="${SOUNIO_SEMANTIC_ORC_TILE_INDEX:-0}"
TILE_STRIDE="${SOUNIO_SEMANTIC_ORC_TILE_STRIDE:-14}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_GRAPH_EDGE_PACK_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-graph-edge-pack.XXXXXX)}"
SOURCE="$OUT_DIR/swow16_graph_edge_fixture_gate.sio"
MANIFEST="$OUT_DIR/swow16_graph_edge_fixture_manifest.json"
KAXI_PACK="$OUT_DIR/swow16_graph_edge_kaxi_pack.json"
RUN_OUT="$OUT_DIR/swow16_graph_edge_fixture_gate.out"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-graph-edge-pack] input=$INPUT_DIR"
echo "[semantic-orc-graph-edge-pack] edge_file=$EDGE_FILE"
echo "[semantic-orc-graph-edge-pack] edge_index=$EDGE_INDEX"
echo "[semantic-orc-graph-edge-pack] edge_stride=$EDGE_STRIDE"
echo "[semantic-orc-graph-edge-pack] tile_index=$TILE_INDEX"
echo "[semantic-orc-graph-edge-pack] tile_stride=$TILE_STRIDE"
echo "[semantic-orc-graph-edge-pack] out=$OUT_DIR"

python3 scripts/research/semantic_orc_swow16_graph_edge_fixture.py \
  --input-dir "$INPUT_DIR" \
  --edge-file "$EDGE_FILE" \
  --edge-index "$EDGE_INDEX" \
  --edge-stride "$EDGE_STRIDE" \
  --tile-index "$TILE_INDEX" \
  --tile-stride "$TILE_STRIDE" \
  --sounio-out "$SOURCE" \
  --manifest "$MANIFEST" \
  --kaxi-pack "$KAXI_PACK"

"$SOUC" check "$SOURCE" >/dev/null
"$SOUC" run "$SOURCE" >"$RUN_OUT"

required_tokens=(
  "SOUNIO_SWOW16_GRAPH_EDGE_TRANSPORT_PASS"
  "claim_level=swow16_graph_edge_fixture_runtime_gate"
  "fixture_source=swow_english_final_graph_edge_neighborhood"
  "swow16_graph_edge_fixture_nodes=16"
  "epistemic_epsilon_gate=true"
  "swow16_marginal_gate=true"
  "approx_math_runtime_gate=true"
  "entropic_regularized_transport_only=true"
  "clinical_claims_not_enforced=true"
  "biomarker_claims_not_enforced=true"
  "statistical_inference_claims_not_enforced=true"
  "no_convergence_theorem=true"
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fqx "$token" "$RUN_OUT"; then
    echo "[semantic-orc-graph-edge-pack] FAIL: missing token: $token" >&2
    cat "$RUN_OUT" >&2
    exit 1
  fi
done

python3 - "$MANIFEST" "$KAXI_PACK" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

manifest_path, pack_path = sys.argv[1:3]
manifest = json.load(open(manifest_path))
pack = json.load(open(pack_path))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if manifest.get("schema") != "sounio.semantic_orc.swow16_graph_edge_fixture_manifest.v1":
    raise SystemExit("unexpected graph-edge manifest schema")
if pack.get("schema") != "sounio.semantic_orc.swow16_kaxi_sinkhorn16_pack.v1":
    raise SystemExit("unexpected K-AXI pack schema")
if manifest.get("fixture_kind") != "graph_edge_neighborhood_lazy_walk":
    raise SystemExit("manifest fixture_kind mismatch")
if pack.get("fixture_kind") != "graph_edge_neighborhood_lazy_walk":
    raise SystemExit("pack fixture_kind mismatch")
selected = [int(value) for value in manifest.get("selected_node_ids", [])]
if len(selected) != 16 or len(set(selected)) != 16:
    raise SystemExit("graph-edge fixture must select 16 unique nodes")
source = int(manifest.get("edge_source_node_id", -1))
target = int(manifest.get("edge_target_node_id", -1))
if source not in selected or target not in selected or source == target:
    raise SystemExit("selected support must include distinct graph edge endpoints")
if pack.get("selected_node_ids") != selected:
    raise SystemExit("manifest/pack selected_node_ids mismatch")
if int(manifest.get("edge_index", -1)) != int(pack.get("edge_index", -2)):
    raise SystemExit("manifest/pack edge_index mismatch")
if int(manifest.get("edge_stride", -1)) != int(pack.get("edge_stride", -2)):
    raise SystemExit("manifest/pack edge_stride mismatch")
if int(manifest.get("tile_index", -1)) != int(pack.get("tile_index", -2)):
    raise SystemExit("manifest/pack tile_index mismatch")
if int(manifest.get("tile_stride", -1)) != int(pack.get("tile_stride", -2)):
    raise SystemExit("manifest/pack tile_stride mismatch")
if float(manifest.get("edge_weight", 0.0)) <= 0.0:
    raise SystemExit("selected graph edge must have positive weight")
support_stats = manifest.get("support_stats", {})
if int(support_stats.get("tile_extra_candidate_count", 0)) != 14:
    raise SystemExit("graph-edge tile must contribute 14 non-endpoint nodes")
if int(support_stats.get("source_incident_support_neighbors", 0)) <= 0:
    raise SystemExit("source endpoint has no support neighbor")
if int(support_stats.get("target_incident_support_neighbors", 0)) <= 0:
    raise SystemExit("target endpoint has no support neighbor")
if int(support_stats.get("induced_undirected_edges", 0)) <= 0:
    raise SystemExit("graph-edge support must include induced graph edges")
if pack.get("init_mem_count") != 320:
    raise SystemExit("K-AXI init_mem_count must be 320")
values = [float(x) for x in pack["init_mem_csv"].split(",")]
if len(values) != 320:
    raise SystemExit("K-AXI init_mem CSV must have 320 values")
if not all(math.isfinite(x) for x in values):
    raise SystemExit("K-AXI init_mem contains a non-finite value")
la = [float(x) for x in pack.get("la_log2", [])]
lb = [float(x) for x in pack.get("lb_log2", [])]
k = [float(x) for x in pack.get("k_log2", [])]
cost = [float(x) for x in pack.get("cost", [])]
if len(la) != 16 or len(lb) != 16 or len(k) != 256 or len(cost) != 256:
    raise SystemExit("K-AXI pack must carry la_log2[16], lb_log2[16], k_log2[256], and cost[256]")
for got, expected in zip(values[0:16], la):
    if abs(got - expected) > 1e-9:
        raise SystemExit("K-AXI init_mem la segment does not match la_log2")
for got, expected in zip(values[16:32], lb):
    if abs(got - expected) > 1e-9:
        raise SystemExit("K-AXI init_mem lb segment does not match lb_log2")
for got, expected in zip(values[32:288], k):
    if abs(got - expected) > 1e-9:
        raise SystemExit("K-AXI init_mem K segment does not match k_log2")
if any(abs(value) > 1e-12 for value in values[288:320]):
    raise SystemExit("K-AXI init_mem scratch segment must be zero")
for i in range(16):
    if abs(cost[i * 16 + i]) > 1e-12:
        raise SystemExit("semantic cost diagonal must be zero")
    for j in range(16):
        if abs(cost[i * 16 + j] - cost[j * 16 + i]) > 1e-12:
            raise SystemExit("semantic cost matrix must be symmetric")
        for mid in range(16):
            if cost[i * 16 + mid] > cost[i * 16 + j] + cost[j * 16 + mid] + 1e-12:
                raise SystemExit("semantic cost matrix violates triangle inequality")
oracle = pack.get("oracle", {})
if oracle.get("iterations") != 16:
    raise SystemExit("K-AXI oracle must use the 16-iteration kernel contract")
if abs(float(oracle.get("mass", 0.0)) - 1.0) >= 0.02:
    raise SystemExit("K-AXI oracle mass is outside prototype tolerance")
if float(oracle.get("max_row_err", 1.0)) >= 0.015:
    raise SystemExit("K-AXI oracle row error is outside prototype tolerance")
if float(oracle.get("max_col_err", 1.0)) >= 0.015:
    raise SystemExit("K-AXI oracle column error is outside prototype tolerance")
edge_probe = manifest.get("edge_transport_probe", {})
if edge_probe.get("probe_role") != "engineering transport-ratio fixture scalar only; not exact Wasserstein-1, not exact Ollivier-Ricci curvature, not an ORC estimator, not a biomarker, and not validated for interpretation":
    raise SystemExit("edge transport-ratio fixture scalar role must be explicitly bounded")
if "not_exact_ollivier_ricci_curvature" not in manifest.get("claim_limits", []):
    raise SystemExit("manifest must reject exact ORC claim")
if "not_exact_wasserstein_distance" not in manifest.get("claim_limits", []):
    raise SystemExit("manifest must reject exact Wasserstein claim")
if "biomarker_claims_not_enforced" not in manifest.get("claim_limits", []):
    raise SystemExit("manifest must reject biomarker claim")
if "pack_only_no_gpu_launch" not in pack.get("boundaries", []):
    raise SystemExit("K-AXI pack must explicitly say no GPU launch was claimed")
for key in ("generator_sha256", "feature_file_sha256", "edge_file_sha256", "generated_sounio_source_sha256"):
    value = manifest.get(key, "")
    if len(value) != 64:
        raise SystemExit(f"manifest key {key} is not a sha256 digest")
path_keys = {
    "generator": "generator_sha256",
    "feature_file": "feature_file_sha256",
    "edge_file": "edge_file_sha256",
    "generated_sounio_source": "generated_sounio_source_sha256",
}
for path_key, digest_key in path_keys.items():
    path = pathlib.Path(manifest[path_key])
    if sha256(path) != manifest[digest_key]:
        raise SystemExit(f"manifest hash mismatch for {path_key}")
PY

echo "[semantic-orc-graph-edge-pack] PASS"
