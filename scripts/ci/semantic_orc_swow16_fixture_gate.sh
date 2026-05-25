#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-bin/souc}"
INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
REGIME="${SOUNIO_SEMANTIC_ORC_REGIME:-normative}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_SWOW16_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-swow16.XXXXXX)}"
SOURCE="$OUT_DIR/swow16_fixture_gate.sio"
OUT_FILE="$OUT_DIR/swow16_fixture_gate.out"
MANIFEST="$OUT_DIR/swow16_fixture_manifest.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-swow16] input=$INPUT_DIR"
echo "[semantic-orc-swow16] regime=$REGIME"
echo "[semantic-orc-swow16] out=$OUT_DIR"

python3 scripts/research/semantic_orc_swow16_fixture.py \
  --input-dir "$INPUT_DIR" \
  --regime "$REGIME" \
  --out "$SOURCE" \
  --manifest "$MANIFEST"

python3 - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    manifest = json.load(handle)
if manifest.get("selected_node_count") != 16:
    raise SystemExit("manifest selected_node_count must be 16")
for key in ("generator_sha256", "template_sha256", "feature_file_sha256", "trajectory_file_sha256"):
    value = manifest.get(key, "")
    if len(value) != 64:
        raise SystemExit(f"manifest key {key} is not a sha256 digest")
PY

"$SOUC" check "$SOURCE" >/dev/null
"$SOUC" run "$SOURCE" >"$OUT_FILE"

required_tokens=(
  "SOUNIO_SWOW16_ENTROPIC_TRANSPORT_PASS"
  "claim_level=swow16_fixture_runtime_gate"
  "fixture_source=cpc2026_sounio_input_node_features_parity"
  "swow16_fixture_nodes=16"
  "epistemic_epsilon_gate=true"
  "epsilon_range_gate=true"
  "swow16_marginal_gate=true"
  "swow16_fixed_point_gate=true"
  "swow16_adaptive_stop=true"
  "entropic_regularized_transport_only=true"
  "clinical_claims_not_enforced=true"
  "no_convergence_theorem=true"
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fqx "$token" "$OUT_FILE"; then
    echo "[semantic-orc-swow16] FAIL: missing token: $token" >&2
    echo "[semantic-orc-swow16] generated source: $SOURCE" >&2
    echo "[semantic-orc-swow16] output:" >&2
    cat "$OUT_FILE" >&2
    exit 1
  fi
done

echo "[semantic-orc-swow16] PASS"
