#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
HANDSHAKE="$ROOT_DIR/tools/loom/process_witness_handshake_payload.freeze.v1"
FIXTURE="$ROOT_DIR/tools/loom/product_exec_cell_fixture.freeze.v1"
EXEC_CELL="$ROOT_DIR/tools/loom/product_exec_cell_host_canary.runtime.v1"
INGRESS="$ROOT_DIR/tools/loom/product_exec_ingress_dark.runtime.v1"
CONTRACT="$ROOT_DIR/tools/loom/HANDSHAKE_EXEC_CELL_PAYLOAD_COHERENCE_V1.md"
HANDSHAKE_REL='tools/loom/process_witness_handshake_payload.freeze.v1'
FIXTURE_REL='tools/loom/product_exec_cell_fixture.freeze.v1'
EMPTY_SHA256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

fail() {
  printf 'sounio-loom-handshake-exec-cell-payload-coherence-selftest: FAIL: %s\n' \
    "$*" >&2
  exit 1
}

field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "${path#$ROOT_DIR/} field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

expect_field() {
  local path="$1" key="$2" expected="$3" actual
  actual="$(field "$path" "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "${path#$ROOT_DIR/} $key expected=$expected actual=$actual"
}

file_hash() {
  local path="$1" sum
  [[ -f "$path" && ! -L "$path" ]] || fail "${path#$ROOT_DIR/} is missing or linked"
  sum="$(sha256sum "$path")"
  printf '%s' "${sum%% *}"
}

expect_sha256() {
  local value="$1" label="$2"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail "$label is not a SHA-256 digest"
  [[ "$value" != "$EMPTY_SHA256" ]] || fail "$label is the empty digest"
}

for path in "$HANDSHAKE" "$FIXTURE" "$EXEC_CELL" "$INGRESS" "$CONTRACT"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required parent is absent: ${path#$ROOT_DIR/}"
done

expect_field "$HANDSHAKE" schema loom-process-witness-handshake-payload-freeze-v1
expect_field "$HANDSHAKE" stage SOUNIO_HANDSHAKE_PAYLOAD_FROZEN
expect_field "$HANDSHAKE" producing_language Sounio
expect_field "$HANDSHAKE" language_role SEMANTIC_PAYLOAD
expect_field "$HANDSHAKE" semantic_authority Sounio
expect_field "$HANDSHAKE" action 9030
expect_field "$HANDSHAKE" material_execution false
expect_field "$HANDSHAKE" parity_open false
expect_field "$HANDSHAKE" claim_ready false

expect_field "$FIXTURE" schema loom-product-exec-cell-fixture-freeze-v1
expect_field "$FIXTURE" stage SEMANTICS_FROZEN
expect_field "$FIXTURE" producing_language Sounio
expect_field "$FIXTURE" language_role SEMANTIC_FIXTURE_PRODUCER
expect_field "$FIXTURE" semantic_authority Sounio
expect_field "$FIXTURE" action 9030
expect_field "$FIXTURE" payload_manifest_path "$HANDSHAKE_REL"
expect_field "$FIXTURE" exec_cell_attached false
expect_field "$FIXTURE" material_execution false
expect_field "$FIXTURE" production_activation false
expect_field "$FIXTURE" parity_open false
expect_field "$FIXTURE" claim_ready false

expect_field "$EXEC_CELL" schema loom-product-exec-cell-host-canary-runtime-v1
expect_field "$EXEC_CELL" stage MATERIAL_EXEC_CELL_CANARY_FROZEN
expect_field "$EXEC_CELL" semantic_authority Sounio
expect_field "$EXEC_CELL" semantic_action 9030
expect_field "$EXEC_CELL" process_witness_manifest_path "$HANDSHAKE_REL"
expect_field "$EXEC_CELL" fixture_manifest_path "$FIXTURE_REL"
expect_field "$EXEC_CELL" exec_cell_attached true
expect_field "$EXEC_CELL" test_only true
expect_field "$EXEC_CELL" material_execution true
expect_field "$EXEC_CELL" production_activation false
expect_field "$EXEC_CELL" exec_attached false
expect_field "$EXEC_CELL" parity_open false
expect_field "$EXEC_CELL" claim_ready false

expect_field "$INGRESS" schema loom-product-exec-ingress-dark-runtime-v1
expect_field "$INGRESS" stage PRODUCT_DARK_ATTACHMENT_FROZEN
expect_field "$INGRESS" semantic_authority Sounio
expect_field "$INGRESS" semantic_action 9031
expect_field "$INGRESS" descriptor_dark_attached true
expect_field "$INGRESS" descriptor_is_bearer false
expect_field "$INGRESS" required_mode_default false
expect_field "$INGRESS" distinct_uid_product_broker false
expect_field "$INGRESS" material_execution false
expect_field "$INGRESS" production_activation false
expect_field "$INGRESS" exec_attached false
expect_field "$INGRESS" parity_open false
expect_field "$INGRESS" claim_ready false

HANDSHAKE_EXECUTABLE="$(field "$HANDSHAKE" executable_sha256)"
FIXTURE_PAYLOAD="$(field "$FIXTURE" payload_sha256)"
CANARY_PAYLOAD="$(field "$EXEC_CELL" payload_sha256)"
expect_sha256 "$HANDSHAKE_EXECUTABLE" 'handshake executable_sha256'
expect_sha256 "$FIXTURE_PAYLOAD" 'fixture payload_sha256'
expect_sha256 "$CANARY_PAYLOAD" 'exec-cell payload_sha256'
[[ "$FIXTURE_PAYLOAD" == "$HANDSHAKE_EXECUTABLE" ]] ||
  fail "fixture payload_sha256 diverged from handshake executable_sha256"
[[ "$CANARY_PAYLOAD" == "$HANDSHAKE_EXECUTABLE" ]] ||
  fail "exec-cell payload_sha256 diverged from handshake executable_sha256"

HANDSHAKE_FILE_SHA="$(file_hash "$HANDSHAKE")"
FIXTURE_FILE_SHA="$(file_hash "$FIXTURE")"
[[ "$HANDSHAKE_FILE_SHA" == "$(field "$FIXTURE" payload_manifest_sha256)" ]] ||
  fail 'fixture payload_manifest_sha256 diverged from the handshake freeze file'
[[ "$HANDSHAKE_FILE_SHA" == "$(field "$EXEC_CELL" process_witness_manifest_sha256)" ]] ||
  fail 'exec-cell process_witness_manifest_sha256 diverged from the handshake freeze file'
[[ "$FIXTURE_FILE_SHA" == "$(field "$EXEC_CELL" fixture_manifest_sha256)" ]] ||
  fail 'exec-cell fixture_manifest_sha256 diverged from the fixture freeze file'

# Live host-canary freeze remains red on this tip: composition/ingress/broker
# source pins do not match HEAD. This lane does not re-freeze those hashes.
COMPOSITION_REL="$(field "$EXEC_CELL" composition_source_path)"
INGRESS_SRC_REL="$(field "$EXEC_CELL" ingress_source_path)"
BROKER_REL="$(field "$EXEC_CELL" broker_source_path)"
COMPOSITION_LIVE="$(file_hash "$ROOT_DIR/$COMPOSITION_REL")"
INGRESS_SRC_LIVE="$(file_hash "$ROOT_DIR/$INGRESS_SRC_REL")"
BROKER_LIVE="$(file_hash "$ROOT_DIR/$BROKER_REL")"
EXEC_CELL_FREEZE_LIVE=true
if [[ "$COMPOSITION_LIVE" != "$(field "$EXEC_CELL" composition_source_sha256)" ||
      "$INGRESS_SRC_LIVE" != "$(field "$EXEC_CELL" ingress_source_sha256)" ||
      "$BROKER_LIVE" != "$(field "$EXEC_CELL" broker_source_sha256)" ]]; then
  EXEC_CELL_FREEZE_LIVE=false
fi
# Gate 5 stays closed on this lane even if Codex-1 later re-pins host hashes.
# A live freeze is reported, not treated as licence to flip required_mode.
[[ "$(field "$INGRESS" required_mode_default)" == false ]] ||
  fail 'required_mode_default flipped; this lane does not authorize gate 5'

SABOTAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-handshake-exec-cell-coherence.XXXXXX")"
cleanup() { rm -rf "$SABOTAGE_DIR"; }
trap cleanup EXIT
SABOTAGE_PAYLOAD="$SABOTAGE_DIR/exec-cell.runtime.v1"
SABOTAGE_MODE="$SABOTAGE_DIR/ingress.runtime.v1"
sed 's/^payload_sha256=.*/payload_sha256='"$EMPTY_SHA256"'/' "$EXEC_CELL" > "$SABOTAGE_PAYLOAD"
[[ "$(field "$SABOTAGE_PAYLOAD" payload_sha256)" != "$HANDSHAKE_EXECUTABLE" ]] ||
  fail 'payload identity sabotage did not change the canary digest'
sed 's/^required_mode_default=.*/required_mode_default=true/' "$INGRESS" > "$SABOTAGE_MODE"
[[ "$(field "$SABOTAGE_MODE" required_mode_default)" == true ]] ||
  fail 'required_mode sabotage did not flip the dark ingress default'

printf 'sounio-loom-handshake-exec-cell-payload-coherence-selftest: PASS semantic_authority=Sounio producer=Bash role=COHERENCE_ONLY actions=9030+9031 handshake_stage=SOUNIO_HANDSHAKE_PAYLOAD_FROZEN fixture_stage=SEMANTICS_FROZEN canary_stage=MATERIAL_EXEC_CELL_CANARY_FROZEN ingress_stage=PRODUCT_DARK_ATTACHMENT_FROZEN payload_sha256=%s handshake_manifest_sha256=%s fixture_manifest_sha256=%s payload_identity=true required_mode_default=false exec_cell_freeze_live=%s canary_exec_cell_attached=true canary_test_only=true product_exec_attached=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false python_executed=false rust_executed=false host_hash_refreeze=false causal_sabotage=PASS next=do-not-flip-required-mode-until-live-exec-cell-freeze-and-activation-garden\n' \
  "$HANDSHAKE_EXECUTABLE" "$HANDSHAKE_FILE_SHA" "$FIXTURE_FILE_SHA" "$EXEC_CELL_FREEZE_LIVE"
