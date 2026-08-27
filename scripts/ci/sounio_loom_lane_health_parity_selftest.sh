#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
FREEZE="$ROOT_DIR/tools/loom/lane_health.freeze.v1"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-lane-health-parity.XXXXXX")"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"
SOUNIO_RUNTIME="$TEST_ROOT/sounio-lane-health-parity"
OCAML_RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-lane-health-parity-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$FREEZE" || true)"
  [[ "$count" == 1 ]] || fail "freeze field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$FREEZE")"
  printf '%s' "${line#*=}"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

[[ -f "$FREEZE" ]] || fail 'freeze manifest is missing'
bash "$ROOT_DIR/scripts/ci/sounio_loom_lane_health_freeze_selftest.sh" >/dev/null
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'Sounio semantics are not frozen'
[[ "$(field parity_open)" == false ]] || fail 'frozen parent was already promoted'

executable_commit="$(field sounio_executable_commit)"
wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$executable_commit" "$wrapper_path" "$compiler_path" |
  tar -x -C "$TOOLCHAIN_ROOT"
[[ "$(file_hash "$TOOLCHAIN_ROOT/$wrapper_path")" == "$(field toolchain_wrapper_sha256)" ]] ||
  fail 'reconstructed compiler wrapper hash differs'
[[ "$(file_hash "$TOOLCHAIN_ROOT/$compiler_path")" == "$(field toolchain_compiler_sha256)" ]] ||
  fail 'reconstructed compiler binary hash differs'

command -v dune >/dev/null 2>&1 || fail 'dune is required for OCaml parity'
command -v ocamlopt >/dev/null 2>&1 || fail 'ocamlopt is required for OCaml parity'
ocamlfind query cryptokit >/dev/null 2>&1 || fail 'OCaml cryptokit is required for parity SHA-256'
dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null

SOUNIO_LOOM_LANE_HEALTH_PARITY_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_LANE_HEALTH_PARITY_OUTPUT="$SOUNIO_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_lane_health_parity.sh" >/dev/null

sounio_result="$($SOUNIO_RUNTIME)"
ocaml_result="$($OCAML_RUNTIME lane-health-parity)"
sounio_payload="${sounio_result#SOUNIO_LANE_HEALTH_PARITY }"
ocaml_payload="${ocaml_result#OCAML_LANE_HEALTH_PARITY }"
[[ "$sounio_payload" == "$ocaml_payload" ]] ||
  fail "OCaml diverged from frozen Sounio: sounio=$sounio_result ocaml=$ocaml_result"
[[ "$sounio_payload" == *'domain=8388608'* ]] || fail 'parity domain is not exhaustive'
[[ "$sounio_payload" == *"parent_semantics_sha256=$(field semantics_sha256)"* ]] ||
  fail 'parity result is not bound to the frozen parent'

# Flip one canonical OCaml decision after classification. The exhaustive
# digest and counts must no longer match the unchanged Sounio witness.
sabotaged_result="$(SOUNIO_LOOM_LANE_HEALTH_PARITY_SABOTAGE_INDEX=0 \
  "$OCAML_RUNTIME" lane-health-parity)"
sabotaged_payload="${sabotaged_result#OCAML_LANE_HEALTH_PARITY }"
[[ "$sabotaged_payload" != "$sounio_payload" ]] ||
  fail 'single-frame OCaml sabotage escaped the exhaustive parity receipt'

digest="$(sed -n 's/.*digest_sha256=\([0-9a-f]\{64\}\).*/\1/p' <<< "$sounio_payload")"
[[ -n "$digest" ]] || fail 'parity result omitted its SHA-256 digest'
printf '%s\n' \
  "sounio-loom-lane-health-parity-selftest: PASS authority=Sounio realization=OCaml domain=8388608 digest_sha256=$digest parent_semantics_sha256=$(field semantics_sha256) single_frame_sabotage=detected"
