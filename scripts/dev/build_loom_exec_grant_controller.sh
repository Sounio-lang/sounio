#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
OUTPUT="${SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-exec-grant-controller}"
OCAMLFIND="${SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_OCAMLFIND:-/usr/bin/ocamlfind}"
OBJCOPY="${SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_OBJCOPY:-/usr/bin/objcopy}"

RESIDENT="$ROOT_DIR/tools/loom/src/loom_resident.ml"
CELL="$ROOT_DIR/tools/loom/src/loom_exec_grant_cell.ml"
CONTROLLER="$ROOT_DIR/tools/loom/src/loom_exec_grant_controller.ml"
STUB="$ROOT_DIR/tools/loom/src/loom_membrane_stubs.c"
RUNTIME_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell.runtime.v1"
FIXTURE_MANIFEST="$ROOT_DIR/tools/loom/host_exec_quorum_fixture.freeze.v1"

fail() {
  printf 'build-loom-exec-grant-controller: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "input is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "input hash drifted: $path"
}

[[ -x "$OCAMLFIND" ]] || fail "ocamlfind is missing: $OCAMLFIND"
[[ -x "$OBJCOPY" ]] || fail "objcopy is missing: $OBJCOPY"
expect_hash "$RESIDENT" a174280d342983b7e7a66eb650178ecd501b924f9f285e75fc2eb2ea81d2696d
expect_hash "$CELL" ca26e248fdad74c67adb0ea13a9b2b314c0e91425c727d389601d114ac099328
expect_hash "$STUB" 61a11fa4bc74e03a4e8c05d4fa979d56f9f628e24c334516de3982addaaece36
expect_hash "$RUNTIME_MANIFEST" e3a9b2ac75c5b8f6eb1c35aec046534f6d6e1d61a2728984e84d0cd36e2ca660
expect_hash "$FIXTURE_MANIFEST" 10401ebe4d302647220433eadb0b1240ce2b3128801f421f2712ae757f5105b5
[[ -f "$CONTROLLER" && ! -L "$CONTROLLER" ]] || fail 'controller source is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-grant-controller.XXXXXX")"
trap 'rm -rf "$work"' EXIT
install -m 0644 "$RESIDENT" "$work/loom_resident.ml"
install -m 0644 "$CELL" "$work/loom_exec_grant_cell.ml"
install -m 0644 "$CONTROLLER" "$work/loom_exec_grant_controller.ml"
install -m 0644 "$STUB" "$work/loom_membrane_stubs.c"

(
  cd "$work"
  "$OCAMLFIND" ocamlopt -package unix,cryptokit -linkpkg \
    loom_membrane_stubs.c loom_resident.ml loom_exec_grant_cell.ml \
    loom_exec_grant_controller.ml -o loom-exec-grant-controller
  "$OBJCOPY" --strip-debug --remove-section=.note.gnu.build-id \
    loom-exec-grant-controller
)
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$work/loom-exec-grant-controller" "$OUTPUT"

printf 'BUILT_LOOM_EXEC_GRANT_CONTROLLER path=%s language=OCaml role=EFFECT_PARITY semantic_authority=Sounio single_resident_controller=true material_grant=false material_execution=false\n' \
  "$OUTPUT"
