#!/usr/bin/env bash
# Verifies the runtime-model <-> generated-type bridge (stdlib/ontology/typed/go.sio):
#   1. the run-pass test compiles, runs, and prints the expected line (a runtime IRI
#      coerced to a nominal GO type and upcast through subclass edges), and
#   2. the compile-fail test is REJECTED with E152 (a GO root cannot satisfy a
#      descendant-typed parameter — the bridge cannot fabricate a bad subsumption).
# x86-64 Linux only.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT_DIR"
case "$(uname -s)/$(uname -m)" in Linux/x86_64|Linux/amd64) ;; *) echo "[ontology-typed-bridge] SKIP: x86-64 Linux only"; exit 0;; esac
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"; sounio_require_souc
TMP="$(mktemp -d)"

PASS_SRC="$ROOT_DIR/tests/run-pass/ontology_typed_bridge_go.sio"
FAIL_SRC="$ROOT_DIR/tests/compile-fail/ontology_typed_bridge_go_reject.sio"
EXPECT="ontology typed bridge go OK"

compile_to() {
  local src="$1"
  local out="$2"
  if "$SOUC_BIN" --help 2>/dev/null | grep -q "compile <file.sio>"; then
    "$SOUC_BIN" compile "$src" -o "$out"
  else
    "$SOUC_BIN" "$src" "$out"
  fi
}

# 1. run-pass: must compile, run, and print EXPECT.
if ! compile_to "$PASS_SRC" "$TMP/bridge" >"$TMP/build.log" 2>&1; then
  echo "[ontology-typed-bridge] FAIL: run-pass test did not compile"; tail -20 "$TMP/build.log"; exit 1
fi
chmod +x "$TMP/bridge"
OUT="$("$TMP/bridge")"
if [ "$OUT" != "$EXPECT" ]; then
  echo "[ontology-typed-bridge] FAIL: got [$OUT] want [$EXPECT]"; exit 1
fi

# 2. compile-fail: must be rejected (E152 subsumption).
if compile_to "$FAIL_SRC" "$TMP/rej" >"$TMP/rej.log" 2>&1; then
  echo "[ontology-typed-bridge] FAIL: bad-subsumption test compiled clean (should reject)"; exit 1
fi
if ! grep -q "E152" "$TMP/rej.log"; then
  echo "[ontology-typed-bridge] FAIL: rejected but not with E152"; tail -10 "$TMP/rej.log"; exit 1
fi

echo "[ontology-typed-bridge] PASS: bridge round-trips and bad subsumption is rejected (E152)"
