#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]] ||
   [[ ! "$(uname -m 2>/dev/null || echo unknown)" =~ ^(x86_64|amd64)$ ]]; then
  printf 'SKIP  module declaration identity source-fresh probe requires Linux x86-64 bootstrap\n'
  exit 0
fi

TMP_DIR="$(mktemp -d /tmp/sounio-module-declaration-identity.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

LOG="$TMP_DIR/probe.log"
PROBE="self-hosted/compiler/module_declaration_identity_probe.sio"

if ! env -u SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE=lean_single bin/souc run "$PROBE" >"$LOG" 2>&1; then
  printf 'FAIL  module declaration identity probe did not execute\n' >&2
  cat "$LOG" >&2
  exit 1
fi

for witness in \
  'declared_identity_left=1' \
  'declared_identity_right=1' \
  'module_decl_ast_left=1' \
  'module_decl_ast_right=1' \
  'same_basename_distinct_logical_identity=1' \
  'legacy_basename_fallback=1' \
  'module_declaration_identity_verdict=0'; do
  if ! grep -Fqx "$witness" "$LOG"; then
    printf 'FAIL  missing module identity witness: %s\n' "$witness" >&2
    cat "$LOG" >&2
    exit 1
  fi
done

printf 'PASS  module declarations retain full AST identity\n'
printf 'PASS  equal basenames retain distinct declared identities\n'
printf 'PASS  undeclared modules retain basename fallback\n'
printf 'source_fresh_path=lean_single_imported_probe\n'
