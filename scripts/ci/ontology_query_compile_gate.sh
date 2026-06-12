#!/usr/bin/env bash
# Verifies stdlib/ontology/query.sio (Vec→Seq migrated, effect-annotated) fully
# COMPILES and RUNS. `souc check` is lenient (misses type errors), so we
# concatenate the module with an exercising main and compile+execute it. The
# main asserts via its exit code. x86-64 Linux only.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT_DIR"
case "$(uname -s)/$(uname -m)" in Linux/x86_64|Linux/amd64) ;; *) echo "[ontology-query] SKIP: x86-64 Linux only"; exit 0;; esac
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"; sounio_require_souc
TMP="$(mktemp -d)"; DRV="$TMP/query_drv.sio"
cat "$ROOT_DIR/stdlib/ontology/query.sio" "$ROOT_DIR/scripts/ci/ontology_fixtures/query_exercise_main.sio" > "$DRV"
if "$SOUC_BIN" --help 2>/dev/null | grep -q "compile <file.sio>"; then
  compile_cmd=( "$SOUC_BIN" compile "$DRV" -o "$TMP/query_drv" )
else
  compile_cmd=( "$SOUC_BIN" "$DRV" "$TMP/query_drv" )
fi
if ! "${compile_cmd[@]}" >"$TMP/build.log" 2>&1; then
  echo "[ontology-query] FAIL: query.sio did not compile"; tail -20 "$TMP/build.log"; exit 1
fi
chmod +x "$TMP/query_drv"
if ! "$TMP/query_drv"; then echo "[ontology-query] FAIL: exercise main returned nonzero"; exit 1; fi
echo "[ontology-query] PASS: query.sio compiles and runs"
