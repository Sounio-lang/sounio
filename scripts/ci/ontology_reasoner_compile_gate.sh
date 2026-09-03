#!/usr/bin/env bash
# Verifies stdlib/ontology/reasoner.sio (Vec→Seq migrated, effect-annotated) fully
# COMPILES and RUNS. `souc check` is lenient (misses type errors), so we
# concatenate the module with an exercising main and compile+execute it. The
# main asserts via its exit code. x86-64 Linux only.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT_DIR"
case "$(uname -s)/$(uname -m)" in Linux/x86_64|Linux/amd64) ;; *) echo "[ontology-reasoner] SKIP: x86-64 Linux only"; exit 0;; esac
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"; sounio_require_souc
TMP="$(mktemp -d)"; DRV="$TMP/reasoner_drv.sio"
cat "$ROOT_DIR/stdlib/ontology/reasoner.sio" "$ROOT_DIR/scripts/ci/ontology_fixtures/reasoner_exercise_main.sio" > "$DRV"
if "$SOUC_BIN" --help 2>/dev/null | grep -q "compile <file.sio>"; then
  compile_cmd=( "$SOUC_BIN" compile "$DRV" -o "$TMP/reasoner_drv" )
else
  compile_cmd=( "$SOUC_BIN" "$DRV" "$TMP/reasoner_drv" )
fi
if ! "${compile_cmd[@]}" >"$TMP/build.log" 2>&1; then
  echo "[ontology-reasoner] FAIL: reasoner.sio did not compile"; tail -20 "$TMP/build.log"; exit 1
fi
chmod +x "$TMP/reasoner_drv"
if ! "$TMP/reasoner_drv"; then echo "[ontology-reasoner] FAIL: exercise main returned nonzero"; exit 1; fi
echo "[ontology-reasoner] PASS: reasoner.sio compiles and runs"
