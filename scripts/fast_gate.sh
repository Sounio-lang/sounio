#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[fast-gate] 1/7 syntax drift scan"
python3 "$ROOT_DIR/skills/sounio-language/scripts/scan_syntax_drift.py" --root "$ROOT_DIR"

echo "[fast-gate] 2/7 workflow script reference check"
bash "$ROOT_DIR/scripts/check_workflow_script_refs.sh"

echo "[fast-gate] 3/7 docs consistency check"
bash "$ROOT_DIR/scripts/check_docs_consistency.sh"

echo "[fast-gate] 4/7 compiler unit tests (cargo test --lib)"
(cd "$ROOT_DIR" && cargo test -p souc --lib)

echo "[fast-gate] 5/7 integration tests"
(cd "$ROOT_DIR" && cargo test -p souc --tests)

echo "[fast-gate] 6/7 check canonical example"
(cd "$ROOT_DIR" && cargo run -p souc --quiet --bin souc -- check examples/hello.sio)

echo "[fast-gate] 7/7 e2e backend gate"
"$ROOT_DIR/scripts/e2e_gate.sh"

echo "[fast-gate] ok"
