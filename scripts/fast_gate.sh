#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[fast-gate] 1/9 cargo preflight (no conflicting workspace cargo jobs)"
bash "$ROOT_DIR/scripts/check_no_active_cargo_jobs.sh"

echo "[fast-gate] 2/9 syntax drift scan"
python3 "$ROOT_DIR/skills/sounio-language/scripts/scan_syntax_drift.py" --root "$ROOT_DIR"

echo "[fast-gate] 3/9 workflow script reference check"
bash "$ROOT_DIR/scripts/check_workflow_script_refs.sh"

echo "[fast-gate] 4/9 docs consistency check"
bash "$ROOT_DIR/scripts/check_docs_consistency.sh"

echo "[fast-gate] 5/9 cultural fidelity (user-facing text leakage)"
python3 "$ROOT_DIR/scripts/cultural_fidelity_gate.py"

echo "[fast-gate] 6/9 compiler unit tests (cargo test --lib)"
(cd "$ROOT_DIR" && cargo test -p souc --lib)

echo "[fast-gate] 7/9 integration tests"
(cd "$ROOT_DIR" && cargo test -p souc --tests)

echo "[fast-gate] 8/9 check canonical example"
(cd "$ROOT_DIR" && cargo run -p souc --quiet --bin souc -- check examples/hello.sio)

echo "[fast-gate] 9/9 e2e backend gate"
"$ROOT_DIR/scripts/e2e_gate.sh"

echo "[fast-gate] ok"
