#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

if [[ "$SKIP_BUILD" = "0" ]]; then
  echo "[fast-gate] 1/10 cargo preflight (no conflicting workspace cargo jobs)"
  bash "$ROOT_DIR/scripts/check_no_active_cargo_jobs.sh"
else
  echo "[fast-gate] 1/10 cargo preflight (skipped, SKIP_BUILD=1)"
fi

echo "[fast-gate] 2/10 syntax drift scan"
python3 "$ROOT_DIR/skills/sounio-language/scripts/scan_syntax_drift.py" --root "$ROOT_DIR"

echo "[fast-gate] 3/10 workflow script reference check"
bash "$ROOT_DIR/scripts/check_workflow_script_refs.sh"

echo "[fast-gate] 4/10 docs consistency check"
bash "$ROOT_DIR/scripts/check_docs_consistency.sh"

echo "[fast-gate] 5/10 issue template contract check"
bash "$ROOT_DIR/scripts/check_issue_template_contracts.sh"

echo "[fast-gate] 6/10 cultural fidelity (user-facing text leakage)"
python3 "$ROOT_DIR/scripts/cultural_fidelity_gate.py"

echo "[fast-gate] 7/10 compiler unit tests (cargo test --lib)"
(cd "$ROOT_DIR" && sounio_cargo test -p souc --lib)

echo "[fast-gate] 8/10 integration tests"
(cd "$ROOT_DIR" && sounio_cargo test -p souc --tests)

echo "[fast-gate] 9/10 check canonical example"
if [[ "$SKIP_BUILD" = "1" ]]; then
  sounio_require_souc
  "$SOUC_BIN" check "$ROOT_DIR/examples/hello.sio"
else
  (cd "$ROOT_DIR" && cargo run -p souc --quiet --bin souc -- check examples/hello.sio)
fi

echo "[fast-gate] 10/10 e2e backend gate"
"$ROOT_DIR/scripts/e2e_gate.sh"

echo "[fast-gate] ok"
