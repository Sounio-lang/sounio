#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

# Ensure stdlib is discoverable for tests that import from it.
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

if [[ "$SKIP_BUILD" = "0" ]]; then
  echo "[fast-gate] 1/14 cargo preflight (no conflicting workspace cargo jobs)"
  bash "$ROOT_DIR/scripts/ci/check_no_active_cargo_jobs.sh"
else
  echo "[fast-gate] 1/14 cargo preflight (skipped, SKIP_BUILD=1)"
fi

echo "[fast-gate] 2/14 syntax drift scan"
python3 "$ROOT_DIR/skills/sounio-language/scripts/scan_syntax_drift.py" --root "$ROOT_DIR"

echo "[fast-gate] 3/14 workflow script reference check"
bash "$ROOT_DIR/scripts/dev/check_workflow_script_refs.sh"

echo "[fast-gate] 4/14 docs consistency check"
bash "$ROOT_DIR/scripts/dev/check_docs_consistency.sh"

echo "[fast-gate] 5/14 docs registry check"
bash "$ROOT_DIR/scripts/dev/check_docs_registry.sh"

echo "[fast-gate] 6/14 ontology validation"
bash "$ROOT_DIR/scripts/ci/run_ontology_validation.sh" --mode rebuilt ontology

echo "[fast-gate] 7/14 issue template contract check"
bash "$ROOT_DIR/scripts/ci/check_issue_template_contracts.sh"

echo "[fast-gate] 8/14 cultural fidelity (user-facing text leakage)"
CULTURAL_FIXTURE_DIR="$ROOT_DIR/scripts/fixtures/cultural_fidelity"
if [[ "${FAST_GATE_CULTURAL_SCAN_DOCS:-0}" == "1" ]]; then
  echo "[fast-gate] cultural fidelity: scanning docs/ (FAST_GATE_CULTURAL_SCAN_DOCS=1)"
  python3 "$ROOT_DIR/scripts/ci/cultural_fidelity_gate.py" --root "$ROOT_DIR" --path "$ROOT_DIR/docs"
elif find "$ROOT_DIR/crates/souc/tests/golden" -type f \( -name '*.txt' -o -name '*.json' \) -print -quit 2>/dev/null | grep -q .; then
  echo "[fast-gate] cultural fidelity: scanning crates/souc/tests/golden defaults"
  python3 "$ROOT_DIR/scripts/ci/cultural_fidelity_gate.py" --root "$ROOT_DIR"
else
  echo "[fast-gate] cultural fidelity: no golden tree under crates/souc/tests/golden; fixture smoke (scanner + allowlist)"
  python3 "$ROOT_DIR/scripts/ci/cultural_fidelity_gate.py" --root "$ROOT_DIR" --no-default-targets \
    --path "$CULTURAL_FIXTURE_DIR/pass_user_output.txt" \
    --path "$CULTURAL_FIXTURE_DIR/dev_build.txt" \
    --allowlist "$CULTURAL_FIXTURE_DIR/allowlist.tsv"
fi

echo "[fast-gate] 9/14 compiler unit tests (cargo test --lib)"
if [[ "$SKIP_BUILD" = "1" ]]; then
  echo "[fast-gate] skipped (no-rust mode)"
else
  (cd "$ROOT_DIR" && sounio_cargo test -p souc --lib)
fi

echo "[fast-gate] 10/14 integration tests"
if [[ "$SKIP_BUILD" = "1" ]]; then
  echo "[fast-gate] skipped (no-rust mode)"
else
  (cd "$ROOT_DIR" && sounio_cargo test -p souc --tests)
fi

echo "[fast-gate] 11/14 check canonical example"
if [[ "$SKIP_BUILD" = "1" ]]; then
  sounio_require_souc
  "$SOUC_BIN" check "$ROOT_DIR/examples/hello.sio"
else
  (cd "$ROOT_DIR" && cargo run -p souc --quiet --bin souc -- check examples/hello.sio)
fi

echo "[fast-gate] 12/14 stdlib science pipeline gate"
bash "$ROOT_DIR/scripts/stdlib_science_pipeline_gate.sh"

echo "[fast-gate] 13/14 stdlib reliability gate"
bash "$ROOT_DIR/scripts/stdlib_reliability_gate.sh"

echo "[fast-gate] 14/14 e2e backend gate"
"$ROOT_DIR/scripts/dev/e2e_gate.sh"

echo "[fast-gate] ok"
