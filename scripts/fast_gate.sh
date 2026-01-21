#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[fast-gate] 1/5 syntax drift scan"
python3 "$ROOT_DIR/skills/sounio-language/scripts/scan_syntax_drift.py" --root "$ROOT_DIR"

echo "[fast-gate] 2/5 compiler unit tests (cargo test --lib)"
(cd "$ROOT_DIR/compiler" && cargo test --lib)

echo "[fast-gate] 3/5 language suite (tests/)"
(cd "$ROOT_DIR/compiler" && cargo test --test language_suite)

echo "[fast-gate] 4/5 check canonical example"
(cd "$ROOT_DIR/compiler" && cargo run --quiet --bin souc -- check "$ROOT_DIR/examples/hello.sio")

echo "[fast-gate] 5/5 e2e backend gate"
"$ROOT_DIR/scripts/e2e_gate.sh"

echo "[fast-gate] ok"
