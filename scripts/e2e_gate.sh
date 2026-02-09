#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  if [[ "$CARGO_TARGET_DIR" = /* ]]; then
    TARGET_DIR="$CARGO_TARGET_DIR"
  else
    TARGET_DIR="$ROOT_DIR/$CARGO_TARGET_DIR"
  fi
else
  TARGET_DIR="$ROOT_DIR/target"
fi

SOUC_BIN="$TARGET_DIR/debug/souc"
EXAMPLE="$ROOT_DIR/examples/simple_test.sio"
GPU_FIXTURE="$ROOT_DIR/scripts/fixtures/gpu_minimal.sio"

OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

echo "[e2e] native build + run"
(cd "$ROOT_DIR" && cargo build -q -p souc --bin souc)
"$SOUC_BIN" build "$EXAMPLE" --backend native -o "$OUT_DIR/simple_test_native"
"$OUT_DIR/simple_test_native"

if [[ "${SOUNIO_SKIP_LLVM:-}" != "1" ]]; then
  echo "[e2e] llvm build + run"
  LLVM_BUILD_LOG="$OUT_DIR/llvm_build.log"
  if (cd "$ROOT_DIR" && cargo build -q -p souc --features llvm --bin souc) >"$LLVM_BUILD_LOG" 2>&1; then
    "$SOUC_BIN" build "$EXAMPLE" --backend llvm -o "$OUT_DIR/simple_test_llvm"
    "$OUT_DIR/simple_test_llvm"
  elif grep -q "No suitable version of LLVM was found" "$LLVM_BUILD_LOG"; then
    echo "[e2e] llvm skipped (LLVM toolchain not available)"
  else
    echo "[e2e] llvm build failed"
    cat "$LLVM_BUILD_LOG"
    exit 1
  fi
else
  echo "[e2e] llvm skipped (SOUNIO_SKIP_LLVM=1)"
fi

if [[ "${SOUNIO_SKIP_GPU:-}" != "1" ]]; then
  echo "[e2e] gpu compile-only"
  (cd "$ROOT_DIR" && cargo build -q -p souc --features gpu --bin souc)
  "$SOUC_BIN" build "$GPU_FIXTURE" --backend gpu -o "$OUT_DIR/gpu_minimal.ptx"
  test -s "$OUT_DIR/gpu_minimal.ptx"
  grep -q "\\.entry" "$OUT_DIR/gpu_minimal.ptx"
else
  echo "[e2e] gpu skipped (SOUNIO_SKIP_GPU=1)"
fi

if [[ "${SOUNIO_SKIP_ONTOLOGY:-}" != "1" ]]; then
  echo "[e2e] ontology cross-check"

  ONTOLOGY_OK="$OUT_DIR/ontology_ok.sio"
  cat >"$ONTOLOGY_OK" <<'EOF'
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.1;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.2)]
fn analyze(d: ChEBIDrug) {
}

fn main() {
  let db_drug: DrugBankDrug = drugbank:DB00945;
  analyze(db_drug);
}
EOF

  "$SOUC_BIN" check "$ONTOLOGY_OK"

  ONTOLOGY_FAIL="$OUT_DIR/ontology_fail.sio"
  cat >"$ONTOLOGY_FAIL" <<'EOF'
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.5;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.3)]
fn strict_analysis(d: ChEBIDrug) {
}

fn main() {
  let db_drug: DrugBankDrug = drugbank:DB00945;
  strict_analysis(db_drug);
}
EOF

  if "$SOUC_BIN" check "$ONTOLOGY_FAIL" >"$OUT_DIR/ontology_fail.stdout" 2>"$OUT_DIR/ontology_fail.stderr"; then
    echo "[e2e] ontology mismatch unexpectedly succeeded"
    exit 1
  fi

  if ! grep -qi "semantic distance" "$OUT_DIR/ontology_fail.stderr"; then
    echo "[e2e] ontology mismatch did not report semantic distance diagnostics"
    cat "$OUT_DIR/ontology_fail.stderr"
    exit 1
  fi
else
  echo "[e2e] ontology skipped (SOUNIO_SKIP_ONTOLOGY=1)"
fi

echo "[e2e] ok"
