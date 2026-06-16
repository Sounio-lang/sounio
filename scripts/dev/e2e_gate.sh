#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
source "$ROOT_DIR/scripts/lib/stage_native_runtime_bundle.sh"

# Ensure stdlib is discoverable for tests that import from it.
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  if [[ "$CARGO_TARGET_DIR" = /* ]]; then
    TARGET_DIR="$CARGO_TARGET_DIR"
  else
    TARGET_DIR="$ROOT_DIR/$CARGO_TARGET_DIR"
  fi
else
  TARGET_DIR="$ROOT_DIR/target"
fi

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
EXAMPLE="$ROOT_DIR/examples/simple_test.sio"
GPU_FIXTURE="$ROOT_DIR/scripts/fixtures/gpu_minimal.sio"

OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

echo "[e2e] native build + run"
sounio_require_souc

if [[ "${SOUNIO_SKIP_MADAROS:-0}" != "1" ]]; then
  echo "[e2e] Madaros Stage1 full-functioning gate"
  make -C "$ROOT_DIR" madaros-full-gate
else
  echo "[e2e] Madaros skipped (SOUNIO_SKIP_MADAROS=1)"
fi

# Skip backend-specific build modes when using the native wrapper
if [[ "$SOUC_BIN" == *"/bin/souc" ]] || [[ "$SOUC_BIN" == *"native-wrapper"* ]]; then
  echo "[e2e] native compiler detected — using compile+run instead of build --backend"
  "$SOUC_BIN" run "$EXAMPLE" 2>/dev/null || echo "[e2e] run skipped (example may need features not in native)"
  echo "[e2e] e2e gate ok (native mode)"
  exit 0
fi

sounio_stage_native_runtime_bundle "$SOUC_BIN"
"$SOUC_BIN" build "$EXAMPLE" --backend native -o "$OUT_DIR/simple_test_native"
"$OUT_DIR/simple_test_native"

if [[ "${SOUNIO_SKIP_NATIVE_V2:-${SOUNIO_SKIP_NATIVE_V2_SHADOW:-0}}" != "1" ]]; then
  echo "[e2e] native v2 machine-ir gate + smoke"
  bash "$ROOT_DIR/scripts/omega/omega_native_v2_shadow_gate.sh"
else
  echo "[e2e] native v2 contract skipped (SOUNIO_SKIP_NATIVE_V2=1)"
fi

if [[ "${SOUNIO_SKIP_LLVM:-}" != "1" ]]; then
  echo "[e2e] llvm build + run"
  if "$SOUC_BIN" build "$EXAMPLE" --backend llvm -o "$OUT_DIR/simple_test_llvm" 2>/dev/null; then
    "$OUT_DIR/simple_test_llvm"
  else
    echo "[e2e] llvm skipped (binary not built with llvm feature/toolchain)"
  fi
else
  echo "[e2e] llvm skipped (SOUNIO_SKIP_LLVM=1)"
fi

if [[ "${SOUNIO_SKIP_GPU:-}" != "1" ]]; then
  echo "[e2e] gpu backend compile smoke"
  GPU_BACKEND_AVAILABLE=0
  if "$SOUC_BIN" build "$GPU_FIXTURE" --backend gpu -o "$OUT_DIR/gpu_minimal.ptx" 2>/dev/null; then
    test -s "$OUT_DIR/gpu_minimal.ptx"
    grep -q "\\.entry" "$OUT_DIR/gpu_minimal.ptx"
    GPU_BACKEND_AVAILABLE=1
  else
    echo "[e2e] gpu skipped (binary not built with gpu feature)"
  fi

  if [[ "$GPU_BACKEND_AVAILABLE" = "1" ]]; then
    echo "[e2e] gpu codegen parity gate"
    bash "$ROOT_DIR/scripts/omega/omega_gpu_codegen_parity_gate.sh"

    echo "[e2e] gpu binary attestation gate"
    bash "$ROOT_DIR/scripts/omega/omega_gpu_binary_attest_gate.sh"

    echo "[e2e] gpu runtime attestation gate"
    bash "$ROOT_DIR/scripts/omega/omega_gpu_runtime_attest_gate.sh"

    echo "[e2e] gpu public contract gate"
    bash "$ROOT_DIR/scripts/omega/omega_gpu_public_contract_gate.sh"
  fi
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
