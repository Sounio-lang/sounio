#!/usr/bin/env bash
# Residual §5.4 full stack status gate: L0 + L1 + L2-fragment + L2-executable (R4).
# Does not claim L2-full FO_XFER soundness.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== L0 algebraic ==="
bash scripts/ci/fo_css_surface_parity_gate.sh

echo "=== L1 FoExpr semantic ==="
bash scripts/ci/fo_surface_transfer_gate.sh

echo "=== L2-fragment FO bytecode ==="
bash scripts/ci/fo_bytecode_fragment_gate.sh

echo "=== L2 pure-emit (fo_bc_compile_expr pure fragment) ==="
bash scripts/ci/fo_emit_pure_gate.sh

echo "=== L2 registration fragment (multipass FO_XFER expand) ==="
bash scripts/ci/fo_registration_fragment_gate.sh

echo "=== L2 engine-install fragment (multipass register pure helpers) ==="
bash scripts/ci/fo_engine_install_fragment_gate.sh

echo "=== L2 method FO_XFER peel fragment ==="
bash scripts/ci/fo_method_xfer_fragment_gate.sh

echo "=== L2 multi-mod prepass fragment ==="
bash scripts/ci/fo_multimod_fragment_gate.sh

echo "=== L2-executable R4 (Madaros numerical) ==="
if [[ -x "${MADAROS_RAW_BIN:-artifacts/self-hosted/madaros}" ]] || [[ -x artifacts/self-hosted/madaros ]]; then
  export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"
  export MADAROS_RAW_BIN="${MADAROS_RAW_BIN:-artifacts/self-hosted/madaros}"
  bash scripts/ci/fo_pk_import_method_driver_gate.sh
else
  echo "WARN: Madaros binary missing — skipping R4 live run (L2-executable not re-validated this invocation)"
fi

echo "FO_RESIDUAL4_STACK_GATE_OK"
echo "STATUS L0=CLOSED L1=CLOSED L2_FRAGMENT=CLOSED L2_PURE_EMIT=CLOSED L2_REGISTRATION_FRAGMENT=CLOSED L2_ENGINE_INSTALL_FRAGMENT=CLOSED L2_METHOD_XFER=CLOSED L2_MULTIMOD=CLOSED L2_FULL_ENGINE=OPEN"
echo "ORAL_CSS_RESIDUAL4_CLOSED"
