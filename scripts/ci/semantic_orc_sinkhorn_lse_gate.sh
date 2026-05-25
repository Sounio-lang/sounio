#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-bin/souc}"
SOURCE="examples/semantic_orc/sinkhorn_lse_orc.sio"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_LSE_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-lse.XXXXXX)}"
OUT_FILE="$OUT_DIR/sinkhorn_lse_orc.out"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-lse] source=$SOURCE"
echo "[semantic-orc-lse] out=$OUT_DIR"

"$SOUC" check "$SOURCE" >/dev/null
"$SOUC" run "$SOURCE" >"$OUT_FILE"

required_tokens=(
  "SOUNIO_SEMANTIC_ORC_LSE_PASS"
  "claim_level=semantic_orc_lse_runtime_gate"
  "epistemic_epsilon_gate=true"
  "epsilon_range_gate=true"
  "same_support_kappa_positive=true"
  "bottleneck_kappa_negative=true"
  "marginal_constraints=true"
  "nonuniform4_fixed_point=true"
  "sinkhorn16_symmetric_fixed_point=true"
  "sinkhorn16_nonuniform_fixed_point=true"
  "sinkhorn16_nonuniform_adaptive_stop=true"
  "instance_checks_passed=true"
  "entropic_regularized_transport_only=true"
  "clinical_claims_not_enforced=true"
  "no_convergence_theorem=true"
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fqx "$token" "$OUT_FILE"; then
    echo "[semantic-orc-lse] FAIL: missing token: $token" >&2
    echo "[semantic-orc-lse] output:" >&2
    cat "$OUT_FILE" >&2
    exit 1
  fi
done

echo "[semantic-orc-lse] PASS"
