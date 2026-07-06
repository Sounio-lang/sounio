#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EISA_REF="${EISA_REF:-origin/gpu/epistemic-tensor-core-next}"
MADAROS_REF="${MADAROS_REF:-origin/canon/madaros-v2-sota}"
BLOCKER_REF="${BLOCKER_REF:-origin/fix/madaros-imported-depclosure-eisa}"

fail() {
  echo "[eisa-metron-canon] FAIL: $*" >&2
  exit 1
}

need_ref() {
  local ref="$1"
  git rev-parse --verify --quiet "$ref" >/dev/null || fail "missing git ref: $ref"
}

need_path() {
  local ref="$1"
  local path="$2"
  git cat-file -e "$ref:$path" 2>/dev/null || fail "missing $ref:$path"
}

need_ref "$EISA_REF"
need_ref "$MADAROS_REF"
need_ref "$BLOCKER_REF"

eisa_paths=(
  docs/research/eisa-stack-architecture-2026-07-05.md
  docs/research/eisa-v1-asbuilt-2026-07-06.md
  docs/research/eisa-v2-arch-2026-07-05.md
  docs/research/eisa-v2-positioning-2026-07-05.md
  docs/handoff/eisa_w4_v2_bridge_continuation.md
  stdlib/eisa/format.sio
  stdlib/eisa/asm.sio
  stdlib/eisa/core.sio
  stdlib/eisa/core_v2.sio
  stdlib/eisa/evm.sio
  stdlib/eisa/backend.sio
  stdlib/eisa/bridge_x86.sio
  tools/eisa/eisa_evm_run.sio
  tools/eisa/eisa_bridge_emit.sio
  scripts/ci/eisa_bridge_conformance_gate.sh
  tests/stdlib/eisa/test_eisa_backend.sio
  tests/stdlib/eisa/test_eisa_backend_v1.sio
  tests/stdlib/eisa/test_eisa_bridge.sio
  tests/stdlib/eisa/test_eisa_bridge_v1.sio
  tests/stdlib/eisa/test_eisa_core.sio
  tests/stdlib/eisa/test_eisa_e5_kernel.sio
  tests/stdlib/eisa/test_eisa_evm.sio
  tests/stdlib/eisa/test_eisa_evm_v1.sio
  tests/stdlib/eisa/test_eisa_evm_v2.sio
  tests/stdlib/eisa/test_eisa_isa.sio
  tests/stdlib/eisa/test_eisa_v1e_showcase.sio
  tests/stdlib/eisa/test_eisax_format.sio
  tests/stdlib/eisa/test_eisax_v1_format.sio
)

for path in "${eisa_paths[@]}"; do
  need_path "$EISA_REF" "$path"
done

madaros_paths=(
  docs/research/madaros-v2-sota-plus-plus-plan-2026-07-04.md
  docs/research/madaros-v2-s1-receipt-implementation-2026-07-04.md
  docs/research/madaros-v2-s2-contract-scaffold-2026-07-04.md
  docs/research/madaros-v2-s3-hlir-serialization-2026-07-04.md
  docs/research/madaros-v2-s4-egraph-ekan-receipts-2026-07-05.md
  scripts/dev/madaros_v2_s1_gate.sh
  scripts/dev/madaros_v2_s2_gate.sh
  scripts/dev/madaros_v2_s3_gate.sh
  scripts/dev/madaros_v2_s4_gate.sh
  scripts/dev/madaros_v2_s5_preflight_gate.sh
)

for path in "${madaros_paths[@]}"; do
  need_path "$MADAROS_REF" "$path"
done

blocker_paths=(
  docs/handoff/continuity/wp-b1-witness/README.md
  docs/handoff/continuity/wp-b1-witness/depclosure_str_a.sio
  docs/handoff/continuity/wp-b1-witness/depclosure_str_b.sio
  docs/handoff/continuity/wp-b1-witness/depclosure_str_main.sio
)

for path in "${blocker_paths[@]}"; do
  need_path "$BLOCKER_REF" "$path"
done

eisa_test_count="$(git ls-tree -r --name-only "$EISA_REF" tests/stdlib/eisa | grep -E '^tests/stdlib/eisa/test_.*[.]sio$' | wc -l | tr -d ' ')"

head_has_eisa_source_tree="no"
if git ls-tree -r --name-only HEAD | awk '/^(stdlib\/eisa\/|tests\/stdlib\/eisa\/|tools\/eisa\/|scripts\/ci\/eisa_bridge_)/ { found=1 } END { exit found ? 0 : 1 }'; then
  head_has_eisa_source_tree="yes"
fi

echo "[eisa-metron-canon] PASS"
echo "  madaros_ref=$MADAROS_REF $(git rev-parse --short "$MADAROS_REF")"
echo "  eisa_ref=$EISA_REF $(git rev-parse --short "$EISA_REF")"
echo "  blocker_ref=$BLOCKER_REF $(git rev-parse --short "$BLOCKER_REF")"
echo "  eisa_test_suites=$eisa_test_count"
echo "  head_has_eisa_source_tree=$head_has_eisa_source_tree"
