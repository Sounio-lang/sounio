#!/usr/bin/env bash
# irfunction_instr_capacity_coherence_gate.sh — pin what IR_MAX_INSTRS now means.
#
# This gate used to assert that IR_MAX_INSTRS, the `pub instrs: [IrInstr; N]`
# field and its `[ir_nop(); N]` initialiser all carried the SAME literal, because
# Sounio array sizes must be literals and could silently diverge from the
# constant.
#
# #1649 removed that field: instruction storage is a variable-size arena region,
# so there is no literal left to keep coherent, and the old gate fails with
# `irfunction_instrs_field_count_0`. The invariant worth pinning changed shape.
#
# WHAT CAN GO WRONG NOW
# ---------------------
# Raising IR_MAX_INSTRS is cheap in storage and dangerous in analysis. Several
# passes populate a FIXED per-instruction context and stop at their own cap:
#
#     while i < func.instr_count && i < DCE_MAX_INSTRS { ... }
#
# A truncated liveness or constant-propagation analysis is not a weaker analysis,
# it is a WRONG one -- a use past the cap is never recorded, so its definition
# looks dead and the sweep deletes live code. Silently, at rc=0. That is the same
# family as #1586 / #1570 / #1577.
#
# So each such pass must REFUSE a function it cannot hold, and this gate proves
# the refusal is present rather than trusting that someone remembered.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${IRFUNCTION_INSTR_CAPACITY_SOURCE:-$ROOT/self-hosted/ir/ir.sio}"
EXPECTED_CAPACITY="${IRFUNCTION_INSTR_CAPACITY_EXPECTED:-16384}"

fail() {
  printf 'IRFUNCTION_INSTR_CAPACITY_FAIL reason=%s\n' "$1" >&2
  exit 1
}

[[ -f "$SOURCE" ]] || fail source_missing
command -v node >/dev/null 2>&1 || fail node_missing
node "$ROOT/scripts/ci/capacity_coherence.mjs" ir "$SOURCE" "$EXPECTED_CAPACITY" "$ROOT" || exit $?

source_sha256="$(sha256sum "$SOURCE" | awk '{print $1}')"
head_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf not_available)"
tree_sha="$(git -C "$ROOT" rev-parse 'HEAD^{tree}' 2>/dev/null || printf not_available)"
worktree_state=not_available
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]; then
    worktree_state=clean
  else
    worktree_state=dirty
  fi
fi

printf '%s\n' \
  'IRFUNCTION_INSTR_CAPACITY_BOUNDARY storage_coherence=proved analysis_refusal=proved runtime_readback=not_claimed soir_storage_pool=checked wire_compatibility=separate_gate'
printf 'IRFUNCTION_INSTR_CAPACITY_PASS source=%s source_sha256=%s expected=%s head=%s tree=%s worktree=%s\n' \
  "$SOURCE" "$source_sha256" "$EXPECTED_CAPACITY" "$head_sha" "$tree_sha" "$worktree_state"
