#!/usr/bin/env bash
# native_capacity_tiers_gate.sh — pin the native x86-64 backend capacity tiers and
# the FAIL-CLOSED handling of each overflow.
#
# History: nc_add_flat_reloc used to stop recording past 65,536 relocations with no
# sentinel and no error, so a program needing more relocations got a WRITTEN ELF
# carrying unpatched call/data sites — jumps to garbage, SIGSEGV at runtime — while
# the compiler reported success. The code buffer was already fail-closed (rc=19);
# only the relocation side was silent.
#
# This gate proves, statically (see native_capacity_tiers_check.py):
#   1. each tier's array DECLARATION matches its named accessor (declarations must
#      stay literals — Sounio array sizes are not expressions);
#   2. no scattered duplicate literal bound-check survives, so the next capacity
#      bump is one edit per tier and cannot become a silent partial no-op;
#   3. nc_add_flat_reloc has an else-branch that raises the reloc_overflow sentinel;
#   4. the native_v2 writer returns a DISTINCT rc=20 for relocation overflow
#      (vs rc=19 for code-buffer overflow), so callers can tell the tiers apart;
#   5. the legacy NATIVE_ELF_BUF writer bounds its byte emitter and consults the
#      emitter overflow flags before writing (it previously did neither);
#   6b. the per-function LABEL tier (NC_V2_LABEL_*) is coherent, fail-closed with
#      its own rc=22, and at least as large as IR_MAX_INSTRS. History: it was
#      [i64; 256] with BOTH bounds silent, so a function with more than ~128 `if`s
#      had jump patches dropped -- the jz/jmp rel32 operand stayed at the
#      placeholder 0 and the branch fell through. Programs exited 0 with wrong
#      answers, and past ~160 `if`s printed nothing at all. That is #1586/#1570,
#      and it is why "move it into a helper function" read as a fix: the tier
#      resets per function. The >= IR_MAX_INSTRS clause is the actual proof of
#      sufficiency -- every label and patch comes from an IR instruction, and the
#      emit loop that reads them is bounded by IR_MAX_INSTRS.
#   7. the ELF LOAD BASE ADDRESS 0x400000 (4194304) was NOT rewritten as a capacity.
#      It numerically collides with the retired 4 MiB ELF tier and must stay a
#      plain literal argument — this clause is the regression guard for that trap.
#
# Static only: it does not run the compiler. rc=20 firing end-to-end is proven in
# the commit that introduced this gate (capacity temporarily lowered in a scratch
# build until the limit tripped, then restored).
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(dirname "$(dirname "$(dirname "$SELF")")")"
CHECKER="$ROOT/scripts/ci/native_capacity_tiers_check.py"

EXPECT_CODE=134217728       # NC_BIG_CODE           128 MiB
EXPECT_RELOC=2097152        # NC_FLAT_RELOC_* x4    2,097,152 entries
EXPECT_ELF=167772160        # NC_BIG_ELF            160 MiB
EXPECT_LEGACY_ELF=16777216  # NATIVE_ELF_BUF        16 MiB
ELF_BASE_ADDR=4194304      # 0x400000 — load address, NOT a capacity
EXPECT_LABEL=16384         # NC_V2_LABEL_* x3     16,384 labels/patches per fn
                           # (must stay >= IR_MAX_INSTRS — see clause 8)

if ! command -v python3 >/dev/null 2>&1; then
  printf 'NATIVE_CAPACITY_TIERS_FAIL reason=python3_missing\n' >&2
  exit 1
fi
if ! test -f "$CHECKER"; then
  printf 'NATIVE_CAPACITY_TIERS_FAIL reason=checker_missing\n' >&2
  exit 1
fi

# NOTE: the checker reads argv POSITIONALLY. Append new tiers at the END; putting
# one in the middle silently rotates every tier that follows it.
python3 "$CHECKER" "$ROOT" \
  "$EXPECT_CODE" "$EXPECT_RELOC" "$EXPECT_ELF" "$EXPECT_LEGACY_ELF" "$ELF_BASE_ADDR" \
  "$EXPECT_LABEL"

head_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf not_available)"
tree_sha="$(git -C "$ROOT" rev-parse 'HEAD^{tree}' 2>/dev/null || printf not_available)"
worktree_state=not_available
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if test -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)"; then
    worktree_state=clean
  else
    worktree_state=dirty
  fi
fi

printf '%s\n' \
  'NATIVE_CAPACITY_TIERS_BOUNDARY declaration_accessor_coherence=proved fail_closed_rc=proved runtime_emission=not_claimed elf_base_addr=preserved'
printf 'NATIVE_CAPACITY_TIERS_PASS code=%s reloc=%s elf=%s legacy_elf=%s head=%s tree=%s worktree=%s\n' \
  "$EXPECT_CODE" "$EXPECT_RELOC" "$EXPECT_ELF" "$EXPECT_LEGACY_ELF" "$head_sha" "$tree_sha" "$worktree_state"
