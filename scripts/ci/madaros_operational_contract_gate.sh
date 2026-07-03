#!/usr/bin/env bash
# Guard the committed operational contract that tells agents how to reason about
# Madaros status. This is intentionally cheap: the heavy compiler proof remains
# `make madaros-full-gate`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GREEN_BASE="17d1157be540d32bb583dd03ca7072a6026e2027"

fail() {
  echo "[madaros-contract] FAIL: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing file: $1"
}

require_executable() {
  [[ -x "$1" ]] || fail "missing executable: $1"
}

require_grep() {
  local pattern="$1"
  local file="$2"
  grep -Fq -- "$pattern" "$file" || fail "missing marker in $file: $pattern"
}

require_file docs/MADAROS_STATUS.md
require_file docs/audit/MADAROS_WORKTREE_CLEANUP_LEDGER_2026-07-03.md
require_file docs/status/madaros_main_proof_17d115.md
require_file AGENTS.md
require_file CLAUDE.md
require_file Makefile
require_file .gitignore
require_executable bin/souc
require_executable bin/madaros
require_file scripts/lib/resolve_madaros.sh
require_file scripts/ci/build_modular_madaros.sh
require_file scripts/ci/madaros_full_gate.sh
require_file scripts/ci/madaros_source_to_elf_gate.sh
require_executable scripts/dev/madaros_two_gate.sh
require_executable scripts/dev/madaros_worktree_cleanup_plan.sh
require_file .github/workflows/ci.yml

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git merge-base --is-ancestor "$GREEN_BASE" HEAD ||
    fail "HEAD does not contain Madaros green base $GREEN_BASE"
fi

require_grep 'bin/souc` routes to Madaros' docs/MADAROS_STATUS.md
require_grep 'receipt-gated' docs/MADAROS_STATUS.md
require_grep 'bin/madaros-relocgate' docs/MADAROS_STATUS.md
require_grep 'artifacts/self-hosted/madaros.gate-receipt' docs/MADAROS_STATUS.md
require_grep 'imported-SMT solver gate (6/6)' docs/MADAROS_STATUS.md
require_grep 'bin/madaros-linux-x86_64' docs/MADAROS_STATUS.md
require_grep 'make madaros-full-gate' docs/MADAROS_STATUS.md
require_grep 'ungated' docs/MADAROS_STATUS.md
require_grep 'artifacts/self-hosted/madaros' docs/MADAROS_STATUS.md
require_grep 'binary is **not evidence**' docs/MADAROS_STATUS.md
require_grep 'scripts/ci/madaros_operational_contract_gate.sh' docs/MADAROS_STATUS.md
require_grep 'scripts/dev/madaros_worktree_cleanup_plan.sh' docs/audit/MADAROS_WORKTREE_CLEANUP_LEDGER_2026-07-03.md

require_grep 'scripts/lib/resolve_madaros.sh' AGENTS.md
require_grep 'docs/MADAROS_STATUS.md' AGENTS.md
require_grep 'default compiler entrypoint and now routes to Madaros' CLAUDE.md

require_grep 'madaros-full-gate: build-madaros' Makefile
require_grep 'bash scripts/ci/madaros_full_gate.sh' Makefile
require_grep 'Madaros Stage1 full-functioning gate' scripts/dev/e2e_gate.sh
require_grep 'make -C "$ROOT_DIR" madaros-full-gate' scripts/dev/e2e_gate.sh
require_grep 'SOUNIO_SKIP_MADAROS' scripts/dev/e2e_gate.sh

require_grep 'RAW_MADAROS=' scripts/ci/madaros_full_gate.sh
require_grep '"$RAW_MADAROS" --check /tmp' scripts/ci/madaros_full_gate.sh
require_grep 'wrapper_tmp_dir.log' scripts/ci/madaros_full_gate.sh
require_grep 'error[E175' scripts/ci/madaros_full_gate.sh
require_grep 'error[E176' scripts/ci/madaros_full_gate.sh
require_grep 'error[E177' scripts/ci/madaros_full_gate.sh
require_grep '--native-v2-emit-sret' scripts/ci/madaros_full_gate.sh
require_grep 'pkg self-test' scripts/ci/madaros_full_gate.sh
require_grep 'resolve_raw_madaros' scripts/ci/madaros_full_gate.sh
require_grep 'receipt_ok' scripts/ci/madaros_full_gate.sh
require_grep 'SOUNIO_MADAROS_FULL_GATE_SKIP_SMT' scripts/ci/madaros_full_gate.sh
require_grep 'imported-SMT solver gate' scripts/ci/madaros_full_gate.sh
require_grep 'write_gate_receipt' scripts/ci/madaros_full_gate.sh
require_grep 'Madaros Greenline Gate' .github/workflows/ci.yml
require_grep 'make madaros-full-gate' .github/workflows/ci.yml
require_grep 'scripts/dev/madaros_two_gate.sh artifacts/self-hosted/madaros' .github/workflows/ci.yml
require_grep 'madaros_source_to_elf_gate.sh' .github/workflows/madaros-prebuilt-refresh.yml
require_grep 'source-to-ELF' .github/workflows/madaros-prebuilt-refresh.yml

require_grep 'DEFAULT ENGINE: Madaros' bin/souc
require_grep '.gate-receipt' bin/souc
require_grep 'bin/madaros-relocgate' bin/souc
require_grep '.gate-receipt' bin/madaros
require_grep 'bin/madaros-relocgate' bin/madaros
require_grep '.gate-receipt' scripts/lib/resolve_madaros.sh
require_grep 'bin/madaros-relocgate' scripts/lib/resolve_madaros.sh
require_grep 'bin/madaros-linux-x86_64' bin/souc
require_grep 'exec env MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros"' bin/souc
require_grep 'artifacts/self-hosted/madaros.gate-receipt' .gitignore
require_grep 'cleanup_plan_command=scripts/dev/madaros_worktree_cleanup_plan.sh' scripts/dev/madaros_readiness_status.sh
require_grep 'It never runs git push, git reset, git clean, git branch -D, or' scripts/dev/madaros_worktree_cleanup_plan.sh
require_grep 'owner confirmation required before any archive, push, or removal' scripts/dev/madaros_worktree_cleanup_plan.sh

echo "[madaros-contract] PASS: status doc, agent contract, default wrapper, and gate wiring are aligned"
