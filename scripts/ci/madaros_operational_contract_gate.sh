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
require_file docs/status/madaros_main_proof_17d115.md
require_file AGENTS.md
require_file CLAUDE.md
require_file Makefile
require_executable bin/souc
require_executable bin/madaros
require_file scripts/lib/resolve_madaros.sh
require_file scripts/ci/build_modular_madaros.sh
require_file scripts/ci/madaros_full_gate.sh
require_file scripts/ci/madaros_source_to_elf_gate.sh

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git merge-base --is-ancestor "$GREEN_BASE" HEAD ||
    fail "HEAD does not contain Madaros green base $GREEN_BASE"
fi

require_grep 'Madaros is **green on `origin/main`' docs/MADAROS_STATUS.md
require_grep 'bin/souc` routes to Madaros' docs/MADAROS_STATUS.md
require_grep 'bin/madaros-linux-x86_64' docs/MADAROS_STATUS.md
require_grep 'make madaros-full-gate' docs/MADAROS_STATUS.md
require_grep 'stale raw `artifacts/self-hosted/madaros`' docs/MADAROS_STATUS.md
require_grep 'binaries are **not** evidence' docs/MADAROS_STATUS.md
require_grep 'scripts/ci/madaros_operational_contract_gate.sh' docs/MADAROS_STATUS.md

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
require_grep 'madaros_source_to_elf_gate.sh' .github/workflows/madaros-prebuilt-refresh.yml
require_grep 'source-to-ELF' .github/workflows/madaros-prebuilt-refresh.yml

require_grep 'DEFAULT ENGINE: Madaros' bin/souc
require_grep 'bin/madaros-linux-x86_64' bin/souc
require_grep 'exec env MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros"' bin/souc

echo "[madaros-contract] PASS: status doc, agent contract, default wrapper, and gate wiring are aligned"
