#!/usr/bin/env bash
# scripts/dev/run_clinical_twin.sh
#
# Build-and-run wrapper for stdlib/clinical/ digital-twin programs.
#
# WHY THIS EXISTS: Madaros native-v2 (the default `bin/souc` engine) has an
# open bug in multi-module native compilation (segfault in
# module_frontend_lower_programs_array_direct_box's array/box lowering seed
# step -- see docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md)
# that blocks ANY program importing another module (`use x::y::*`), which
# includes every file under stdlib/clinical/. The `lean_single` engine
# (bin/souc-lean-single-x86_64 -- the verified bootstrap seed compiler used
# to build Madaros itself) does not have this bug and produces
# hand-verified-correct output. Use this script, not `souc run`, for clinical
# / PBPK code until the Madaros bug is fixed.
#
# Usage: scripts/dev/run_clinical_twin.sh <file.sio>

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${1:?Usage: $0 <file.sio>}"
ENGINE="$ROOT_DIR/bin/souc-lean-single-x86_64"

if [[ ! -x "$ENGINE" ]]; then
    echo "error: $ENGINE not found or not executable" >&2
    exit 1
fi

TMP_ELF="$(mktemp /tmp/clinical_twin_run_XXXXXX)"
trap 'rm -f "$TMP_ELF"' EXIT

"$ENGINE" "$SRC" "$TMP_ELF"
chmod +x "$TMP_ELF"
exec "$TMP_ELF"
