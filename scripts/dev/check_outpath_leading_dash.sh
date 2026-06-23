#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Pattern-B output-path scripts: the first positional is the OUTPUT path
# (OUT="${1:-default}"), with no flag grammar. A stray leading-dash token
# (e.g. an accidental --help) must be rejected rather than silently used as
# the write target — otherwise the script writes a dash-named file (or a
# binary) into the workspace. See docs/audit/WORKTREE_AUDIT_CHECK_LEAK_2026-06-23.md
# for the sibling Pattern-A guard.
OUTPATH_LD_SCRIPTS=(
  scripts/ontology/build_dontology_ffi_importer.sh
  scripts/ci/build_modular_madaros.sh
  scripts/ci/build_native_souc.sh
)

BOGUS="-bogus-outpath-ld"
failed=0

check_script() {
  local script="$1"

  if [[ ! -f "$script" ]]; then
    echo "missing script: $script" >&2
    failed=1
    return
  fi

  if ! grep -qE 'OUT="\$\{1:-' "$script"; then
    echo "regression: $script no longer uses the OUT=\"\${1:-...}\" positional-output form"
    failed=1
  fi

  # Only run the dynamic check when the guard is actually present, so a
  # regressed script is never invoked with a dash arg (which would itself
  # trigger the build-leak this test guards against).
  if ! grep -qE '== -\*' "$script"; then
    echo "regression: $script has no leading-dash output-path guard"
    failed=1
    return
  fi

  # Dynamic (fast): a leading-dash first arg must exit non-zero during arg
  # handling, before any build/compile work, and must not leave a residue.
  if bash "$script" "$BOGUS" >/dev/null 2>&1; then
    echo "regression: $script accepted a leading-dash output path"
    failed=1
  fi
  if [[ -e "$ROOT_DIR/$BOGUS" ]]; then
    echo "regression: $script wrote a dash-named residue ($BOGUS)"
    failed=1
    rm -f -- "$ROOT_DIR/$BOGUS"
  fi
}

for script in "${OUTPATH_LD_SCRIPTS[@]}"; do
  check_script "$script"
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "output-path leading-dash check passed (${#OUTPATH_LD_SCRIPTS[@]} scripts)"
