# scripts/lib/lib_sounio_claim_oracle.sh
# ADR-008 helpers for claim gates that still carry a foreign corroboration path.
#
#   source this file after `set -euo pipefail` and `cd` to repo root.
#
# Claim clock: Sounio-native sentinels / constants / OK tokens (must hard-fail).
# Foreign path: Python/mpmath/diff — report by default; hard-fail only if
#   SOUNIO_FOREIGN_ORACLE_HARD=1

# shellcheck disable=SC2034
SOUNIO_FOREIGN_ORACLE_HARD="${SOUNIO_FOREIGN_ORACLE_HARD:-0}"

# Record a foreign mismatch. Returns 1 only when HARD=1 (caller may set fail=1).
sounio_foreign_mismatch() {
  local msg="${1:-foreign mismatch}"
  echo "FOREIGN CORROBORATION MISMATCH (ADR-008): $msg"
  if [ "${SOUNIO_FOREIGN_ORACLE_HARD}" = "1" ]; then
    echo "  hard-fail: SOUNIO_FOREIGN_ORACLE_HARD=1"
    return 1
  fi
  echo "  report-only (claim clock is Sounio); export SOUNIO_FOREIGN_ORACLE_HARD=1 to fail"
  return 0
}

# Soft-diff two files: non-zero only under HARD=1.
sounio_foreign_diff() {
  local a="$1" b="$2"
  local label="${3:-diff}"
  if diff -q "$a" "$b" >/dev/null 2>&1; then
    return 0
  fi
  echo "FOREIGN CORROBORATION DIFF ($label):"
  diff "$a" "$b" | head -40 || true
  if [ "${SOUNIO_FOREIGN_ORACLE_HARD}" = "1" ]; then
    return 1
  fi
  return 0
}
