#!/usr/bin/env bash
# Dissertation compiler pin check.
#
# Verifies the selected native souc ELF, not only the wrapper script. The
# wrapper hash is identical across known A/B compiler divergences.

set -euo pipefail

EXPECTED_SHA="${SOUNIO_EXPECTED_SOUC_SHA:-3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93}"

if [[ "${SKIP_COMPILER_PIN_CHECK:-0}" == "1" ]]; then
  echo "verify_compiler_pin: SKIPPED (SKIP_COMPILER_PIN_CHECK=1)"
  exit 0
fi

candidate="${SOUC_BIN:-${SOUC:-./bin/souc}}"
if [[ ! -x "$candidate" ]]; then
  echo "FATAL: compiler candidate is not executable: $candidate" >&2
  exit 2
fi

selected="$candidate"
if selected_probe="$(SOUNIO_SOUC_PRINT_PATH=1 "$candidate" 2>/dev/null)"; then
  if [[ -n "$selected_probe" ]]; then
    selected="$selected_probe"
  fi
fi

if [[ ! -f "$selected" ]]; then
  echo "FATAL: selected compiler artifact does not exist: $selected" >&2
  echo "  Candidate wrapper: $candidate" >&2
  exit 2
fi

actual_sha="$(sha256sum "$selected" | awk '{print $1}')"
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
  echo "FATAL: SOUC compiler SHA mismatch" >&2
  echo "  Expected native SHA: $EXPECTED_SHA" >&2
  echo "  Actual native SHA:   $actual_sha" >&2
  echo "  Candidate wrapper:   $candidate" >&2
  echo "  Selected native:     $selected" >&2
  exit 1
fi

echo "verify_compiler_pin: PASS"
echo "  selected_native=$selected"
echo "  sha256=$actual_sha"
