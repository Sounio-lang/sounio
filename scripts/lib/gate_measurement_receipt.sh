#!/usr/bin/env bash
# Positive-measurement receipts for CI gates.
#
# A gate that exits 0 without recording how many assertions it exercised is
# indistinguishable from a gate that did no work. This library is the minimal
# contract that collapses that ambiguity:
#
#   GATE_MEASURED schema=sounio.gate.measurement.v1 gate=<name> assertions=<N> status=pass|fail|skip [reason=...]
#
# Rules (enforced by gate_measurement_emit and by the meta-gate):
#   - status=pass requires assertions >= 1
#   - status=skip may have assertions=0 only with a non-empty reason=
#   - status=fail may have any count (failure is already a signal)
#
# Source after gate_assert.sh:
#   . scripts/lib/gate_assert.sh
#   . scripts/lib/gate_measurement_receipt.sh
#   gate_name "my_gate"
#   gate_measurement_add 3    # or gate_measurement_set N
#   gate_measurement_emit pass

if [[ -n "${_SOUNIO_GATE_MEASUREMENT_SOURCED:-}" ]]; then return 0; fi
_SOUNIO_GATE_MEASUREMENT_SOURCED=1

_GATE_MEASURED_ASSERTIONS=0

gate_measurement_reset() {
  _GATE_MEASURED_ASSERTIONS=0
}

gate_measurement_set() {
  _GATE_MEASURED_ASSERTIONS="${1:?gate_measurement_set: need integer count}"
}

gate_measurement_add() {
  local n="${1:-1}"
  _GATE_MEASURED_ASSERTIONS=$((_GATE_MEASURED_ASSERTIONS + n))
}

gate_measurement_count() {
  printf '%s' "${_GATE_MEASURED_ASSERTIONS}"
}

# Emit one machine line. Refuses pass-with-zero (success without work).
gate_measurement_emit() {
  local status="${1:-pass}"
  local reason="${2:-}"
  local gate="${_GATE_NAME:-gate}"
  local n="${_GATE_MEASURED_ASSERTIONS}"

  case "$status" in
    pass)
      if [[ "$n" -lt 1 ]]; then
        if declare -F gate_fail >/dev/null 2>&1; then
          gate_fail "refusing status=pass with assertions=${n} — that is success without work (absence read as positive signal)"
        fi
        echo "GATE_MEASUREMENT_EMIT_FAIL: pass with assertions=${n}" >&2
        exit 1
      fi
      ;;
    skip)
      if [[ "$n" -lt 1 && -z "$reason" ]]; then
        if declare -F gate_fail >/dev/null 2>&1; then
          gate_fail "status=skip with assertions=0 requires reason= (otherwise it is a silent green)"
        fi
        echo "GATE_MEASUREMENT_EMIT_FAIL: skip without reason" >&2
        exit 1
      fi
      ;;
    fail) ;;
    *)
      if declare -F gate_fail >/dev/null 2>&1; then
        gate_fail "unknown measurement status='$status'"
      fi
      echo "GATE_MEASUREMENT_EMIT_FAIL: bad status" >&2
      exit 1
      ;;
  esac

  if [[ -n "$reason" ]]; then
    # reason must be a single token (no spaces) for line-oriented parsing
    reason="${reason// /_}"
    printf 'GATE_MEASURED schema=sounio.gate.measurement.v1 gate=%s assertions=%s status=%s reason=%s\n' \
      "$gate" "$n" "$status" "$reason"
  else
    printf 'GATE_MEASURED schema=sounio.gate.measurement.v1 gate=%s assertions=%s status=%s\n' \
      "$gate" "$n" "$status"
  fi
}

# Parse a log/stream; print assertions for the last GATE_MEASURED pass line, or empty.
gate_measurement_last_pass_assertions() {
  local file="${1:-}"
  local line n
  if [[ -n "$file" ]]; then
    line="$(grep -E '^GATE_MEASURED ' "$file" 2>/dev/null | grep ' status=pass' | tail -1 || true)"
  else
    line="$(grep -E '^GATE_MEASURED ' | grep ' status=pass' | tail -1 || true)"
  fi
  [[ -n "$line" ]] || return 0
  n="$(sed -n 's/.* assertions=\([0-9][0-9]*\).*/\1/p' <<<"$line")"
  printf '%s' "$n"
}

# Validate: a successful subject run (rc=0) must carry GATE_MEASURED assertions>=1.
# Usage: gate_measurement_require_positive_receipt <log_file> <subject_rc> <label>
gate_measurement_require_positive_receipt() {
  local log="$1"
  local rc="$2"
  local label="${3:-subject}"
  local line n status

  if [[ "$rc" -ne 0 ]]; then
    # Failure is already a signal; receipt optional.
    return 0
  fi

  line="$(grep -E '^GATE_MEASURED ' "$log" 2>/dev/null | tail -1 || true)"
  if [[ -z "$line" ]]; then
    if declare -F gate_fail >/dev/null 2>&1; then
      gate_fail "$label exited 0 with no GATE_MEASURED line — success without a measurement receipt"
    fi
    return 1
  fi

  status="$(sed -n 's/.* status=\([a-z]*\).*/\1/p' <<<"$line")"
  n="$(sed -n 's/.* assertions=\([0-9][0-9]*\).*/\1/p' <<<"$line")"

  if [[ "$status" == "pass" ]]; then
    if [[ -z "$n" || "$n" -lt 1 ]]; then
      if declare -F gate_fail >/dev/null 2>&1; then
        gate_fail "$label status=pass with assertions=${n:-empty} — measured nothing"
      fi
      return 1
    fi
  elif [[ "$status" == "skip" ]]; then
    if ! grep -q ' reason=' <<<"$line"; then
      if declare -F gate_fail >/dev/null 2>&1; then
        gate_fail "$label status=skip without reason= — silent skip is green without work"
      fi
      return 1
    fi
  fi
  return 0
}
