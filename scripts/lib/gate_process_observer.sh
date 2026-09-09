#!/usr/bin/env bash
# Observe a long command's log without changing its argv, log bytes or exit status.
# The observer does not impose a timeout or retry the command.
gate_observe_command() (
  local label="$1" log="$2"; shift 2
  local interval="${SOUNIO_GATE_OBSERVE_SECONDS:-30}" observer="" rc
  [[ "$interval" =~ ^[1-9][0-9]*$ ]] || {
    echo "gate observer: interval must be a positive integer" >&2
    exit 64
  }
  cleanup_observer() {
    if [[ -n "$observer" ]]; then
      kill "$observer" 2>/dev/null || true
      wait "$observer" 2>/dev/null || true
    fi
  }
  trap cleanup_observer EXIT
  (
    sleeper=""
    trap '[[ -z "$sleeper" ]] || kill "$sleeper" 2>/dev/null; exit 0' TERM INT
    started=$SECONDS
    while :; do
      sleep "$interval" & sleeper=$!
      wait "$sleeper" || exit 0
      sleeper=""
      elapsed=$((SECONDS - started))
      printf 'GATE_OBSERVATION label=%s elapsed_seconds=%s log_bytes=' "$label" "$elapsed"
      if [[ -f "$log" ]]; then
        wc -c < "$log"
        tail -n 2 "$log"
      else
        printf '0\n'
      fi
    done
  ) &
  observer=$!
  printf 'GATE_OBSERVER_START label=%s observer_pid=%s interval_seconds=%s\n' "$label" "$observer" "$interval"
  if "$@" >"$log" 2>&1; then rc=0; else rc=$?; fi
  cleanup_observer
  observer=""
  printf 'GATE_OBSERVER_END label=%s command_rc=%s\n' "$label" "$rc"
  exit "$rc"
)
