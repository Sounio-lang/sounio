#!/usr/bin/env bash
# Observe a long command's log without changing its argv, log bytes or exit status.
# The observer does not impose a timeout or retry the command.
# Best-effort Linux host/root-cgroup counters. These are sampled infrastructure
# evidence, not command-attributed allocation or proof of a termination cause.
# Missing files remain explicitly unavailable; telemetry must never set gate rc.
gate_observe_resources() (
  local label="$1" phase="$2" file line found
  for file in /proc/meminfo /proc/vmstat /proc/loadavg /proc/pressure/memory /sys/fs/cgroup/memory.events; do
    found=0
    if [[ -r "$file" ]]; then
      while IFS= read -r line; do
        case "$file:$line" in
          /proc/meminfo:MemTotal:*|/proc/meminfo:MemAvailable:*|/proc/meminfo:SwapFree:*|/proc/vmstat:oom_kill\ *|/proc/loadavg:*|/proc/pressure/memory:*|/sys/fs/cgroup/memory.events:*)
            printf 'GATE_RESOURCE label=%s phase=%s source=%s value=%s\n' "$label" "$phase" "$file" "$line"
            found=1 ;;
        esac
      done < "$file"
    fi
    if [[ "$found" == 0 ]]; then
      printf 'GATE_RESOURCE label=%s phase=%s source=%s unavailable=true\n' "$label" "$phase" "$file"
    fi
  done
  return 0
)

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
  gate_observe_resources "$label" start || true
  (
    sleeper=""
    trap '[[ -z "$sleeper" ]] || kill "$sleeper" 2>/dev/null; exit 0' TERM INT
    started=$SECONDS
    while :; do
      sleep "$interval" & sleeper=$!
      wait "$sleeper" || exit 0
      sleeper=""
      elapsed=$((SECONDS - started))
      gate_observe_resources "$label" sample || true
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
  gate_observe_resources "$label" end || true
  printf 'GATE_OBSERVER_END label=%s command_rc=%s\n' "$label" "$rc"
  exit "$rc"
)
