#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="${1:-$ROOT_DIR/artifacts/sprint1/int_to_string_perf_gate.v1.json}"
REQUIRE_JIT_RUNNER="${SOUNIO_SPRINT1_REQUIRE_JIT_RUNNER:-0}"
SOUC_JIT_VERSION="${SOUNIO_SOUC_JIT_VERSION:-0.100.3-jit.1}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$(dirname "$OUT_JSON")"

emit_json() {
  local status="$1"
  local reason="$2"
  local mode="$3"
  local runner="$4"
  local base_s="$5"
  local full_s="$6"
  local net_s="$7"
  local blockers_json="$8"
  python3 - "$OUT_JSON" "$status" "$reason" "$mode" "$runner" "$base_s" "$full_s" "$net_s" "$blockers_json" <<'PY'
import datetime as dt
import json
import sys

out_path, status, reason, mode, runner, base_s, full_s, net_s, blockers_json = sys.argv[1:10]

def parse_float(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None

try:
    blockers = json.loads(blockers_json)
    if not isinstance(blockers, list):
        blockers = [str(blockers)]
except Exception:
    blockers = [str(blockers_json)]

payload = {
    "schema": "sounio.sprint1.int_to_string_perf_gate.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": status,
    "reason": reason,
    "mode": mode,
    "runner": runner,
    "target_seconds": 1.0,
    "metrics": {
        "base_seconds": parse_float(base_s),
        "full_seconds": parse_float(full_s),
        "net_seconds": parse_float(net_s),
    },
    "blockers": blockers,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
print(f"wrote: {out_path}")
print(f"status={status} reason={reason}")
PY
}

if [[ ! -x /usr/bin/time ]]; then
  emit_json "not_run" "time_binary_unavailable" "none" "" "" "" "" "[\"/usr/bin/time_missing\"]"
  exit 0
fi

run_probe_mode() {
  local mode="$1"
  local bin="$2"
  set +e
  timeout 30 "$bin" "$mode" self-hosted/compiler/main.sio -- --bench-int-to-string 0 > "$TMP_DIR/probe.out" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$TMP_DIR/probe.rc"
}

declare -a candidates=()
if [[ -n "${SOUNIO_SOUC_JIT_BIN:-}" ]]; then
  candidates+=("${SOUNIO_SOUC_JIT_BIN}")
fi
if [[ -n "${SOUNIO_SOUC_BIN:-}" ]]; then
  candidates+=("${SOUNIO_SOUC_BIN}")
fi
candidates+=(
  "$ROOT_DIR/souc"
  "$ROOT_DIR/target/debug/souc"
  "$ROOT_DIR/target/release/souc"
  "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64"
  "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
)

resolver="$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh"
resolver_failed_note=""
if [[ -x "$resolver" ]]; then
  set +e
  resolver_candidate="$(
    SOUNIO_SOUC_VERSION="$SOUC_JIT_VERSION" \
    SOUNIO_SOUC_VARIANT="jit" \
    OMEGA_SOUC_REQUIRE_PINNED=1 \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
      "$resolver" --print-path 2>"$TMP_DIR/jit_resolver.log"
  )"
  resolver_rc=$?
  set -e
  if [[ "$resolver_rc" == "0" && -n "$resolver_candidate" ]]; then
    candidates=("$resolver_candidate" "${candidates[@]}")
  else
    resolver_failed_note="jit_resolver_failed:version=${SOUC_JIT_VERSION}"
  fi
fi

jit_runner=""
bench_mode="jit"
blockers=()
has_jit_capable_candidate=0

for c in "${candidates[@]}"; do
  if [[ ! -x "$c" ]]; then
    continue
  fi
  run_probe_mode "jit" "$c"
  probe_rc="$(cat "$TMP_DIR/probe.rc")"
  if [[ "$probe_rc" == "0" ]] && rg -F -q "bench_int_to_string iterations=0" "$TMP_DIR/probe.out"; then
    jit_runner="$c"
    break
  fi
  if rg -F -q "JIT backend not enabled" "$TMP_DIR/probe.out"; then
    blockers+=("jit_backend_not_enabled:$c")
  elif [[ "$probe_rc" == "124" ]]; then
    has_jit_capable_candidate=1
    blockers+=("jit_probe_timeout:$c")
  else
    has_jit_capable_candidate=1
    blockers+=("jit_probe_failed:$c")
  fi
done

if [[ -z "$jit_runner" ]]; then
  if [[ -n "$resolver_failed_note" ]]; then
    blockers+=("$resolver_failed_note")
  fi
  blockers_json="$(python3 - <<'PY' "${blockers[@]:-}"
import json
import sys
vals = [v for v in sys.argv[1:] if v]
if not vals:
    vals = ["jit_runner_unavailable"]
print(json.dumps(vals))
PY
)"
  if [[ "$REQUIRE_JIT_RUNNER" == "1" ]]; then
    if [[ "$has_jit_capable_candidate" == "1" ]]; then
      emit_json "fail" "jit_runner_unusable" "none" "" "" "" "" "$blockers_json"
    else
      emit_json "fail" "jit_runner_required_missing" "none" "" "" "" "" "$blockers_json"
    fi
    exit 0
  fi

  run_runner=""
  for c in "${candidates[@]}"; do
    if [[ ! -x "$c" ]]; then
      continue
    fi
    run_probe_mode "run" "$c"
    probe_rc="$(cat "$TMP_DIR/probe.rc")"
    if [[ "$probe_rc" == "0" ]] && rg -F -q "bench_int_to_string iterations=0" "$TMP_DIR/probe.out"; then
      run_runner="$c"
      break
    fi
    if [[ "$probe_rc" == "124" ]]; then
      blockers+=("run_probe_timeout:$c")
    else
      blockers+=("run_probe_failed:$c")
    fi
  done

  if [[ -z "$run_runner" ]]; then
    blockers_json="$(python3 - <<'PY' "${blockers[@]:-}"
import json
import sys
vals = [v for v in sys.argv[1:] if v]
if not vals:
    vals = ["runner_unavailable"]
print(json.dumps(vals))
PY
)"
    emit_json "not_run" "runner_unavailable" "none" "" "" "" "" "$blockers_json"
    exit 0
  else
    jit_runner="$run_runner"
    bench_mode="run"
  fi
fi

run_bench_mode() {
  local name="$1"
  local iterations="$2"
  set +e
  /usr/bin/time -f '%e' -o "$TMP_DIR/${name}.time" \
    timeout 180 "$jit_runner" "$bench_mode" self-hosted/compiler/main.sio -- --bench-int-to-string "$iterations" \
    > "$TMP_DIR/${name}.out" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$TMP_DIR/${name}.rc"
}

run_bench_mode "bench_base" 0
run_bench_mode "bench_full" 1000000

base_rc="$(cat "$TMP_DIR/bench_base.rc")"
full_rc="$(cat "$TMP_DIR/bench_full.rc")"

if [[ "$base_rc" != "0" || "$full_rc" != "0" ]]; then
  blockers_json="$(python3 - <<'PY' "$base_rc" "$full_rc"
import json
import sys
b, f = sys.argv[1:3]
vals = []
if b != "0":
    vals.append(f"bench_base_rc={b}")
if f != "0":
    vals.append(f"bench_full_rc={f}")
print(json.dumps(vals))
PY
)"
  emit_json "not_run" "${bench_mode}_benchmark_command_failed" "$bench_mode" "$jit_runner" "" "" "" "$blockers_json"
  exit 0
fi

if ! rg -F -q "bench_int_to_string iterations=0" "$TMP_DIR/bench_base.out"; then
  emit_json "not_run" "missing_base_output_marker" "$bench_mode" "$jit_runner" "" "" "" "[\"marker_missing:iterations=0\"]"
  exit 0
fi
if ! rg -F -q "bench_int_to_string iterations=1000000" "$TMP_DIR/bench_full.out"; then
  emit_json "not_run" "missing_full_output_marker" "$bench_mode" "$jit_runner" "" "" "" "[\"marker_missing:iterations=1000000\"]"
  exit 0
fi

base_s="$(cat "$TMP_DIR/bench_base.time")"
full_s="$(cat "$TMP_DIR/bench_full.time")"
net_s="$(python3 - "$base_s" "$full_s" <<'PY'
import sys
base = float(sys.argv[1])
full = float(sys.argv[2])
net = full - base
if net < 0:
    net = 0.0
print(f"{net:.6f}")
PY
)"

if python3 - "$net_s" <<'PY'
import sys
net = float(sys.argv[1])
raise SystemExit(0 if net < 1.0 else 1)
PY
then
  emit_json "pass" "target_met" "$bench_mode" "$jit_runner" "$base_s" "$full_s" "$net_s" "[]"
else
  emit_json "fail" "target_not_met" "$bench_mode" "$jit_runner" "$base_s" "$full_s" "$net_s" "[]"
fi
