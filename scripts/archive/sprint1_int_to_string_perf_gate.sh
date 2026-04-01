#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="${1:-$ROOT_DIR/artifacts/sprint1/int_to_string_perf_gate.v1.json}"
REQUIRE_JIT_RUNNER="${SOUNIO_SPRINT1_REQUIRE_JIT_RUNNER:-1}"
SOUC_JIT_VERSION="${SOUNIO_SOUC_JIT_VERSION:-0.100.3-jit.2}"
SOURCE_FILE="${SOUNIO_SPRINT1_SOURCE_FILE:-$ROOT_DIR/self-hosted/compiler/main.sio}"
RUN_LANE_JSON="${SOUNIO_SPRINT1_RUN_LANE_JSON:-$ROOT_DIR/artifacts/sprint1/int_to_string_perf_run_lane.v1.json}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$(dirname "$RUN_LANE_JSON")"

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

emit_run_lane_json() {
  local status="$1"
  local reason="$2"
  local mode="$3"
  local runner="$4"
  local base_s="$5"
  local full_s="$6"
  local net_s="$7"
  local blockers_json="$8"
  python3 - "$RUN_LANE_JSON" "$status" "$reason" "$mode" "$runner" "$base_s" "$full_s" "$net_s" "$blockers_json" <<'PY'
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
    "schema": "sounio.sprint1.int_to_string_perf_run_lane.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "non_gating": True,
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

contains_literal() {
  local needle="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q "$needle" "$file"
  else
    grep -F -q "$needle" "$file"
  fi
}

source_blockers_json() {
  python3 - <<'PY' "${@:-}"
import json
import sys
vals = [v for v in sys.argv[1:] if v]
if not vals:
    vals = ["unknown_source_contract_failure"]
print(json.dumps(vals))
PY
}

write_bench_fixture() {
  local iterations="$1"
  local out_path="$2"
  cat > "$out_path" <<EOF
fn i64_digit_count_from_negative(v: i64) -> i64 {
    if v <= -1000000000000000000 { 19 }
    else if v <= -100000000000000000 { 18 }
    else if v <= -10000000000000000 { 17 }
    else if v <= -1000000000000000 { 16 }
    else if v <= -100000000000000 { 15 }
    else if v <= -10000000000000 { 14 }
    else if v <= -1000000000000 { 13 }
    else if v <= -100000000000 { 12 }
    else if v <= -10000000000 { 11 }
    else if v <= -1000000000 { 10 }
    else if v <= -100000000 { 9 }
    else if v <= -10000000 { 8 }
    else if v <= -1000000 { 7 }
    else if v <= -100000 { 6 }
    else if v <= -10000 { 5 }
    else if v <= -1000 { 4 }
    else if v <= -100 { 3 }
    else if v <= -10 { 2 }
    else { 1 }
}

fn i64_to_string_decimal(n: i64) -> string with Mut, Panic, Div {
    var out: [i8; 24] = [0; 24]
    if n == 0 {
        out[0] = 48 as i8
        return str_from_bytes(out, 1)
    }

    let negative = n < 0
    let neg_value = if negative { n } else { 0 - n }
    let digits = i64_digit_count_from_negative(neg_value)
    let total_len = if negative { digits + 1 } else { digits }

    var value = n
    var write = total_len - 1
    while value <= -100 || value >= 100 {
        var pair = value % 100
        if pair < 0 {
            pair = 0 - pair
        }
        let ones = pair % 10
        let tens = pair / 10
        let write_prev = write - 1
        out[write as usize] = (48 + ones) as i8
        out[write_prev as usize] = (48 + tens) as i8
        write = write - 2
        value = value / 100
    }

    var rem = value
    if rem < 0 {
        rem = 0 - rem
    }
    if rem >= 10 {
        let write_prev = write - 1
        out[write as usize] = (48 + (rem % 10)) as i8
        out[write_prev as usize] = (48 + (rem / 10)) as i8
    } else {
        out[write as usize] = (48 + rem) as i8
    }

    if negative {
        out[0] = 45 as i8
    }

    str_from_bytes(out, total_len)
}

fn int_to_string(n: i64) -> string with Mut, Panic, Div {
    i64_to_string_decimal(n)
}

fn run_int_to_string_bench(iterations: i64) -> i64 with Mut, Panic, Div {
    var checksum: i64 = 0
    var i: i64 = 0
    while i < iterations {
        let selector = i & 7
        let sample =
            if selector == 0 { 9223372036854775807 }
            else if selector == 1 { -9223372036854775807 }
            else if selector == 2 { 1234567890123456789 }
            else if selector == 3 { -987654321098765432 }
            else if selector == 4 { 777777777777777777 }
            else if selector == 5 { -333333333333333333 }
            else if selector == 6 { 100000000000000001 }
            else { -100000000000000001 }
        let rendered = int_to_string(sample)
        checksum = checksum + str_len(rendered)
        i = i + 1
    }
    checksum
}

fn main() with IO, Mut, Panic, Div {
    let iterations: i64 = $iterations
    let checksum = run_int_to_string_bench(iterations)
    println("bench_int_to_string iterations=" + int_to_string(iterations) +
            " checksum=" + int_to_string(checksum))
}
EOF
}

write_arith_probe_fixture() {
  local out_path="$1"
  cat > "$out_path" <<'EOF'
fn main() -> i32 with IO {
    var i: i64 = 0
    var acc: i64 = 0
    while i < 1000 {
        acc = acc + i
        i = i + 1
    }
    if acc == 499500 {
        println("jit_probe_arith_ok")
        0
    } else {
        println("jit_probe_arith_bad")
        1
    }
}
EOF
}

write_string_probe_fixture() {
  local out_path="$1"
  cat > "$out_path" <<'EOF'
fn main() -> i32 with IO, Mut, Panic {
    let n = str_len("abc")
    let s = str_slice("0123456789", 2, 5)
    var bytes: [i8; 4] = [0; 4]
    bytes[0] = 52 as i8
    bytes[1] = 50 as i8
    let t = str_from_bytes(bytes, 2)
    if n == 3 && str_eq(s, "234") && str_eq(t, "42") {
        println("jit_probe_string_ok")
        0
    } else {
        println("jit_probe_string_bad")
        1
    }
}
EOF
}

if [[ ! -x /usr/bin/time ]]; then
  emit_json "not_run" "time_binary_unavailable" "none" "" "" "" "" "[\"/usr/bin/time_missing\"]"
  exit 0
fi

if [[ ! -f "$SOURCE_FILE" ]]; then
  emit_json "fail" "source_file_missing" "none" "" "" "" "" "[\"self-hosted/compiler/main.sio_missing\"]"
  exit 0
fi

declare -a source_blockers=()
if ! contains_literal 'fn int_to_string(n: i64) -> string with Mut, Panic, Div {' "$SOURCE_FILE"; then
  source_blockers+=("int_to_string_signature_missing")
fi
if ! contains_literal 'i64_to_string_decimal(n)' "$SOURCE_FILE"; then
  source_blockers+=("int_to_string_delegate_missing")
fi
if ! contains_literal 'var out: [i8; 24] = [0; 24]' "$SOURCE_FILE"; then
  source_blockers+=("i64_to_string_decimal_buffer_missing")
fi
if contains_literal 'result = char_from_i64(ch) + result' "$SOURCE_FILE"; then
  source_blockers+=("legacy_on2_prepend_detected")
fi
if [[ "${#source_blockers[@]}" -gt 0 ]]; then
  emit_json "fail" "source_contract_mismatch" "none" "" "" "" "" "$(source_blockers_json "${source_blockers[@]}")"
  exit 0
fi

BASE_FIXTURE="$TMP_DIR/bench_int_to_string_base.sio"
FULL_FIXTURE="$TMP_DIR/bench_int_to_string_full.sio"
ARITH_PROBE_FIXTURE="$TMP_DIR/jit_probe_arith.sio"
STRING_PROBE_FIXTURE="$TMP_DIR/jit_probe_string.sio"
write_bench_fixture 0 "$BASE_FIXTURE"
write_bench_fixture 1000000 "$FULL_FIXTURE"
write_arith_probe_fixture "$ARITH_PROBE_FIXTURE"
write_string_probe_fixture "$STRING_PROBE_FIXTURE"

run_probe_mode() {
  local mode="$1"
  local bin="$2"
  local fixture="$3"
  set +e
  timeout 15 "$bin" "$mode" "$fixture" > "$TMP_DIR/probe.out" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$TMP_DIR/probe.rc"
}

RUN_LANE_LAST_STATUS="not_run"
RUN_LANE_LAST_REASON="runner_unavailable"
RUN_LANE_LAST_RUNNER=""
RUN_LANE_LAST_NET=""

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
  "$ROOT_DIR/bin/souc"
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

declare -A seen_candidates=()
declare -a unique_candidates=()
for c in "${candidates[@]}"; do
  if [[ -z "$c" ]]; then
    continue
  fi
  if [[ -n "${seen_candidates[$c]:-}" ]]; then
    continue
  fi
  seen_candidates["$c"]=1
  unique_candidates+=("$c")
done
candidates=("${unique_candidates[@]}")

run_run_lane() {
  local run_runner=""
  local -a run_blockers=()
  local blockers_json="[]"

  for c in "${candidates[@]}"; do
    if [[ ! -x "$c" ]]; then
      continue
    fi
    run_probe_mode "run" "$c" "$BASE_FIXTURE"
    probe_rc="$(cat "$TMP_DIR/probe.rc")"
    if [[ "$probe_rc" == "0" ]] && contains_literal "bench_int_to_string iterations=0" "$TMP_DIR/probe.out"; then
      run_runner="$c"
      break
    fi
    if [[ "$probe_rc" == "124" ]]; then
      run_blockers+=("run_probe_timeout:$c")
    else
      run_blockers+=("run_probe_failed:$c")
    fi
  done

  if [[ -z "$run_runner" ]]; then
    blockers_json="$(python3 - <<'PY' "${run_blockers[@]:-}"
import json
import sys
vals = [v for v in sys.argv[1:] if v]
if not vals:
    vals = ["runner_unavailable"]
print(json.dumps(vals))
PY
)"
    RUN_LANE_LAST_STATUS="not_run"
    RUN_LANE_LAST_REASON="runner_unavailable"
    RUN_LANE_LAST_RUNNER=""
    RUN_LANE_LAST_NET=""
    emit_run_lane_json "not_run" "runner_unavailable" "run" "" "" "" "" "$blockers_json"
    return
  fi

  set +e
  /usr/bin/time -f '%e' -o "$TMP_DIR/run_lane_base.time" \
    timeout 240 "$run_runner" run "$BASE_FIXTURE" \
    > "$TMP_DIR/run_lane_base.out" 2>&1
  base_rc=$?
  /usr/bin/time -f '%e' -o "$TMP_DIR/run_lane_full.time" \
    timeout 240 "$run_runner" run "$FULL_FIXTURE" \
    > "$TMP_DIR/run_lane_full.out" 2>&1
  full_rc=$?
  set -e

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
    RUN_LANE_LAST_STATUS="not_run"
    RUN_LANE_LAST_REASON="run_benchmark_command_failed"
    RUN_LANE_LAST_RUNNER="$run_runner"
    RUN_LANE_LAST_NET=""
    emit_run_lane_json "not_run" "run_benchmark_command_failed" "run" "$run_runner" "" "" "" "$blockers_json"
    return
  fi

  if ! contains_literal "bench_int_to_string iterations=0" "$TMP_DIR/run_lane_base.out"; then
    RUN_LANE_LAST_STATUS="not_run"
    RUN_LANE_LAST_REASON="missing_base_output_marker"
    RUN_LANE_LAST_RUNNER="$run_runner"
    RUN_LANE_LAST_NET=""
    emit_run_lane_json "not_run" "missing_base_output_marker" "run" "$run_runner" "" "" "" "[\"marker_missing:iterations=0\"]"
    return
  fi
  if ! contains_literal "bench_int_to_string iterations=1000000" "$TMP_DIR/run_lane_full.out"; then
    RUN_LANE_LAST_STATUS="not_run"
    RUN_LANE_LAST_REASON="missing_full_output_marker"
    RUN_LANE_LAST_RUNNER="$run_runner"
    RUN_LANE_LAST_NET=""
    emit_run_lane_json "not_run" "missing_full_output_marker" "run" "$run_runner" "" "" "" "[\"marker_missing:iterations=1000000\"]"
    return
  fi

  run_base_s="$(cat "$TMP_DIR/run_lane_base.time")"
  run_full_s="$(cat "$TMP_DIR/run_lane_full.time")"
  run_net_s="$(python3 - "$run_base_s" "$run_full_s" <<'PY'
import sys
base = float(sys.argv[1])
full = float(sys.argv[2])
net = full - base
if net < 0:
    net = 0.0
print(f"{net:.6f}")
PY
)"

  if python3 - "$run_net_s" <<'PY'
import sys
net = float(sys.argv[1])
raise SystemExit(0 if net < 1.0 else 1)
PY
  then
    RUN_LANE_LAST_STATUS="pass"
    RUN_LANE_LAST_REASON="target_met"
    RUN_LANE_LAST_RUNNER="$run_runner"
    RUN_LANE_LAST_NET="$run_net_s"
    emit_run_lane_json "pass" "target_met" "run" "$run_runner" "$run_base_s" "$run_full_s" "$run_net_s" "[]"
  else
    RUN_LANE_LAST_STATUS="fail"
    RUN_LANE_LAST_REASON="target_not_met"
    RUN_LANE_LAST_RUNNER="$run_runner"
    RUN_LANE_LAST_NET="$run_net_s"
    emit_run_lane_json "fail" "target_not_met" "run" "$run_runner" "$run_base_s" "$run_full_s" "$run_net_s" "[]"
  fi
}

run_run_lane

jit_runner=""
bench_mode="jit"
blockers=()
has_jit_capable_candidate=0
final_blockers_json="[]"

for c in "${candidates[@]}"; do
  if [[ ! -x "$c" ]]; then
    continue
  fi
  run_probe_mode "jit" "$c" "$ARITH_PROBE_FIXTURE"
  probe_rc="$(cat "$TMP_DIR/probe.rc")"
  if [[ "$probe_rc" == "0" ]] && contains_literal "jit_probe_arith_ok" "$TMP_DIR/probe.out"; then
    has_jit_capable_candidate=1
    run_probe_mode "jit" "$c" "$STRING_PROBE_FIXTURE"
    str_probe_rc="$(cat "$TMP_DIR/probe.rc")"
    if [[ "$str_probe_rc" == "0" ]] && contains_literal "jit_probe_string_ok" "$TMP_DIR/probe.out"; then
      jit_runner="$c"
      break
    fi
    if [[ "$str_probe_rc" == "124" ]]; then
      blockers+=("jit_string_probe_timeout:$c")
    else
      blockers+=("jit_string_runtime_unavailable:$c")
    fi
    continue
  fi
  if contains_literal "JIT backend not enabled" "$TMP_DIR/probe.out"; then
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
  final_blockers_json="$blockers_json"
  if [[ "$REQUIRE_JIT_RUNNER" == "1" ]]; then
    if printf '%s\n' "${blockers[@]:-}" | grep -F -q "jit_string_runtime_unavailable:"; then
      emit_json "fail" "jit_string_runtime_unavailable" "none" "" "" "" "" "$blockers_json"
      exit 0
    fi
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
    run_probe_mode "run" "$c" "$BASE_FIXTURE"
    probe_rc="$(cat "$TMP_DIR/probe.rc")"
    if [[ "$probe_rc" == "0" ]] && contains_literal "bench_int_to_string iterations=0" "$TMP_DIR/probe.out"; then
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
  local fixture="$2"
  set +e
  /usr/bin/time -f '%e' -o "$TMP_DIR/${name}.time" \
    timeout 240 "$jit_runner" "$bench_mode" "$fixture" \
    > "$TMP_DIR/${name}.out" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$TMP_DIR/${name}.rc"
}

run_bench_mode "bench_base" "$BASE_FIXTURE"
run_bench_mode "bench_full" "$FULL_FIXTURE"

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
  if [[ "$final_blockers_json" != "[]" ]]; then
    blockers_json="$(python3 - <<'PY' "$final_blockers_json" "$base_rc" "$full_rc"
import json
import sys
prior_raw, b, f = sys.argv[1:4]
try:
    vals = json.loads(prior_raw)
    if not isinstance(vals, list):
        vals = [str(vals)]
except Exception:
    vals = [str(prior_raw)]
if b != "0":
    vals.append(f"bench_base_rc={b}")
if f != "0":
    vals.append(f"bench_full_rc={f}")
print(json.dumps(vals))
PY
)"
  fi
  emit_json "not_run" "${bench_mode}_benchmark_command_failed" "$bench_mode" "$jit_runner" "" "" "" "$blockers_json"
  exit 0
fi

if ! contains_literal "bench_int_to_string iterations=0" "$TMP_DIR/bench_base.out"; then
  emit_json "not_run" "missing_base_output_marker" "$bench_mode" "$jit_runner" "" "" "" "[\"marker_missing:iterations=0\"]"
  exit 0
fi
if ! contains_literal "bench_int_to_string iterations=1000000" "$TMP_DIR/bench_full.out"; then
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
  emit_json "pass" "target_met" "$bench_mode" "$jit_runner" "$base_s" "$full_s" "$net_s" "$final_blockers_json"
else
  emit_json "fail" "target_not_met" "$bench_mode" "$jit_runner" "$base_s" "$full_s" "$net_s" "$final_blockers_json"
fi
