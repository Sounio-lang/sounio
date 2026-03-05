#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_JSON="${1:-$ROOT_DIR/artifacts/sprint1/jit_runtime_debug.v1.json}"
SOURCE_FILE="${SOUNIO_SPRINT1_SOURCE_FILE:-$ROOT_DIR/self-hosted/compiler/main.sio}"
SOUC_JIT_VERSION="${SOUNIO_SOUC_JIT_VERSION:-0.100.3-jit.1}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$(dirname "$OUT_JSON")"

contains_literal() {
  local needle="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q "$needle" "$file"
  else
    grep -F -q "$needle" "$file"
  fi
}

write_arith_probe_fixture() {
  local out_path="$1"
  cat > "$out_path" <<'PROBE'
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
PROBE
}

write_string_probe_fixture() {
  local out_path="$1"
  cat > "$out_path" <<'PROBE'
fn main() -> i32 with IO, Mut, Panic {
    let n = str_len("abc")
    let s = str_slice("0123456789", 2, 5)
    var bytes: [i8; 4] = [0; 4]
    bytes[0] = 52 as i8
    bytes[1] = 50 as i8
    let t = str_from_bytes(bytes, 2)
    if n == 3 && s == "234" && t == "42" {
        println("jit_probe_string_ok")
        0
    } else {
        println("jit_probe_string_bad")
        1
    }
}
PROBE
}

run_probe() {
  local candidate="$1"
  local probe="$2"
  local timeout_s="$3"
  local marker="$4"
  shift 4

  local out_file="$TMP_DIR/${probe}.out"
  set +e
  timeout "${timeout_s}s" "$@" > "$out_file" 2>&1
  local rc=$?
  set -e

  local status="fail"
  local reason="rc_nonzero"

  if contains_literal "JIT backend not enabled" "$out_file"; then
    status="not_jit"
    reason="jit_backend_not_enabled"
  elif contains_literal "overflowed its stack" "$out_file"; then
    status="fail"
    reason="stack_overflow"
  elif [[ "$rc" == "124" ]]; then
    status="fail"
    reason="timeout"
  elif [[ "$rc" == "139" || "$rc" == "134" ]] || contains_literal "Segmentation fault" "$out_file" || contains_literal "dumped core" "$out_file"; then
    status="fail"
    reason="segfault"
  elif [[ "$rc" == "0" ]]; then
    if [[ -z "$marker" ]] || contains_literal "$marker" "$out_file"; then
      status="pass"
      reason="ok"
    else
      status="fail"
      reason="missing_marker"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "$candidate" "$probe" "$status" "$reason" "$rc" >> "$TMP_DIR/results.tsv"
}

ARITH_PROBE_FIXTURE="$TMP_DIR/jit_probe_arith.sio"
STRING_PROBE_FIXTURE="$TMP_DIR/jit_probe_string.sio"
write_arith_probe_fixture "$ARITH_PROBE_FIXTURE"
write_string_probe_fixture "$STRING_PROBE_FIXTURE"

: > "$TMP_DIR/results.tsv"

declare -a candidates=()
if [[ -n "${SOUNIO_SOUC_JIT_BIN:-}" ]]; then
  candidates+=("${SOUNIO_SOUC_JIT_BIN}")
fi
if [[ -n "${SOUNIO_SOUC_BIN:-}" ]]; then
  candidates+=("${SOUNIO_SOUC_BIN}")
fi

candidates+=(
  "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
  "$ROOT_DIR/souc"
  "$ROOT_DIR/target/debug/souc"
  "$ROOT_DIR/target/release/souc"
  "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64"
  "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
)

resolver="$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh"
if [[ -x "$resolver" ]]; then
  set +e
  resolver_candidate="$(
    SOUNIO_SOUC_VERSION="$SOUC_JIT_VERSION" \
    SOUNIO_SOUC_VARIANT="jit" \
    OMEGA_SOUC_REQUIRE_PINNED=1 \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
      "$resolver" --print-path 2>/dev/null
  )"
  resolver_rc=$?
  set -e
  if [[ "$resolver_rc" == "0" && -n "$resolver_candidate" ]]; then
    candidates=("$resolver_candidate" "${candidates[@]}")
  fi
fi

declare -A seen_candidates=()
declare -a unique_candidates=()
for c in "${candidates[@]}"; do
  [[ -n "$c" ]] || continue
  [[ -x "$c" ]] || continue
  if [[ -n "${seen_candidates[$c]:-}" ]]; then
    continue
  fi
  seen_candidates["$c"]=1
  unique_candidates+=("$c")
done

for c in "${unique_candidates[@]}"; do
  run_probe "$c" "arith" 20 "jit_probe_arith_ok" "$c" jit "$ARITH_PROBE_FIXTURE"
  run_probe "$c" "string" 20 "jit_probe_string_ok" "$c" jit "$STRING_PROBE_FIXTURE"
  if [[ -f "$SOURCE_FILE" ]]; then
    run_probe "$c" "source" 30 "bench_int_to_string iterations=0" "$c" jit "$SOURCE_FILE" -- --bench-int-to-string 0
  else
    printf '%s\t%s\t%s\t%s\t%s\n' "$c" "source" "fail" "source_file_missing" "-1" >> "$TMP_DIR/results.tsv"
  fi
done

python3 - "$TMP_DIR/results.tsv" "$OUT_JSON" <<'PY'
import datetime as dt
import json
import sys
from collections import defaultdict

results_path, out_path = sys.argv[1:3]
rows = []
for raw in open(results_path, "r", encoding="utf-8"):
    raw = raw.rstrip("\n")
    if not raw:
        continue
    candidate, probe, status, reason, rc = raw.split("\t", 4)
    rows.append(
        {
            "candidate": candidate,
            "probe": probe,
            "status": status,
            "reason": reason,
            "rc": int(rc),
        }
    )

by_candidate = defaultdict(dict)
for row in rows:
    by_candidate[row["candidate"]][row["probe"]] = {
        "status": row["status"],
        "reason": row["reason"],
        "rc": row["rc"],
    }

candidates = []
for candidate, probes in by_candidate.items():
    arith = probes.get("arith", {}).get("status")
    string = probes.get("string", {}).get("status")
    source = probes.get("source", {}).get("status")
    usable = arith == "pass" and string == "pass" and source == "pass"
    candidates.append(
        {
            "path": candidate,
            "probes": probes,
            "jit_usable_for_sprint1": usable,
        }
    )

blockers = sorted({
    f"{row['probe']}:{row['reason']}:{row['candidate']}"
    for row in rows
    if row["status"] != "pass"
})

payload = {
    "schema": "sounio.sprint1.jit_runtime_debug.v1",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "summary": {
        "candidate_count": len(candidates),
        "usable_candidate_count": sum(1 for c in candidates if c["jit_usable_for_sprint1"]),
        "jit_runtime_blocked": not any(c["jit_usable_for_sprint1"] for c in candidates),
    },
    "candidates": candidates,
    "blockers": blockers,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

print(f"wrote: {out_path}")
print(f"jit_runtime_blocked={payload['summary']['jit_runtime_blocked']}")
PY
