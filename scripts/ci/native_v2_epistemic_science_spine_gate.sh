#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-science-spine] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-science-spine] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR:-$(mktemp -d /tmp/sounio-native-v2-science-spine.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
PASS1_DIR="$ARTIFACT_DIR/pass1"
PASS2_DIR="$ARTIFACT_DIR/pass2"
MANIFEST_PATH="${SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST:-tests/native-v2/science_spine/manifest.tsv}"
DRIVER_SRC="self-hosted/compiler/native_compile_driver.sio"
STAGE1_DRIVER="$ARTIFACT_DIR/native_compile_driver.stage1"
STAGE1_DRIVER_2="$ARTIFACT_DIR/native_compile_driver.stage1.replay"
RESULTS_TSV="$ARTIFACT_DIR/results.tsv"
SUMMARY_JSON="$ARTIFACT_DIR/summary.json"
SELF_GATE_LOG="$LOG_DIR/native_v2_driver_self_compile_gate.log"
STAGE1_LOG="$LOG_DIR/stage1.compile.log"
STAGE1_REPLAY_LOG="$LOG_DIR/stage1.replay.compile.log"
STAGE1_FILE_LOG="$LOG_DIR/stage1.file.txt"

mkdir -p "$LOG_DIR" "$PASS1_DIR" "$PASS2_DIR"

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

portable_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

assert_elf_x86_64() {
  local path="$1"
  local label="$2"
  local file_log="$LOG_DIR/${label}.file.txt"

  if command -v file >/dev/null 2>&1; then
    file "$path" >"$file_log"
    if ! grep -q 'ELF 64-bit LSB executable, x86-64' "$file_log"; then
      echo "[native-v2-science-spine] FAIL: unexpected file kind for $path" >&2
      cat "$file_log" >&2
      exit 1
    fi
  fi
}

append_result_header() {
  cat >"$RESULTS_TSV" <<'EOF'
case_id	program	claim_class	expected_exit	pass1_exit	pass2_exit	stdout	status	pass1_sha256	pass2_sha256	bytes
EOF
}

append_result_row() {
  local case_id="$1"
  local program="$2"
  local claim_class="$3"
  local expected_exit="$4"
  local pass1_exit="$5"
  local pass2_exit="$6"
  local stdout_status="$7"
  local status="$8"
  local pass1_sha="$9"
  local pass2_sha="${10}"
  local bytes="${11}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$program" "$claim_class" "$expected_exit" "$pass1_exit" "$pass2_exit" \
    "$stdout_status" "$status" "$pass1_sha" "$pass2_sha" "$bytes" >>"$RESULTS_TSV"
}

emit_summary_json() {
  python3 - "$SUMMARY_JSON" "$RESULTS_TSV" "$SOUC_BIN" "$MANIFEST_PATH" "$STAGE1_DRIVER" "$STAGE1_DRIVER_2" "$OUT_DIR" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])
souc_bin = sys.argv[3]
manifest_path = pathlib.Path(sys.argv[4])
stage1_path = pathlib.Path(sys.argv[5])
stage1_replay_path = pathlib.Path(sys.argv[6])
out_dir = pathlib.Path(sys.argv[7])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
with open(results_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

payload = {
    "schema": "sounio.native_v2_epistemic_science_spine.v1",
    "status": "pass" if all(r["status"] == "ok" for r in rows) else "fail",
    "compiler_resolved": souc_bin,
    "target": "x86_64-linux",
    "fallback_path": "none",
    "host_callback": "none",
    "driver_source": "self-hosted/compiler/native_compile_driver.sio",
    "manifest": str(manifest_path),
    "manifest_sha256": sha256(manifest_path),
    "stage1_driver": str(stage1_path),
    "stage1_driver_sha256": sha256(stage1_path),
    "stage1_driver_replay_sha256": sha256(stage1_replay_path),
    "stage1_driver_deterministic": sha256(stage1_path) == sha256(stage1_replay_path),
    "case_count": len(rows),
    "pass_count": sum(1 for r in rows if r["status"] == "ok"),
    "fail_count": sum(1 for r in rows if r["status"] != "ok"),
    "results_tsv": str(results_path),
    "artifact_dir": str(out_dir),
    "cases": rows,
}

summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] != "pass" or not payload["stage1_driver_deterministic"]:
    raise SystemExit(1)
PY
}

echo "[native-v2-science-spine] souc=$SOUC_BIN"
echo "[native-v2-science-spine] out=$OUT_DIR"
echo "[native-v2-science-spine] manifest=$MANIFEST_PATH"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[native-v2-science-spine] FAIL: missing manifest $MANIFEST_PATH" >&2
  exit 1
fi

bash scripts/ci/native_v2_driver_self_compile_gate.sh >"$SELF_GATE_LOG" 2>&1

"$SOUC_BIN" run "$DRIVER_SRC" -- "$DRIVER_SRC" -o "$STAGE1_DRIVER" >"$STAGE1_LOG" 2>&1
chmod +x "$STAGE1_DRIVER" 2>/dev/null || true
assert_elf_x86_64 "$STAGE1_DRIVER" "stage1"

"$SOUC_BIN" run "$DRIVER_SRC" -- "$DRIVER_SRC" -o "$STAGE1_DRIVER_2" >"$STAGE1_REPLAY_LOG" 2>&1
chmod +x "$STAGE1_DRIVER_2" 2>/dev/null || true
assert_elf_x86_64 "$STAGE1_DRIVER_2" "stage1.replay"

if ! cmp -s "$STAGE1_DRIVER" "$STAGE1_DRIVER_2"; then
  echo "[native-v2-science-spine] FAIL: stage1 driver replay is not byte-identical" >&2
  exit 1
fi

append_result_header

while IFS=$'\t' read -r case_id program_path expected_exit expected_stdout claim_class; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi

  if [[ ! -f "$program_path" ]]; then
    echo "[native-v2-science-spine] FAIL: missing program for $case_id: $program_path" >&2
    exit 1
  fi
  if [[ "$expected_stdout" != "-" && ! -f "$expected_stdout" ]]; then
    echo "[native-v2-science-spine] FAIL: missing stdout fixture for $case_id: $expected_stdout" >&2
    exit 1
  fi

  pass1_bin="$PASS1_DIR/$case_id"
  pass2_bin="$PASS2_DIR/$case_id"
  pass1_compile_log="$LOG_DIR/$case_id.pass1.compile.log"
  pass2_compile_log="$LOG_DIR/$case_id.pass2.compile.log"
  pass1_stdout="$LOG_DIR/$case_id.pass1.stdout"
  pass2_stdout="$LOG_DIR/$case_id.pass2.stdout"
  pass1_stderr="$LOG_DIR/$case_id.pass1.stderr"
  pass2_stderr="$LOG_DIR/$case_id.pass2.stderr"

  if ! "$STAGE1_DRIVER" "$program_path" -o "$pass1_bin" >"$pass1_compile_log" 2>&1; then
    echo "[native-v2-science-spine] FAIL: pass1 compile failed for $case_id" >&2
    tail -n 80 "$pass1_compile_log" >&2 || true
    exit 1
  fi
  chmod +x "$pass1_bin" 2>/dev/null || true
  assert_elf_x86_64 "$pass1_bin" "$case_id.pass1"

  if ! "$STAGE1_DRIVER" "$program_path" -o "$pass2_bin" >"$pass2_compile_log" 2>&1; then
    echo "[native-v2-science-spine] FAIL: pass2 compile failed for $case_id" >&2
    tail -n 80 "$pass2_compile_log" >&2 || true
    exit 1
  fi
  chmod +x "$pass2_bin" 2>/dev/null || true
  assert_elf_x86_64 "$pass2_bin" "$case_id.pass2"

  set +e
  "$pass1_bin" >"$pass1_stdout" 2>"$pass1_stderr"
  pass1_exit=$?
  "$pass2_bin" >"$pass2_stdout" 2>"$pass2_stderr"
  pass2_exit=$?
  set -e

  stdout_status="mismatch"
  if [[ "$expected_stdout" == "-" ]]; then
    if [[ ! -s "$pass1_stdout" && ! -s "$pass2_stdout" ]]; then
      stdout_status="ok"
    fi
  elif cmp -s "$expected_stdout" "$pass1_stdout" && cmp -s "$expected_stdout" "$pass2_stdout"; then
    stdout_status="ok"
  fi

  pass1_sha="$(portable_sha256 "$pass1_bin")"
  pass2_sha="$(portable_sha256 "$pass2_bin")"
  byte_count="$(portable_size "$pass1_bin")"
  status="ok"

  if [[ "$pass1_exit" != "$expected_exit" || "$pass2_exit" != "$expected_exit" ]]; then
    status="exit_mismatch"
  elif [[ "$stdout_status" != "ok" ]]; then
    status="stdout_mismatch"
  elif ! cmp -s "$pass1_bin" "$pass2_bin"; then
    status="binary_mismatch"
  fi

  append_result_row \
    "$case_id" "$program_path" "$claim_class" "$expected_exit" "$pass1_exit" "$pass2_exit" \
    "$stdout_status" "$status" "$pass1_sha" "$pass2_sha" "$byte_count"

  if [[ "$status" != "ok" ]]; then
    echo "[native-v2-science-spine] FAIL: $case_id status=$status" >&2
    if [[ "$stdout_status" != "ok" && "$expected_stdout" != "-" ]]; then
      diff -u "$expected_stdout" "$pass1_stdout" >&2 || true
      diff -u "$expected_stdout" "$pass2_stdout" >&2 || true
    fi
    exit 1
  fi
done <"$MANIFEST_PATH"

if command -v rg >/dev/null 2>&1; then
  if rg -i 'fallback|host_callback|rust' "$LOG_DIR" >/dev/null; then
    echo "[native-v2-science-spine] FAIL: fallback/host/Rust marker appeared in logs" >&2
    rg -i 'fallback|host_callback|rust' "$LOG_DIR" >&2 || true
    exit 1
  fi
fi

file "$STAGE1_DRIVER" >"$STAGE1_FILE_LOG" 2>/dev/null || true
emit_summary_json

echo "[native-v2-science-spine] PASS: stage1 self-compile, corpus ELF/runtime parity, deterministic replay, and summary evidence"
echo "[native-v2-science-spine] summary=$SUMMARY_JSON"
