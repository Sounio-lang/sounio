#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-gum] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-gum] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_GUM_DIR:-$(mktemp -d /tmp/sounio-native-v2-gum.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
PASS1_DIR="$ARTIFACT_DIR/pass1"
PASS2_DIR="$ARTIFACT_DIR/pass2"
MANIFEST_PATH="${SOUNIO_NATIVE_V2_GUM_MANIFEST:-tests/native-v2/gum_primitives/manifest.tsv}"
DRIVER_SRC="self-hosted/compiler/native_compile_driver.sio"
STAGE1_DRIVER="$ARTIFACT_DIR/native_compile_driver.stage1"
STAGE1_DRIVER_2="$ARTIFACT_DIR/native_compile_driver.stage1.replay"
RESULTS_TSV="$ARTIFACT_DIR/results.tsv"
SUMMARY_JSON="$ARTIFACT_DIR/summary.json"
SELF_GATE_LOG="$LOG_DIR/native_v2_driver_self_compile_gate.log"
STAGE1_LOG="$LOG_DIR/stage1.compile.log"
STAGE1_REPLAY_LOG="$LOG_DIR/stage1.replay.compile.log"

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
      echo "[native-v2-gum] FAIL: unexpected file kind for $path" >&2
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
  local case_id="$1" program="$2" claim_class="$3" expected_exit="$4"
  local pass1_exit="$5" pass2_exit="$6" stdout_status="$7" status="$8"
  local pass1_sha="$9" pass2_sha="${10}" bytes="${11}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$program" "$claim_class" "$expected_exit" "$pass1_exit" "$pass2_exit" \
    "$stdout_status" "$status" "$pass1_sha" "$pass2_sha" "$bytes" >>"$RESULTS_TSV"
}

echo "[native-v2-gum] souc=$SOUC_BIN"
echo "[native-v2-gum] out=$OUT_DIR"
echo "[native-v2-gum] manifest=$MANIFEST_PATH"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[native-v2-gum] FAIL: missing manifest $MANIFEST_PATH" >&2
  exit 1
fi
# Manifest row floor: the case loop appends exactly one results row per
# non-comment data row. An empty manifest runs zero cases, case_count reads
# 0, and 0 == 0 computes status "pass" -- an instrument that answered
# nothing certifying itself.
gum_manifest_rows="$(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST_PATH" | grep -c . || true)"
if [[ "$gum_manifest_rows" -lt 1 ]]; then
  echo "[native-v2-gum] FAIL: manifest $MANIFEST_PATH has no data rows -- nothing to verify" >&2
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
  echo "[native-v2-gum] FAIL: stage1 driver replay is not byte-identical" >&2
  exit 1
fi

append_result_header

while IFS=$'\t' read -r case_id program_path expected_exit expected_stdout claim_class; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi

  if [[ ! -f "$program_path" ]]; then
    echo "[native-v2-gum] FAIL: missing program for $case_id: $program_path" >&2
    exit 1
  fi
  if [[ "$expected_stdout" != "-" && ! -f "$expected_stdout" ]]; then
    echo "[native-v2-gum] FAIL: missing stdout fixture for $case_id: $expected_stdout" >&2
    exit 1
  fi

  pass1_bin="$PASS1_DIR/$case_id"
  pass2_bin="$PASS2_DIR/$case_id"
  pass1_compile_log="$LOG_DIR/$case_id.pass1.compile.log"
  pass2_compile_log="$LOG_DIR/$case_id.pass2.compile.log"
  pass1_stdout="$LOG_DIR/$case_id.pass1.stdout"
  pass2_stdout="$LOG_DIR/$case_id.pass2.stdout"

  if ! "$STAGE1_DRIVER" "$program_path" -o "$pass1_bin" >"$pass1_compile_log" 2>&1; then
    echo "[native-v2-gum] FAIL: pass1 compile failed for $case_id" >&2
    tail -n 80 "$pass1_compile_log" >&2 || true
    exit 1
  fi
  chmod +x "$pass1_bin" 2>/dev/null || true
  assert_elf_x86_64 "$pass1_bin" "$case_id.pass1"

  if ! "$STAGE1_DRIVER" "$program_path" -o "$pass2_bin" >"$pass2_compile_log" 2>&1; then
    echo "[native-v2-gum] FAIL: pass2 compile failed for $case_id" >&2
    tail -n 80 "$pass2_compile_log" >&2 || true
    exit 1
  fi
  chmod +x "$pass2_bin" 2>/dev/null || true
  assert_elf_x86_64 "$pass2_bin" "$case_id.pass2"

  set +e
  "$pass1_bin" >"$pass1_stdout" 2>/dev/null
  pass1_exit=$?
  "$pass2_bin" >"$pass2_stdout" 2>/dev/null
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
    echo "[native-v2-gum] FAIL: $case_id status=$status" >&2
    if [[ "$stdout_status" != "ok" && "$expected_stdout" != "-" ]]; then
      diff -u "$expected_stdout" "$pass1_stdout" >&2 || true
    fi
    exit 1
  fi

  echo "[native-v2-gum] ok: $case_id ($claim_class)"
done <"$MANIFEST_PATH"

# SSE2 sqrtsd proof: verify the generated ELF uses sqrtsd instruction
GUM_SSE2_VERIFIED=false
SQRTSD_LOG="$LOG_DIR/sqrtsd_scan.log"
if command -v objdump >/dev/null 2>&1; then
  sqrt_bin="$PASS1_DIR/sqrt_f64_smoke"
  if [[ -f "$sqrt_bin" ]]; then
    objdump -d "$sqrt_bin" >"$SQRTSD_LOG" 2>&1 || true
    if grep -q 'sqrtsd' "$SQRTSD_LOG"; then
      GUM_SSE2_VERIFIED=true
      echo "[native-v2-gum] sqrtsd confirmed in generated ELF (ISO GUM SSE2 proof)"
    else
      echo "[native-v2-gum] NOTE: sqrtsd not confirmed by objdump (objdump may not be available or format differs)" >&2
    fi
  fi
else
  echo "[native-v2-gum] NOTE: objdump not available -- sqrtsd scan skipped"
fi

# Pure-bash summary JSON emitter (replaces python3 csv.DictReader + json.dump heredoc).
# TSV header: case_id($1) program($2) claim_class($3) expected_exit($4) pass1_exit($5)
#   pass2_exit($6) stdout($7) status($8) pass1_sha256($9) pass2_sha256($10) bytes($11)
# Keys sorted alphabetically per sort_keys=True.
MANIFEST_SHA256="$(sha256sum "$MANIFEST_PATH" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$MANIFEST_PATH" | awk '{print $1}')"
STAGE1_SHA256="$(sha256sum "$STAGE1_DRIVER" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$STAGE1_DRIVER" | awk '{print $1}')"
STAGE1_2_SHA256="$(sha256sum "$STAGE1_DRIVER_2" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$STAGE1_DRIVER_2" | awk '{print $1}')"
if [[ "$STAGE1_SHA256" == "$STAGE1_2_SHA256" ]]; then
  stage1_deterministic=true
else
  stage1_deterministic=false
fi
gum_sse2_bool=false
[[ "${GUM_SSE2_VERIFIED,,}" == "true" ]] && gum_sse2_bool=true
case_count="$(awk 'NR>1 && NF>0' "$RESULTS_TSV" | wc -l | tr -d ' ')"
# Every manifest data row must have answered exactly once; a shortfall is
# lost rows, not a smaller corpus (0 == 0 would otherwise read "pass").
if [[ "$case_count" -ne "$gum_manifest_rows" ]]; then
  echo "[native-v2-gum] FAIL: $case_count of $gum_manifest_rows manifest cases produced results -- incomplete pass" >&2
  exit 1
fi
pass_count_gum="$(awk -F'\t' 'NR>1 && $8=="ok"' "$RESULTS_TSV" | wc -l | tr -d ' ')"
fail_count_gum=$(( case_count - pass_count_gum ))
if [[ "$pass_count_gum" -eq "$case_count" ]]; then
  status_str="pass"
else
  status_str="fail"
fi

CASES_JSON="$(awk -F'\t' '
function jsesc(s,    out, i, c) {
  out = ""
  for (i = 1; i <= length(s); i++) {
    c = substr(s, i, 1)
    if (c == "\\") out = out "\\\\"
    else if (c == "\"") out = out "\\\""
    else if (c == "\n") out = out "\\n"
    else if (c == "\r") out = out "\\r"
    else if (c == "\t") out = out "\\t"
    else out = out c
  }
  return out
}
NR==1 { next }
NR>1 && NF>=8 {
  if (printed) printf ",\n"
  printed = 1
  printf "    {\n"
  printf "      \"bytes\": \"%s\",\n",          jsesc($11)
  printf "      \"case_id\": \"%s\",\n",        jsesc($1)
  printf "      \"claim_class\": \"%s\",\n",    jsesc($3)
  printf "      \"expected_exit\": \"%s\",\n",  jsesc($4)
  printf "      \"pass1_exit\": \"%s\",\n",     jsesc($5)
  printf "      \"pass1_sha256\": \"%s\",\n",   jsesc($9)
  printf "      \"pass2_exit\": \"%s\",\n",     jsesc($6)
  printf "      \"pass2_sha256\": \"%s\",\n",   jsesc($10)
  printf "      \"program\": \"%s\",\n",        jsesc($2)
  printf "      \"status\": \"%s\",\n",         jsesc($8)
  printf "      \"stdout\": \"%s\"\n",          jsesc($7)
  printf "    }"
}
END { if (printed) printf "\n" }
' "$RESULTS_TSV")"
CASES_JSON="[
$CASES_JSON
  ]"

"$ROOT_DIR/bin/kretikos" json-emit \
  --string  "artifact_dir=$OUT_DIR" \
  --int     "case_count=$case_count" \
  --raw-json "cases=$CASES_JSON" \
  --string  "compiler_resolved=$SOUC_BIN" \
  --string  "driver_source=self-hosted/compiler/native_compile_driver.sio" \
  --int     "fail_count=$fail_count_gum" \
  --string  "fallback_path=none" \
  --bool    "gum_sse2_verified=$gum_sse2_bool" \
  --string  "host_callback=none" \
  --string  "manifest=$MANIFEST_PATH" \
  --string  "manifest_sha256=$MANIFEST_SHA256" \
  --int     "pass_count=$pass_count_gum" \
  --string  "results_tsv=$RESULTS_TSV" \
  --string  "schema=sounio.native_v2_gum_primitives.v1" \
  --string  "sota_claim=ISO JCGM 100:2008 GUM uncertainty propagation as native x86-64 SSE2 IR primitive" \
  --bool    "stage1_driver_deterministic=$stage1_deterministic" \
  --string  "stage1_driver_sha256=$STAGE1_SHA256" \
  --string  "status=$status_str" \
  --string  "target=x86_64-linux" \
  > "$SUMMARY_JSON"

if [[ "$status_str" != "pass" || "$stage1_deterministic" != "true" ]]; then
  exit 1
fi

echo "[native-v2-gum] PASS: sqrt_f64 SSE2, ISO GUM addition, GUM multiplication, epistemic PBPK confidence gate, deterministic replay"
echo "[native-v2-gum] summary=$SUMMARY_JSON"
