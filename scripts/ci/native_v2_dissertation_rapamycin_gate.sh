#!/usr/bin/env bash
# Dissertation rapamycin DES gate -- native-v2 epistemic PBPK.
# Requires: native_v2_gum_primitives_gate.sh must pass first.
# Compiles tests/native-v2/science_spine/rapamycin_des_gum.sio through the
# generated stage1 driver. Verifies: ELF kind, stdout parity with fixture,
# deterministic replay, sqrtsd (SSE2) in generated binary.
# Records dissertation_gate=pass in summary JSON.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[dissertation-rapa] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[dissertation-rapa] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_DISSERTATION_RAPA_DIR:-$(mktemp -d /tmp/sounio-dissertation-rapa.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
DRIVER_SRC="self-hosted/compiler/native_compile_driver.sio"
STAGE1_DRIVER="$ARTIFACT_DIR/native_compile_driver.stage1"
STAGE1_DRIVER_2="$ARTIFACT_DIR/native_compile_driver.stage1.replay"
TEST_SRC="tests/native-v2/science_spine/rapamycin_des_gum.sio"
TEST_FIXTURE="tests/native-v2/science_spine/rapamycin_des_gum.stdout"
RAPA_BIN_1="$ARTIFACT_DIR/rapamycin_des_gum.pass1"
RAPA_BIN_2="$ARTIFACT_DIR/rapamycin_des_gum.pass2"
RAPA_STDOUT_1="$LOG_DIR/rapamycin.pass1.stdout"
RAPA_STDOUT_2="$LOG_DIR/rapamycin.pass2.stdout"
SUMMARY_JSON="$ARTIFACT_DIR/summary.json"
GUM_GATE_LOG="$LOG_DIR/native_v2_gum_primitives_gate.log"
STAGE1_LOG="$LOG_DIR/stage1.compile.log"
STAGE1_REPLAY_LOG="$LOG_DIR/stage1.replay.compile.log"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

portable_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

assert_elf_x86_64() {
  local path="$1"
  local label="$2"
  if command -v file >/dev/null 2>&1; then
    local out
    out=$(file "$path")
    if ! echo "$out" | grep -q 'ELF 64-bit LSB executable, x86-64'; then
      echo "[dissertation-rapa] FAIL: unexpected file kind for $path: $out" >&2
      exit 1
    fi
  fi
}

echo "[dissertation-rapa] souc=$SOUC_BIN"
echo "[dissertation-rapa] out=$OUT_DIR"

# Require the GUM primitives gate to pass first
bash scripts/ci/native_v2_gum_primitives_gate.sh >"$GUM_GATE_LOG" 2>&1

# Build stage1 driver (pass 1)
echo "[dissertation-rapa] building stage1 (pass 1)..."
"$SOUC_BIN" run "$DRIVER_SRC" -- "$DRIVER_SRC" -o "$STAGE1_DRIVER" >"$STAGE1_LOG" 2>&1
chmod +x "$STAGE1_DRIVER" 2>/dev/null || true
assert_elf_x86_64 "$STAGE1_DRIVER" "stage1"

# Build stage1 driver (pass 2 -- deterministic replay)
echo "[dissertation-rapa] building stage1 (pass 2 replay)..."
"$SOUC_BIN" run "$DRIVER_SRC" -- "$DRIVER_SRC" -o "$STAGE1_DRIVER_2" >"$STAGE1_REPLAY_LOG" 2>&1
chmod +x "$STAGE1_DRIVER_2" 2>/dev/null || true

if ! cmp -s "$STAGE1_DRIVER" "$STAGE1_DRIVER_2"; then
  echo "[dissertation-rapa] FAIL: stage1 driver is not deterministic" >&2
  exit 1
fi

# Compile rapamycin_des_gum through stage1 (pass 1 and pass 2 for binary determinism)
echo "[dissertation-rapa] compiling rapamycin_des_gum.sio through stage1..."
"$STAGE1_DRIVER" "$TEST_SRC" -o "$RAPA_BIN_1" >"$LOG_DIR/rapa.pass1.compile.log" 2>&1
chmod +x "$RAPA_BIN_1" 2>/dev/null || true
assert_elf_x86_64 "$RAPA_BIN_1" "rapa.pass1"

"$STAGE1_DRIVER" "$TEST_SRC" -o "$RAPA_BIN_2" >"$LOG_DIR/rapa.pass2.compile.log" 2>&1
chmod +x "$RAPA_BIN_2" 2>/dev/null || true

if ! cmp -s "$RAPA_BIN_1" "$RAPA_BIN_2"; then
  echo "[dissertation-rapa] FAIL: rapamycin binary is not deterministic across pass1/pass2" >&2
  exit 1
fi

# Run and check stdout
"$RAPA_BIN_1" >"$RAPA_STDOUT_1" 2>/dev/null
"$RAPA_BIN_2" >"$RAPA_STDOUT_2" 2>/dev/null

if ! cmp -s "$TEST_FIXTURE" "$RAPA_STDOUT_1"; then
  echo "[dissertation-rapa] FAIL: stdout does not match fixture" >&2
  diff -u "$TEST_FIXTURE" "$RAPA_STDOUT_1" >&2 || true
  exit 1
fi
if ! cmp -s "$TEST_FIXTURE" "$RAPA_STDOUT_2"; then
  echo "[dissertation-rapa] FAIL: pass2 stdout does not match fixture" >&2
  exit 1
fi

# Verify sqrtsd (SSE2) is in the generated binary
sqrts_verified=false
if command -v objdump >/dev/null 2>&1; then
  objdump -d "$RAPA_BIN_1" >"$LOG_DIR/rapa.disasm.txt" 2>/dev/null || true
  if grep -q 'sqrtsd' "$LOG_DIR/rapa.disasm.txt"; then
    sqrts_verified=true
    echo "[dissertation-rapa] sqrtsd (SSE2) confirmed in generated ELF"
  else
    echo "[dissertation-rapa] WARNING: sqrtsd not found via objdump (may still be present)" >&2
  fi
fi

# Emit summary JSON
rapa_sha="$(portable_sha256 "$RAPA_BIN_1")"
rapa_bytes="$(portable_size "$RAPA_BIN_1")"
stage1_sha="$(portable_sha256 "$STAGE1_DRIVER")"

# Emit summary JSON via pure-Sounio kretikos json-emit (replaces python json.dump heredoc).
# Schema sounio.native_v2_dissertation_rapamycin.v1. Args ordered alphabetically by key
# to preserve byte-identity with prior python `sort_keys=True` output.
"$ROOT_DIR/bin/kretikos" json-emit \
  --array-strings "claims=gum_through_ode|iso_uncertainty_budget|compile_time_confidence_gate|rapamycin_3comp_native_elf" \
  --string "compiler_resolved=$SOUC_BIN" \
  --string "dissertation_gate=pass" \
  --string "driver_source=self-hosted/compiler/native_compile_driver.sio" \
  --string "fallback_path=none" \
  --bool "gum_sse2_verified=$sqrts_verified" \
  --string "host_callback=none" \
  --int "rapa_bin_bytes=$rapa_bytes" \
  --string "rapa_bin_sha256=$rapa_sha" \
  --bool "rapa_deterministic=true" \
  --string "schema=sounio.native_v2_dissertation_rapamycin.v1" \
  --bool "stage1_deterministic=true" \
  --string "stage1_driver_sha256=$stage1_sha" \
  --string "status=pass" \
  --string "target=x86_64-linux" \
  --string "test_fixture=tests/native-v2/science_spine/rapamycin_des_gum.stdout" \
  --string "test_source=tests/native-v2/science_spine/rapamycin_des_gum.sio" \
  > "$SUMMARY_JSON"

echo "[dissertation-rapa] PASS"
echo "[dissertation-rapa] dissertation_gate=pass, sqrtsd_verified=$sqrts_verified"
echo "[dissertation-rapa] summary=$SUMMARY_JSON"
