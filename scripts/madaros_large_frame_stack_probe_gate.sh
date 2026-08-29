#!/usr/bin/env bash
# Madaros native gate — large-frame stack probe (page-by-page sub+touch).
#
# Acceptance:
#   - Default ./bin/souc (Madaros) compile+run of the witness
#   - Sentinel LARGE_FRAME_STACK_PROBE_OK
#   - Optional: unsplit oct_mul full body (informational; #1274 split remains)
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-./bin/souc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_large_frame_stack_probe_gate =="
engine_line="$($SOUC --version 2>&1 | head -1 || true)"
echo "engine: $engine_line"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single"
  exit 1
fi

RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$(pwd)/artifacts/self-hosted/madaros" "$(pwd)/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null || true)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
if [[ -z "$RAW" ]]; then
  echo "MADAROS_LARGE_FRAME_STACK_PROBE_GATE_BLOCKED reason=no_raw_madaros" >&2
  exit 1
fi
echo "raw_elf=$RAW"
echo "raw_elf_sha256=$(sha256sum "$RAW" | awk '{print $1}')"
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

SRC=tests/madaros/large_frame_stack_probe/large_frame_witness.sio

echo "== compile $SRC =="
if ! $SOUC compile "$SRC" -o "$OUT/large.elf" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: native compile"
  tail -40 "$OUT/compile.log" || true
  fail=1
else
  echo "PASS: compile"
  # Report largest sub rsp imm in the ELF (diagnostic)
  python3 - "$OUT/large.elf" <<'PY' || true
import struct, sys
d=open(sys.argv[1],'rb').read()
frames=[]
for i in range(len(d)-7):
    if d[i:i+3]==bytes([0x48,0x81,0xec]):
        frames.append(struct.unpack_from('<I',d,i+3)[0])
# also detect page-probe sequence (mov r11, imm)
probes=0
for i in range(len(d)-7):
    if d[i:i+3]==bytes([0x49,0xc7,0xc3]):
        probes+=1
print(f"sub_rsp_imm32_frames={[hex(f) for f in frames[:8]]} max={hex(max(frames)) if frames else None}")
print(f"mov_r11_imm_sites={probes}")
PY
  echo "== run =="
  set +e
  "$OUT/large.elf" >"$OUT/run.out" 2>"$OUT/run.err"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL: run rc=$rc (SEGV=139 is the historical large-frame mode)"
    cat "$OUT/run.err" 2>/dev/null || true
    fail=1
  else
    if ! grep -q 'LARGE_FRAME_STACK_PROBE_OK' "$OUT/run.out"; then
      echo "FAIL: missing sentinel"
      cat "$OUT/run.out" || true
      fail=1
    else
      echo "PASS: run + sentinel"
      head -5 "$OUT/run.out" || true
    fi
  fi
fi

if [[ $fail -ne 0 ]]; then
  echo "MADAROS_LARGE_FRAME_STACK_PROBE_GATE_FAIL"
  exit 1
fi
echo "MADAROS_LARGE_FRAME_STACK_PROBE_GATE_OK"
exit 0
