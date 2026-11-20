#!/usr/bin/env bash
# Root C, seed loader (bootstrap/roots/c), cross-platform gate:
#   - Both platforms (Darwin arm64 and Linux x86_64):
#       * verifies SeedC.lean bytes match bootstrap/roots/c/seed-c
#       * builds seed_c_ref.cpp and Lean aarch64run (RunAArch64.lean)
#       * checks aarch64run self-reproduction, cross-assembly (seed-a, seed-b,
#         and spr1-check when present), fixed vectors, all 256 bytes, and 5000-case fuzz
#   - Darwin arm64:
#       * builds aarch64_jit_exec.cpp (MAP_JIT + svc #0 Linux syscall hook)
#       * checks native CPU execution of seed-c: self-reproduction, cross-assembly,
#         fixed vectors, all 256 bytes, and 5000-case differential fuzz
#   - Linux x86_64:
#       * runs native seed-a < seed-c.hex == seed-c and seed-b < seed-c.oct == seed-c
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
A="$ROOT/bootstrap/roots/a"
B="$ROOT/bootstrap/roots/b"
C="$ROOT/bootstrap/roots/c"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

FUZZ_N="${ROOT_C_FUZZ_N:-5000}"
FUZZ_SEED="${ROOT_C_FUZZ_SEED:-20261116}"
OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ -z "${CXX:-}" ]]; then
  if [[ "$OS" == "Darwin" && -x "/usr/bin/clang++" ]]; then
    CXX_BIN="/usr/bin/clang++"
  elif [[ "$OS" == "Darwin" ]]; then
    CXX_BIN="clang++"
  else
    CXX_BIN="g++"
  fi
else
  CXX_BIN="$CXX"
fi

if [[ -n "${LEAN_BIN:-}" ]]; then
  lean="$LEAN_BIN/lean"
  lake="$LEAN_BIN/lake"
elif command -v lean >/dev/null 2>&1; then
  lean="$(command -v lean)"
  lake="$(command -v lake || true)"
elif [[ -x "$HOME/.elan/bin/lean" ]]; then
  lean="$HOME/.elan/bin/lean"
  lake="$HOME/.elan/bin/lake"
else
  echo "ROOT_C_SEED_GATE=FAIL: lean not found" >&2
  exit 2
fi

echo "== root_c_seed_gate ($OS $ARCH) =="
fail=0

# 0. Verify SeedC.lean literal bytes match bootstrap/roots/c/seed-c
python3 - "$C/seed-c" "$ROOT/formal/Bootstrap/SeedC.lean" << 'PY' || fail=1
import re, sys
raw = list(open(sys.argv[1], "rb").read())
txt = open(sys.argv[2], "r").read()
m0 = re.search(r"def seedC_0 : List Nat := \[(.*?)\]", txt, re.S)
m1 = re.search(r"def seedC_1 : List Nat := \[(.*?)\]", txt, re.S)
if not m0 or not m1:
    print("FAIL: could not locate seedC_0/seedC_1 in SeedC.lean")
    sys.exit(1)
bs = [int(x.strip()) for x in (m0.group(1) + "," + m1.group(1)).split(",") if x.strip()]
if bs != raw:
    print(f"FAIL: SeedC.lean bytes ({len(bs)}) != seed-c ({len(raw)})")
    sys.exit(1)
print(f"ok   SeedC.lean matches seed-c ({len(raw)} bytes)")
PY

# 1. Build seed_c_ref and fuzz corpus
"$CXX_BIN" -std=c++23 -O2 -o "$OUT/ref" "$C/seed_c_ref.cpp"
mkdir -p "$OUT/fz"
"$OUT/ref" fuzz "$FUZZ_SEED" "$FUZZ_N" "$OUT/fz"
"$OUT/ref" all256 > "$OUT/all.bin"
"$OUT/ref" decode < "$OUT/all.bin" > "$OUT/all.want"

# 2. Build Lean runner (aarch64run)
if [[ -n "$lake" ]] && grep -q 'aarch64run' "$ROOT/formal/lakefile.lean" 2>/dev/null; then
  (cd "$ROOT/formal" && "$lake" build aarch64run >/dev/null)
  RUN_LEAN=("$ROOT/formal/.lake/build/bin/aarch64run")
else
  mkdir -p "$OUT/olean/Bootstrap" "$OUT/c"
  prefix="$("$lean" --print-prefix)"
  leanc="$prefix/bin/leanc"
  (
    cd "$ROOT/formal"
    LEAN_PATH="$OUT/olean" "$lean" "Bootstrap/AArch64Min.lean" \
      -o "$OUT/olean/Bootstrap/AArch64Min.olean" -c "$OUT/c/AArch64Min.c"
    LEAN_PATH="$OUT/olean" "$lean" "Bootstrap/RunAArch64.lean" \
      -o "$OUT/olean/Bootstrap/RunAArch64.olean" -c "$OUT/c/RunAArch64.c"
  )
  if [[ -x "$leanc" ]]; then
    "$leanc" -O2 -o "$OUT/aarch64run" "$OUT/c/AArch64Min.c" "$OUT/c/RunAArch64.c"
    RUN_LEAN=("$OUT/aarch64run")
  else
    RUN_LEAN=(env "LEAN_PATH=$OUT/olean" "$lean" --run "$ROOT/formal/Bootstrap/RunAArch64.lean")
  fi
fi

check_runner() {
  local tag="$1"
  shift
  local -a cmd=("$@")

  assemble_one() {
    local name="$1" listing="$2" expected="$3" rc=0
    "${cmd[@]}" "$C/seed-c" < "$listing" > "$OUT/$tag.$name.out" || rc=$?
    if [[ "$rc" -eq 0 ]] && cmp -s "$OUT/$tag.$name.out" "$expected"; then
      echo "ok   [$tag] $name ($(wc -c < "$expected") bytes)"
    else
      echo "FAIL [$tag] $name: rc=$rc"; fail=1
    fi
  }

  assemble_one self-reproduction "$C/seed-c.bin" "$C/seed-c"
  assemble_one cross-c-assembles-a "$C/seed-a.bin" "$A/seed-a"
  assemble_one cross-c-assembles-b "$C/seed-b.bin" "$B/seed-b"
  if [[ -f "$C/spr1-check.bin" && -f "$ROOT/bootstrap/model/spr1_check" ]]; then
    assemble_one cross-c-assembles-spr1 "$C/spr1-check.bin" "$ROOT/bootstrap/model/spr1_check"
  fi

  vec() {
    local name="$1" in="$2" want_rc="$3" want_out="$4" rc=0
    printf "$in" > "$OUT/v.in"; printf "$want_out" > "$OUT/v.want"
    "${cmd[@]}" "$C/seed-c" < "$OUT/v.in" > "$OUT/v.out" || rc=$?
    if [[ "$rc" -ne "$want_rc" ]]; then
      echo "FAIL [$tag] $name: rc=$rc, want $want_rc"; fail=1
    elif [[ "$want_rc" -eq 0 ]] && ! cmp -s "$OUT/v.out" "$OUT/v.want"; then
      echo "FAIL [$tag] $name: wrong bytes"; fail=1
    else
      echo "ok   [$tag] $name"
    fi
  }
  vec empty                ''                          0 ''
  vec byte                 '01000001'                  0 'A'
  vec bytes_ws             '01000001 01000010\t01000011\n' 0 'ABC'
  vec max_byte             '11111111'                  0 '\377'
  vec comment              ': x\n01000001 : y\n01000010' 0 'AB'
  vec comment_at_eof       '01000001 : no newline'     0 'A'
  vec comment_any_bytes    ':\001\377\r\n01111010'     0 'z'
  vec space_in_group       '0100 0001'                 1 ''
  vec colon_in_group       '0100:0001\n'               1 ''
  vec digit_2              '01000002'                  1 ''
  vec letter_a             '0100000a'                  1 ''
  vec cr_refused           '01000001\r\n'              1 ''
  vec incomplete_group     '010000010100'              1 ''
  vec hash_refused         '# not a comment'           1 ''
  vec semi_refused         '; not a comment'           1 ''
  vec nul_refused          '\000'                      1 ''

  local all_rc=0
  "${cmd[@]}" "$C/seed-c" < "$OUT/all.bin" > "$OUT/all.$tag.out" || all_rc=$?
  if [[ "$all_rc" -eq 0 ]] && [[ "$(wc -c < "$OUT/all.$tag.out")" -eq 256 ]] && cmp -s "$OUT/all.$tag.out" "$OUT/all.want"; then
    echo "ok   [$tag] all 256 byte values"
  else
    echo "FAIL [$tag] all 256 byte values: rc=$all_rc"; fail=1
  fi
}

# Run Lean AArch64Min checks (both platforms)
check_runner "aarch64run" "${RUN_LEAN[@]}"
"${RUN_LEAN[@]}" --batch "$C/seed-c" "$OUT/fz" "$FUZZ_N" || fail=1

# Platform-specific native execution checks
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  "$CXX_BIN" -std=c++23 -O2 -o "$OUT/jit" "$ROOT/bootstrap/model/aarch64_jit_exec.cpp"
  check_runner "jit" "$OUT/jit"
  "$OUT/jit" --batch "$C/seed-c" "$OUT/fz" "$FUZZ_N" || fail=1
elif [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
  cp "$A/seed-a" "$OUT/seed-a"; cp "$B/seed-b" "$OUT/seed-b"
  chmod +x "$OUT/seed-a" "$OUT/seed-b"
  rc_a=0; "$OUT/seed-a" < "$C/seed-c.hex" > "$OUT/a_from_hex" || rc_a=$?
  if [[ "$rc_a" -eq 0 ]] && cmp -s "$OUT/a_from_hex" "$C/seed-c"; then
    echo "ok   [linux-x86_64] seed-a < seed-c.hex == seed-c ($(wc -c < "$C/seed-c") bytes)"
  else
    echo "FAIL [linux-x86_64] seed-a < seed-c.hex: rc=$rc_a"; fail=1
  fi
  rc_b=0; "$OUT/seed-b" < "$C/seed-c.oct" > "$OUT/b_from_oct" || rc_b=$?
  if [[ "$rc_b" -eq 0 ]] && cmp -s "$OUT/b_from_oct" "$C/seed-c"; then
    echo "ok   [linux-x86_64] seed-b < seed-c.oct == seed-c ($(wc -c < "$C/seed-c") bytes)"
  else
    echo "FAIL [linux-x86_64] seed-b < seed-c.oct: rc=$rc_b"; fail=1
  fi
else
  echo "note: native runner skipped on $OS $ARCH (aarch64run checks already passed)"
fi

if [[ $fail -ne 0 ]]; then
  echo "ROOT_C_SEED_GATE=FAIL"
  exit 1
fi
echo "ROOT_C_SEED_GATE=PASS"
