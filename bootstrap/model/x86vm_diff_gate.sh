#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FORMAL_DIR="$ROOT/formal"
CPU_SRC="$ROOT/bootstrap/model/x86vm_diff_cpu.cpp"
MODEL_LEAN="$FORMAL_DIR/Bootstrap/X86VmDiffModel.lean"
LAKE_BIN="${LAKE_BIN:-$(command -v lake || true)}"
if [ -z "$LAKE_BIN" ] && [ -x "$HOME/.elan/bin/lake" ]; then
  LAKE_BIN="$HOME/.elan/bin/lake"
fi

CASES="${DIFF_CASES:-66000}"
SEED="${DIFF_SEED:-0x583836564d3236}"

arch="$(uname -m)"
os="$(uname -s)"
if [ "$os" != "Linux" ] || [ "$arch" != "x86_64" ]; then
  echo "SKIP/FAIL: x86vm_diff_gate.sh requires Linux x86_64 hardware (got $os $arch)" >&2
  exit 1
fi

if [ -z "$LAKE_BIN" ]; then
  echo "FAIL: lake binary not found in PATH or \$HOME/.elan/bin" >&2
  exit 1
fi

work="$(mktemp -d /tmp/sounio-x86vm-diff.XXXXXX)"
cleanup() {
  rm -rf "$work"
}
trap cleanup EXIT

cpu_bin="$work/x86vm_diff_cpu"
cases_file="$work/cases.txt"
sweep_in="$work/sweep_in.txt"
sweep_out="$work/sweep_out.txt"
neg_file="$work/neg_cases.txt"
build_log="$work/lake_build.log"

CXX="${CXX:-g++}"
"$CXX" -O2 -std=c++23 -Wall -Wextra -Werror "$CPU_SRC" -o "$cpu_bin"

VM_DIFF_BIN="$FORMAL_DIR/.lake/build/bin/x86vm-diff"
ensure_model_ready() {
  if [ -x "$VM_DIFF_BIN" ] && [ "${X86VM_FORCE_LEAN_RUN:-0}" != "1" ] &&
     [ "$VM_DIFF_BIN" -nt "$FORMAL_DIR/Bootstrap/X86Vm.lean" ] &&
     [ "$VM_DIFF_BIN" -nt "$MODEL_LEAN" ]; then
    return 0
  fi
  if ! (cd "$FORMAL_DIR" && "$LAKE_BIN" build Bootstrap.X86Dec >"$build_log" 2>&1); then
    cat "$build_log" >&2
    exit 1
  fi
  vm_olean="$FORMAL_DIR/.lake/build/lib/lean/Bootstrap/X86Vm.olean"
  if [ ! -f "$vm_olean" ] || [ "$FORMAL_DIR/Bootstrap/X86Vm.lean" -nt "$vm_olean" ]; then
    mkdir -p "$(dirname "$vm_olean")"
    (cd "$FORMAL_DIR" && "$LAKE_BIN" env lean -R . -o "$vm_olean" "Bootstrap/X86Vm.lean")
  fi
}

run_model() {
  if [ -x "$VM_DIFF_BIN" ] && [ "${X86VM_FORCE_LEAN_RUN:-0}" != "1" ] &&
     [ "$VM_DIFF_BIN" -nt "$FORMAL_DIR/Bootstrap/X86Vm.lean" ] &&
     [ "$VM_DIFF_BIN" -nt "$MODEL_LEAN" ]; then
    "$VM_DIFF_BIN" "$@"
  else
    (cd "$FORMAL_DIR" && "$LAKE_BIN" env lean --run "Bootstrap/X86VmDiffModel.lean" "$@")
  fi
}

ensure_model_ready

# 1. Randomized differential suite across all 31 X86Vm + 33 X86Dec forms.
"$cpu_bin" "$cases_file" "$CASES" "$SEED"
MIN_VM=1000
if [ "$CASES" -lt 40000 ]; then
  MIN_VM=0
fi
run_model "$cases_file" "$CASES" "$MIN_VM"

# 2. Single-site Hamming-1 fetch disturbance sweep (`X86Vm.altsAt`).
if [ "${X86VM_SKIP_SWEEP:-0}" != "1" ]; then
  run_model --emit-sweep "$sweep_in"
  sweep_n="$(wc -l < "$sweep_in" | tr -d ' ')"
  if [ "$sweep_n" -eq 0 ]; then
    echo "FAIL: Hamming-1 disturbance sweep emitted 0 executable cases" >&2
    exit 1
  fi
  "$cpu_bin" --exec "$sweep_in" "$sweep_out"
  run_model "$sweep_out" "$sweep_n" 0
  echo "HAMMING1_SWEEP_PASS: verified $sweep_n single-bit fetch disturbances with 0 disagreements"
fi

# 3. Built-in negative control (pure POSIX awk):
#    corrupt 1 nibble of post-RAX on the first normal retirement line and verify rejection.
if [ "${X86VM_SKIP_NEG_CONTROL:-0}" != "1" ]; then
  head -n 64 "$cases_file" | awk '
    BEGIN { mutated = 0 }
    {
      if (!mutated && NF == 40 && $21 == "|") {
        s = $23
        last = substr(s, length(s), 1)
        rep = (last == "0") ? "1" : "0"
        $23 = substr(s, 1, length(s) - 1) rep
        mutated = 1
      }
      print
    }
    END { if (!mutated) exit 2 }
  ' > "$neg_file"
  if run_model "$neg_file" 1 0 >"$work/neg.out" 2>"$work/neg.err"; then
    echo "FAIL: negative control unexpectedly PASSED!" >&2
    exit 1
  fi
  echo "NEG_CONTROL_PASS: corrupted hardware trace rejected cleanly"
fi
