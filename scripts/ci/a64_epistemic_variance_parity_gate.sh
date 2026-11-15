#!/usr/bin/env bash
# a64_epistemic_variance_parity_gate.sh — the arm64 lane of lean_single must
# propagate GUM variance the way the x86 lane does.
#
# Why this exists
# ---------------
# compile_all_arm64 mirrors the epistemic shadow lanes, and the mirror had
# drifted: + and − used the blind var(a)+var(b) (e+e read 2·var(e), x−x read
# 2·var(x)), / kept no variance or sensitivity at all (variance_of(e/100) read
# 0), sqrt/exp/ln/... carried the argument's variance through as if f' == 1,
# and a loop-carried accumulator correlated every iteration's fresh measurement
# with the previous one (619.444 for an analytic 169.444). Nothing noticed,
# because no gate ever RAN an aarch64 binary: a64_compile_fail_parity_gate
# compares refusals, and tests/selfhost/aarch64_compile only checks that an
# ARM ELF is produced.
#
# GATE_CONTRACT: v0
# GATE_ID: a64_epistemic_variance_parity
# GATE_CLAIMS: for every case in epistemic_variance_parity_cases.txt, the
#              aarch64-linux binary (run under qemu-aarch64) prints the same
#              stdout and exits with the same code as the x86-64 binary built
#              by the same lean_single from the same source
# GATE_ENGINE: lean_single (bin/souc-lean-single-x86_64, or SOUNIO_LEAN_SINGLE_BIN)
# GATE_RESULT_ON_SKIP: fail
#
# The list is positive only: a case enters when the arm64 lane reaches parity
# on it. Equality with x86 is the claim, not correctness; the x86 values of
# these cases are themselves checked against analytic oracles by the cases.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUNIO_LEAN_SINGLE_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
CASES="${CASES:-$ROOT_DIR/tests/selfhost/aarch64_compile/epistemic_variance_parity_cases.txt}"
QEMU="${QEMU_AARCH64:-$(command -v qemu-aarch64-static || command -v qemu-aarch64 || true)}"
TIMEOUT_SECS="${TIMEOUT_SECS:-120}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

echo "A64_EPISTEMIC_VARIANCE_PARITY_GATE_START"
echo "souc=$SOUC"
echo "cases=$CASES"
echo "qemu=$QEMU"

if [ ! -x "$SOUC" ]; then echo "FAIL: lean_single compiler not executable: $SOUC"; exit 1; fi
if [ -z "$QEMU" ]; then echo "FAIL: no qemu-aarch64 found (install qemu-user-static); this gate does not skip"; exit 1; fi
if [ ! -f "$CASES" ]; then echo "FAIL: case list missing: $CASES"; exit 1; fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-a64-var-parity.XXXXXX")"
trap 'rm -rf "${WORK_DIR:?}"' EXIT

run_case() {
    local src="$1" tag="$2" out="$3"
    local elf="$WORK_DIR/$tag.elf"
    local rc=0
    if [ "$tag" = "${tag%.a64}" ]; then
        ( ulimit -s unlimited; timeout "$TIMEOUT_SECS" "$SOUC" "$src" "$elf" ) > "$WORK_DIR/$tag.build.log" 2>&1 || true
    else
        ( ulimit -s unlimited; timeout "$TIMEOUT_SECS" "$SOUC" "$src" "$elf" --target aarch64-linux ) > "$WORK_DIR/$tag.build.log" 2>&1 || true
    fi
    if [ ! -s "$elf" ]; then
        echo "NOBUILD" > "$out"
        return
    fi
    chmod +x "$elf"
    if [ "$tag" = "${tag%.a64}" ]; then
        timeout "$TIMEOUT_SECS" "$elf" > "$out" 2>/dev/null || rc=$?
    else
        timeout "$TIMEOUT_SECS" "$QEMU" "$elf" > "$out" 2>/dev/null || rc=$?
    fi
    echo "rc=$rc" >> "$out"
}

total=0
fail=0
while IFS= read -r src; do
    case "$src" in ''|'#'*) continue ;; esac
    total=$((total + 1))
    if [ ! -f "$src" ]; then
        echo "FAIL $src: listed case does not exist"
        fail=$((fail + 1))
        continue
    fi
    tag="$(echo "$src" | tr '/' '_' | sed 's/\.sio$//')"
    run_case "$src" "$tag" "$WORK_DIR/$tag.x86.out"
    run_case "$src" "$tag.a64" "$WORK_DIR/$tag.a64.out"
    if grep -qx "NOBUILD" "$WORK_DIR/$tag.x86.out"; then
        echo "FAIL $src: x86 build failed"; sed -n "1,5p" "$WORK_DIR/$tag.build.log" || true
        fail=$((fail + 1))
    elif cmp -s "$WORK_DIR/$tag.x86.out" "$WORK_DIR/$tag.a64.out"; then
        echo "ok   $src"
    else
        echo "FAIL $src: aarch64 output differs from x86"
        { diff "$WORK_DIR/$tag.x86.out" "$WORK_DIR/$tag.a64.out" || true; } | sed -n "1,12p"
        fail=$((fail + 1))
    fi
done < "$CASES"

echo "cases=$total divergent=$fail"
if [ "$total" -eq 0 ]; then echo "FAIL: empty case list"; exit 1; fi
if [ "$fail" -ne 0 ]; then
    echo "A64_EPISTEMIC_VARIANCE_PARITY_GATE_FAIL"
    exit 1
fi
echo "A64_EPISTEMIC_VARIANCE_PARITY_GATE_PASS"
