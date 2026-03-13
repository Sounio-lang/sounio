#!/usr/bin/env bash
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PKG=self-hosted/interop/package.sio
PASS=0 FAIL=0

run_test() {
    local name="$1"; shift
    if "$@" > /dev/null 2>&1; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name"; FAIL=$((FAIL+1))
    fi
}

grep_test() {
    local name="$1" pattern="$2"
    if grep -q "$pattern" "$PKG"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 122+123 Package Gate ==="

# T1: souc check passes
run_test "T1_souc_check" $SOUC check "$PKG"

# T2-T8: struct/constant/function existence
grep_test "T2_SnpkgHeader_struct"     "struct SnpkgHeader"
grep_test "T3_SnpkgEntry_struct"      "struct SnpkgEntry"
grep_test "T4_SnpkgBuffer_struct"     "struct SnpkgBuffer"
grep_test "T5_SNPKG_MAGIC_0"         "SNPKG_MAGIC_0"
grep_test "T6_snpkg_write_header"    "fn snpkg_write_header"
grep_test "T7_snpkg_verify"          "fn snpkg_verify("
grep_test "T8_snpkg_verify_magic"    "fn snpkg_verify_magic"

echo ""
echo "PASS=$PASS FAIL=$FAIL TOTAL=$((PASS+FAIL))"
exit $FAIL
