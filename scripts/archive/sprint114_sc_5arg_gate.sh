#!/usr/bin/env bash
# Sprint 114: Short-circuit &&/|| + 5-argument function calls
# Tests: sc_and, sc_or, merge (5-arg), merge_sort (multi-fn)
set -eo pipefail

SOUC=./bin/souc
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

PASS=0; FAIL=0; NOT_RUN=0

check() {
    local name="$1" src="$2" key="$3" expected="$4"
    local out elf rc=0
    elf=$(mktemp /tmp/lean114_XXXXXX.elf)
    out=$($SOUC run "$LEAN" -- "$src" "$elf" 2>&1 && chmod +x "$elf" && "$elf" 2>&1) || rc=$?
    rm -f "$elf"
    if [ $rc -ne 0 ]; then NOT_RUN=$((NOT_RUN+1)); echo "NOT_RUN $name (exit $rc)"; return; fi
    local val; val=$(echo "$out" | grep "^${key}=" | head -1 | cut -d= -f2)
    if [ "$val" = "$expected" ]; then
        PASS=$((PASS+1)); echo "PASS $name: $key=$val"
    else
        FAIL=$((FAIL+1)); echo "FAIL $name: $key expected=$expected got=$val"
    fi
}

# Short-circuit && tests
cat > /tmp/s114_sc.sio << 'EOF'
fn main() -> i64 {
    var a: i64 = 5
    var b: i64 = 0
    var c: i64 = 3
    let r1 = if a > 0 && c > 0 { 1 } else { 0 }
    let r2 = if b > 0 && c > 0 { 1 } else { 0 }
    let r3 = if b > 0 || c > 0 { 1 } else { 0 }
    let r4 = if b > 0 || b > 0 { 1 } else { 0 }
    print("and_tt=")
    print_int(r1)
    print("\n")
    print("and_ft=")
    print_int(r2)
    print("\n")
    print("or_ft=")
    print_int(r3)
    print("\n")
    print("or_ff=")
    print_int(r4)
    print("\n")
    0
}
EOF
check "sc_and_tt" /tmp/s114_sc.sio "and_tt" "1"
check "sc_and_ft" /tmp/s114_sc.sio "and_ft" "0"
check "sc_or_ft"  /tmp/s114_sc.sio "or_ft"  "1"
check "sc_or_ff"  /tmp/s114_sc.sio "or_ff"  "0"

# 5-argument function: simple merge
cat > /tmp/s114_merge5.sio << 'EOF'
fn merge(dst: i64, src: i64, lo: i64, mid: i64, hi: i64) {
    var i: i64 = lo
    var j: i64 = mid
    var k: i64 = lo
    while i < mid && j < hi {
        if (*src)[i] <= (*src)[j] {
            (*dst)[k] = (*src)[i]
            i = i + 1
        } else {
            (*dst)[k] = (*src)[j]
            j = j + 1
        }
        k = k + 1
    }
    while i < mid {
        (*dst)[k] = (*src)[i]
        i = i + 1
        k = k + 1
    }
    while j < hi {
        (*dst)[k] = (*src)[j]
        j = j + 1
        k = k + 1
    }
}
fn main() -> i64 {
    var src: [i64; 8] = [0; 8]
    var dst: [i64; 8] = [0; 8]
    src[0] = 1
    src[1] = 3
    src[2] = 5
    src[3] = 2
    src[4] = 4
    src[5] = 6
    merge(&dst, &src, 0, 3, 6)
    print("m0=")
    print_int(dst[0])
    print("\n")
    print("m2=")
    print_int(dst[2])
    print("\n")
    print("m5=")
    print_int(dst[5])
    print("\n")
    0
}
EOF
check "merge5_m0" /tmp/s114_merge5.sio "m0" "1"
check "merge5_m2" /tmp/s114_merge5.sio "m2" "3"
check "merge5_m5" /tmp/s114_merge5.sio "m5" "6"

# Full merge sort (16 elements)
cat > /tmp/s114_msort.sio << 'EOF'
fn merge(dst: i64, src: i64, lo: i64, mid: i64, hi: i64) {
    var i: i64 = lo
    var j: i64 = mid
    var k: i64 = lo
    while i < mid && j < hi {
        if (*src)[i] <= (*src)[j] {
            (*dst)[k] = (*src)[i]
            i = i + 1
        } else {
            (*dst)[k] = (*src)[j]
            j = j + 1
        }
        k = k + 1
    }
    while i < mid {
        (*dst)[k] = (*src)[i]
        i = i + 1
        k = k + 1
    }
    while j < hi {
        (*dst)[k] = (*src)[j]
        j = j + 1
        k = k + 1
    }
}
fn copy_range(dst: i64, src: i64, lo: i64, hi: i64) {
    var i: i64 = lo
    while i < hi {
        (*dst)[i] = (*src)[i]
        i = i + 1
    }
}
fn merge_sort(arr: i64, tmp: i64, n: i64) {
    var width: i64 = 1
    while width < n {
        var lo: i64 = 0
        while lo < n {
            let mid = lo + width
            var hi: i64 = lo + width + width
            if mid > n { hi = lo } else { if hi > n { hi = n } }
            if mid < n {
                merge(tmp, arr, lo, mid, hi)
                copy_range(arr, tmp, lo, hi)
            }
            lo = lo + width + width
        }
        width = width + width
    }
}
fn main() -> i64 {
    var arr: [i64; 16] = [0; 16]
    var tmp: [i64; 16] = [0; 16]
    arr[0] = 38
    arr[1] = 27
    arr[2] = 43
    arr[3] = 3
    arr[4] = 9
    arr[5] = 82
    arr[6] = 10
    arr[7] = 1
    arr[8] = 55
    arr[9] = 17
    arr[10] = 66
    arr[11] = 4
    arr[12] = 50
    arr[13] = 22
    arr[14] = 33
    arr[15] = 7
    merge_sort(&arr, &tmp, 16)
    print("ms_min=")
    print_int(arr[0])
    print("\n")
    print("ms_max=")
    print_int(arr[15])
    print("\n")
    print("ms_4=")
    print_int(arr[3])
    print("\n")
    0
}
EOF
check "msort_min" /tmp/s114_msort.sio "ms_min" "1"
check "msort_max" /tmp/s114_msort.sio "ms_max" "82"
check "msort_4"   /tmp/s114_msort.sio "ms_4"   "7"

echo ""
echo "Sprint 114: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN total=$((PASS+FAIL+NOT_RUN))"
[ $FAIL -eq 0 ]
