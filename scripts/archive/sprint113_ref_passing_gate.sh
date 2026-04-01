#!/usr/bin/env bash
# Sprint 113: Reference-Passing Gate — &arr / (*ptr)[i] across function calls
set -eo pipefail

SOUC=./bin/souc
LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio
TMPDIR=/tmp/sprint113_gate
mkdir -p "$TMPDIR"

pass=0; fail=0; not_run=0

compile() {
    local src=$1 out=$2
    timeout 120 $SOUC run $LEAN -- "$src" "$out" 2>/dev/null
}

check() {
    local name=$1 elf=$2 expected=$3
    local actual
    actual=$("$elf" 2>/dev/null || true)
    if echo "$actual" | grep -qF "$expected"; then
        echo "PASS: $name"
        pass=$((pass+1))
    else
        echo "FAIL: $name  expected='$expected'"
        echo "  got: $(echo "$actual" | tail -3)"
        fail=$((fail+1))
    fi
}

_ec=0

# ── T1: fill + sum via pointer ────────────────────────────────────────────────
cat > "$TMPDIR/ptr_sum.sio" << 'EOF'
fn arr_fill(ptr: i64, n: i64) {
    var i: i64 = 0
    while i < n {
        (*ptr)[i] = i + 1
        i = i + 1
    }
}
fn arr_sum(ptr: i64, n: i64) -> i64 {
    var total: i64 = 0
    var i: i64 = 0
    while i < n {
        total = total + (*ptr)[i]
        i = i + 1
    }
    total
}
fn main() -> i64 {
    var arr: [i64; 20] = [0; 20]
    arr_fill(&arr, 10)
    let s = arr_sum(&arr, 10)
    print("ptr_sum=")
    print_int(s)
    print("\n")
    0
}
EOF
compile "$TMPDIR/ptr_sum.sio" "$TMPDIR/ptr_sum.elf" || _ec=$?
chmod +x "$TMPDIR/ptr_sum.elf" 2>/dev/null
check "ptr_fill_sum" "$TMPDIR/ptr_sum.elf" "ptr_sum=55"

# ── T2: binary search ─────────────────────────────────────────────────────────
cat > "$TMPDIR/bsearch.sio" << 'EOF'
fn binary_search(ptr: i64, n: i64, target: i64) -> i64 {
    var lo: i64 = 0
    var hi: i64 = n - 1
    var result: i64 = -1
    while lo <= hi {
        let mid = (lo + hi) / 2
        let val = (*ptr)[mid]
        if val == target {
            result = mid
            lo = hi + 1
        } else if val < target {
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    result
}
fn main() -> i64 {
    var arr: [i64; 10] = [0; 10]
    arr[0] = 2
    arr[1] = 5
    arr[2] = 8
    arr[3] = 12
    arr[4] = 16
    arr[5] = 23
    arr[6] = 38
    arr[7] = 56
    arr[8] = 72
    arr[9] = 91
    let i1 = binary_search(&arr, 10, 23)
    let i2 = binary_search(&arr, 10, 100)
    print("bsearch_23=")
    print_int(i1)
    print("\n")
    print("bsearch_100=")
    print_int(i2)
    print("\n")
    0
}
EOF
compile "$TMPDIR/bsearch.sio" "$TMPDIR/bsearch.elf" || _ec=$?
chmod +x "$TMPDIR/bsearch.elf" 2>/dev/null
check "binary_search" "$TMPDIR/bsearch.elf" "bsearch_23=5"

# ── T3: selection sort via pointer ────────────────────────────────────────────
cat > "$TMPDIR/sel_sort.sio" << 'EOF'
fn find_min_idx(ptr: i64, start: i64, n: i64) -> i64 {
    var min_idx: i64 = start
    var i: i64 = start + 1
    while i < n {
        if (*ptr)[i] < (*ptr)[min_idx] { min_idx = i }
        i = i + 1
    }
    min_idx
}
fn sel_sort(ptr: i64, n: i64) {
    var i: i64 = 0
    while i < n - 1 {
        let m = find_min_idx(ptr, i, n)
        if m != i {
            var tmp: i64 = (*ptr)[i]
            (*ptr)[i] = (*ptr)[m]
            (*ptr)[m] = tmp
        }
        i = i + 1
    }
}
fn main() -> i64 {
    var arr: [i64; 8] = [0; 8]
    arr[0] = 64
    arr[1] = 25
    arr[2] = 12
    arr[3] = 22
    arr[4] = 11
    arr[5] = 90
    arr[6] = 3
    arr[7] = 45
    sel_sort(&arr, 8)
    print("sorted_first=")
    print_int(arr[0])
    print("\n")
    print("sorted_last=")
    print_int(arr[7])
    print("\n")
    0
}
EOF
compile "$TMPDIR/sel_sort.sio" "$TMPDIR/sel_sort.elf" || _ec=$?
chmod +x "$TMPDIR/sel_sort.elf" 2>/dev/null
check "selection_sort" "$TMPDIR/sel_sort.elf" "sorted_first=3"

# ── T4: max-subarray (Kadane's algorithm) ────────────────────────────────────
cat > "$TMPDIR/kadane.sio" << 'EOF'
fn max_subarray(ptr: i64, n: i64) -> i64 {
    var max_so_far: i64 = (*ptr)[0]
    var max_ending_here: i64 = (*ptr)[0]
    var i: i64 = 1
    while i < n {
        let v = (*ptr)[i]
        if max_ending_here + v > v {
            max_ending_here = max_ending_here + v
        } else {
            max_ending_here = v
        }
        if max_ending_here > max_so_far {
            max_so_far = max_ending_here
        }
        i = i + 1
    }
    max_so_far
}
fn main() -> i64 {
    var arr: [i64; 9] = [0; 9]
    arr[0] = 0 - 2
    arr[1] = 1
    arr[2] = 0 - 3
    arr[3] = 4
    arr[4] = 0 - 1
    arr[5] = 2
    arr[6] = 1
    arr[7] = 0 - 5
    arr[8] = 4
    let result = max_subarray(&arr, 9)
    print("max_subarray=")
    print_int(result)
    print("\n")
    0
}
EOF
compile "$TMPDIR/kadane.sio" "$TMPDIR/kadane.elf" || _ec=$?
chmod +x "$TMPDIR/kadane.elf" 2>/dev/null
check "kadane" "$TMPDIR/kadane.elf" "max_subarray=6"

echo ""
echo "=== Sprint 113 Ref-Passing Gate: $pass PASS, $fail FAIL, $not_run NOT_RUN ==="
[ $fail -eq 0 ]
