#!/usr/bin/env bash
# Sub-PTX Step 0 gate — verify the K-AXI `round=rN` attribute lowers
# to the matching PTX rounding-mode suffix on f64 ops, AND lines
# without the attribute still emit un-suffixed PTX (backward compat).
#
# This gate is the prerequisite for:
#   * B1 (locked-rounding sedenion-mul → re-run S-SSM ZD harness)
#   * B2 (locked-rounding Sinkhorn-LSE → reproducible-across-centers
#         ORC on ABIDE-I)
#
# Acceptance:
#   1. souc check + compile of self-hosted/gpu/test_kaxi_round_mode.sio rc=0
#   2. compiled program emits 5 kernel stanzas (no_attr, rn, rz, rp, rm)
#   3. no_attr stanza contains "add.f64", "sub.f64", "mul.f64", and
#      fma-decomposed "mul.f64" + "add.f64" — and DOES NOT contain any
#      add.r[nzpm].f64 / sub.r[nzpm].f64 / mul.r[nzpm].f64 lines
#   4. rN stanza (N ∈ {n,z,p,m}) contains add.rN.f64, sub.rN.f64,
#      mul.rN.f64, and fma-decomposed mul.rN.f64 + add.rN.f64
#   5. The kaxi_ptx_golden_gate.sh continues to pass (re-run as
#      a backward-compat smoke).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
SRC="self-hosted/gpu/test_kaxi_round_mode.sio"
GOLDEN_GATE="scripts/ci/kaxi_ptx_golden_gate.sh"

PASS=0
FAIL=0
FIRST_FAILURES=()

note_fail() {
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("$1")
}

if [[ ! -x "$SOUC" ]]; then
    echo "FAIL: souc not executable at $SOUC" >&2
    exit 2
fi
if [[ ! -f "$SRC" ]]; then
    echo "FAIL: required file missing: $SRC" >&2
    exit 2
fi

# (1) souc check + compile
if "$SOUC" check "$SRC" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc check failed: $SRC"
fi

TMP_BIN="$(mktemp -t kaxi_round.XXXXXX)"
TMP_OUT="$(mktemp -t kaxi_round.XXXXXX.out)"
trap 'rm -f "$TMP_BIN" "$TMP_OUT"' EXIT

if "$SOUC" compile "$SRC" -o "$TMP_BIN" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "souc compile failed: $SRC"
fi

if [[ -x "$TMP_BIN" ]]; then
    if "$TMP_BIN" >"$TMP_OUT" 2>&1; then
        PASS=$((PASS + 1))
    else
        note_fail "compiled test exited non-zero"
    fi
else
    note_fail "compiled binary not executable"
fi

# (2) 5 KERNEL stanzas
stanza_count=$(grep -c '^---KERNEL ' "$TMP_OUT" 2>/dev/null || echo 0)
if [[ "$stanza_count" -eq 5 ]]; then
    PASS=$((PASS + 1))
else
    note_fail "expected 5 KERNEL stanzas, got $stanza_count"
fi

# Helper: extract one stanza into a tmp file
extract_stanza() {
    local name="$1"
    local out="$2"
    awk -v n="$name" '
        $0 == "---KERNEL " n { keep=1; next }
        $0 == "---KERNEL_END" { keep=0 }
        keep
    ' "$TMP_OUT" > "$out"
}

# (3) no_attr backward-compat: only un-suffixed .f64 ops
NO_ATTR="$(mktemp -t kaxi_round.XXXXXX.no_attr)"
trap 'rm -f "$TMP_BIN" "$TMP_OUT" "$NO_ATTR"' EXIT
extract_stanza no_attr "$NO_ATTR"
if grep -qE '^\s*add\.f64\s' "$NO_ATTR" && \
   grep -qE '^\s*sub\.f64\s' "$NO_ATTR" && \
   grep -qE '^\s*mul\.f64\s' "$NO_ATTR" && \
   ! grep -qE '^\s*(add|sub|mul)\.r[nzpm]\.f64\s' "$NO_ATTR"; then
    PASS=$((PASS + 1))
else
    note_fail "no_attr kernel missing un-suffixed .f64 ops or has unexpected suffixes"
fi

# (4) Each of rn/rz/rp/rm stanzas has matching suffix on all four ops
for mode in rn rz rp rm; do
    M="$(mktemp -t kaxi_round.XXXXXX.$mode)"
    extract_stanza "$mode" "$M"
    if grep -qE "^\s*add\.${mode}\.f64\s" "$M" && \
       grep -qE "^\s*sub\.${mode}\.f64\s" "$M" && \
       grep -qE "^\s*mul\.${mode}\.f64\s" "$M" && \
       [[ $(grep -cE "^\s*mul\.${mode}\.f64\s" "$M") -ge 2 ]] && \
       [[ $(grep -cE "^\s*add\.${mode}\.f64\s" "$M") -ge 2 ]]; then
        # the >=2 mul and >=2 add catch fma decomposition emitting
        # both mul.<mode>.f64 and add.<mode>.f64 alongside the standalone
        # mul/add/sub.
        PASS=$((PASS + 1))
    else
        note_fail "$mode kernel missing expected add.${mode}.f64 / sub.${mode}.f64 / mul.${mode}.f64 (incl. fma decomp)"
    fi
    rm -f "$M"
done

# (5) Backward-compat sanity: re-run the golden PTX gate.
if [[ -x "$GOLDEN_GATE" ]] || [[ -f "$GOLDEN_GATE" ]]; then
    if bash "$GOLDEN_GATE" >/dev/null 2>&1; then
        PASS=$((PASS + 1))
    else
        note_fail "kaxi_ptx_golden_gate.sh regressed (sub-PTX Step 0 broke byte-stability)"
    fi
else
    note_fail "kaxi_ptx_golden_gate.sh not found at $GOLDEN_GATE"
fi

echo "=== sub-PTX Step 0 round-mode gate ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [[ "${#FIRST_FAILURES[@]}" -gt 0 ]]; then
    echo "  first failures:"
    for f in "${FIRST_FAILURES[@]}"; do
        echo "    $f"
    done
fi

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
