#!/usr/bin/env bash
# CI gate for rung R20: provenance binding.
#   Z1_AUDIT_REPRODUCES  Z2_EXECUTOR_SURFACE  Z3_BEHAVIOUR_RECEIPT
# Compile arm: SFCL_R20_RUN_COMPILE=1, needs artifacts/self-hosted/madaros-provenance.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r20_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r20_2026-07-28.md"
EXECUTOR="self-hosted/compiler/claim_executor.sio"
FIX="scripts/ci/fixtures"
PV_ELF="$REPO_ROOT/artifacts/self-hosted/madaros-provenance"
RECEIPT="$REPO_ROOT/artifacts/self_falsifying_r20_receipt.txt"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R20_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"; [[ -f "$CONTRACT" ]] || fail "contract missing"

if [[ "${SFCL_R20_RUN_COMPILE:-0}" == "1" ]]; then
    RAW="${MADAROS_RAW_BIN:-$PV_ELF}"
    [[ -x "$RAW" ]] || fail "compile arm: no provenance-binding compiler at $RAW"
    ulimit -s unlimited 2>/dev/null || true
    export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$REPO_ROOT/stdlib}"
    T="$(mktemp -d "${TMPDIR:-/tmp}/sfcl_r20.XXXXXX")"; trap 'rm -rf "$T"' EXIT
    probe(){ local n="$1" s="$2" rcw="$3" mk="$4" ew="$5"; rm -f "$T/$n.elf"
        set +e; timeout 300 "$RAW" build --verify-claims "$s" -o "$T/$n.elf" >"$T/$n.log" 2>&1; local rc=$?; set -e
        [[ "$rcw" == zero && $rc -ne 0 ]] && { cat "$T/$n.log"; fail "$n: wanted rc 0, got $rc"; }
        [[ "$rcw" == nonzero && $rc -eq 0 ]] && { cat "$T/$n.log"; fail "$n: wanted non-zero rc"; }
        grep -q "$mk" "$T/$n.log" || { cat "$T/$n.log"; fail "$n: no '$mk'"; }
        [[ "$ew" == yes && ! -f "$T/$n.elf" ]] && fail "$n: expected an ELF"
        [[ "$ew" == no && -f "$T/$n.elf" ]] && fail "$n: an ELF was emitted and must not have been"
        echo "  [OK] $n rc=$rc elf=$ew $mk"; }
    probe Z_PRESENT_PASSES  "$FIX/self_falsifying_provenance_present.sio" zero    CLAIM_PASS                 yes
    probe Z_MISSING_BLOCKS  "$FIX/self_falsifying_provenance_missing.sio" nonzero CLAIM_PROVENANCE_MISSING   no
    probe Z_BACKWARD_COMPAT "$FIX/self_falsifying_provenance_compat.sio"  zero    CLAIM_PASS                 yes
    # shared code changed again: R17 and R2 paths must be re-verified
    probe R17_REGRESSION_WITNESS_DRIFT "$FIX/self_falsifying_witness_drift.sio" nonzero CLAIM_WITNESS_MISMATCH no
    probe R2_REGRESSION_TOKEN_DRIFT    "$FIX/self_falsifying_token_drift.sio"   nonzero CLAIM_TOKEN_MISMATCH   no
    SHA="$(sha256sum "$EXECUTOR" | cut -d' ' -f1)"
    { echo "R20 provenance-binding behaviour receipt"; echo "executor_sha256=$SHA";
      echo "compiler=$RAW"; echo "";
      echo "Z_PRESENT_PASSES             rc=0 elf=yes CLAIM_PASS";
      echo "Z_MISSING_BLOCKS             rc=1 elf=no  CLAIM_PROVENANCE_MISSING   <- the rung";
      echo "Z_BACKWARD_COMPAT            rc=0 elf=yes CLAIM_PASS (no provenance field)";
      echo "R17_REGRESSION_WITNESS_DRIFT rc=1 elf=no  CLAIM_WITNESS_MISMATCH";
      echo "R2_REGRESSION_TOKEN_DRIFT    rc=1 elf=no  CLAIM_TOKEN_MISMATCH"; echo "";
      echo "In Z_MISSING_BLOCKS the gate exited 0 and emitted the declared token.";
      echo "The cited artifact -- the real one, the parity-collapse Phi -- is on";
      echo "another branch, and the build was refused."; } > "$RECEIPT"
    echo "compile arm: receipt written"
fi

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in Z1_FINDING_CLOSED Z2_EXECUTOR_SURFACE Z3_BEHAVIOUR_RECEIPT; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
grep -q "build REFUSED on the absent artifact" <<<"$OUT" \
    || fail "the missing-artifact probe no longer demonstrates refusal"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R20_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: spec '${SPEC_TOKEN}' vs contract '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R20_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
# The concessions, and the instrument-failure disclosure that is half the rung.
grep -q "The instrument failed first" "$SPEC" || fail "the audit's own filter defect was deleted"
grep -q "precondition was the negation" "$SPEC" || fail "the statement of that defect was deleted"
grep -q "Not a claim that the completeness result is wrong" "$SPEC" || fail "the scope limit was deleted"
grep -q "Not a defect count of 93" "$SPEC" || fail "the prose-vs-dependency distinction was deleted"
# The correction of this spec's own first reading is one deletion away.
grep -q "was corrected by looking" "$SPEC" \
    || fail "the correction of the 'mostly planned names' dismissal was deleted"
grep -q "never committed anywhere, on any branch" "$SPEC" \
    || fail "the never-committed dependency finding was deleted"
grep -q "self-destroys when the defect is fixed" "$SPEC" \
    || fail "the self-destroying-fixture lesson was deleted"
grep -q "Not content verification" "$SPEC" || fail "the existence-only concession was deleted"
echo "SELF_FALSIFYING_COMPILATION_LINE_R20_GATE_OK"
