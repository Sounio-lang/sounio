#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-bin/souc}"
SOURCE=tools/pireus/twist_gauge_factorization.sio
FROZEN=tools/pireus/twist_gauge_factorization.values.v1
RECEIPT=tools/pireus/evidence/pireus_twist_gauge_factorization.receipt.v1
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { printf 'PIREUS_TWIST_GAUGE_FACTORIZATION_GATE_FAIL: %s\n' "$*" >&2; exit 1; }
semantic_tail() { sed -n '/^schema=pireus-twist-gauge-factorization-v1/,$p' "$1"; }

"$SOUC" run "$SOURCE" >"$work/run-1.log" 2>&1 || fail 'Sounio authority did not execute'
"$SOUC" run "$SOURCE" >"$work/run-2.log" 2>&1 || fail 'Sounio authority was not repeatable'
semantic_tail "$work/run-1.log" >"$work/actual-1"
semantic_tail "$work/run-2.log" >"$work/actual-2"
cmp -s "$work/actual-1" "$work/actual-2" || fail 'two Sounio executions diverged'
cmp -s "$FROZEN" "$work/actual-1" || fail 'current Sounio result diverges from frozen semantics'

source_sha="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
frozen_sha="$(sha256sum "$FROZEN" | cut -d' ' -f1)"
grep -qx "sounio_source_sha256=$source_sha" "$RECEIPT" || fail 'receipt source hash drifted'
grep -qx "frozen_semantics_sha256=$frozen_sha" "$RECEIPT" || fail 'receipt semantics hash drifted'
grep -qx 'producer_language=Sounio' "$RECEIPT" || fail 'semantic producer is not Sounio'
grep -qx 'producer_role=SEMANTIC_AUTHORITY' "$RECEIPT" || fail 'authority role drifted'
grep -qx 'promotion=GAUGE_WHT_CANDIDATE_REFUSED' "$RECEIPT" || fail 'refusal was not recorded'
grep -qx 'claim_boundary=no-general-complexity-lower-bound' "$RECEIPT" || fail 'claim boundary is absent'

sed 's/if bits <= 1 { return 0 - 1 }/if bits <= 1 { return 1 }/' "$SOURCE" >"$work/mutated.sio"
cmp -s "$SOURCE" "$work/mutated.sio" && fail 'negative mutation did not apply'
if "$SOUC" run "$work/mutated.sio" >"$work/mutated.log" 2>&1; then
    fail 'altered Cayley-Dickson convention was accepted'
fi
grep -q '^PIREUS_TWIST_GAUGE_FACTORIZATION_FAIL$' "$work/mutated.log" || fail 'mutation failed for an unrelated reason'
if grep -q '^PIREUS_TWIST_GAUGE_FACTORIZATION_PASS$' "$work/mutated.log"; then
    fail 'mutated convention retained the authority marker'
fi

printf 'PIREUS_TWIST_GAUGE_FACTORIZATION_GATE_PASS source_sha256=%s frozen_sha256=%s\n' "$source_sha" "$frozen_sha"
