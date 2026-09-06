#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-bin/souc}"
SOURCE=tools/pireus/walsh_twist_channel_spectrum.sio
FROZEN=tools/pireus/walsh_twist_channel_spectrum.values.v1
RECEIPT=tools/pireus/evidence/pireus_walsh_twist_channel_spectrum.receipt.v1
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { printf 'PIREUS_WALSH_TWIST_CHANNEL_SPECTRUM_GATE_FAIL: %s\n' "$*" >&2; exit 1; }
semantic_tail() { sed -n '/^schema=pireus-walsh-twist-channel-spectrum-v1/,$p' "$1"; }

"$SOUC" run "$SOURCE" >"$work/current.log" 2>&1 || fail 'Sounio authority did not execute'
semantic_tail "$work/current.log" >"$work/current.values"
cmp -s "$FROZEN" "$work/current.values" || fail 'current spectrum diverges from frozen semantics'

source_sha="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
frozen_sha="$(sha256sum "$FROZEN" | cut -d' ' -f1)"
grep -qx "sounio_source_sha256=$source_sha" "$RECEIPT" || fail 'receipt source hash drifted'
grep -qx "frozen_semantics_sha256=$frozen_sha" "$RECEIPT" || fail 'receipt semantics hash drifted'
grep -qx 'producer_language=Sounio' "$RECEIPT" || fail 'semantic producer is not Sounio'
grep -qx 'producer_role=SEMANTIC_AUTHORITY' "$RECEIPT" || fail 'authority role drifted'
grep -qx 'generated_candidate=WalshCharacterChannels' "$RECEIPT" || fail 'generated operator identity is absent'
grep -qx 'equivalence=EXACT' "$RECEIPT" || fail 'candidate equivalence drifted'
grep -qx 'sparsity_promotion=REFUSED_AT_DIMENSION_16' "$RECEIPT" || fail 'dense-spectrum refusal is absent'
grep -qx 'claim_boundary=dimension-16-spectrum-not-asymptotic-complexity' "$RECEIPT" || fail 'claim boundary is absent'

sed 's/if bits <= 1 { return 0 - 1 }/if bits <= 1 { return 1 }/' "$SOURCE" >"$work/convention-mutant.sio"
"$SOUC" run "$work/convention-mutant.sio" >"$work/convention-mutant.log" 2>&1 || fail 'convention mutant failed before producing a spectrum'
semantic_tail "$work/convention-mutant.log" >"$work/convention-mutant.values"
cmp -s "$FROZEN" "$work/convention-mutant.values" && fail 'frozen spectrum does not bind Convention X'

sed 's/if bit_parity(k & i, bits) == 0/if true/' "$SOURCE" >"$work/character-mutant.sio"
if "$SOUC" run "$work/character-mutant.sio" >"$work/character-mutant.log" 2>&1; then
    fail 'non-Walsh character basis was accepted'
fi
grep -q '^PIREUS_WALSH_TWIST_CHANNEL_SPECTRUM_FAIL$' "$work/character-mutant.log" || fail 'character mutant failed for an unrelated reason'

printf 'PIREUS_WALSH_TWIST_CHANNEL_SPECTRUM_GATE_PASS source_sha256=%s frozen_sha256=%s\n' "$source_sha" "$frozen_sha"
