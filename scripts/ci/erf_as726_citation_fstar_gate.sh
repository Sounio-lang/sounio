#!/usr/bin/env bash
# ADR-009 verified_foreign_reference gate: F* pilot for special_scipy_parity.
#
# Does NOT attempt to re-derive the Abramowitz & Stegun (1964) 7.1.26
# error bound (1.5e-7) from real analysis first principles -- that
# bound is cited, trusted authority, the same way a formal proof cites
# a well-known theorem rather than reproving all of mathematics.
#
# What this DOES mechanically verify: that Sounio's
# stdlib/special/erf.sio rational-approximation coefficients (p,
# a1..a5) are byte-identical to the published constants the cited
# bound applies to. A coefficient transcription error would silently
# produce an uncited, unbounded rational function that mpmath sampling
# at a handful of test points might not catch; this equality proof
# catches it unconditionally, regardless of sample points.

set -euo pipefail
umask 077

fail() {
  printf 'erf-as726-citation-fstar: FAIL: %s\n' "$*" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOURCE="$ROOT_DIR/tools/fstar/ErfAs726Citation.fst"
ERF_SIO="$ROOT_DIR/stdlib/special/erf.sio"
EXPECTED_FSTAR_VERSION="2026.08.30"

[[ -r "$SOURCE" ]] || fail "oracle source not found: $SOURCE"
[[ -r "$ERF_SIO" ]] || fail "erf.sio not found: $ERF_SIO"

command -v fstar.exe >/dev/null 2>&1 || \
  fail "fstar.exe not on PATH; install ${EXPECTED_FSTAR_VERSION} from https://github.com/FStarLang/FStar/releases"

observed_version="$(fstar.exe --version 2>&1 | sed -n '1s/^F\* //p')"
[[ "$observed_version" == "$EXPECTED_FSTAR_VERSION" ]] || \
  fail "fstar version drift: expected $EXPECTED_FSTAR_VERSION, observed ${observed_version:-unknown}"

# Manual sync check: the six numeric literals cited in the .fst must
# still appear verbatim in erf.sio. This is a coarse grep, not a
# semantic diff -- if erf.sio's coefficient block is ever restructured,
# this check (and the .fst) need a human to re-sync them.
for literal in 0.3275911 0.254829592 0.284496736 1.421413741 1.453152027 1.061405429; do
  grep -Fq "$literal" "$ERF_SIO" || \
    fail "cited coefficient $literal not found verbatim in erf.sio -- .fst is out of sync, re-check by hand"
done

source_sha256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"

work="$(mktemp -d "${TMPDIR:-/tmp}/erf-as726-fstar.XXXXXX")"
trap 'rm -rf "$work"' EXIT

fstar_output="$(fstar.exe "$SOURCE" 2>&1)" || {
  printf '%s\n' "$fstar_output" >&2
  fail "F* verification failed (see output above)"
}

printf '%s\n' "$fstar_output" | grep -q "Verified module: ErfAs726Citation" || \
  fail "F* did not report successful verification"

printf 'ERF_AS726_CITATION_FSTAR_V1\n'
printf 'oracle_class=verified_foreign_reference\n'
printf 'producer_language=F*\n'
printf 'producer_role=COEFFICIENT_CITATION_EQUALITY\n'
printf 'semantic_authority_language=Sounio\n'
printf 'citation=Abramowitz and Stegun 1964, Handbook of Mathematical Functions, 7.1.26, p.299\n'
printf 'cited_error_bound=1.5e-7\n'
printf 'error_bound_proved_here=no (trusted citation, see comment header)\n'
printf 'coefficient_equality_proved_here=yes\n'
printf 'fstar_version=%s\n' "$observed_version"
printf 'oracle_source_sha256=%s\n' "$source_sha256"
printf 'result=PASS\n'
