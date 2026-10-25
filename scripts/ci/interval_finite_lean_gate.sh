#!/usr/bin/env bash
# SIO-178: kernel-check the finite rational model and its explicit axiom footprint.
# The rd/ru enclosure premises are local assumptions, NOT hardware certificates.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -gt 1 ]]; then echo "usage: $0 [FRESH_ABSOLUTE_OUTPUT]" >&2; exit 2; fi
if [[ $# == 1 ]]; then
  OUT="$1"
  [[ "$OUT" == /* && ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'output must be a fresh absolute path' >&2; exit 2; }
  mkdir -p -- "$OUT"
else
  OUT="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/interval-finite-lean.XXXXXXXX")"
fi
printf 'phase\texit\n' > "$OUT/RESULTS.tsv"
finish() { local rc=$?; trap - EXIT; printf '%s\n' "$rc" > "$OUT/STATUS"; echo "INTERVAL_FINITE_LEAN_GATE exit=$rc receipts=$OUT"; exit "$rc"; }
trap finish EXIT
phase() {
  local label="$1"; shift
  local rc=0
  "$@" > "$OUT/$label.stdout" 2> "$OUT/$label.stderr" || rc=$?
  printf '%s\t%s\n' "$label" "$rc" >> "$OUT/RESULTS.tsv"
  if [[ "$rc" != 0 ]]; then cat "$OUT/$label.stderr" >&2; return "$rc"; fi
}
cd "$ROOT"
command -v node >/dev/null
command -v sha256sum >/dev/null
[[ "$(cat formal/lean4/lean-toolchain)" == 'leanprover/lean4:v4.33.0' ]]
git rev-parse HEAD > "$OUT/SOURCE_HEAD"
git status --porcelain --untracked-files=no > "$OUT/SOURCE_STATUS"
printf 'UTC_START=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/PROTOCOL.txt"
printf '%s\n' 'scope=finite Rat model; conditional local rounding enclosure; no IEEE/hardware qualification' >> "$OUT/PROTOCOL.txt"
sha256sum formal/lean4/{SounioIntervalFinite.lean,SounioIntervalFiniteAxiomAudit.lean,lakefile.lean,lean-toolchain} scripts/ci/{interval_finite_axiom_inventory.mjs,interval_finite_lean_gate.sh,axiom_inventory_source.mjs} scripts/dev/gen_axiom_inventory.sh > "$OUT/INPUTS.sha256"
cd "$ROOT/formal/lean4"
# Resolve elan's selected toolchain rather than hashing only the elan proxy.
PREFIX="$(lean --print-prefix)"
LEAN="$PREFIX/bin/lean"
LAKE="$PREFIX/bin/lake"
[[ -x "$LEAN" && -x "$LAKE" ]]
export PATH="$PREFIX/bin:$PATH"
"$LEAN" --version > "$OUT/LEAN_VERSION"
"$LAKE" --version > "$OUT/LAKE_VERSION"
grep -Eq '^Lean \(version 4\.33\.0,.*commit d8b18978322de05a8f3dba51ef03cf5461676c17,' "$OUT/LEAN_VERSION"
grep -Fq '(Lean version 4.33.0)' "$OUT/LAKE_VERSION"
printf '%s\n' "$PREFIX" > "$OUT/TOOLCHAIN_PREFIX"
sha256sum "$LEAN" "$LAKE" > "$OUT/BINARIES.sha256"
# Dynamic runtime libraries are part of the actual toolchain, not just launchers.
find "$PREFIX/lib/lean" -maxdepth 1 -type f -name '*.so' -print0 | sort -z | xargs -0 -r sha256sum > "$OUT/TOOLCHAIN_LIBRARIES.sha256"
phase scoped-build "$LAKE" build SounioIntervalFinite SounioIntervalFiniteAxiomAudit
# Unconditional fresh source check and fresh #print axioms; cached audit text
# from a prior job or the evidence directory is never accepted as the input.
phase fresh-source "$LAKE" env lean SounioIntervalFinite.lean
phase fresh-audit "$LAKE" env lean SounioIntervalFiniteAxiomAudit.lean
phase source-scanner-selftest node "$ROOT/scripts/ci/axiom_inventory_source.mjs" --selftest
phase guard-selftest node "$ROOT/scripts/ci/interval_finite_axiom_inventory.mjs" --selftest
phase expected-control-count grep -Fxq 'INTERVAL_AXIOM_INVENTORY_SELFTEST_PASS controls=9' "$OUT/guard-selftest.stdout"
phase axiom-footprint node "$ROOT/scripts/ci/interval_finite_axiom_inventory.mjs" SounioIntervalFinite.lean "$OUT/fresh-audit.stdout"
# Exact cardinality also rejects a vacuous empty source/audit pair.
phase expected-theorem-count grep -Fxq 'INTERVAL_AXIOM_INVENTORY_PASS theorems=32 allowed=propext,Classical.choice,Quot.sound primitive_rounding_premises=explicit_NOT_ESTABLISHED_hardware' "$OUT/axiom-footprint.stdout"
sha256sum .lake/build/lib/lean/{SounioIntervalFinite,SounioIntervalFiniteAxiomAudit}.olean > "$OUT/ARTIFACTS.sha256"
cd "$ROOT"
phase unchanged-inputs sha256sum --check "$OUT/INPUTS.sha256"
printf 'UTC_END=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/PROTOCOL.txt"
cat "$OUT/RESULTS.tsv" "$OUT/LEAN_VERSION" "$OUT/BINARIES.sha256" "$OUT/axiom-footprint.stdout" "$OUT/guard-selftest.stdout"
echo 'INTERVAL_FINITE_LEAN_PASS theorems=32 negative_controls=9'
