#!/usr/bin/env bash
# Exact rational unbounded additive interval model; not IEEE/backend certification.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: interval_extended_lean_gate.sh FRESH_ABSOLUTE_OUTPUT}"
[[ "$OUT" == /* && ! -e "$OUT" && ! -L "$OUT" ]] || exit 64
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
trap 'code=$?;echo "$code" > "$OUT/STATUS"' EXIT
phase() {
 local label="$1";shift
 local code=0
 "$@" > "$OUT/$label.stdout" 2> "$OUT/$label.stderr" || code=$?
 printf '%s\t%s\n' "$label" "$code" >> "$OUT/RESULTS.tsv"
 [[ "$code" == 0 ]] || { cat "$OUT/$label.stdout" "$OUT/$label.stderr" >&2;return "$code"; }
}
cd "$ROOT"
id > "$OUT/identity.txt"
sha256sum formal/lean4/{SounioIntervalFinite,SounioIntervalFiniteAxiomAudit,SounioIntervalExtended,SounioIntervalExtendedAxiomAudit}.lean formal/lean4/{lakefile.lean,lean-toolchain} scripts/ci/interval_{finite_axiom_inventory,extended_axiom_inventory}.mjs scripts/ci/interval_extended_lean_gate.sh > "$OUT/SOURCE.sha256"
[[ -s "$OUT/SOURCE.sha256" ]] || { echo "empty source manifest" >&2;exit 65; }
printf '%s  %s\n' 17cf45617ddf5aacbcafe8a553cc6b6cd02bb5bfa12eb14103950438fddc172b formal/lean4/SounioIntervalFinite.lean fb2abaecb5c45f6110195ef03a2414199d21e8269431c68bcbbca205f7ea75a6 formal/lean4/SounioIntervalFiniteAxiomAudit.lean 68b816784f992312bec5ad27235a0f8c36797ba5e0b84cfa3af2603d08ea0200 scripts/ci/interval_finite_axiom_inventory.mjs > "$OUT/FINITE_DEPENDENCY.sha256"
phase finite-source-identity sha256sum -c "$OUT/FINITE_DEPENDENCY.sha256"
[[ "$(cat formal/lean4/lean-toolchain)" == leanprover/lean4:v4.33.0 ]]
PREFIX="${LEAN_PREFIX:-$(lean --print-prefix)}"
LEAN="$PREFIX/bin/lean";LAKE="$PREFIX/bin/lake"
[[ -x "$LEAN" && -x "$LAKE" ]]
export PATH="$PREFIX/bin:$PATH"
"$LEAN" --version > "$OUT/LEAN_VERSION"
grep -Eq '^Lean \(version 4\.33\.0,.*commit d8b18978322de05a8f3dba51ef03cf5461676c17,' "$OUT/LEAN_VERSION"
printf '%s  %s\n' e8baaa71855a616dc351028f3ad2200051b0671f423a1696a100e809302d5550 "$LEAN" 60330ab6f07dce20f3fa9ebb08e8b984ea9549eac172afeb15d9d2227060e2b3 "$LAKE" > "$OUT/TOOLCHAIN_BINARIES.sha256"
phase toolchain-pin sha256sum -c "$OUT/TOOLCHAIN_BINARIES.sha256"
find "$PREFIX/lib/lean" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$OUT/TOOLCHAIN_LIBRARIES.sha256"
[[ -s "$OUT/TOOLCHAIN_LIBRARIES.sha256" ]] || { echo "empty toolchain library manifest" >&2;exit 65; }
NODE="$(command -v node)"
sha256sum "$NODE" > "$OUT/NODE.sha256"
printf '%s\n' 'scope=Rat plus infinite endpoints, finite-rational membership, exact and conditional-outward addition/subtraction; general multiplication/division and IEEE linkage NOT ESTABLISHED' > "$OUT/SCOPE.txt"
cd formal/lean4
phase scoped-build "$LAKE" build SounioIntervalFinite SounioIntervalFiniteAxiomAudit SounioIntervalExtended SounioIntervalExtendedAxiomAudit
phase fresh-finite "$LAKE" env lean SounioIntervalFinite.lean
phase fresh-finite-audit "$LAKE" env lean SounioIntervalFiniteAxiomAudit.lean
phase finite-guard-selftest "$NODE" "$ROOT/scripts/ci/interval_finite_axiom_inventory.mjs" --selftest
phase finite-footprint "$NODE" "$ROOT/scripts/ci/interval_finite_axiom_inventory.mjs" SounioIntervalFinite.lean "$OUT/fresh-finite-audit.stdout"
grep -Fq 'theorems=32 ' "$OUT/finite-footprint.stdout"
phase fresh-extended "$LAKE" env lean SounioIntervalExtended.lean
phase fresh-extended-audit "$LAKE" env lean SounioIntervalExtendedAxiomAudit.lean
phase guard-selftest "$NODE" "$ROOT/scripts/ci/interval_extended_axiom_inventory.mjs" --selftest
grep -Fxq 'INTERVAL_EXTENDED_AXIOM_SELFTEST_PASS controls=10' "$OUT/guard-selftest.stdout"
phase extended-footprint "$NODE" "$ROOT/scripts/ci/interval_extended_axiom_inventory.mjs" SounioIntervalExtended.lean "$OUT/fresh-extended-audit.stdout"
grep -Fq 'theorems=40 ' "$OUT/extended-footprint.stdout"
"$NODE" "$ROOT/scripts/ci/interval_extended_axiom_inventory.mjs" --prepare-mutants SounioIntervalExtended.lean "$OUT"
while IFS=$'\t' read -r name file first last;do
 code=0
 "$LAKE" env lean "$file" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr" || code=$?
 echo "$code" > "$OUT/$name.exit"
 [[ "$code" == 1 ]]
 cat "$OUT/$name.stdout" "$OUT/$name.stderr" > "$OUT/$name.diagnostic"
 phase "mutant-$name" "$NODE" "$ROOT/scripts/ci/interval_extended_axiom_inventory.mjs" --check-mutant-error "$OUT/$name.diagnostic" "$first" "$last"
done < "$OUT/mutants.tsv"
[[ "$(wc -l < "$OUT/mutants.tsv")" == 4 ]]
sha256sum .lake/build/lib/lean/{SounioIntervalFinite,SounioIntervalFiniteAxiomAudit,SounioIntervalExtended,SounioIntervalExtendedAxiomAudit}.olean > "$OUT/PROOF_ARTIFACTS.sha256"
cd "$ROOT"
phase unchanged-source sha256sum -c "$OUT/SOURCE.sha256"
phase unchanged-toolchain sha256sum -c "$OUT/TOOLCHAIN_BINARIES.sha256"
phase unchanged-libraries sha256sum -c "$OUT/TOOLCHAIN_LIBRARIES.sha256"
phase unchanged-node sha256sum -c "$OUT/NODE.sha256"
printf '%s\n' 'INTERVAL_EXTENDED_LEAN_PASS theorems=40 finite_dependency=32 real_mutants=4 guard_controls=10' > "$OUT/result.txt"
cat "$OUT/result.txt"
