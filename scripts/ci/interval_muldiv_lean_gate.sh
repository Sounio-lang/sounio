#!/usr/bin/env bash
# Exact rational set multiplication/division, not IEEE/backend certification.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: interval_muldiv_lean_gate.sh FRESH_ABSOLUTE_OUTPUT}"
[[ "$OUT" == /* && ! -e "$OUT" && ! -L "$OUT" ]] || exit 64
mkdir -p "$(dirname "$OUT")";mkdir "$OUT"
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
sha256sum formal/lean4/{SounioIntervalFinite,SounioIntervalFiniteAxiomAudit,SounioIntervalExtended,SounioIntervalExtendedAxiomAudit,SounioIntervalExtendedMulDiv,SounioIntervalExtendedMulDivAxiomAudit}.lean formal/lean4/{lakefile.lean,lean-toolchain} scripts/ci/interval_{finite_axiom_inventory,extended_axiom_inventory,muldiv_axiom_inventory}.mjs scripts/ci/{interval_extended_lean_gate,interval_muldiv_lean_gate}.sh scripts/ci/axiom_inventory_source.mjs > "$OUT/SOURCE.sha256"
[[ -s "$OUT/SOURCE.sha256" ]] || { echo "empty source manifest" >&2;exit 65; }
printf '%s  %s\n' b1eb2fdd6e623528d3c2064574e63137b782ed8da2a138253d044dc239a4a838 formal/lean4/SounioIntervalExtended.lean 7db7099327adad37db0635320f3eb029feb2d26ba390ed24045ca831481d0288 formal/lean4/SounioIntervalExtendedAxiomAudit.lean > "$OUT/ADDITIVE_SOURCE.sha256"
phase frozen-additive-source sha256sum -c "$OUT/ADDITIVE_SOURCE.sha256"
# The parent gate checks exact Lean4.33.0 binaries and libraries, finite32 identity,
# fresh finite/additive source and audits, and its four actual semantic mutants.
phase additive-dependency bash scripts/ci/interval_extended_lean_gate.sh "$OUT/additive"
PREFIX="$(cat "$OUT/additive/TOOLCHAIN_BINARIES.sha256" | head -n1 | sed 's/^[^ ]*  //;s@/bin/lean$@@')"
[[ -n "$PREFIX" && -x "$PREFIX/bin/lake" ]] || { echo "missing verified Lean prefix" >&2;exit 65; }
LAKE="$PREFIX/bin/lake"
NODE="$(command -v node)"
cd formal/lean4
phase scoped-build "$LAKE" build SounioIntervalExtendedMulDiv SounioIntervalExtendedMulDivAxiomAudit
phase fresh-source "$LAKE" env lean SounioIntervalExtendedMulDiv.lean
phase fresh-audit "$LAKE" env lean SounioIntervalExtendedMulDivAxiomAudit.lean
phase guard-selftest "$NODE" "$ROOT/scripts/ci/interval_muldiv_axiom_inventory.mjs" --selftest
grep -Fxq 'INTERVAL_MULDIV_AXIOM_SELFTEST_PASS controls=11' "$OUT/guard-selftest.stdout"
phase footprint "$NODE" "$ROOT/scripts/ci/interval_muldiv_axiom_inventory.mjs" SounioIntervalExtendedMulDiv.lean "$OUT/fresh-audit.stdout"
grep -Fq 'theorems=53 ' "$OUT/footprint.stdout"
printf 'formal/lean4/SounioIntervalExtendedMulDiv.lean\0' > "$OUT/selected-source.files0"
cd "$ROOT"
phase source-scanner-selftest "$NODE" scripts/ci/axiom_inventory_source.mjs --selftest
phase no-proof-escapes "$NODE" scripts/ci/axiom_inventory_source.mjs "$OUT/source-inventory.md" "$OUT/selected-source.files0"
grep -Fxq 'files=1 axioms=0 axiom_files=0 native_decide=0 native_decide_files=0 sorry=0' "$OUT/no-proof-escapes.stderr"
cd formal/lean4
"$NODE" "$ROOT/scripts/ci/interval_muldiv_axiom_inventory.mjs" --prepare-mutants SounioIntervalExtendedMulDiv.lean "$OUT"
while IFS=$'\t' read -r name file first last;do
 code=0
 "$LAKE" env lean "$file" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr" || code=$?
 echo "$code" > "$OUT/$name.exit"
 [[ "$code" == 1 ]]
 cat "$OUT/$name.stdout" "$OUT/$name.stderr" > "$OUT/$name.diagnostic"
 phase "mutant-$name" "$NODE" "$ROOT/scripts/ci/interval_muldiv_axiom_inventory.mjs" --check-mutant-error "$OUT/$name.diagnostic" "$first" "$last"
done < "$OUT/mutants.tsv"
[[ "$(wc -l < "$OUT/mutants.tsv")" == 12 ]]
sha256sum .lake/build/lib/lean/{SounioIntervalFinite,SounioIntervalFiniteAxiomAudit,SounioIntervalExtended,SounioIntervalExtendedAxiomAudit,SounioIntervalExtendedMulDiv,SounioIntervalExtendedMulDivAxiomAudit}.olean > "$OUT/PROOF_ARTIFACTS.sha256"
cd "$ROOT"
phase unchanged-source sha256sum -c "$OUT/SOURCE.sha256"
phase unchanged-toolchain sha256sum -c "$OUT/additive/TOOLCHAIN_BINARIES.sha256"
phase unchanged-libraries sha256sum -c "$OUT/additive/TOOLCHAIN_LIBRARIES.sha256"
phase unchanged-node sha256sum -c "$OUT/additive/NODE.sha256"
printf '%s\n' 'INTERVAL_MULDIV_LEAN_PASS new_theorems=53 finite_dependency=32 additive_dependency=40 real_mutants=12 guard_controls=11' > "$OUT/result.txt"
cat "$OUT/result.txt"
