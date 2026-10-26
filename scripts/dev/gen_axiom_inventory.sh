#!/usr/bin/env bash
# gen_axiom_inventory.sh -- regenerate formal/AXIOM_INVENTORY.md from source.
#
# What it counts, over every tracked `.lean` file under formal/ (excluding the
# vendored formal/lake/packages/ tree):
#
#   * every `axiom` declaration, with its name and line, grouped by file;
#   * `native_decide` uses per file;
#   * `sorry` in code (expected: 0).
#
# Comments (`-- ...`, nested `/- ... -/`, doc comments `/-- -/`, `/-! -/`) and
# string literals are blanked out BEFORE matching, preserving line numbers. A
# raw `grep` over-counts: many files say in their doc comments that they use
# "no sorry" or "no axioms". The method matches the one formal/README.md
# documents (declarations matching `^\s*axiom\s`, optionally preceded by
# `private`/`protected`/`noncomputable` modifiers, which no file currently uses).
#
# What it does NOT do: it does not run Lean. The "#print axioms" section of the
# inventory is not regenerated here; CI produces it (lean-proofs job, step
# "Axiom report for headline theorems", from formal/AxiomReport.lean and
# formal/lean4/AxiomReport.lean). The inventory carries the last reported
# result, explicitly marked as reported.
#
# Usage:
#   scripts/dev/gen_axiom_inventory.sh            # rewrite formal/AXIOM_INVENTORY.md
#   scripts/dev/gen_axiom_inventory.sh --check    # exit 1 if the file is stale
#   scripts/dev/gen_axiom_inventory.sh --stdout   # print, write nothing
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT="$ROOT/formal/AXIOM_INVENTORY.md"
MODE="${1:-write}"

case "$MODE" in
  write|--check|--stdout) ;;
  *) echo "usage: $0 [--check|--stdout]" >&2; exit 2 ;;
esac

TMP="$(mktemp)"
LIST="$(mktemp)"
trap 'rm -f "$TMP" "$LIST"' EXIT

cd "$ROOT"
git ls-files -z -- 'formal/*.lean' 'formal/**/*.lean' >"$LIST"
node "$ROOT/scripts/ci/axiom_inventory_source.mjs" "$TMP" "$LIST"

case "$MODE" in
  --stdout) cat "$TMP" ;;
  --check)
    if ! cmp -s "$TMP" "$OUT"; then
      echo "formal/AXIOM_INVENTORY.md is stale; run scripts/dev/gen_axiom_inventory.sh" >&2
      diff -u "$OUT" "$TMP" | head -40 >&2 || true
      exit 1
    fi
    echo "formal/AXIOM_INVENTORY.md is up to date" ;;
  write) cp "$TMP" "$OUT"; echo "wrote ${OUT#"$ROOT"/}" ;;
esac
