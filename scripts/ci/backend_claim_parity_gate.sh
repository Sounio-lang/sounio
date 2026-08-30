#!/usr/bin/env bash
# A document may not claim a backend the binary reports as not compiled.
#
# For months the docs said "Cranelift JIT enabled" and named a "Default JIT
# profile" artifact `souc-linux-x86_64-jit`. Measured 2026-08-27: that artifact is
# tracked nowhere, no build path passes `--features jit`, the binary exports no
# Cranelift symbol, and its own `info` prints
#
#     [-] Cranelift JIT - rebuild with --features jit
#
# The seven `cranelift` strings inside it are the messages that say it is absent.
# The claim survived a policy commit that demoted it from "default" to "legacy"
# instead of deleting it, and it had reached a list of RECOMMENDED PHRASINGS in
# docs/implementation/SELF_HOSTED_COMPILER.md -- so the documentation was not
# merely wrong, it was instructing people to repeat the error.
#
# The binary says which backends it has, in two characters. Nothing had to read it.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "backend_claim_parity"

ART="${BACKEND_CLAIM_ARTIFACT:-$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-gpu}"
if [[ ! -x "$ART" ]]; then
  gate_skip "no checked artifact at $ART to ask about its backends"
fi
echo "  artifact=$ART"

INFO="$(timeout 60 "$ART" info 2>&1)"
require_nonempty "$INFO" "souc info output"

# ONLY the "Enabled Backends:" block. `info` also prints an "Enabled Features:"
# block, and reading both was the first version's mistake: it took `[-] Ontology`
# on THIS artifact as a statement that no souc has an ontology CLI -- while
# `bin/souc ontology resolve GO:0008150` returns real data. One binary's report is
# a fact about that binary. Narrowing the parse is the whole correction.
BACKENDS="$(sed -n '/^Enabled Backends:/,/^$/p' <<<"$INFO")"
mapfile -t DISABLED < <(grep -oE '^\s*\[-\]\s*[A-Za-z0-9 ]+' <<<"$BACKENDS" | sed -E 's/^\s*\[-\]\s*//; s/\s+$//')
mapfile -t ENABLED  < <(grep -oE '^\s*\[\+\]\s*[A-Za-z0-9 ]+' <<<"$BACKENDS" | sed -E 's/^\s*\[\+\]\s*//; s/\s+$//')

# Controls. The binary is known to report at least one of each; if the parse
# yields none, it is the pattern and not the binary, and a clean report would be
# a lie of the same kind this gate exists to catch.
if ((${#DISABLED[@]} == 0)); then
  gate_fail "parse control failed: no '[-]' backend lines found in \`$ART info\`.
       Either the output format changed or the grep is wrong. Refusing to report
       parity against an empty set."
fi
if ((${#ENABLED[@]} == 0)); then
  gate_fail "parse control failed: no '[+]' backend lines found -- the parser cannot
       tell enabled from disabled, so its verdict about either is meaningless."
fi
echo "  compiled in: ${ENABLED[*]}"
echo "  not compiled: ${DISABLED[*]}"

ALLOW="$ROOT_DIR/scripts/ci/fixtures/backend_claim_historical.txt"
require_file "$ALLOW"

# An assertion is a doc line naming a NOT-COMPILED backend together with a word
# that puts it in the present. A line that also carries a negation is the repair,
# not the defect.
# `available` is deliberately NOT here. It is a conditional -- "skip unless LLVM
# available", "LLVM when available" -- not a claim that the backend is compiled in,
# and including it flagged four test-infrastructure tables that are correct.
ASSERT='enabled|by default|\(default\)|is the default'
# A repair must carry the canonical phrase `not compiled` on the SAME line as the
# backend name. Keeping this list small and literal is deliberate: the first
# version tried to enumerate every way a correction might be phrased, and my own
# corrections then read as defects because I had worded them differently.
NEGATED='not compiled|rebuild with|disabled'   # "X disabled" is a true report, not a claim

declare -a HITS=()
for b in "${DISABLED[@]}"; do
  key="$(awk '{print $1}' <<<"$b")"          # "Cranelift JIT" -> "Cranelift"
  # Guard against a one-word generic key matching unrelated prose.
  [[ ${#key} -lt 4 ]] && continue
  [[ -z "$key" ]] && continue
  while IFS= read -r line; do
    f="${line%%:*}"
    grep -qxF "$f" "$ALLOW" && continue      # recorded as historical
    # Test the negation on the WHOLE line; truncate only for display. Truncating
    # first hid the "not compiled" half of a multi-line repair and reported the
    # repair as the defect.
    grep -qE "$NEGATED" <<<"$line" && continue
    HITS+=("$(cut -c1-160 <<<"$line")")
  done < <(grep -rniE "$key[^.]{0,60}($ASSERT)|($ASSERT)[^.]{0,40}$key" \
             --include='*.md' docs/ README.md 2>/dev/null)
done

if ((${#HITS[@]})); then
  echo "  FAIL: a document asserts a backend the binary reports as NOT compiled:" >&2
  printf '        %s\n' "${HITS[@]}" >&2
  echo "" >&2
  echo "        Ask the binary: \`$ART info\`. If the doc is a record of a past state," >&2
  echo "        add its path to scripts/ci/fixtures/backend_claim_historical.txt, which" >&2
  echo "        is read in review. Otherwise fix the claim." >&2
  gate_fail "${#HITS[@]} live document(s) claim a backend that is not compiled"
fi

gate_pass "no live document claims a backend the binary reports as not compiled"
