#!/usr/bin/env bash
# Emit artifacts/self-hosted/madaros.gate-receipt for a binary that has just
# been built and gated.
#
# The receipt existed for weeks with no producer and no consumer — written by
# hand, then left behind when the binary was rebuilt, so it described a file
# that no longer existed while looking exactly like provenance. This script is
# the producer; scripts/ci/madaros_receipt_gate.sh is the consumer.
#
# Run it ONLY after the binary has passed its gates, and record which gates
# those were. A receipt is a claim about evidence; writing one for an ungated
# binary reintroduces the problem in a tidier font.
#
#   usage: madaros_write_receipt.sh <binary> [gate-name] [gate-checks-csv]
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "madaros_write_receipt"

BINARY="${1:-$ROOT_DIR/artifacts/self-hosted/madaros}"
GATE="${2:-madaros_full_gate.sh}"
CHECKS="${3:-}"
RECEIPT="$ROOT_DIR/artifacts/self-hosted/madaros.gate-receipt"

require_nonempty_file "$BINARY" "no binary to write a receipt for: $BINARY"
require_file "$ROOT_DIR/scripts/ci/$GATE" "the receipt would name a gate that does not exist: $GATE"

if head -c2 "$BINARY" 2>/dev/null | grep -q '#!'; then
  gate_fail "$BINARY is a wrapper script, not a raw ELF — a receipt for a wrapper certifies nothing"
fi

sha="$(sha256sum "$BINARY" | awk '{print $1}')"
require_nonempty "$sha" "sha256sum produced nothing"

rel="$(realpath --relative-to="$ROOT_DIR" "$BINARY" 2>/dev/null || printf '%s' "$BINARY")"

# Provenance must describe the tree the binary was BUILT from, not the tree that
# happens to be checked out when someone runs this. Two cases:
#
#   the artifact is tracked and unmodified -> the commit that last changed it.
#     This is the committed prebuilt; HEAD may be a thousand commits later and
#     claiming HEAD would be exactly the lie this whole gate exists to catch.
#   otherwise -> HEAD, which is the build-and-commit flow the weekly refresh
#     workflow uses.
if git ls-files --error-unmatch "$rel" >/dev/null 2>&1 && git diff --quiet HEAD -- "$rel" 2>/dev/null; then
  commit="$(git log -1 --format=%H -- "$rel" 2>/dev/null)"
  origin="the commit that last changed $rel"
else
  commit="$(git rev-parse HEAD 2>/dev/null)"
  origin="HEAD at build time"
  if ! git diff --quiet HEAD -- self-hosted stdlib 2>/dev/null; then
    gate_fail "self-hosted/ or stdlib/ has uncommitted changes: source_commit=$commit would not describe the tree this binary was built from. Commit the sources first, or the receipt is a guess."
  fi
fi
require_nonempty "$commit" "could not determine a source commit — a receipt with no provenance is unverifiable"

# created_utc comes from the clock; everything else is derived from the artifact
# and the repository, so the receipt cannot drift from them without this script
# being re-run.
{
  echo "madaros_full_gate_receipt_v1"
  echo "artifact=$(realpath --relative-to="$ROOT_DIR" "$BINARY" 2>/dev/null || printf '%s' "$BINARY")"
  echo "sha256=$sha"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gate=$GATE"
  echo "gate_result=pass"
  [[ -n "$CHECKS" ]] && echo "gate_checks=$CHECKS"
  echo "source_commit=$commit"
  echo "note=written by scripts/ci/madaros_write_receipt.sh; verified by scripts/ci/madaros_receipt_gate.sh"
} >"$RECEIPT"

echo "  wrote $RECEIPT"
echo "    sha256=$sha"
echo "    source_commit=$commit"

# Prove the pair agrees before claiming success. A writer whose output its own
# verifier rejects is worse than no writer.
if ! bash "$ROOT_DIR/scripts/ci/madaros_receipt_gate.sh" >/dev/null 2>&1; then
  bash "$ROOT_DIR/scripts/ci/madaros_receipt_gate.sh" 2>&1 | sed 's/^/    /' >&2
  gate_fail "the receipt just written does not satisfy madaros_receipt_gate.sh"
fi

gate_pass "receipt written and verified against its own gate"
