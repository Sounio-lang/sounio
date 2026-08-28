#!/usr/bin/env bash
# Keep the cross-thread founder context discoverable without treating it as a
# substitute for repository evidence or operational policy.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "[founder-intent-contract] FAIL: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing file: $1"
}

require_marker() {
  local marker="$1"
  local file="$2"
  grep -Fq -- "$marker" "$file" || fail "missing marker in $file: $marker"
}

require_file FOUNDER_INTENT.md
require_file AGENTS.md
require_file CLAUDE.md
require_file ONBOARDING.md

require_marker 'FOUNDER_INTENT.md' AGENTS.md
require_marker 'FOUNDER_INTENT.md' CLAUDE.md
require_marker 'FOUNDER_INTENT.md' ONBOARDING.md

require_marker 'Do not diminish the intuition. Do not spare it from the test.' FOUNDER_INTENT.md
require_marker 'intuition != analogy != formal model != executable implementation' FOUNDER_INTENT.md
require_marker '## The Garden and butterflies' FOUNDER_INTENT.md
require_marker 'Garden -> Hypothesis -> Executable -> Claim-ready' FOUNDER_INTENT.md
require_marker '`f128`, `f256`' FOUNDER_INTENT.md
require_marker '`val`, `err`, and `u` are different facts' FOUNDER_INTENT.md
require_marker 'Do not reassociate a non-associative expression' FOUNDER_INTENT.md
require_marker 'Octonions as a model of psychopharmacology are a research hypothesis' FOUNDER_INTENT.md
require_marker '### Disagreement protocol' FOUNDER_INTENT.md
require_marker '### Drift check before changing a research primitive' FOUNDER_INTENT.md

echo '[founder-intent-contract] PASS: entrypoints and semantic intent markers are present'
