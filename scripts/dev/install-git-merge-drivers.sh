#!/usr/bin/env bash
# scripts/dev/install-git-merge-drivers.sh
#
# Registers this repo's custom git merge drivers in the local git config.
# Run once per clone/worktree:
#
#   bash scripts/dev/install-git-merge-drivers.sh
#
# Custom merge-driver *commands* cannot live in a committed file (git only reads
# them from git config), so each working copy must install them. The .gitattributes
# entries that select these drivers ARE committed.
#
# Driver: governance-regen — regenerates the generated docs/governance artifacts
# instead of attempting a meaningless textual 3-way merge. See
# scripts/dev/git-merge-governance.sh and .gitattributes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

git -C "$ROOT" config merge.governance-regen.name \
    "Regenerate generated docs/governance artifacts (no textual merge)"
# Relative command path so it resolves correctly in every worktree that shares
# this config (git runs merge drivers with CWD at the working-tree root).
git -C "$ROOT" config merge.governance-regen.driver \
    "bash scripts/dev/git-merge-governance.sh %A %P"

echo "Installed git merge driver: governance-regen"
echo "  selects: docs/governance/{DOCS_ACCEPTANCE_REPORT.md,DOCS_AUTHORITY_MATRIX.md,topic-registry.v1.json}"

# post-merge hook: the driver runs mid-merge and can leave the generated
# governance files stale, so regenerate them from the fully-merged tree after
# the merge completes. Installed into the per-worktree hooks dir.
HOOKS_DIR="$(git -C "$ROOT" rev-parse --git-path hooks)"
mkdir -p "$HOOKS_DIR"
if [[ -e "$HOOKS_DIR/post-merge" ]] && ! grep -q "regenerate governance metadata" "$HOOKS_DIR/post-merge" 2>/dev/null; then
    echo "WARNING: $HOOKS_DIR/post-merge already exists and is not ours; leaving it untouched." >&2
    echo "  Append the body of scripts/dev/git-hooks/post-merge manually if desired." >&2
else
    cp "$ROOT/scripts/dev/git-hooks/post-merge" "$HOOKS_DIR/post-merge"
    chmod +x "$HOOKS_DIR/post-merge"
    echo "Installed git hook: post-merge (regenerates governance metadata)"
fi
