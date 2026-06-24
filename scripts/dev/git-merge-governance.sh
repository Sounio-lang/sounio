#!/usr/bin/env bash
# scripts/dev/git-merge-governance.sh
#
# Git merge driver for the GENERATED docs/governance artifacts:
#   - docs/governance/DOCS_ACCEPTANCE_REPORT.md
#   - docs/governance/DOCS_AUTHORITY_MATRIX.md
#   - docs/governance/topic-registry.v1.json
#
# These files are produced by scripts/docs/sync_governance_metadata.mjs from the
# repository's doc set. A textual 3-way merge of them is meaningless and they
# conflict on nearly every merge from an active main (each docs-touching commit
# regenerates them). Instead of merging text, this driver regenerates them from
# the already-merged source docs and hands git the fresh content.
#
# Wired up via .gitattributes (`merge=governance-regen`) plus a one-time
#   bash scripts/dev/install-git-merge-drivers.sh
# per clone/worktree (the driver command lives in git config, which is not
# committed). NOTE: GitHub's server-side merge does NOT run custom drivers, so
# this resolves LOCAL merges only — push the result to clear the PR's conflict.
#
# Invoked by git as: git-merge-governance.sh %A %P
#   $1 = %A  temp file holding "our" version; the driver MUST leave the final
#            merged content here for git to consume.
#   $2 = %P  repo-relative pathname of the file being merged.
set -euo pipefail

# Git runs merge drivers with CWD at the top of the working tree.
ROOT="$(pwd)"
out="$1"
pathname="${2:-}"

# Regenerate all governance artifacts from the merged source docs. Best-effort:
# if regeneration fails we fall back to leaving "our" version in place ($out is
# already our version), so the merge never hard-fails on these generated files.
if [[ -f "$ROOT/scripts/docs/sync_governance_metadata.mjs" ]]; then
    node "$ROOT/scripts/docs/sync_governance_metadata.mjs" >/dev/null 2>&1 || true
fi

# Hand git the freshly regenerated content for this specific file.
if [[ -n "$pathname" && -f "$ROOT/$pathname" ]]; then
    cp "$ROOT/$pathname" "$out"
fi

exit 0
