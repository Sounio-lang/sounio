#!/usr/bin/env bash
# scripts/ci/generated_snapshot_guard.sh
#
# A pull request may not edit a generated snapshot.
#
# Why: these files are functions of the whole tree. Every PR that regenerated
# them conflicted with every other PR that did (measured 2026-10-05: of 89 open
# PRs, 23 touched topic-registry.v1.json, 21 DOCS_AUTHORITY_MATRIX.md and 11
# datasets/sounio-code-examples). GitHub's server-side merge runs no custom
# merge driver, so .gitattributes alone could not stop it. The checkers now
# validate against the registry rebuilt in memory (scripts/docs/
# check_docs_registry.mjs), so a stale snapshot hides nothing, and the snapshot
# is refreshed in its own PR:
#
#   git checkout -b snapshot/refresh-$(date +%F) origin/main
#   node scripts/docs/sync_governance_metadata.mjs --snapshot
#   python3 scripts/dev/export_hf_dataset.py
#   bash docs/training/finetune/prepare_corpus.sh
#
# A head branch named snapshot/refresh-* is the only one allowed through.
#
# Usage: SOUNIO_GUARD_BASE=<base sha> SOUNIO_GUARD_HEAD_REF=<branch> \
#        bash scripts/ci/generated_snapshot_guard.sh
# With no base (push, merge_group, local run without a PR) it checks nothing
# and says so.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9

SNAPSHOT_PATTERN='^(docs/governance/topic-registry\.v1\.json|docs/governance/DOCS_AUTHORITY_MATRIX\.md|datasets/sounio-code-examples/(train|validation)\.jsonl|datasets/sounio-code-examples/manifest\.json|docs/training/finetune/sounio_corpus\.txt|artifacts/gates/witness_declares_its_sabotage\.json|\.claude/llm_offload_log\.md)$'

base="${SOUNIO_GUARD_BASE:-}"
head_ref="${SOUNIO_GUARD_HEAD_REF:-}"

if [[ -z "$base" ]]; then
    echo "generated_snapshot_guard: SKIP (no pull-request base; nothing to compare)"
    exit 0
fi
if [[ "$head_ref" == snapshot/refresh-* ]]; then
    echo "generated_snapshot_guard: PASS (snapshot refresh branch $head_ref)"
    exit 0
fi

if ! touched="$(git diff --name-only "$base"...HEAD 2>&1)"; then
    echo "generated_snapshot_guard: FAIL (cannot diff $base...HEAD: $touched)" >&2
    exit 2
fi
offending="$(grep -E "$SNAPSHOT_PATTERN" <<<"$touched" || true)"

if [[ -z "$offending" ]]; then
    echo "generated_snapshot_guard: PASS (no generated snapshot in the diff)"
    exit 0
fi

fix=""
while IFS= read -r f; do
    if git cat-file -e "$base:$f" 2>/dev/null; then
        fix+="  git checkout $base -- $f"$'\n'
    else
        fix+="  git rm --cached --quiet $f    # not tracked on the base"$'\n'
    fi
done <<<"$offending"

cat >&2 <<EOF
generated_snapshot_guard: FAIL — this PR edits generated snapshot(s):
$(sed 's/^/  /' <<<"$offending")

Restore them to the base and push again:

${fix}  git commit -m "Drop regenerated snapshots (refreshed in their own PR)"

Docs metadata in the documents themselves still belongs in your PR:
node scripts/docs/sync_governance_metadata.mjs no longer writes the snapshot.
EOF
exit 1
