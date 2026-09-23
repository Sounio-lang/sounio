#!/usr/bin/env bash
# Refuse any workflow job that combines a self-hosted (or runner-pool) label
# with a bare `pull_request:` trigger and no same-repo-origin guard.
#
# WHY THIS EXISTS
#
# `pull_request` runs against the MERGE of a fork's proposed changes; a
# self-hosted runner executing that code has whatever access the runner's
# environment carries -- for this repo, potentially a path into the project's
# own Kubernetes/Slurm cluster. GitHub's own guidance is explicit that
# self-hosted runners on a repo reachable by fork PRs must not run untrusted
# code unguarded. No such guard existed anywhere in this repo before this
# gate: a live instance was found and fixed in the same PR that adds this
# script (.github/workflows/gpu-research.yml's `validate-on-gpu` job had
# `runs-on: [self-hosted, gpu, cuda]` under a bare `pull_request:` trigger,
# gated only by an on/off repo variable -- no repo-origin check at all).
#
# WHAT COUNTS AS "SELF-HOSTED-ISH"
#
# A job's `runs-on:` mentioning the literal `self-hosted`, or a `${{ vars.* }}`
# expression whose variable name contains RUNNER (the kaxi-ptxas-accept /
# ci-fabric-pool-smoke pattern: `${{ vars.SOME_RUNNER_LABELS || 'ubuntu-latest' }}`).
# The latter is flagged even though today's instances all default safely to a
# GH-hosted label and are workflow_dispatch-only (no pull_request trigger at
# all) -- the point is to catch it BEFORE a future edit adds a pull_request
# trigger to such a job without also adding the guard.
#
# WHAT COUNTS AS GUARDED
#
# The job's own `if:` (checked in full, spanning `run:`/`env:` blocks is not
# needed -- only the job-level `if:` line(s) matter) contains the substring
# `head.repo.full_name == github.repository`. This is a text scan of the
# workflow YAML, not a full parse -- the same idiom
# scripts/ci/impact_ci_selftest.sh already uses to check ci-decision's needs
# list against evaluate_ci_decision.py's required map.
#
# `pull_request_target` is a related but separate risk class (it runs with
# base-repo secrets against a fork's code by design) and is deliberately out
# of scope here -- see docs/ops/fork_pr_self_hosted_runner_policy.md.
#
# Usage:
#   bash scripts/dev/check_self_hosted_runner_fork_exposure.sh              # check the real tree
#   bash scripts/dev/check_self_hosted_runner_fork_exposure.sh --selftest   # positive/negative fixtures
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCANNER="$ROOT_DIR/scripts/dev/_fork_exposure_scan.py"

# GitHub Actions loads both .yml and .yaml from .github/workflows/; scanning
# only *.yml would let an unguarded self-hosted job in a .yaml-suffixed
# workflow bypass this gate entirely. nullglob so a (today hypothetical)
# absence of one extension doesn't pass a literal unmatched glob pattern
# through to python3 as a nonexistent path.
real_workflow_files() {
    local files=()
    shopt -s nullglob
    files=("$ROOT_DIR"/.github/workflows/*.yml "$ROOT_DIR"/.github/workflows/*.yaml)
    shopt -u nullglob
    printf '%s\n' "${files[@]}"
}

run_real() {
    local files=()
    mapfile -t files < <(real_workflow_files)
    python3 "$SCANNER" "${files[@]}"
}

selftest() {
    local tmp rc=0
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    # POSITIVE: self-hosted job, bare pull_request trigger, no guard -- must flag.
    cat > "$tmp/positive.yml" <<'EOF'
on:
  pull_request:
jobs:
  danger:
    runs-on: [self-hosted, gpu, cuda]
    if: vars.SOME_FLAG == '1'
    steps:
      - run: echo hi
EOF

    # NEGATIVE 1: same shape, but guarded -- must NOT flag.
    cat > "$tmp/negative_guarded.yml" <<'EOF'
on:
  pull_request:
jobs:
  safe:
    runs-on: [self-hosted, gpu, cuda]
    if: vars.SOME_FLAG == '1' && (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)
    steps:
      - run: echo hi
EOF

    # NEGATIVE 2: self-hosted, but the workflow has no pull_request trigger at
    # all (workflow_dispatch only) -- must NOT flag.
    cat > "$tmp/negative_no_pr_trigger.yml" <<'EOF'
on:
  workflow_dispatch:
jobs:
  dispatch_only:
    runs-on: [self-hosted, linux, kaxi-slurm]
    steps:
      - run: echo hi
EOF

    # NEGATIVE 3: pull_request_target is a separate, out-of-scope risk class.
    cat > "$tmp/negative_pull_request_target.yml" <<'EOF'
on:
  pull_request_target:
jobs:
  automation:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - run: echo hi
EOF

    # NEGATIVE 4: GH-hosted runner, bare pull_request trigger -- must NOT flag
    # (this is the overwhelmingly common shape in the repo).
    cat > "$tmp/negative_gh_hosted.yml" <<'EOF'
on:
  pull_request:
jobs:
  ordinary:
    runs-on: ubuntu-24.04
    steps:
      - run: echo hi
EOF

    # NEGATIVE 5: `on: pull_request_target` as a bare inline scalar (not
    # bracketed) -- same out-of-scope risk class as NEGATIVE 3, exercised
    # through the inline-scalar code path instead of the block-form one.
    cat > "$tmp/negative_inline_scalar_target.yml" <<'EOF'
on: pull_request_target
jobs:
  automation:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - run: echo hi
EOF

    # POSITIVE 2: `pull_request: {}` (explicit empty mapping) is YAML-equivalent
    # to bare `pull_request:` -- must still flag an unguarded self-hosted job.
    cat > "$tmp/positive_empty_mapping_trigger.yml" <<'EOF'
on:
  pull_request: {}
jobs:
  danger:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - run: echo hi
EOF

    # POSITIVE 3: a 4-space-indented jobs map (nonstandard for this repo, but
    # valid YAML) -- the job-key indent is detected, not hardcoded to 2, so
    # this must still flag an unguarded self-hosted job.
    cat > "$tmp/positive_four_space_indent.yml" <<'EOF'
on:
  pull_request:
jobs:
    danger:
        runs-on: [self-hosted, gpu, cuda]
        steps:
            - run: echo hi
EOF

    # POSITIVE 4: a column-0 comment between jobs is not a dedent -- the
    # unguarded job after it must still be reached and flagged.
    cat > "$tmp/positive_comment_between_jobs.yml" <<'EOF'
on:
  pull_request:
jobs:
  safe:
    runs-on: ubuntu-24.04
    steps:
      - run: echo hi
# separator
  danger:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - run: echo hi
EOF

    # POSITIVE 5: the guard text present, but only inside an || with an
    # unrelated term -- `guard || vars.ENABLE == '1'` is true whenever EITHER
    # side is true, so it does NOT restrict to same-repo pull_request events.
    # Must still flag: presence of the guard text is not the same as the
    # guard actually gating anything.
    cat > "$tmp/positive_or_bypass.yml" <<'EOF'
on:
  pull_request:
jobs:
  danger:
    runs-on: [self-hosted, gpu, cuda]
    if: github.event.pull_request.head.repo.full_name == github.repository || vars.ENABLE == '1'
    steps:
      - run: echo hi
EOF

    # POSITIVE 6: `on: pull_request` as a bare scalar on the `on:` line itself
    # (no block, no brackets) -- a form the outer `on:` matcher used to miss
    # entirely (it required either nothing or a `[...]` after `on:`), so this
    # workflow's trigger was invisible to the scanner and an unguarded
    # self-hosted job under it went unflagged.
    cat > "$tmp/positive_inline_scalar_trigger.yml" <<'EOF'
on: pull_request
jobs:
  danger:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - run: echo hi
EOF

    # POSITIVE 7: `on: [pull_request, pull_request_target]` -- a combined
    # inline list where `pull_request` is still fork-reachable. The naive
    # `"pull_request_target" not in line` check used to disqualify this
    # whole line just because pull_request_target ALSO appeared in it,
    # missing the bare pull_request that was also present.
    cat > "$tmp/positive_inline_combined_trigger.yml" <<'EOF'
on: [pull_request, pull_request_target]
jobs:
  danger:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - run: echo hi
EOF

    local out
    out="$(python3 "$SCANNER" "$tmp"/*.yml || true)"

    if grep -q 'positive.yml:danger' <<<"$out"; then
        echo "  ok   POSITIVE: unguarded self-hosted job under pull_request is flagged"
    else
        echo "  FAIL POSITIVE: unguarded self-hosted job under pull_request was NOT flagged"; rc=1
    fi
    if grep -q 'positive_empty_mapping_trigger.yml:danger' <<<"$out"; then
        echo "  ok   POSITIVE: pull_request: {} (empty mapping) trigger is flagged"
    else
        echo "  FAIL POSITIVE: pull_request: {} (empty mapping) trigger was NOT flagged"; rc=1
    fi
    if grep -q 'positive_four_space_indent.yml:danger' <<<"$out"; then
        echo "  ok   POSITIVE: a 4-space-indented jobs map is flagged"
    else
        echo "  FAIL POSITIVE: a 4-space-indented jobs map was NOT flagged"; rc=1
    fi
    if grep -q 'positive_comment_between_jobs.yml:danger' <<<"$out"; then
        echo "  ok   POSITIVE: a job after a column-0 comment is flagged"
    else
        echo "  FAIL POSITIVE: a job after a column-0 comment was NOT flagged"; rc=1
    fi
    if grep -q 'positive_or_bypass.yml:danger' <<<"$out"; then
        echo "  ok   POSITIVE: guard-text-inside-an-|| bypass is still flagged"
    else
        echo "  FAIL POSITIVE: guard-text-inside-an-|| bypass was NOT flagged"; rc=1
    fi
    if grep -q 'positive_inline_scalar_trigger.yml:danger' <<<"$out"; then
        echo "  ok   POSITIVE: bare scalar 'on: pull_request' trigger is flagged"
    else
        echo "  FAIL POSITIVE: bare scalar 'on: pull_request' trigger was NOT flagged"; rc=1
    fi
    if grep -q 'positive_inline_combined_trigger.yml:danger' <<<"$out"; then
        echo "  ok   POSITIVE: 'on: [pull_request, pull_request_target]' is flagged"
    else
        echo "  FAIL POSITIVE: 'on: [pull_request, pull_request_target]' was NOT flagged"; rc=1
    fi
    for job in "negative_guarded.yml:safe" "negative_no_pr_trigger.yml:dispatch_only" \
               "negative_pull_request_target.yml:automation" "negative_gh_hosted.yml:ordinary" \
               "negative_inline_scalar_target.yml:automation"; do
        if grep -q "$job" <<<"$out"; then
            echo "  FAIL NEGATIVE: $job was flagged but should not have been"; rc=1
        else
            echo "  ok   NEGATIVE: $job correctly not flagged"
        fi
    done

    # Anti-vacuity: the scanner must actually find jobs in the real tree, or a
    # broken scanner (matches nothing, ever) would pass everything silently.
    local real_job_count
    local real_files=()
    mapfile -t real_files < <(real_workflow_files)
    real_job_count="$(python3 "$SCANNER" --count-jobs "${real_files[@]}")"
    if [[ "$real_job_count" -lt 20 ]]; then
        echo "  FAIL NEGATIVE: only $real_job_count jobs found across the real workflow tree -- scanner is dead"; rc=1
    else
        echo "  ok   NEGATIVE: scanner finds $real_job_count jobs across the real workflow tree"
    fi

    echo "failures: $rc"
    return $rc
}

if [[ "${1:-}" == "--selftest" ]]; then
    selftest
    exit $?
fi

selftest >/dev/null 2>&1 || {
    echo "ABORT: the gate's own controls fail -- its verdict would be noise, not evidence." >&2
    selftest
    exit 2
}

# No `|| true`: the scanner communicates violations via stdout, not its exit
# code (main() returns 0 whether it found violations or none) -- the only way
# run_real exits nonzero is a genuine scanner crash (a malformed/unreadable
# workflow file, a bug). Swallowing that here would make a crash read as "no
# violations found", the exact fail-open this gate exists to prevent. Let
# `set -e` propagate it as a hard failure instead.
violations="$(run_real)"
if [[ -n "$violations" ]]; then
    echo "CHECK_SELF_HOSTED_RUNNER_FORK_EXPOSURE_FAIL: unguarded self-hosted runner(s) reachable from fork PRs:" >&2
    echo "$violations" | sed 's/^/  /' >&2
    echo "" >&2
    echo "Each job above combines a self-hosted/runner-pool label with a bare" >&2
    echo "pull_request: trigger and no same-repo-origin guard. Add:" >&2
    echo "  (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)" >&2
    echo "to the job's if:, or move it off self-hosted infrastructure, or remove" >&2
    echo "the pull_request trigger. See docs/ops/fork_pr_self_hosted_runner_policy.md." >&2
    exit 1
fi

echo "CHECK_SELF_HOSTED_RUNNER_FORK_EXPOSURE_OK: no unguarded self-hosted runner reachable from fork PRs"
