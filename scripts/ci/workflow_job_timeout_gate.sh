#!/usr/bin/env bash
# workflow_job_timeout_gate.sh — every job in ci.yml carries timeout-minutes.
#
# WHY (#2443). The Contracts job ran 124 steps with no timeout-minutes, so any
# one of them stalling held a runner for GitHub's 6-hour default before the
# PR saw a verdict. Five more jobs in ci.yml were unbounded the same way. The
# bounds were sized from measured durations (see the comments at each job);
# this gate keeps a newly added job from silently reintroducing the 6-hour
# default.
#
# A job that calls a reusable workflow (`uses:` at job level) cannot carry
# timeout-minutes and is exempt.
#
#   bash scripts/ci/workflow_job_timeout_gate.sh [workflow.yml]
#   bash scripts/ci/workflow_job_timeout_gate.sh --selftest
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "workflow_job_timeout_gate"

# Prints "<job> <has_timeout 0|1> <reusable 0|1>" per job under top-level jobs:.
list_jobs() {
  awk '
    /^[^ #]/ { in_jobs = ($0 ~ /^jobs:[[:space:]]*$/); if (job != "") emit(); next }
    !in_jobs { next }
    /^  [A-Za-z0-9_-]+:[[:space:]]*$/ {
      if (job != "") emit()
      job = $1; sub(/:$/, "", job); t = 0; u = 0; next
    }
    # Unsupported quoted keys and flow mappings must fail closed.
    /^  [^ #]/ {
      if (job != "") emit()
      print "unsupported-job-at-line-" NR, 0, 0; next
    }
    /^    timeout-minutes:[[:space:]]*[0-9]+/ { t = 1 }
    /^    uses:/ { u = 1 }
    function emit() { print job, t, u; job = "" }
    END { if (job != "") emit() }
  ' "$1"
}

check() {
  local wf="$1" jobs bad n
  require_file "$wf"
  jobs="$(list_jobs "$wf")"
  require_nonempty "$jobs" "no jobs parsed from $wf — the parser is not reading this file"
  n="$(printf '%s\n' "$jobs" | grep -c .)"
  bad="$(printf '%s\n' "$jobs" | awk '$2 == 0 && $3 == 0 { print $1 }')"
  if [[ -n "$bad" ]]; then
    echo "  jobs in $wf without timeout-minutes (GitHub default: 360 min):"
    printf '    %s\n' $bad
    return 1
  fi
  echo "  $n jobs in $wf, all bounded"
  return 0
}

selftest() {
  local d rc=0
  d="$(mktemp -d)"; trap 'rm -rf "$d"' RETURN
  cat >"$d/good.yml" <<'EOF'
name: t
on: push
jobs:
  a:
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    steps:
      - run: true
  b:
    uses: ./.github/workflows/other.yml
EOF
  cat >"$d/bad.yml" <<'EOF'
name: t
on: push
jobs:
  a:
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    steps:
      - run: true
  unbounded:
    runs-on: ubuntu-24.04
    steps:
      - name: step with a nested timeout does not count
        timeout-minutes: 3
        run: true
EOF
  if check "$d/good.yml" >/dev/null; then echo "  ok   POSITIVE: bounded workflow accepted"
  else echo "  FAIL POSITIVE: bounded workflow rejected"; rc=1; fi
  if out="$(check "$d/bad.yml")"; then echo "  FAIL NEGATIVE: unbounded job accepted"; rc=1
  elif grep -qx '    unbounded' <<<"$out"; then echo "  ok   NEGATIVE: unbounded job named and rejected"
  else echo "  FAIL NEGATIVE: rejected but did not name the job"; rc=1; fi
  for shape in quoted flow topflow; do
    case "$shape" in
      quoted) printf '\n  "quoted_unbounded":\n    runs-on: ubuntu-24.04\n    steps: []\n' >"$d/extra" ;;
      flow) printf '\n  flow_unbounded: {runs-on: ubuntu-24.04, steps: []}\n' >"$d/extra" ;;
      topflow) printf 'jobs: {flow_unbounded: {runs-on: ubuntu-24.04, steps: []}}\n' >"$d/extra" ;;
    esac
    if [[ "$shape" == topflow ]]; then cp "$d/extra" "$d/bypass.yml"
    else cat "$d/good.yml" "$d/extra" >"$d/bypass.yml"; fi
    if (check "$d/bypass.yml") >/dev/null 2>&1; then
      echo "  FAIL NEGATIVE: $shape mapping accepted"; rc=1
    else echo "  ok   NEGATIVE: $shape mapping rejected"; fi
  done
  return "$rc"
}

if [[ "${1:-}" == "--selftest" ]]; then
  selftest || gate_fail "selftest failed"
  gate_pass "selftest"
  exit 0
fi

selftest >/dev/null || gate_fail "selftest failed — the instrument is broken; run with --selftest"
check "${1:-$ROOT_DIR/.github/workflows/ci.yml}" || gate_fail "unbounded job(s) above (#2443)"
gate_pass
