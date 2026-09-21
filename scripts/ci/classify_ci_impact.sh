#!/usr/bin/env bash

set -euo pipefail

EVENT_NAME="${CI_EVENT_NAME:-${GITHUB_EVENT_NAME:-pull_request}}"
BASE_SHA="${CI_BASE_SHA:-}"
HEAD_SHA="${CI_HEAD_SHA:-HEAD}"
OUTPUT_FILE="${GITHUB_OUTPUT:-}"

usage() {
  cat <<'USAGE'
Usage: classify_ci_impact.sh [path ...]

With explicit paths, classifies those paths. Without paths, reads the changed
paths from CI_BASE_SHA..CI_HEAD_SHA. Non-pull-request events select the full CI.
Writes key=value rows to stdout and, when set, GITHUB_OUTPUT.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

declare -A impact=()
keys=(docs website compiler runtime stdlib tests lean math ontology clinical sio full)
for key in "${keys[@]}"; do impact["$key"]=false; done

mark() { impact["$1"]=true; }

if [[ "$EVENT_NAME" != "pull_request" ]]; then
  for key in "${keys[@]}"; do impact["$key"]=true; done
else
  paths=()
  if (($#)); then
    paths=("$@")
  else
    [[ -n "$BASE_SHA" ]] || { echo "error: CI_BASE_SHA is required for pull_request classification" >&2; exit 2; }
    # The diff is this classifier's only evidence, and its failure used to be
    # invisible: fed through process substitution, a failed `git diff` left
    # paths=(), every output false, and the script exited 0 -- a run that
    # downstream reads identically to "no jobs needed" and silently skips the
    # whole matrix (CI_TRUST_CONTRACT: an instrument that did not answer is
    # unavailable, not an empty selected set). Capture its status, and refuse
    # an empty PR diff for the same reason.
    diff_list="$(mktemp)"
    diff_err="$(mktemp)"
    if ! git diff --name-only "$BASE_SHA" "$HEAD_SHA" >"$diff_list" 2>"$diff_err"; then
      echo "error: git diff --name-only $BASE_SHA $HEAD_SHA failed -- cannot classify impact:" >&2
      sed 's/^/  git: /' "$diff_err" >&2
      rm -f "$diff_list" "$diff_err"
      exit 3
    fi
    if [[ ! -s "$diff_list" ]]; then
      echo "error: git diff --name-only $BASE_SHA $HEAD_SHA is empty -- a pull request with no changed paths would silently skip every job" >&2
      rm -f "$diff_list" "$diff_err"
      exit 4
    fi
    mapfile -t paths <"$diff_list"
    rm -f "$diff_list" "$diff_err"
  fi

  for path in "${paths[@]}"; do
    [[ -n "$path" ]] || continue
    recognized=false

    case "$path" in
      .github/workflows/*|scripts/ci/classify_ci_impact.sh|scripts/ci/evaluate_ci_decision.py|scripts/ci/impact_ci_selftest.sh|scripts/dev/check_workflow_script_refs.sh)
        mark full
        recognized=true
        ;;
    esac

    case "$path" in
      *.md|docs/*|AGENTS.md|CLAUDE.md|CLAUDE_HANDOFF.md|FOUNDER_INTENT.md|ONBOARDING.md|README.md)
        mark docs
        recognized=true
        ;;
    esac
    case "$path" in website/*) mark website; recognized=true ;; esac
    case "$path" in
      self-hosted/*|bin/*|scripts/lib/*|scripts/ci/build_*|scripts/ci/selfhost_*|scripts/selfhost/*|scripts/run_sio_test_suite.sh)
        mark compiler
        recognized=true
        ;;
    esac
    case "$path" in stdlib/runtime/*|self-hosted/native/*) mark runtime; recognized=true ;; esac
    case "$path" in stdlib/*) mark stdlib; recognized=true ;; esac
    case "$path" in tests/*|check_sounio.sh) mark tests; recognized=true ;; esac
    case "$path" in formal/lean4/*) mark lean; recognized=true ;; esac
    case "$path" in
      formal/lean4/*|scripts/ci/sedenion_*|scripts/ci/cd_tower_*|scripts/ci/gresnigt_*|scripts/ci/furey_*|scripts/ci/octonion_probes_gate.sh|scripts/research/sedenion_*|scripts/research/cd_tower_*|scripts/research/oct_*|scripts/research/ossm_*|scripts/ci/ade_wildgen_*|scripts/research/ade_wildgen_*)
        mark math
        recognized=true
        ;;
    esac
    case "$path" in stdlib/compiler/ontology/*|scripts/ci/*ontology*|docs/ontology/*) mark ontology; recognized=true ;; esac
    case "$path" in stdlib/clinical/*|tests/*vancomycin*|docs/clinical/*|docs/*pbpk*|scripts/ci/*pbpk*|scripts/ci/*clinical*) mark clinical; recognized=true ;; esac
    case "$path" in *.sio) mark sio; recognized=true ;; esac

    # A newly introduced surface must receive the full matrix until this
    # classifier explicitly learns its dependency boundary.
    [[ "$recognized" == true ]] || mark full
  done

  if [[ "${impact[full]}" == true ]]; then
    for key in "${keys[@]}"; do impact["$key"]=true; done
  fi
fi

for key in "${keys[@]}"; do
  row="$key=${impact[$key]}"
  echo "$row"
  [[ -z "$OUTPUT_FILE" ]] || echo "$row" >>"$OUTPUT_FILE"
done
