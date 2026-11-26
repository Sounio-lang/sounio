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
paths from CI_BASE_SHA..CI_HEAD_SHA. pull_request and merge_group events are
classified by that diff; every other event (push, schedule, dispatch) selects
the full CI.
Writes key=value rows to stdout and, when set, GITHUB_OUTPUT.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

reused_from=""
declare -A impact=()
keys=(docs website compiler runtime stdlib tests lean bootstrap math ontology clinical sio full)
for key in "${keys[@]}"; do impact["$key"]=false; done

mark() { impact["$1"]=true; }

# A merge-queue run (merge_group) is classified by its diff exactly like the
# pull request it carries: base_sha..head_sha of the merge group is the set of
# queued changes on top of the current main. Forcing it to `full` (as every
# non-PR event still is) re-ran every job -- including the ~78 min current-source
# Madaros job -- for docs-, website- and lean-only PRs, after the PR run had
# already skipped them for the same paths. Measured 2026-09-25: 20/20 recent
# merge-group runs green at 47 min-1 h 32 min. push/schedule/dispatch stay full,
# so main itself is still covered in full after every merge.
if [[ "$EVENT_NAME" != "pull_request" && "$EVENT_NAME" != "merge_group" ]]; then
  for key in "${keys[@]}"; do impact["$key"]=true; done
else
  paths=()
  if (($#)); then
    paths=("$@")
  else
    [[ -n "$BASE_SHA" ]] || { echo "error: CI_BASE_SHA is required for $EVENT_NAME classification" >&2; exit 2; }
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
      echo "error: git diff --name-only $BASE_SHA $HEAD_SHA is empty -- a $EVENT_NAME with no changed paths would silently skip every job" >&2
      rm -f "$diff_list" "$diff_err"
      exit 4
    fi
    mapfile -t paths <"$diff_list"
    rm -f "$diff_list" "$diff_err"
    # Reuse (decision 2026-11-26): if an earlier head of this PR already ran,
    # and passed, every lane HEAD needs, and differs from HEAD only in
    # website/docs/changelog paths, its heavy-job verdict still holds. The jobs
    # are scheduled by that delta. The full BASE..HEAD classification is computed
    # first (a recursive run with reuse disabled). It is what R must cover, and it
    # is published as policy_* for the policy gates. See
    # ci_reuse_green_ancestor.sh. Fail-closed: no qualifying head means the full
    # classification.
    if [[ "$EVENT_NAME" == "pull_request" && "${CI_REUSE_DISABLE:-0}" != 1 ]]; then
      full_rows="$(CI_REUSE_DISABLE=1 GITHUB_OUTPUT= bash "${BASH_SOURCE[0]}" 2>/dev/null)" || full_rows=""
      required=""
      while IFS== read -r k v; do
        case "$k" in policy_*|reused_from) continue ;; esac
        [[ "$v" == true ]] && required="$required $k"
      done <<<"$full_rows"
      reuse_sha=""
      if [[ -n "$full_rows" ]]; then
        reuse_sha="$(CI_HEAD_SHA="$HEAD_SHA" CI_TESTED_SHA="${CI_TESTED_SHA:-$(git rev-parse HEAD)}" CI_REUSE_REQUIRED="$required" bash "$(dirname "${BASH_SOURCE[0]}")/ci_reuse_green_ancestor.sh" || true)"
      fi
      if [[ -n "$reuse_sha" ]]; then
        mapfile -t reuse_paths < <(git diff --name-only "$reuse_sha" "$HEAD_SHA")
        if ((${#reuse_paths[@]})); then
          echo "reuse: $reuse_sha already ran and passed every lane HEAD needs (${required# }); classifying the ${#reuse_paths[@]} path(s) of $reuse_sha..$HEAD_SHA instead of ${#paths[@]}" >&2
          paths=("${reuse_paths[@]}")
          reused_from="$reuse_sha"
        fi
      fi
    fi
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
      # llms.txt is the model-context sibling of README.md, not a new surface.
      *.md|docs/*|AGENTS.md|CLAUDE.md|CLAUDE_HANDOFF.md|FOUNDER_INTENT.md|ONBOARDING.md|README.md|llms.txt)
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
    # The SCO examples and their expected stdout are scientific regression fixtures.
    case "$path" in tests/*|examples/sco_corpus/*|check_sounio.sh) mark tests; recognized=true ;; esac
    # The bootstrap chain (Bootstrap Proofs job) and the rest of formal/ (Lean Proofs job).
    # The package files and the toolchain belong to both, as does the axiom report script.
    case "$path" in
      formal/Bootstrap/*|formal/Sporos/*|formal/Sporos.lean|bootstrap/model/*) mark bootstrap; recognized=true ;;
      formal/lakefile.lean|formal/lean-toolchain|formal/lake-manifest.json|scripts/ci/lean_axiom_report.sh) mark bootstrap; mark lean; recognized=true ;;
      formal/lean4/*|formal/*.lean) mark lean; recognized=true ;;
    esac
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
row="reused_from=${reused_from:-}"
echo "$row"
[[ -z "$OUTPUT_FILE" ]] || echo "$row" >>"$OUTPUT_FILE"
# policy_<key>: the full BASE..HEAD classification, never reduced by reuse.
# Policy gates (the changelog fragment gate, ADR-018) read these, so a reused
# execution verdict cannot hide what the pull request changes.
declare -A policy=()
for key in "${keys[@]}"; do policy["$key"]="${impact[$key]}"; done
if [[ -n "${full_rows:-}" ]]; then
  while IFS== read -r k v; do
    [[ -n "${policy[$k]+set}" ]] && policy["$k"]="$v"
  done <<<"$full_rows"
fi
for key in "${keys[@]}"; do
  row="policy_$key=${policy[$key]}"
  echo "$row"
  [[ -z "$OUTPUT_FILE" ]] || echo "$row" >>"$OUTPUT_FILE"
done
