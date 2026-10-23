#!/usr/bin/env bash
# Extract fenced sio/sounio snippets and require a positive compiler check verdict.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Preserve the existing explicit SOUC override through canonical resolution.
if [[ -n "${SOUC:-}" ]]; then SOUC_BIN="$SOUC"; fi
source "$REPO_ROOT/scripts/lib/resolve_souc.sh"
sounio_require_souc
SOUC="$SOUC_BIN"
source "$REPO_ROOT/scripts/lib/classify_compile.sh"
SNIPPET_TIMEOUT_SECONDS="${SNIPPET_TIMEOUT_SECONDS:-30}"
[[ "$SNIPPET_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: invalid snippet timeout" >&2; exit 1; }
command -v timeout >/dev/null || { echo "FAIL: timeout command unavailable" >&2; exit 1; }
VERBOSE=0
AUDIT_ALL=0
TARGET_DIR=""
for arg in "$@"; do
  if [[ "$arg" == --verbose ]]; then VERBOSE=1; elif [[ "$arg" == --audit-all ]]; then AUDIT_ALL=1; else TARGET_DIR="$arg"; fi
done
if [[ -z "$TARGET_DIR" ]]; then
  TARGET_DIRS=("$REPO_ROOT/docs/training/rosetta" "$REPO_ROOT/website/src/content/docs")
else
  TARGET_DIRS=("$TARGET_DIR")
fi
SNIPPET_WORK="$(mktemp -d)"
trap 'rm -rf "$SNIPPET_WORK"' EXIT
pass=0 fail=0 skip=0 total=0 contextual=0 uncontracted=0 fixture_pass=0
extract_snippets() {
  local md_file="$1" line inside=false snippet_idx=0 buf="" out
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^\`\`\`(sio|sounio)[[:space:]]*$ ]]; then
      if $inside; then echo "FAIL: nested Sounio fence in $md_file" >&2; return 1; fi
      inside=true; buf=""; continue
    fi
    if $inside && [[ "$line" == '```' ]]; then
      if [[ -n "$buf" ]]; then
        # Unique per snippet, not basename: translations and equal filenames
        # must never overwrite an earlier document's code.
        out="$(mktemp "$SNIPPET_WORK/snippet.XXXXXX")"
        mv "$out" "$out.sio"
        printf '%s' "$buf" > "$out.sio"
        printf '%s:%s\n' "$md_file" "$snippet_idx" > "$out.origin"
        snippet_idx=$((snippet_idx + 1))
      fi
      inside=false; buf=""; continue
    fi
    if $inside; then buf="${buf}${line}"$'\n'; fi
  done < "$md_file"
  if $inside; then echo "FAIL: unclosed Sounio fence in $md_file" >&2; return 1; fi
}
for dir in "${TARGET_DIRS[@]}"; do
  [[ -d "$dir" ]] || { echo "FAIL: snippet target directory missing: $dir" >&2; exit 1; }
  while IFS= read -r -d '' md; do extract_snippets "$md"; done < <(
    find "$dir" -type f \( -name '*.md' -o -name '*.mdx' \) -not -path '*/archived/*' -not -path '*/internal/*' -print0
  )
done
# Explicit selected sample; uncontracted corpus is visible, never validated.
CONTRACTS="${DOC_SNIPPET_CONTRACTS:-$REPO_ROOT/scripts/ci/doc_snippet_contracts.tsv}"
if [[ -z "$TARGET_DIR" && "$AUDIT_ALL" == 0 ]]; then
  [[ -f "$CONTRACTS" ]] || { echo "FAIL: selected-sample contract missing" >&2; exit 1; }
  declare -A contract_kind contract_hash contract_fixture contract_fhash contract_verdict contract_note seen
  while IFS=$'\t' read -r doc idx hash kind fixture fhash verdict note extra; do
    [[ -z "$doc" || "$doc" == \#* ]] && continue
    [[ -z "$extra" && "$doc" != /* && "$doc" != *..* && "$doc" =~ ^(docs/|website/) && "$idx" =~ ^[0-9]+$ && "$hash" =~ ^[0-9a-f]{64}$ && -n "$note" ]] || { echo "FAIL: malformed contract row $doc:$idx" >&2; exit 1; }
    if [[ "$kind" == negative && ! "$note" =~ ^E[0-9]{3,4}$ ]]; then echo "FAIL: negative requires reviewed diagnostic code" >&2; exit 1; fi
    if [[ "$kind" == linked-fixture && "$verdict" == negative && "$note" != "confidence|epsilon|ε|E036|bound is not tight" ]]; then echo "FAIL: linked-negative requires canonical KL5 refusal matcher" >&2; exit 1; fi
    key="$REPO_ROOT/$doc:$idx"
    [[ -z "${contract_kind[$key]+x}" ]] || { echo "FAIL: duplicate contract $key" >&2; exit 1; }
    case "$kind:$verdict" in standalone:positive|negative:negative|contextual:none|linked-fixture:positive|linked-fixture:negative) ;; *) echo "FAIL: unsupported contract kind/verdict $key" >&2; exit 1;; esac
    if [[ "$kind" == linked-fixture ]]; then
      [[ "$fixture" =~ ^tests/ && "$fixture" != *..* && -f "$REPO_ROOT/$fixture" && "$fhash" =~ ^[0-9a-f]{64}$ ]] || { echo "FAIL: invalid linked fixture $key" >&2; exit 1; }
      [[ "$(sha256sum "$REPO_ROOT/$fixture" | cut -d' ' -f1)" == "$fhash" ]] || { echo "FAIL: fixture hash drift $key" >&2; exit 1; }
    else
      [[ "$fixture" == - && "$fhash" == - ]] || { echo "FAIL: unexpected fixture fields $key" >&2; exit 1; }
    fi
    contract_kind[$key]="$kind"; contract_hash[$key]="$hash"; contract_fixture[$key]="$fixture"; contract_fhash[$key]="$fhash"; contract_verdict[$key]="$verdict"; contract_note[$key]="$note"
  done < "$CONTRACTS"
  [[ ${#contract_kind[@]} -gt 0 ]] || { echo 'FAIL: empty selected sample'; exit 1; }
fi
shopt -s nullglob
for f in "$SNIPPET_WORK"/*.sio; do
  total=$((total + 1))
  origin="$(cat "${f%.sio}.origin")"
  kind=standalone; verdict=positive; note=-; input="$f"
  if [[ -z "$TARGET_DIR" && "$AUDIT_ALL" == 0 ]]; then
    if [[ -z "${contract_kind[$origin]+x}" ]]; then
      selected_doc=false
      for key in "${!contract_kind[@]}"; do [[ "${key%:*}" == "${origin%:*}" ]] && selected_doc=true; done
      if $selected_doc; then echo "FAIL: missing selected-document contract $origin"; fail=$((fail + 1)); else echo "UNCONTRACTED (NOT VALIDATED): $origin"; uncontracted=$((uncontracted + 1)); fi
      continue
    fi
    seen[$origin]=1
    [[ "$(sha256sum "$f" | cut -d' ' -f1)" == "${contract_hash[$origin]}" ]] || { echo "FAIL: snippet hash drift $origin"; fail=$((fail + 1)); continue; }
    kind="${contract_kind[$origin]}"; verdict="${contract_verdict[$origin]}"; note="${contract_note[$origin]}"
    if [[ "$kind" == contextual ]]; then echo "CONTEXTUAL (NOT VALIDATED): $origin reason=$note"; contextual=$((contextual + 1)); continue; fi
    if [[ "$kind" == linked-fixture ]]; then input="$REPO_ROOT/${contract_fixture[$origin]}"; fi
  fi
  if [[ -n "$TARGET_DIR" || "$AUDIT_ALL" == 1 ]] && ! grep -Eq '\bfn[[:space:]]' "$f"; then skip=$((skip + 1)); continue; fi
  log="${f%.sio}.log"
  rc=0
  timeout --kill-after=5s "${SNIPPET_TIMEOUT_SECONDS}s" "$SOUC" check "$input" > "$log" 2>&1 || rc=$?
  ok=false
  if [[ "$verdict" == positive && "$rc" == 0 ]] && ! sounio_is_fatal_log "$log" && grep -Eq '^[[:space:]]*check:[[:space:]]*OK[[:space:]]*$' "$log" && ! grep -Eiq 'parse error|type error|check:[[:space:]]*(FAIL|ERROR)|(^|[[:space:]])error([[:space:]:\[]|$)|error\[E[0-9]+\]' "$log"; then ok=true; fi
  rule="$note"
  if [[ "$kind" == linked-fixture ]]; then rule=""; fi
  if [[ "$verdict" == negative && "$note" != - ]] && sounio_expect_refusal "$rc" "$log" "" "$rule" && grep -Eiq "$note" "$log" && ! grep -Eq '^[[:space:]]*check:[[:space:]]*OK[[:space:]]*$' "$log"; then ok=true; fi
  if $ok; then
    if [[ "$kind" == linked-fixture ]]; then fixture_pass=$((fixture_pass + 1)); echo "FIXTURE_CHECK_PASS (DISPLAYED SNIPPET NOT VALIDATED): $origin fixture=$input"; else pass=$((pass + 1)); echo "SNIPPET_CHECK_PASS kind=$kind: $origin"; fi
  else
    fail=$((fail + 1)); echo "FAIL kind=$kind exit=$rc: $origin"
    cat "$log"
  fi
done
if [[ -z "$TARGET_DIR" && "$AUDIT_ALL" == 0 ]]; then
  for key in "${!contract_kind[@]}"; do [[ -n "${seen[$key]+x}" ]] || { echo "FAIL: contracted fence missing $key"; fail=$((fail + 1)); }; done
  echo "SELECTED_SAMPLE_ONLY snippet_checks=$pass fixture_checks=$fixture_pass contextual=$contextual uncontracted=$uncontracted; NO GLOBAL DOCS VALIDATION"
fi
printf '\nDoc snippet check: %s pass, %s fail, %s skip, %s total\n' "$pass" "$fail" "$skip" "$total"
[[ "$fail" == 0 ]] || exit 1
[[ "$((pass + fixture_pass))" -gt 0 ]] || { echo 'FAIL: no measured checks'; exit 1; }
echo "GATE_MEASURED assertions=$((pass + fixture_pass))"
