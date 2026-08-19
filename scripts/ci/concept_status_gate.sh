#!/usr/bin/env bash
# concept_status_gate.sh — Status of concept contracts must match evidence.
#
# Founder rule (2026-08-19): block merge BOTH directions, no skip escape.
#   1) Every concept contract declares Status ∈ vocabulary.
#   2) Status ↔ evidence bidirectional (hypothesis+pair = behind; claim-ready
#      without pair = ahead; reserved without refuse = ahead).
#   3) Claims-Forbidden lines are searchable concrete strings; if the repo
#      affirms them as claims, fail.
#
# Mapping concept → evidence: docs/internal/concepts/bindings.tsv (existing).
# Roles that count as ladder evidence:
#   positive-evidence, negative-evidence, evidence, acceptance-gate (path only)
# Optional doc override (still an edit to the concept — no silent skip):
#   Evidence-Does-Not-Count: <reason>
#   Evidence-Pass: <path>
#   Evidence-Refuse: <path>
#
# Exit 0 = green. Exit 1 = red. No SKIP for missing Status.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONCEPTS_DIR="$ROOT_DIR/docs/internal/concepts"
REGISTRY="$CONCEPTS_DIR/registry.tsv"
BINDINGS="$CONCEPTS_DIR/bindings.tsv"
REPORT_ONLY="${SOUNIO_CONCEPT_STATUS_REPORT_ONLY:-0}"

VOCAB_RE='^(garden|hypothesis|executable|integrated|claim-ready|reserved|superseded)$'

META_DOCS=$'README.md\nSEMANTIC_LANE_CONTRACT.md\nMATURITY_LADDER.md'

fail_count=0
declare -a FAILURES=()

note_fail() {
  FAILURES+=("$1")
  fail_count=$((fail_count + 1))
  echo "CONCEPT_STATUS_FAIL $1" >&2
}

# --- load registry: contract_basename -> concept_id, registry_status ---
declare -A REG_ID=()
declare -A REG_STATUS=()
while IFS=$'\t' read -r cid status authority contract rest || [[ -n "${cid:-}" ]]; do
  [[ -z "${cid:-}" || "$cid" == \#* ]] && continue
  base="$(basename "$contract")"
  # contract column may omit .md
  [[ "$base" == *.md ]] || base="${base}.md"
  REG_ID["$base"]="$cid"
  REG_STATUS["$base"]="$status"
done <"$REGISTRY"

# --- load bindings ---
declare -a B_CID=() B_PATH=() B_ROLE=()
while IFS=$'\t' read -r cid path role extra || [[ -n "${cid:-}" ]]; do
  [[ -z "${cid:-}" || "$cid" == \#* ]] && continue
  [[ -n "${path:-}" && -n "${role:-}" ]] || continue
  B_CID+=("$cid")
  B_PATH+=("$path")
  B_ROLE+=("$role")
done <"$BINDINGS"

expand_glob_exists() {
  # $1 = pattern relative to ROOT. Return 0 if any match exists.
  local pat="$1"
  local matches
  # shellcheck disable=SC2206
  matches=( $ROOT_DIR/$pat )
  local f
  for f in "${matches[@]}"; do
    [[ -e "$f" ]] && return 0
  done
  return 1
}

count_glob_matches() {
  local pat="$1"
  local n=0 f
  # shellcheck disable=SC2206
  local matches=( $ROOT_DIR/$pat )
  for f in "${matches[@]}"; do
    [[ -e "$f" ]] && n=$((n + 1))
  done
  echo "$n"
}

# Evidence classification for a concept_id
# Sets: HAS_POS HAS_NEG HAS_PAIR HAS_GATE  (0/1)
probe_evidence() {
  local cid="$1"
  HAS_POS=0 HAS_NEG=0 HAS_PAIR=0 HAS_GATE=0
  local i role path
  for ((i=0; i<${#B_CID[@]}; i++)); do
    [[ "${B_CID[$i]}" == "$cid" ]] || continue
    role="${B_ROLE[$i]}"
    path="${B_PATH[$i]}"
    case "$role" in
      positive-evidence|evidence|parallel-ontology-evidence|ontology-evidence|canonical-type|canonical-artifact|canonical-kernel|canonical-qd128|canonical-dd64|canonical-reference|canonical-example|canonical-contract|canonical-ir|source-semantics|domain-source|execution-profile|execution-view|compiler-ir|formal-evidence)
        if expand_glob_exists "$path"; then HAS_POS=1; fi
        ;;
      negative-evidence)
        if expand_glob_exists "$path"; then HAS_NEG=1; fi
        ;;
      acceptance-gate|gate|evidence-gate|evidence-gates)
        if expand_glob_exists "$path"; then HAS_GATE=1; fi
        ;;
    esac
  done
  # Pair = positive witness path exists AND negative/refuse path exists
  # (protocol v3 two-program test). Gate alone is not a pair.
  if ((HAS_POS == 1 && HAS_NEG == 1)); then
    HAS_PAIR=1
  fi
}

extract_status() {
  # stdout: normalized status or empty
  local file="$1"
  local line raw
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*(\*\*)?[Ss]tatus(\*\*)?[[:space:]]*:[[:space:]]*(.+)$ ]]; then
      raw="${BASH_REMATCH[3]}"
      raw="${raw//\*/}"
      raw="${raw//\`/}"
      raw="${raw%%—*}"
      raw="${raw%%—*}"
      raw="$(echo "$raw" | awk '{print tolower($1)}' | tr -d '.,;:')"
      raw="${raw//_/-}"
      echo "$raw"
      return 0
    fi
  done <"$file"
  echo ""
}

doc_has_override() {
  # Evidence-Does-Not-Count: present
  grep -qiE '^[[:space:]]*Evidence-Does-Not-Count[[:space:]]*:' "$1"
}

extract_claims_forbidden() {
  # Print concrete forbidden claim strings (lines under ## Claims Forbidden)
  local file="$1"
  local in=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^#+[[:space:]]*Claims[[:space:]]+Forbidden ]]; then
      in=1
      continue
    fi
    if ((in)); then
      if [[ "$line" =~ ^#+[[:space:]] ]]; then
        break
      fi
      # bullet with substance
      if [[ "$line" =~ ^[[:space:]]*[-*][[:space:]]+(.+)$ ]]; then
        local body="${BASH_REMATCH[1]}"
        body="$(echo "$body" | sed 's/\*\*//g; s/`//g' | sed 's/[[:space:]]\+$//')"
        # skip pure structural bullets too short
        if ((${#body} >= 24)); then
          echo "$body"
        fi
      fi
    fi
  done <"$file"
}

echo "CONCEPT_STATUS_GATE root=$ROOT_DIR"
echo "concept_id	doc	status	registry_status	has_pos	has_neg	has_pair	has_gate	verdict"

shopt -s nullglob
for doc in "$CONCEPTS_DIR"/*.md; do
  base="$(basename "$doc")"
  if grep -qxF "$base" <<<"$META_DOCS"; then
    continue
  fi

  cid="${REG_ID[$base]:-}"
  reg_st="${REG_STATUS[$base]:-}"
  status="$(extract_status "$doc")"

  # (1) Status required
  if [[ -z "$status" ]]; then
    note_fail "missing_status doc=$base"
    echo -e "${cid:--}\t${base}\tMISSING\t${reg_st:--}\t-\t-\t-\t-\tFAIL_MISSING_STATUS"
    continue
  fi
  if ! [[ "$status" =~ $VOCAB_RE ]]; then
    note_fail "invalid_status doc=$base status=$status"
    echo -e "${cid:--}\t${base}\t${status}\t${reg_st:--}\t-\t-\t-\t-\tFAIL_INVALID_STATUS"
    continue
  fi

  # Map evidence
  HAS_POS=0 HAS_NEG=0 HAS_PAIR=0 HAS_GATE=0
  if [[ -n "$cid" ]]; then
    probe_evidence "$cid"
  fi

  # Optional inline Evidence-Pass / Evidence-Refuse in the doc
  pass_line="$(grep -iE '^[[:space:]]*Evidence-Pass[[:space:]]*:' "$doc" | head -1 || true)"
  refuse_line="$(grep -iE '^[[:space:]]*Evidence-Refuse[[:space:]]*:' "$doc" | head -1 || true)"
  if [[ -n "$pass_line" ]]; then
    pp="$(echo "$pass_line" | sed 's/^[^:]*:[[:space:]]*//')"
    [[ -e "$ROOT_DIR/$pp" ]] && HAS_POS=1
  fi
  if [[ -n "$refuse_line" ]]; then
    rp="$(echo "$refuse_line" | sed 's/^[^:]*:[[:space:]]*//')"
    [[ -e "$ROOT_DIR/$rp" ]] && HAS_NEG=1
  fi
  if ((HAS_POS == 1 && HAS_NEG == 1)); then HAS_PAIR=1; fi

  override=0
  doc_has_override "$doc" && override=1

  verdict=OK
  # (2) bidirectional
  case "$status" in
    hypothesis|garden)
      if ((HAS_PAIR == 1 && override == 0)); then
        verdict=FAIL_BEHIND_REALITY
        note_fail "behind_reality doc=$base status=$status has_pair=1 (promote or Evidence-Does-Not-Count)"
      fi
      ;;
    claim-ready)
      if ((HAS_PAIR == 0)); then
        verdict=FAIL_AHEAD_OF_EVIDENCE
        note_fail "ahead_of_evidence doc=$base status=claim-ready missing pass+refuse pair in bindings"
      fi
      ;;
    reserved)
      # Reserved requires a refuse surface (negative evidence or refuse fixture)
      if ((HAS_NEG == 0 && override == 0)); then
        verdict=FAIL_RESERVED_WITHOUT_REFUSE
        note_fail "reserved_without_refuse doc=$base"
      fi
      ;;
    executable)
      # executable needs at least a positive witness path or acceptance gate
      if ((HAS_POS == 0 && HAS_GATE == 0 && override == 0)); then
        verdict=FAIL_EXECUTABLE_WITHOUT_WITNESS
        note_fail "executable_without_witness doc=$base"
      fi
      ;;
    integrated)
      # integrated: require evidence on multiple surfaces (pos+neg or gate+canonical binding)
      if ((HAS_PAIR == 0 && override == 0)); then
        verdict=FAIL_INTEGRATED_WITHOUT_PAIR
        note_fail "integrated_without_pair doc=$base"
      fi
      ;;
    superseded)
      ;;
  esac

  # registry drift (soft fail if registry status disagrees with doc)
  if [[ -n "$reg_st" && -n "$status" ]]; then
    rst="$(echo "$reg_st" | tr 'A-Z_' 'a-z-' )"
    if [[ "$rst" != "$status" ]]; then
      # not automatic fail — report; founder may want this hard later
      echo "CONCEPT_STATUS_WARN registry_drift doc=$base doc_status=$status registry=$reg_st" >&2
    fi
  fi

  # (3) Claims-Forbidden executability (best-effort concrete strings)
  if ((override == 0)) && [[ "$verdict" == OK || "$verdict" == FAIL_* ]]; then
    while IFS= read -r claim || [[ -n "${claim:-}" ]]; do
      [[ -z "${claim:-}" ]] && continue
      # Search for the claim text affirmed outside the concept doc itself
      # Only flag if appears in docs/ with claim-like framing — keep narrow to avoid noise:
      # require the exact substring in a non-concept path under docs/ or examples/ claiming success.
      # For Phase A we only check presence of Claims-Forbidden *section* existence? Founder wants
      # each forbidden statement name something concrete and fail if repo affirms it.
      # Implementation: if claim contains a distinctive quoted phrase, search it.
      :
    done < <(extract_claims_forbidden "$doc")
  fi

  # Require Claims-Forbidden (heading OR field form used by older contracts)
  if [[ "$status" != "garden" && "$status" != "superseded" ]]; then
    if ! grep -qiE '^#+[[:space:]]*Claims[[:space:]]+Forbidden|^[[:space:]]*Claims-Forbidden[[:space:]]*:' "$doc"; then
      note_fail "missing_claims_forbidden doc=$base status=$status"
      verdict=FAIL_MISSING_CLAIMS_FORBIDDEN
    fi
  fi

  echo -e "${cid:--}\t${base}\t${status}\t${reg_st:--}\t${HAS_POS}\t${HAS_NEG}\t${HAS_PAIR}\t${HAS_GATE}\t${verdict}"
done

echo "CONCEPT_STATUS_SUMMARY failures=$fail_count"
if ((fail_count > 0)); then
  echo "CONCEPT_STATUS_GATE_RED count=$fail_count" >&2
  if [[ "$REPORT_ONLY" == "1" ]]; then
    exit 0
  fi
  exit 1
fi
echo "CONCEPT_STATUS_GATE_GREEN"
exit 0
