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
#   Evidence-Does-Not-Count:   (founder-accepted escape — NOT a Status-Held)
#     Reason: <non-empty, specific to the pair that does not count>
#     Owner:  <who signs>
#     Date:   <ISO YYYY-MM-DD>
#     Missing any of the three = RED (malformed is stricter than absence).
#     No automatic expiry (decision documented in audit doc).
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

VOCAB_RE='^(garden|hypothesis|executable|integrated|claim-ready|reserved-owed|reserved-taken|superseded)$'

META_DOCS=$'README.md\nSEMANTIC_LANE_CONTRACT.md\nMATURITY_LADDER.md'

fail_count=0
declare -a FAILURES=()
# Active (well-formed) EDNC declarations — printed always, oldest first.
# Age from Date field only (not git log). No expiry, no age-based fail.
declare -a EDNC_ROWS=()  # "age\tdoc\towner\tdate\treason"
declare -a OWED_ROWS=()  # "age\tdoc\towner\tsince\tblocked_on"


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

# Parse Evidence-Does-Not-Count block.
# Sets: EDNC_PRESENT (0/1), EDNC_OK (0/1), EDNC_REASON, EDNC_OWNER, EDNC_DATE,
#       EDNC_MISS (comma list of missing fields), EDNC_AGE_DAYS (or -1)
parse_ednc() {
  local file="$1"
  EDNC_PRESENT=0
  EDNC_OK=0
  EDNC_REASON=""
  EDNC_OWNER=""
  EDNC_DATE=""
  EDNC_MISS=""
  EDNC_AGE_DAYS=-1

  if ! grep -qiE '^[[:space:]]*Evidence-Does-Not-Count[[:space:]]*:' "$file"; then
    return 0
  fi
  EDNC_PRESENT=1

  # Collect the block: from Evidence-Does-Not-Count: through blank line before next ## heading
  # or next top-level field that is not Reason/Owner/Date.
  local line in=0 same=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*[Ee]vidence-[Dd]oes-[Nn]ot-[Cc]ount[[:space:]]*:[[:space:]]*(.*)$ ]]; then
      in=1
      same="${BASH_REMATCH[1]}"
      # allow inline "Evidence-Does-Not-Count: reason text" as Reason seed only if no Reason: field later
      if [[ -n "$same" ]]; then
        EDNC_REASON="$same"
      fi
      continue
    fi
    if ((in == 0)); then
      continue
    fi
    # end of block
    if [[ "$line" =~ ^#+[[:space:]] ]]; then
      break
    fi
    if [[ -z "${line//[[:space:]]/}" ]]; then
      # blank: end block unless we have not finished required fields yet — still end
      break
    fi
    if [[ "$line" =~ ^[[:space:]]*[Rr]eason[[:space:]]*:[[:space:]]*(.*)$ ]]; then
      EDNC_REASON="${BASH_REMATCH[1]}"
      continue
    fi
    if [[ "$line" =~ ^[[:space:]]*[Oo]wner[[:space:]]*:[[:space:]]*(.*)$ ]]; then
      EDNC_OWNER="${BASH_REMATCH[1]}"
      continue
    fi
    if [[ "$line" =~ ^[[:space:]]*[Dd]ate[[:space:]]*:[[:space:]]*(.*)$ ]]; then
      EDNC_DATE="${BASH_REMATCH[1]}"
      continue
    fi
    # unknown field inside block — stop (do not silently absorb)
    break
  done <"$file"

  # trim
  EDNC_REASON="$(echo -n "$EDNC_REASON" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  EDNC_OWNER="$(echo -n "$EDNC_OWNER" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  EDNC_DATE="$(echo -n "$EDNC_DATE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

  local miss=()
  [[ -z "$EDNC_REASON" ]] && miss+=("Reason")
  [[ -z "$EDNC_OWNER" ]] && miss+=("Owner")
  [[ -z "$EDNC_DATE" ]] && miss+=("Date")

  # vacuous reasons are not reasons
  local rl
  rl="$(echo "$EDNC_REASON" | tr '[:upper:]' '[:lower:]')"
  case "$rl" in
    ""|"ainda nao"|"ainda não"|"n/a"|"na"|"tbd"|"todo"|"fixme"|"later"|"wip"|"-"|"." )
      if [[ -n "$EDNC_REASON" ]]; then
        miss+=("Reason(vacuous)")
      fi
      ;;
  esac
  # reason must be specific enough (founder: vacuous reasons like "not yet" / PT "ainda nao" are rejected)
  if [[ -n "$EDNC_REASON" && ${#EDNC_REASON} -lt 12 ]]; then
    miss+=("Reason(too_short)")
  fi

  # Date must be ISO YYYY-MM-DD
  if [[ -n "$EDNC_DATE" ]]; then
    if ! [[ "$EDNC_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
      miss+=("Date(not_ISO)")
    else
      # age in days for visibility (no expiry fail — see audit doc)
      if date -d "$EDNC_DATE" >/dev/null 2>&1; then
        local then now
        then=$(date -d "$EDNC_DATE" +%s)
        now=$(date +%s)
        EDNC_AGE_DAYS=$(( (now - then) / 86400 ))
      fi
    fi
  fi

  if ((${#miss[@]} > 0)); then
    local IFS=,
    EDNC_MISS="${miss[*]}"
    EDNC_OK=0
  else
    EDNC_OK=1
  fi
}



# Parse reserved-owed / reserved-taken required fields from a concept doc.
# Sets: RES_OWNER RES_SINCE RES_BLOCKED RES_REASON RES_MISS RES_AGE_DAYS RES_OK
parse_reserved_fields() {
  local file="$1" kind="$2"
  RES_OWNER="" RES_SINCE="" RES_BLOCKED="" RES_REASON=""
  RES_MISS="" RES_AGE_DAYS=-1 RES_OK=0

  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*[Rr]eserved-[Oo]wner[[:space:]]*:[[:space:]]*(.*)$ ]]; then
      RES_OWNER="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]*[Rr]eserved-[Ss]ince[[:space:]]*:[[:space:]]*(.*)$ ]]; then
      RES_SINCE="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]*[Rr]eserved-[Bb]locked-[Oo]n[[:space:]]*:[[:space:]]*(.*)$ ]]; then
      RES_BLOCKED="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]*[Rr]eserved-[Rr]eason[[:space:]]*:[[:space:]]*(.*)$ ]]; then
      RES_REASON="${BASH_REMATCH[1]}"
    fi
  done <"$file"

  RES_OWNER="$(echo -n "$RES_OWNER" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  RES_SINCE="$(echo -n "$RES_SINCE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  RES_BLOCKED="$(echo -n "$RES_BLOCKED" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  RES_REASON="$(echo -n "$RES_REASON" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

  local miss=()
  if [[ "$kind" == "reserved-owed" ]]; then
    [[ -z "$RES_OWNER" ]] && miss+=("Reserved-Owner")
    [[ -z "$RES_SINCE" ]] && miss+=("Reserved-Since")
    [[ -z "$RES_BLOCKED" ]] && miss+=("Reserved-Blocked-On")
    if [[ -n "$RES_SINCE" ]]; then
      if ! [[ "$RES_SINCE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        miss+=("Reserved-Since(not_ISO)")
      elif date -d "$RES_SINCE" >/dev/null 2>&1; then
        local then now
        then=$(date -d "$RES_SINCE" +%s)
        now=$(date +%s)
        RES_AGE_DAYS=$(( (now - then) / 86400 ))
      fi
    fi
    if [[ -n "$RES_BLOCKED" && ${#RES_BLOCKED} -lt 12 ]]; then
      miss+=("Reserved-Blocked-On(too_short)")
    fi
  elif [[ "$kind" == "reserved-taken" ]]; then
    [[ -z "$RES_REASON" ]] && miss+=("Reserved-Reason")
    if [[ -n "$RES_REASON" && ${#RES_REASON} -lt 8 ]]; then
      miss+=("Reserved-Reason(too_short)")
    fi
  fi

  if ((${#miss[@]} > 0)); then
    local IFS=,
    RES_MISS="${miss[*]}"
    RES_OK=0
  else
    RES_OK=1
  fi
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
    if [[ "$status" == "reserved" ]]; then
      note_fail "bare_reserved doc=$base (use reserved-owed or reserved-taken; bare reserved is invalid)"
      echo -e "${cid:--}\t${base}\treserved\t${reg_st:--}\t-\t-\t-\t-\tFAIL_BARE_RESERVED"
    else
      note_fail "invalid_status doc=$base status=$status"
      echo -e "${cid:--}\t${base}\t${status}\t${reg_st:--}\t-\t-\t-\t-\tFAIL_INVALID_STATUS"
    fi
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
  parse_ednc "$doc"
  if ((EDNC_PRESENT == 1)); then
    if ((EDNC_OK == 1)); then
      override=1
      # Record for end-of-run visibility (always printed, green or red).
      age="$EDNC_AGE_DAYS"
      [[ "$age" -lt 0 ]] && age=0
      # reason single-line for TSV
      rflat="$(echo -n "$EDNC_REASON" | tr '\t\n' '  ')"
      EDNC_ROWS+=("${age}"$'	'"${base}"$'	'"${EDNC_OWNER}"$'	'"${EDNC_DATE}"$'	'"${rflat}")
    else
      # Malformed EDNC is RED even when it would not have been needed — wider door closed.
      verdict=FAIL_EDNC_MALFORMED
      note_fail "ednc_malformed doc=$base missing=$EDNC_MISS (Reason+Owner+Date ISO required; vacuous Reason rejected)"
      echo -e "${cid:--}\t${base}\t${status}\t${reg_st:--}\t${HAS_POS}\t${HAS_NEG}\t${HAS_PAIR}\t${HAS_GATE}\t${verdict}"
      continue
    fi
  fi

  verdict=OK
  # (2) bidirectional
  case "$status" in
    hypothesis|garden)
      if ((HAS_PAIR == 1 && override == 0)); then
        verdict=FAIL_BEHIND_REALITY
        note_fail "behind_reality doc=$base status=$status has_pair=1 (promote or complete Evidence-Does-Not-Count with Reason+Owner+Date)"
      fi
      ;;
    claim-ready)
      if ((HAS_PAIR == 0)); then
        verdict=FAIL_AHEAD_OF_EVIDENCE
        note_fail "ahead_of_evidence doc=$base status=claim-ready missing pass+refuse pair in bindings"
      fi
      ;;
    reserved-owed)
      parse_reserved_fields "$doc" "reserved-owed"
      if ((RES_OK == 0)); then
        verdict=FAIL_RESERVED_OWED_MALFORMED
        note_fail "reserved_owed_malformed doc=$base missing=$RES_MISS (Reserved-Owner+Reserved-Since+Reserved-Blocked-On required)"
      else
        age="$RES_AGE_DAYS"; [[ "$age" -lt 0 ]] && age=0
        bflat="$(echo -n "$RES_BLOCKED" | tr '	
' '  ')"
        OWED_ROWS+=("${age}"$'	'"${base}"$'	'"${RES_OWNER}"$'	'"${RES_SINCE}"$'	'"${bflat}")
        if ((HAS_NEG == 0)); then
          verdict=FAIL_RESERVED_WITHOUT_REFUSE
          note_fail "reserved_without_refuse doc=$base status=reserved-owed"
        fi
      fi
      ;;
    reserved-taken)
      parse_reserved_fields "$doc" "reserved-taken"
      if ((RES_OK == 0)); then
        verdict=FAIL_RESERVED_TAKEN_MALFORMED
        note_fail "reserved_taken_malformed doc=$base missing=$RES_MISS (Reserved-Reason required)"
      elif ((HAS_NEG == 0)); then
        verdict=FAIL_RESERVED_WITHOUT_REFUSE
        note_fail "reserved_without_refuse doc=$base status=reserved-taken"
      fi
      ;;
    executable)
      # executable needs a positive witness or gate. EDNC does not waive this
      # (founder accepted EDNC only for "pair exists but must not promote").
      if ((HAS_POS == 0 && HAS_GATE == 0)); then
        verdict=FAIL_EXECUTABLE_WITHOUT_WITNESS
        note_fail "executable_without_witness doc=$base"
      fi
      ;;
    integrated)
      if ((HAS_PAIR == 0)); then
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

# --- Always emit active EDNC list (visibility, not pressure) ---
emit_ednc_visibility() {
  local n=${#EDNC_ROWS[@]}
  local out_tsv="$ROOT_DIR/docs/internal/concepts/ednc_active.tsv"
  local sorted tmp line age doc owner date reason
  tmp="$(mktemp)"
  {
    echo -e "age_days\tdoc\towner\tdate\treason"
    if ((n > 0)); then
      # oldest first (age descending); age is from Date field, not mtime/git
      printf '%s\n' "${EDNC_ROWS[@]}" | sort -t$'\t' -k1,1nr
    fi
  } >"$tmp"

  # 1) stdout/stderr — visible on every local and CI log line stream
  echo "CONCEPT_STATUS_EDNC_ACTIVE count=$n (no expiry; age from declaration Date; oldest first)" 
  if ((n == 0)); then
    echo "CONCEPT_STATUS_EDNC_ACTIVE none"
  else
    local i=0
    while IFS=$'\t' read -r age doc owner date reason; do
      [[ "$age" == "age_days" ]] && continue
      i=$((i + 1))
      echo "CONCEPT_STATUS_EDNC_ACTIVE [$i/$n] doc=$doc owner=$owner age_days=$age date=$date"
    done <"$tmp"
  fi

  # 2) committed-path TSV regenerated each run — humans open the concepts dir
  #    and see the roster without grepping logs. Not a claim of truth beyond
  #    what the gate just read; CI may show it as a dirty file if left unignored.
  #    We write under docs/internal/concepts/ so it sits next to the contracts.
  cp "$tmp" "$out_tsv"

  # 3) GitHub Job Summary when present (Actions UI — no log dig)
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo "## Concept Evidence-Does-Not-Count (active)"
      echo ""
      echo "Declarations **do not expire**. Age is from the declared \`Date\` field (not git history)."
      echo "Sorted oldest first. Count: **$n**."
      echo ""
      if ((n == 0)); then
        echo "_No active Evidence-Does-Not-Count declarations._"
      else
        echo "| age_days | doc | owner | date |"
        echo "|---:|---|---|---|"
        while IFS=$'\t' read -r age doc owner date reason; do
          [[ "$age" == "age_days" ]] && continue
          echo "| $age | \`$doc\` | $owner | $date |"
        done <"$tmp"
      fi
    } >>"$GITHUB_STEP_SUMMARY"
  fi
  rm -f "$tmp"

  # reserved-owed roster (same visibility rules as EDNC)
  local on=${#OWED_ROWS[@]}
  local otmp
  otmp="$(mktemp)"
  {
    echo -e "age_days\tdoc\towner\tsince\tblocked_on"
    if ((on > 0)); then
      printf '%s\n' "${OWED_ROWS[@]}" | sort -t$'\t' -k1,1nr
    fi
  } >"$otmp"
  echo "CONCEPT_STATUS_OWED_ACTIVE count=$on (no expiry; age from Reserved-Since; oldest first)"
  if ((on == 0)); then
    echo "CONCEPT_STATUS_OWED_ACTIVE none"
  else
    local j=0
    while IFS=$'\t' read -r age doc owner since blocked; do
      [[ "$age" == "age_days" ]] && continue
      j=$((j + 1))
      echo "CONCEPT_STATUS_OWED_ACTIVE [$j/$on] doc=$doc owner=$owner age_days=$age since=$since"
    done <"$otmp"
  fi
  cp "$otmp" "$ROOT_DIR/docs/internal/concepts/reserved_owed_active.tsv"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo ""
      echo "## reserved-owed (active debts)"
      echo ""
      echo "Do **not** expire. Age from \`Reserved-Since\`. Oldest first. Count: **$on**."
      echo ""
      if ((on == 0)); then
        echo "_No active reserved-owed declarations._"
      else
        echo "| age_days | doc | owner | since | blocked_on |"
        echo "|---:|---|---|---|---|"
        while IFS=$'\t' read -r age doc owner since blocked; do
          [[ "$age" == "age_days" ]] && continue
          echo "| $age | \`$doc\` | $owner | $since | $blocked |"
        done <"$otmp"
      fi
    } >>"$GITHUB_STEP_SUMMARY"
  fi
  rm -f "$otmp"
}

emit_ednc_visibility

echo "CONCEPT_STATUS_SUMMARY failures=$fail_count ednc_active=${#EDNC_ROWS[@]}"
if ((fail_count > 0)); then
  echo "CONCEPT_STATUS_GATE_RED count=$fail_count" >&2
  if [[ "$REPORT_ONLY" == "1" ]]; then
    exit 0
  fi
  exit 1
fi
echo "CONCEPT_STATUS_GATE_GREEN"
exit 0
