#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-attention-compiler.XXXXXX")"
STATE_DIR="$WORK/state"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-attention-compiler: FAIL: %s\n' "$*" >&2
  exit 1
}

digest() {
  printf '%s' "$1" | sha256sum | awk '{print $1}'
}

expect_refusal() {
  local label="$1" expected="$2"
  shift 2
  local rc=0
  set +e
  "$@" >"$WORK/$label.out" 2>"$WORK/$label.err"
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || fail "$label returned rc=$rc"
  rg -q "$expected" "$WORK/$label.err" || {
    sed -n '1,180p' "$WORK/$label.err" >&2
    fail "$label was refused by an unrelated rule"
  }
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

provenance="$(digest attention-provenance)"
claim_evidence="$(digest claim-evidence)"
candidate_evidence="$(digest candidate-evidence)"
candidate_falsifier="$(digest candidate-falsifier)"
outcome="$(digest attention-outcome)"

"$LOOM" world-create --state-dir "$STATE_DIR" --world scheduler \
  --agent codex --lane attention-v0 >/dev/null

for world in info falsify counter; do
  "$LOOM" world-create --state-dir "$STATE_DIR" --world "$world" \
    --agent codex --lane "attention-$world" >/dev/null
  "$LOOM" knowledge-observe --state-dir "$STATE_DIR" --world "$world" \
    --knowledge "knowledge-$world" --value "observation-$world" --error 0.01 \
    --uncertainty bounded --confidence 0.5 --provenance "$provenance" >/dev/null
  "$LOOM" epistemic-claim-open --state-dir "$STATE_DIR" --world "$world" \
    --claim "claim-$world" --knowledge "knowledge-$world" \
    --evidence "$claim_evidence" >/dev/null
done

header='candidate_id\ttarget_world\tclaim\tprovider\tresource\tinformation\tfalsification\tdivergence\tcost\trisk\tevidence_sha256\tfalsifier_sha256'
candidates="$WORK/candidates.tsv"
printf '%b\n' "$header" \
  "candidate-info\tinfo\tclaim-info\tcodex\t/attention/info\t900\t400\t300\t40\t100\t$candidate_evidence\t$candidate_falsifier" \
  "candidate-falsify\tfalsify\tclaim-falsify\tclaude\t/attention/falsify\t500\t950\t400\t50\t80\t$candidate_evidence\t$candidate_falsifier" \
  "candidate-counter\tcounter\tclaim-counter\tgrok\t/attention/counter\t400\t500\t980\t60\t120\t$candidate_evidence\t$candidate_falsifier" \
  > "$candidates"

compiled="$($LOOM attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-info --candidates "$candidates" --budget 100 \
  --policy information-first --owner codex --generation generation-1)"
rg -q 'policy=information-first selected=candidate-info ' <<< "$compiled" || \
  fail 'information-first selected the wrong candidate'

expect_refusal attention-to-capability-conflict 'epistemic-global-resource-conflict:/attention/info' \
  "$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability cap-conflict --resource /attention/info \
  --owner codex --generation generation-1

compiled="$($LOOM attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-falsification --candidates "$candidates" --budget 100 \
  --policy falsification-first --owner claude --generation generation-2)"
rg -q 'policy=falsification-first selected=candidate-falsify ' <<< "$compiled" || \
  fail 'falsification-first selected the wrong candidate'

"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability cap-counter --resource /attention/counter \
  --owner codex --generation generation-1 >/dev/null
expect_refusal capability-to-attention-conflict 'epistemic-global-resource-conflict:/attention/counter' \
  "$LOOM" attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-counter-blocked --candidates "$candidates" --budget 100 \
  --policy counterfactual-first --owner grok --generation generation-3
"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" \
  --world scheduler --capability cap-counter --owner codex \
  --generation generation-1 >/dev/null

compiled="$($LOOM attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-counter --candidates "$candidates" --budget 100 \
  --policy counterfactual-first --owner grok --generation generation-3)"
rg -q 'policy=counterfactual-first selected=candidate-counter ' <<< "$compiled" || \
  fail 'counterfactual-first selected the wrong candidate'

expect_refusal duplicate-live-resource 'epistemic-global-resource-conflict:/attention/info' \
  "$LOOM" attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-info-conflict --candidates "$candidates" --budget 100 \
  --policy information-first --owner codex --generation generation-4

absent="$WORK/absent.tsv"
printf '%b\n' "$header" \
  "candidate-absent\tinfo\tclaim-absent\tcodex\t/attention/absent\t900\t900\t900\t10\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$absent"
expect_refusal absent-claim 'epistemic-attention-target-claim-missing:info/claim-absent' \
  "$LOOM" attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-absent --candidates "$absent" --budget 100 \
  --policy information-first --owner codex --generation generation-4

over_budget="$WORK/over-budget.tsv"
printf '%b\n' "$header" \
  "candidate-expensive\tinfo\tclaim-info\tcodex\t/attention/expensive\t900\t900\t900\t101\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$over_budget"
expect_refusal over-budget-only 'epistemic-attention-no-feasible-candidate' \
  "$LOOM" attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-expensive --candidates "$over_budget" --budget 100 \
  --policy information-first --owner codex --generation generation-4

malformed="$WORK/malformed.tsv"
printf 'wrong-header\n' > "$malformed"
expect_refusal malformed-candidates 'epistemic-attention-candidate-header-invalid' \
  "$LOOM" attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-malformed --candidates "$malformed" --budget 100 \
  --policy information-first --owner codex --generation generation-4

expect_refusal completion-identity-drift 'epistemic-attention-completion-identity-drift:plan-info' \
  "$LOOM" attention-complete --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-info --owner intruder --generation generation-1 --outcome "$outcome"

"$LOOM" attention-complete --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-info --owner codex --generation generation-1 \
  --outcome "$outcome" >/dev/null
compiled="$($LOOM attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-info-reuse --candidates "$candidates" --budget 100 \
  --policy information-first --owner codex --generation generation-4)"
rg -q 'selected=candidate-info ' <<< "$compiled" || \
  fail 'completed plan did not release its resource'

"$LOOM" attention-complete --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-falsification --owner claude --generation generation-2 \
  --outcome "$outcome" >/dev/null
"$LOOM" attention-complete --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-counter --owner grok --generation generation-3 \
  --outcome "$outcome" >/dev/null
"$LOOM" attention-complete --state-dir "$STATE_DIR" --world scheduler \
  --plan plan-info-reuse --owner codex --generation generation-4 \
  --outcome "$outcome" >/dev/null

status="$($LOOM world-status --state-dir "$STATE_DIR" --world scheduler)"
rg -q 'attention_plans=4 live_attention=0 ' <<< "$status" || \
  fail "attention plan state is wrong: $status"

"$LOOM" export-events-arrow --state-dir "$STATE_DIR" \
  --out "$WORK/attention.arrow" > "$WORK/export.out"
rg -q 'authority=verified-derived rows=20 ' "$WORK/export.out" || {
  sed -n '1,120p' "$WORK/export.out" >&2
  fail 'Arrow projection did not contain the twenty verified events'
}
inspect="$($LOOM verify-events-arrow --file "$WORK/attention.arrow")"
rg -q 'schema=loom-spectral-events-v1 rows=20 batches=1' <<< "$inspect" || \
  fail "native Arrow reader disagreed: $inspect"

scheduler_journal="$STATE_DIR/loom-epistemic/worlds/scheduler/journal.tsv"
awk -F '\t' 'BEGIN { OFS="\t" }
  NR == 2 { $6 = (substr($6, 1, 1) == "0" ? "1" : "0") substr($6, 2) }
  { print }
' "$scheduler_journal" > "$scheduler_journal.tampered"
mv "$scheduler_journal.tampered" "$scheduler_journal"

expect_refusal journal-sabotage 'epistemic-journal-event-digest-mismatch:seq=2' \
  "$LOOM" world-verify --state-dir "$STATE_DIR" --world scheduler
expect_refusal arrow-laundering 'epistemic-journal-event-digest-mismatch:seq=2' \
  "$LOOM" export-events-arrow --state-dir "$STATE_DIR" \
  --out "$WORK/laundered.arrow"
[[ ! -e "$WORK/laundered.arrow" ]] || fail 'journal sabotage produced Arrow output'

printf 'SOUNIO_LOOM_ATTENTION_COMPILER_GATE_PASS=true schema=loom-attention-compiler-v0 policies=3 candidates=3 plans=4 events=20 comparator=DETERMINISTIC resources=LINEAR completion=RELEASES journal_sabotage=REFUSED arrow_laundering=REFUSED runtime=OCaml+Sounio\n'
