#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-portfolio.XXXXXX")"
STATE_DIR="$WORK/state"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-portfolio-attention: FAIL: %s\n' "$*" >&2
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

rehash_last_event_after_payload_mutation() {
  local journal="$1" mutation="$2"
  local sequence observed previous kind payload old_digest body new_digest
  IFS=$'\t' read -r sequence observed previous kind payload old_digest \
    < <(tail -n 1 "$journal")
  printf '%s' "$payload" | perl -0777 -ne 'print pack("H*", $_)' \
    > "$WORK/payload.bin"
  case "$mutation" in
    candidate-digest)
      perl -0777 -pi -e '
        $changed = s/(candidate_set_digest=)([0-9a-f])/$1 . ($2 eq "0" ? "1" : "0")/e;
        END { exit 42 unless $changed == 1 }
      ' "$WORK/payload.bin" || fail 'candidate digest mutation missed its field'
      ;;
    frontier)
      perl -0777 -pi -e '
        $changed = s/frontier=selected_ids/frontier=xelected_ids/;
        END { exit 42 unless $changed == 1 }
      ' "$WORK/payload.bin" || fail 'frontier mutation missed its field'
      ;;
    *) fail "unknown semantic mutation: $mutation" ;;
  esac
  payload="$(perl -0777 -ne 'print unpack("H*", $_)' "$WORK/payload.bin")"
  body="$(printf '%s\t%s\t%s\t%s\t%s' \
    "$sequence" "$observed" "$previous" "$kind" "$payload")"
  new_digest="$({ printf '%s\000%s' 'loom-epistemic-journal-v0' "$body"; } \
    | sha256sum | awk '{print $1}')"
  sed '$d' "$journal" > "$journal.rehashed"
  printf '%s\t%s\n' "$body" "$new_digest" >> "$journal.rehashed"
  mv "$journal.rehashed" "$journal"
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

provenance="$(digest portfolio-provenance)"
claim_evidence="$(digest portfolio-claim-evidence)"
candidate_evidence="$(digest portfolio-candidate-evidence)"
candidate_falsifier="$(digest portfolio-candidate-falsifier)"
outcome="$(digest portfolio-outcome)"

"$LOOM" world-create --state-dir "$STATE_DIR" --world target \
  --agent codex --lane portfolio-target >/dev/null
"$LOOM" knowledge-observe --state-dir "$STATE_DIR" --world target \
  --knowledge knowledge-target --value observation --error 0.01 \
  --uncertainty bounded --confidence 0.5 --provenance "$provenance" >/dev/null
"$LOOM" epistemic-claim-open --state-dir "$STATE_DIR" --world target \
  --claim claim-target --knowledge knowledge-target \
  --evidence "$claim_evidence" >/dev/null
"$LOOM" world-create --state-dir "$STATE_DIR" --world scheduler \
  --agent codex --lane portfolio-v0 >/dev/null

header='candidate_id\ttarget_world\tclaim\tprovider\tresources\tinformation\tfalsification\tdivergence\ttoken_cost\twall_cost\tgpu_cost\tquota_cost\trisk\tevidence_sha256\tfalsifier_sha256'
candidates="$WORK/candidates.tsv"
printf '%b\n' "$header" \
  "A\ttarget\tclaim-target\tclaude\t/portfolio/a\t100\t10\t10\t6\t6\t1\t1\t10\t$candidate_evidence\t$candidate_falsifier" \
  "B\ttarget\tclaim-target\tcodex\t/portfolio/b1,/portfolio/b2\t60\t20\t20\t4\t4\t1\t1\t3\t$candidate_evidence\t$candidate_falsifier" \
  "C\ttarget\tclaim-target\tkimi\t/portfolio/c1,/portfolio/c2\t60\t20\t20\t4\t4\t1\t1\t3\t$candidate_evidence\t$candidate_falsifier" \
  "D\ttarget\tclaim-target\tgrok\t/portfolio/d\t10\t150\t20\t8\t8\t2\t2\t5\t$candidate_evidence\t$candidate_falsifier" \
  "E\ttarget\tclaim-target\tminimax\t/portfolio/e\t10\t20\t150\t8\t8\t2\t2\t5\t$candidate_evidence\t$candidate_falsifier" \
  > "$candidates"

compiled="$($LOOM attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-info --candidates "$candidates" \
  --token-budget 8 --wall-budget 8 --gpu-budget 2 --quota-budget 2 \
  --policy information-first --owner codex --generation generation-1)"
rg -q 'policy=information-first selected=B,C selected_count=2 resources=4 candidates=5 enumerated=31 ' \
  <<< "$compiled" || fail "non-greedy portfolio was not selected: $compiled"

expect_refusal portfolio-to-capability-conflict \
  'epistemic-global-resource-conflict:/portfolio/b1' \
  "$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability cap-after-portfolio --resource /portfolio/b1 \
  --owner codex --generation generation-1

compiled="$($LOOM attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-falsification --candidates "$candidates" \
  --token-budget 8 --wall-budget 8 --gpu-budget 2 --quota-budget 2 \
  --policy falsification-first --owner claude --generation generation-2)"
rg -q 'policy=falsification-first selected=D selected_count=1 ' <<< "$compiled" || \
  fail "falsification-first selected the wrong portfolio: $compiled"

compiled="$($LOOM attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-counter --candidates "$candidates" \
  --token-budget 8 --wall-budget 8 --gpu-budget 2 --quota-budget 2 \
  --policy counterfactual-first --owner grok --generation generation-3)"
rg -q 'policy=counterfactual-first selected=E selected_count=1 ' <<< "$compiled" || \
  fail "counterfactual-first selected the wrong portfolio: $compiled"

capability_candidates="$WORK/capability-conflict.tsv"
printf '%b\n' "$header" \
  "CAP\ttarget\tclaim-target\tcodex\t/portfolio/capability-first\t900\t900\t900\t1\t1\t0\t0\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$capability_candidates"
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability cap-before-portfolio \
  --resource /portfolio/capability-first --owner codex \
  --generation generation-cap >/dev/null
expect_refusal capability-to-portfolio-conflict \
  'epistemic-global-resource-conflict:/portfolio/capability-first' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-capability-blocked \
  --candidates "$capability_candidates" --token-budget 1 --wall-budget 1 \
  --gpu-budget 0 --quota-budget 0 --policy information-first \
  --owner codex --generation generation-cap
"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" \
  --world scheduler --capability cap-before-portfolio --owner codex \
  --generation generation-cap >/dev/null

attention_header='candidate_id\ttarget_world\tclaim\tprovider\tresource\tinformation\tfalsification\tdivergence\tcost\trisk\tevidence_sha256\tfalsifier_sha256'
attention_candidates="$WORK/single-attention.tsv"
printf '%b\n' "$attention_header" \
  "single\ttarget\tclaim-target\tcodex\t/portfolio/single-first\t900\t900\t900\t1\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$attention_candidates"
"$LOOM" attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan single-first --candidates "$attention_candidates" --budget 1 \
  --policy information-first --owner codex --generation generation-single >/dev/null
single_conflict="$WORK/single-conflict.tsv"
printf '%b\n' "$header" \
  "SINGLE\ttarget\tclaim-target\tcodex\t/portfolio/single-first\t900\t900\t900\t1\t1\t0\t0\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$single_conflict"
expect_refusal attention-to-portfolio-conflict \
  'epistemic-global-resource-conflict:/portfolio/single-first' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-single-blocked \
  --candidates "$single_conflict" --token-budget 1 --wall-budget 1 \
  --gpu-budget 0 --quota-budget 0 --policy information-first \
  --owner codex --generation generation-single
"$LOOM" attention-complete --state-dir "$STATE_DIR" --world scheduler \
  --plan single-first --owner codex --generation generation-single \
  --outcome "$outcome" >/dev/null

portfolio_to_attention="$WORK/portfolio-to-attention.tsv"
printf '%b\n' "$attention_header" \
  "blocked\ttarget\tclaim-target\tcodex\t/portfolio/b2\t900\t900\t900\t1\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$portfolio_to_attention"
expect_refusal portfolio-to-attention-conflict \
  'epistemic-global-resource-conflict:/portfolio/b2' \
  "$LOOM" attention-compile --state-dir "$STATE_DIR" --world scheduler \
  --plan single-blocked --candidates "$portfolio_to_attention" --budget 1 \
  --policy information-first --owner codex --generation generation-single

internal="$WORK/internal-collision.tsv"
printf '%b\n' "$header" \
  "X\ttarget\tclaim-target\tcodex\t/internal/shared,/internal/x\t80\t80\t80\t2\t2\t0\t0\t1\t$candidate_evidence\t$candidate_falsifier" \
  "Y\ttarget\tclaim-target\tclaude\t/internal/shared,/internal/y\t80\t80\t80\t2\t2\t0\t0\t1\t$candidate_evidence\t$candidate_falsifier" \
  > "$internal"
compiled="$($LOOM attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-internal --candidates "$internal" \
  --token-budget 4 --wall-budget 4 --gpu-budget 0 --quota-budget 0 \
  --policy information-first --owner codex --generation generation-internal)"
rg -q 'selected=X selected_count=1 resources=2 candidates=2 enumerated=3 feasible=2 frontier=2 ' \
  <<< "$compiled" || fail "resource-colliding subset entered the frontier: $compiled"
"$LOOM" attention-portfolio-complete --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-internal --owner codex \
  --generation generation-internal --outcome "$outcome" >/dev/null

atomic="$WORK/atomic.tsv"
printf '%b\n' "$header" \
  "ATOMIC\ttarget\tclaim-target\tcodex\t/atomic/busy,/atomic/free\t900\t900\t900\t1\t1\t0\t0\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$atomic"
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability atomic-busy --resource /atomic/busy \
  --owner codex --generation generation-atomic >/dev/null
expect_refusal atomic-reservation-conflict \
  'epistemic-global-resource-conflict:/atomic/busy' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-atomic --candidates "$atomic" \
  --token-budget 1 --wall-budget 1 --gpu-budget 0 --quota-budget 0 \
  --policy information-first --owner codex --generation generation-atomic
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability atomic-free --resource /atomic/free \
  --owner codex --generation generation-atomic >/dev/null
"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" \
  --world scheduler --capability atomic-busy --owner codex \
  --generation generation-atomic >/dev/null
"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" \
  --world scheduler --capability atomic-free --owner codex \
  --generation generation-atomic >/dev/null

budget_candidate="$WORK/budget.tsv"
printf '%b\n' "$header" \
  "BUDGET\ttarget\tclaim-target\tcodex\t/budget/all\t900\t900\t900\t9\t9\t3\t3\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$budget_candidate"
expect_refusal token-budget 'epistemic-portfolio-no-feasible-subset' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio budget-token --candidates "$budget_candidate" \
  --token-budget 8 --wall-budget 9 --gpu-budget 3 --quota-budget 3 \
  --policy information-first --owner codex --generation generation-budget
expect_refusal wall-budget 'epistemic-portfolio-no-feasible-subset' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio budget-wall --candidates "$budget_candidate" \
  --token-budget 9 --wall-budget 8 --gpu-budget 3 --quota-budget 3 \
  --policy information-first --owner codex --generation generation-budget
expect_refusal gpu-budget 'epistemic-portfolio-no-feasible-subset' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio budget-gpu --candidates "$budget_candidate" \
  --token-budget 9 --wall-budget 9 --gpu-budget 2 --quota-budget 3 \
  --policy information-first --owner codex --generation generation-budget
expect_refusal quota-budget 'epistemic-portfolio-no-feasible-subset' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio budget-quota --candidates "$budget_candidate" \
  --token-budget 9 --wall-budget 9 --gpu-budget 3 --quota-budget 2 \
  --policy information-first --owner codex --generation generation-budget

pathological="$WORK/pathological-frontier.tsv"
printf '%b\n' "$header" > "$pathological"
for index in 1 2 3 4 5 6 7 8 9; do
  printf 'P%s\ttarget\tclaim-target\tcodex\t/pathological/%s\t1\t1\t1\t1\t1\t0\t0\t0\t%s\t%s\n' \
    "$index" "$index" "$candidate_evidence" "$candidate_falsifier" \
    >> "$pathological"
done
expect_refusal pathological-frontier \
  'epistemic-portfolio-frontier-limit-exceeded:257' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-pathological \
  --candidates "$pathological" --token-budget 9 --wall-budget 9 \
  --gpu-budget 0 --quota-budget 0 --policy information-first \
  --owner codex --generation generation-pathological

absent="$WORK/absent.tsv"
printf '%b\n' "$header" \
  "ABSENT\ttarget\tclaim-absent\tcodex\t/portfolio/absent\t900\t900\t900\t1\t1\t0\t0\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$absent"
expect_refusal absent-claim \
  'epistemic-attention-target-claim-missing:target/claim-absent' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-absent --candidates "$absent" \
  --token-budget 1 --wall-budget 1 --gpu-budget 0 --quota-budget 0 \
  --policy information-first --owner codex --generation generation-absent

malformed="$WORK/malformed.tsv"
printf 'wrong-header\n' > "$malformed"
expect_refusal malformed-candidates 'epistemic-portfolio-candidate-header-invalid' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-malformed --candidates "$malformed" \
  --token-budget 1 --wall-budget 1 --gpu-budget 0 --quota-budget 0 \
  --policy information-first --owner codex --generation generation-malformed

empty_resource="$WORK/empty-resource.tsv"
printf '%b\n' "$header" \
  "EMPTY\ttarget\tclaim-target\tcodex\t\t900\t900\t900\t1\t1\t0\t0\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$empty_resource"
expect_refusal empty-resource 'epistemic-portfolio-resources-invalid' \
  "$LOOM" attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-empty-resource \
  --candidates "$empty_resource" --token-budget 1 --wall-budget 1 \
  --gpu-budget 0 --quota-budget 0 --policy information-first \
  --owner codex --generation generation-empty-resource

expect_refusal completion-identity-drift \
  'epistemic-portfolio-completion-identity-drift:portfolio-info' \
  "$LOOM" attention-portfolio-complete --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-info --owner intruder \
  --generation generation-1 --outcome "$outcome"

"$LOOM" attention-portfolio-complete --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-info --owner codex \
  --generation generation-1 --outcome "$outcome" >/dev/null
compiled="$($LOOM attention-portfolio-compile --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-info-reuse --candidates "$candidates" \
  --token-budget 8 --wall-budget 8 --gpu-budget 2 --quota-budget 2 \
  --policy information-first --owner codex --generation generation-4)"
rg -q 'selected=B,C selected_count=2 resources=4 ' <<< "$compiled" || \
  fail 'completion did not release the entire selected resource union'

"$LOOM" attention-portfolio-complete --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-falsification --owner claude \
  --generation generation-2 --outcome "$outcome" >/dev/null
"$LOOM" attention-portfolio-complete --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-counter --owner grok \
  --generation generation-3 --outcome "$outcome" >/dev/null
"$LOOM" attention-portfolio-complete --state-dir "$STATE_DIR" \
  --world scheduler --portfolio portfolio-info-reuse --owner codex \
  --generation generation-4 --outcome "$outcome" >/dev/null

status="$($LOOM world-status --state-dir "$STATE_DIR" --world scheduler)"
rg -q 'attention_plans=1 live_attention=0 attention_portfolios=5 live_portfolios=0 ' \
  <<< "$status" || fail "portfolio state is wrong: $status"

"$LOOM" export-events-arrow --state-dir "$STATE_DIR" \
  --out "$WORK/portfolio.arrow" > "$WORK/export.out"
rg -q 'authority=verified-derived rows=22 ' "$WORK/export.out" || {
  sed -n '1,120p' "$WORK/export.out" >&2
  fail 'Arrow projection did not contain the twenty-two verified events'
}
inspect="$($LOOM verify-events-arrow --file "$WORK/portfolio.arrow")"
rg -q 'schema=loom-spectral-events-v1 rows=22 batches=1' <<< "$inspect" || \
  fail "native Arrow reader disagreed: $inspect"

TAMPER_STATE="$WORK/tamper-base"
"$LOOM" world-create --state-dir "$TAMPER_STATE" --world tamper-target \
  --agent codex --lane tamper-target >/dev/null
"$LOOM" knowledge-observe --state-dir "$TAMPER_STATE" --world tamper-target \
  --knowledge tamper-knowledge --value tamper --error 0.01 \
  --uncertainty bounded --confidence 0.5 --provenance "$provenance" >/dev/null
"$LOOM" epistemic-claim-open --state-dir "$TAMPER_STATE" --world tamper-target \
  --claim tamper-claim --knowledge tamper-knowledge \
  --evidence "$claim_evidence" >/dev/null
"$LOOM" world-create --state-dir "$TAMPER_STATE" --world tamper-scheduler \
  --agent codex --lane tamper-scheduler >/dev/null
tamper_candidates="$WORK/tamper.tsv"
printf '%b\n' "$header" \
  "T\ttamper-target\ttamper-claim\tcodex\t/tamper/resource\t900\t900\t900\t1\t1\t0\t0\t0\t$candidate_evidence\t$candidate_falsifier" \
  > "$tamper_candidates"
"$LOOM" attention-portfolio-compile --state-dir "$TAMPER_STATE" \
  --world tamper-scheduler --portfolio tamper-portfolio \
  --candidates "$tamper_candidates" --token-budget 1 --wall-budget 1 \
  --gpu-budget 0 --quota-budget 0 --policy information-first \
  --owner codex --generation tamper-generation >/dev/null

cp -a "$TAMPER_STATE" "$WORK/candidate-tamper"
candidate_journal="$WORK/candidate-tamper/loom-epistemic/worlds/tamper-scheduler/journal.tsv"
rehash_last_event_after_payload_mutation "$candidate_journal" candidate-digest
expect_refusal candidate-set-sabotage \
  'epistemic-portfolio-candidate-set-digest-mismatch:tamper-portfolio' \
  "$LOOM" world-verify --state-dir "$WORK/candidate-tamper" \
  --world tamper-scheduler
expect_refusal arrow-semantic-laundering \
  'epistemic-portfolio-candidate-set-digest-mismatch:tamper-portfolio' \
  "$LOOM" export-events-arrow --state-dir "$WORK/candidate-tamper" \
  --out "$WORK/semantic-laundered.arrow"
[[ ! -e "$WORK/semantic-laundered.arrow" ]] || \
  fail 'semantic portfolio sabotage produced Arrow output'

cp -a "$TAMPER_STATE" "$WORK/frontier-tamper"
frontier_journal="$WORK/frontier-tamper/loom-epistemic/worlds/tamper-scheduler/journal.tsv"
rehash_last_event_after_payload_mutation "$frontier_journal" frontier
expect_refusal frontier-sabotage \
  'epistemic-portfolio-frontier-mismatch:tamper-portfolio' \
  "$LOOM" world-verify --state-dir "$WORK/frontier-tamper" \
  --world tamper-scheduler

cp -a "$TAMPER_STATE" "$WORK/journal-tamper"
raw_journal="$WORK/journal-tamper/loom-epistemic/worlds/tamper-scheduler/journal.tsv"
awk -F '\t' 'BEGIN { OFS="\t" }
  NR == 2 { $6 = (substr($6, 1, 1) == "0" ? "1" : "0") substr($6, 2) }
  { print }
' "$raw_journal" > "$raw_journal.tampered"
mv "$raw_journal.tampered" "$raw_journal"
expect_refusal journal-sabotage 'epistemic-journal-event-digest-mismatch:seq=2' \
  "$LOOM" world-verify --state-dir "$WORK/journal-tamper" \
  --world tamper-scheduler

printf 'SOUNIO_LOOM_PORTFOLIO_ATTENTION_GATE_PASS=true schema=loom-pareto-portfolio-v0 frame=9010 candidates=5 subsets=31 policies=3 budgets=4 non_greedy=B,C atomic_resources=4 internal_collision=REFUSED external_conflicts=BIDIRECTIONAL partial_reservation=NONE frontier_cap=REFUSED replay=RECOMPUTED candidate_sabotage=REFUSED frontier_sabotage=REFUSED journal_sabotage=REFUSED arrow_laundering=REFUSED rows=22 runtime=OCaml+Sounio\n'
