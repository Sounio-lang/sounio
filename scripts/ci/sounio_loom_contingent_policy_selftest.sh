#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-contingent.XXXXXX")"
STATE_DIR="$WORK/state"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-contingent-policy: FAIL: %s\n' "$*" >&2
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
    sed -n '1,220p' "$WORK/$label.err" >&2
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
    action-digest)
      perl -0777 -pi -e '
        $changed = s/(action_set_digest=)([0-9a-f])/$1 . ($2 eq "0" ? "1" : "0")/e;
        END { exit 42 unless $changed == 1 }
      ' "$WORK/payload.bin" || fail 'action digest mutation missed its field'
      ;;
    outcome-digest)
      perl -0777 -pi -e '
        $changed = s/(outcome_set_digest=)([0-9a-f])/$1 . ($2 eq "0" ? "1" : "0")/e;
        END { exit 42 unless $changed == 1 }
      ' "$WORK/payload.bin" || fail 'outcome digest mutation missed its field'
      ;;
    frontier)
      perl -0777 -pi -e '
        $changed = s/frontier=policy_sha256/frontier=xolicy_sha256/;
        END { exit 42 unless $changed == 1 }
      ' "$WORK/payload.bin" || fail 'frontier mutation missed its field'
      ;;
    selected-policy)
      perl -0777 -pi -e '
        $changed = s/selected_policy=path/selected_policy=xath/;
        END { exit 42 unless $changed == 1 }
      ' "$WORK/payload.bin" || fail 'selected policy mutation missed its field'
      ;;
    route)
      perl -0777 -pi -e '
        $changed = s/next_action=B/next_action=C/;
        END { exit 42 unless $changed == 1 }
      ' "$WORK/payload.bin" || fail 'route mutation missed its field'
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

provenance="$(digest contingent-provenance)"
claim_evidence="$(digest contingent-claim-evidence)"
action_evidence="$(digest contingent-action-evidence)"
action_falsifier="$(digest contingent-action-falsifier)"
branch_evidence="$(digest contingent-branch-evidence)"
outcome_x="$(digest contingent-outcome-x)"
outcome_y="$(digest contingent-outcome-y)"
outcome_done="$(digest contingent-outcome-done)"

"$LOOM" world-create --state-dir "$STATE_DIR" --world target \
  --agent codex --lane contingent-target >/dev/null
"$LOOM" knowledge-observe --state-dir "$STATE_DIR" --world target \
  --knowledge knowledge-target --value observation --error 0.01 \
  --uncertainty bounded --confidence 0.5 --provenance "$provenance" >/dev/null
"$LOOM" epistemic-claim-open --state-dir "$STATE_DIR" --world target \
  --claim claim-target --knowledge knowledge-target \
  --evidence "$claim_evidence" >/dev/null
"$LOOM" world-create --state-dir "$STATE_DIR" --world scheduler \
  --agent codex --lane contingent-v0 >/dev/null

action_header='state\taction_id\ttarget_world\tclaim\tprovider\tresource\tinformation\tfalsification\tdivergence\ttoken_cost\twall_cost\tgpu_cost\tquota_cost\trisk\tevidence_sha256\tfalsifier_sha256'
outcome_header='action_id\tvariant_index\tvariant_count\toutcome_id\tsuccessor_state\tbranch_evidence_sha256'
actions="$WORK/actions.tsv"
outcomes="$WORK/outcomes.tsv"
printf '%b\n' "$action_header" \
  "root\tA\ttarget\tclaim-target\tcodex\t/cont/a\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  "root\tD\ttarget\tclaim-target\tclaude\t/cont/d\t20\t150\t20\t2\t2\t0\t0\t2\t$action_evidence\t$action_falsifier" \
  "root\tE\ttarget\tclaim-target\tgrok\t/cont/e\t20\t20\t150\t2\t2\t0\t0\t2\t$action_evidence\t$action_falsifier" \
  "state-x\tB\ttarget\tclaim-target\tkimi\t/cont/b\t100\t30\t30\t2\t2\t0\t0\t2\t$action_evidence\t$action_falsifier" \
  "state-y\tC\ttarget\tclaim-target\tminimax\t/cont/c\t120\t40\t40\t3\t3\t0\t0\t3\t$action_evidence\t$action_falsifier" \
  > "$actions"
printf '%b\n' "$outcome_header" \
  "A\t0\t2\tX\tstate-x\t$branch_evidence" \
  "A\t1\t2\tY\tstate-y\t$branch_evidence" \
  "B\t0\t1\tDONE\t-\t$branch_evidence" \
  "C\t0\t1\tDONE\t-\t$branch_evidence" \
  "D\t0\t1\tDONE\t-\t$branch_evidence" \
  "E\t0\t1\tDONE\t-\t$branch_evidence" \
  > "$outcomes"

compiled="$($LOOM contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-info --root-state root \
  --actions "$actions" --outcomes "$outcomes" --token-budget 10 \
  --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation generation-1)"
rg -q 'root_action=A current_resource=/cont/a states=3 actions=5 outcomes=6 .*nodes=3 guaranteed_information=110 guaranteed_falsification=40 guaranteed_divergence=40 worst_risk=4 token=4/10 wall=4/10 ' \
  <<< "$compiled" || fail "robust branch-conditioned policy was wrong: $compiled"
if rg -q 'guaranteed_information=230|token=6/10|current_resource=/cont/b|current_resource=/cont/c' \
  <<< "$compiled"; then
  fail 'mutually exclusive branches were laundered into an open-loop sum/reservation'
fi

# A future branch resource is not reserved before its outcome exists.
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability future-c --resource /cont/c \
  --owner codex --generation generation-1 >/dev/null
"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" \
  --world scheduler --capability future-c --owner codex \
  --generation generation-1 >/dev/null

# A failed handoff keeps A reserved and never partially reserves/releases.
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability busy-b --resource /cont/b \
  --owner blocker --generation generation-busy >/dev/null
expect_refusal atomic-next-resource-conflict \
  'epistemic-global-resource-conflict:/cont/b' \
  "$LOOM" contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-info --outcome X \
  --owner codex --generation generation-1 --outcome-digest "$outcome_x"
expect_refusal current-resource-still-owned \
  'epistemic-global-resource-conflict:/cont/a' \
  "$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability steal-a --resource /cont/a \
  --owner intruder --generation generation-intruder
"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" \
  --world scheduler --capability busy-b --owner blocker \
  --generation generation-busy >/dev/null

compiled="$($LOOM contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-falsification \
  --root-state root --actions "$actions" --outcomes "$outcomes" \
  --token-budget 10 --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order falsification-first --owner claude --generation generation-2)"
rg -q 'root_action=D current_resource=/cont/d ' <<< "$compiled" || \
  fail "falsification order chose the wrong root policy: $compiled"
compiled="$($LOOM contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-counter --root-state root \
  --actions "$actions" --outcomes "$outcomes" --token-budget 10 \
  --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order counterfactual-first --owner grok --generation generation-3)"
rg -q 'root_action=E current_resource=/cont/e ' <<< "$compiled" || \
  fail "counterfactual order chose the wrong root policy: $compiled"

expect_refusal wrong-owner \
  'epistemic-contingent-observation-identity-drift:policy-info' \
  "$LOOM" contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-info --outcome X \
  --owner intruder --generation generation-1 --outcome-digest "$outcome_x"
expect_refusal wrong-outcome \
  'epistemic-contingent-outcome-not-in-partition:A/UNKNOWN' \
  "$LOOM" contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-info --outcome UNKNOWN \
  --owner codex --generation generation-1 --outcome-digest "$outcome_x"

advanced="$($LOOM contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-info --outcome X \
  --owner codex --generation generation-1 --outcome-digest "$outcome_x")"
rg -q 'state=advanced path=root current_action=A outcome=X next_path=root.0 next_action=B released_resource=/cont/a next_resource=/cont/b ' \
  <<< "$advanced" || fail "X did not route atomically to B: $advanced"
"$LOOM" contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-falsification --outcome DONE \
  --owner claude --generation generation-2 --outcome-digest "$outcome_done" \
  >/dev/null
"$LOOM" contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-counter --outcome DONE \
  --owner grok --generation generation-3 --outcome-digest "$outcome_done" \
  >/dev/null
completed="$($LOOM contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-info --outcome DONE \
  --owner codex --generation generation-1 --outcome-digest "$outcome_done")"
rg -q 'state=completed path=root.0 current_action=B outcome=DONE .*released_resource=/cont/b next_resource=- ' \
  <<< "$completed" || fail "terminal B did not release its resource: $completed"

# Completion releases both the historical and final resources.
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability reuse-a --resource /cont/a \
  --owner codex --generation generation-4 >/dev/null
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" \
  --world scheduler --capability reuse-b --resource /cont/b \
  --owner codex --generation generation-4 >/dev/null
"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" \
  --world scheduler --capability reuse-a --owner codex \
  --generation generation-4 >/dev/null
"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" \
  --world scheduler --capability reuse-b --owner codex \
  --generation generation-4 >/dev/null

# The other nominal branch is separately executable and routes only to C.
"$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-y --root-state root \
  --actions "$actions" --outcomes "$outcomes" --token-budget 10 \
  --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation generation-5 >/dev/null
advanced="$($LOOM contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-y --outcome Y \
  --owner codex --generation generation-5 --outcome-digest "$outcome_y")"
rg -q 'outcome=Y next_path=root.1 next_action=C released_resource=/cont/a next_resource=/cont/c ' \
  <<< "$advanced" || fail "Y did not route atomically to C: $advanced"
"$LOOM" contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-y --outcome DONE \
  --owner codex --generation generation-5 --outcome-digest "$outcome_done" \
  >/dev/null

status="$($LOOM world-status --state-dir "$STATE_DIR" --world scheduler)"
rg -q 'contingent_policies=4 live_contingent=0 ' <<< "$status" || \
  fail "contingent policy state is wrong: $status"

# Closed nominal partitions are structural, not comments.
incomplete="$WORK/incomplete.tsv"
printf '%b\n' "$outcome_header" \
  "A\t0\t2\tX\tstate-x\t$branch_evidence" \
  "B\t0\t1\tDONE\t-\t$branch_evidence" \
  "C\t0\t1\tDONE\t-\t$branch_evidence" \
  "D\t0\t1\tDONE\t-\t$branch_evidence" \
  "E\t0\t1\tDONE\t-\t$branch_evidence" > "$incomplete"
expect_refusal incomplete-partition \
  'epistemic-contingent-partition-incomplete:A' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy incomplete --root-state root \
  --actions "$actions" --outcomes "$incomplete" --token-budget 10 \
  --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation negative

duplicate="$WORK/duplicate.tsv"
printf '%b\n' "$outcome_header" \
  "A\t0\t2\tX\tstate-x\t$branch_evidence" \
  "A\t0\t2\tY\tstate-y\t$branch_evidence" \
  "B\t0\t1\tDONE\t-\t$branch_evidence" \
  "C\t0\t1\tDONE\t-\t$branch_evidence" \
  "D\t0\t1\tDONE\t-\t$branch_evidence" \
  "E\t0\t1\tDONE\t-\t$branch_evidence" > "$duplicate"
expect_refusal duplicate-variant \
  'epistemic-contingent-partition-noncanonical:A' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy duplicate --root-state root \
  --actions "$actions" --outcomes "$duplicate" --token-budget 10 \
  --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation negative

cycle_actions="$WORK/cycle-actions.tsv"
cycle_outcomes="$WORK/cycle-outcomes.tsv"
printf '%b\n' "$action_header" \
  "cycle\tCA\ttarget\tclaim-target\tcodex\t/cycle/a\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  > "$cycle_actions"
printf '%b\n' "$outcome_header" \
  "CA\t0\t1\tAGAIN\tcycle\t$branch_evidence" > "$cycle_outcomes"
expect_refusal graph-cycle 'epistemic-contingent-graph-cycle:cycle' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy cycle --root-state cycle \
  --actions "$cycle_actions" --outcomes "$cycle_outcomes" \
  --token-budget 10 --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation negative

unreachable_actions="$WORK/unreachable-actions.tsv"
unreachable_outcomes="$WORK/unreachable-outcomes.tsv"
printf '%b\n' "$action_header" \
  "visible\tVA\ttarget\tclaim-target\tcodex\t/visible/a\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  "hidden\tHA\ttarget\tclaim-target\tcodex\t/hidden/a\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  > "$unreachable_actions"
printf '%b\n' "$outcome_header" \
  "HA\t0\t1\tDONE\t-\t$branch_evidence" \
  "VA\t0\t1\tDONE\t-\t$branch_evidence" > "$unreachable_outcomes"
expect_refusal graph-unreachable \
  'epistemic-contingent-state-unreachable:hidden' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy unreachable --root-state visible \
  --actions "$unreachable_actions" --outcomes "$unreachable_outcomes" \
  --token-budget 10 --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation negative

absent_actions="$WORK/absent-actions.tsv"
printf '%b\n' "$action_header" \
  "absent\tAA\ttarget\tclaim-absent\tcodex\t/absent/a\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  > "$absent_actions"
absent_outcomes="$WORK/absent-outcomes.tsv"
printf '%b\n' "$outcome_header" \
  "AA\t0\t1\tDONE\t-\t$branch_evidence" > "$absent_outcomes"
expect_refusal absent-claim \
  'epistemic-attention-target-claim-missing:target/claim-absent' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy absent --root-state absent \
  --actions "$absent_actions" --outcomes "$absent_outcomes" \
  --token-budget 10 --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation negative

budget_actions="$WORK/budget-actions.tsv"
budget_outcomes="$WORK/budget-outcomes.tsv"
printf '%b\n' "$action_header" \
  "budget\tBA\ttarget\tclaim-target\tcodex\t/budget/a\t10\t10\t10\t9\t9\t3\t3\t1\t$action_evidence\t$action_falsifier" \
  > "$budget_actions"
printf '%b\n' "$outcome_header" \
  "BA\t0\t1\tDONE\t-\t$branch_evidence" > "$budget_outcomes"
expect_refusal token-budget 'epistemic-contingent-no-feasible-policy' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy budget-token --root-state budget \
  --actions "$budget_actions" --outcomes "$budget_outcomes" \
  --token-budget 8 --wall-budget 9 --gpu-budget 3 --quota-budget 3 \
  --order information-first --owner codex --generation negative
expect_refusal wall-budget 'epistemic-contingent-no-feasible-policy' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy budget-wall --root-state budget \
  --actions "$budget_actions" --outcomes "$budget_outcomes" \
  --token-budget 9 --wall-budget 8 --gpu-budget 3 --quota-budget 3 \
  --order information-first --owner codex --generation negative
expect_refusal gpu-budget 'epistemic-contingent-no-feasible-policy' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy budget-gpu --root-state budget \
  --actions "$budget_actions" --outcomes "$budget_outcomes" \
  --token-budget 9 --wall-budget 9 --gpu-budget 2 --quota-budget 3 \
  --order information-first --owner codex --generation negative
expect_refusal quota-budget 'epistemic-contingent-no-feasible-policy' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy budget-quota --root-state budget \
  --actions "$budget_actions" --outcomes "$budget_outcomes" \
  --token-budget 9 --wall-budget 9 --gpu-budget 3 --quota-budget 2 \
  --order information-first --owner codex --generation negative

# Seven child choices across three mutually exclusive branches yield 7^3
# exact history-conditioned trees. The 257th skyline member refuses.
pathological_actions="$WORK/pathological-actions.tsv"
pathological_outcomes="$WORK/pathological-outcomes.tsv"
printf '%b\n' "$action_header" \
  "proot\tPA\ttarget\tclaim-target\tcodex\t/path/root\t10\t10\t10\t1\t1\t0\t0\t0\t$action_evidence\t$action_falsifier" \
  > "$pathological_actions"
for index in 1 2 3 4 5 6 7; do
  printf '%b\n' \
    "child\tP$index\ttarget\tclaim-target\tcodex\t/path/$index\t101\t100\t100\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
    >> "$pathological_actions"
done
printf '%b\n' "$outcome_header" \
  "PA\t0\t3\tO0\tchild\t$branch_evidence" \
  "PA\t1\t3\tO1\tchild\t$branch_evidence" \
  "PA\t2\t3\tO2\tchild\t$branch_evidence" \
  > "$pathological_outcomes"
for index in 1 2 3 4 5 6 7; do
  printf '%b\n' "P$index\t0\t1\tDONE\t-\t$branch_evidence" \
    >> "$pathological_outcomes"
done
expect_refusal pathological-frontier \
  'epistemic-contingent-frontier-limit-exceeded:proot:257' \
  "$LOOM" contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy pathological --root-state proot \
  --actions "$pathological_actions" --outcomes "$pathological_outcomes" \
  --token-budget 100 --wall-budget 100 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation negative

# The machine lock serializes load, next-resource precheck, append, fsync, and
# replay. Two independent policies racing for one free successor therefore
# behave as one compare-and-swap: exactly one advances and the loser appends
# nothing while retaining its current resource.
RACE_STATE="$WORK/race-state"
"$LOOM" world-create --state-dir "$RACE_STATE" --world race-target \
  --agent codex --lane race-target >/dev/null
"$LOOM" knowledge-observe --state-dir "$RACE_STATE" --world race-target \
  --knowledge k --value v --error 0.01 --uncertainty bounded \
  --confidence 0.5 --provenance "$provenance" >/dev/null
"$LOOM" epistemic-claim-open --state-dir "$RACE_STATE" \
  --world race-target --claim c --knowledge k \
  --evidence "$claim_evidence" >/dev/null
"$LOOM" world-create --state-dir "$RACE_STATE" --world race-scheduler \
  --agent codex --lane race-scheduler >/dev/null
race_actions_one="$WORK/race-actions-one.tsv"
race_actions_two="$WORK/race-actions-two.tsv"
race_outcomes_one="$WORK/race-outcomes-one.tsv"
race_outcomes_two="$WORK/race-outcomes-two.tsv"
printf '%b\n' "$action_header" \
  "root-one\tA1\trace-target\tc\tcodex\t/race/root-one\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  "child-one\tB1\trace-target\tc\tcodex\t/race/shared\t20\t20\t20\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  > "$race_actions_one"
printf '%b\n' "$action_header" \
  "root-two\tA2\trace-target\tc\tclaude\t/race/root-two\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  "child-two\tB2\trace-target\tc\tclaude\t/race/shared\t20\t20\t20\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  > "$race_actions_two"
printf '%b\n' "$outcome_header" \
  "A1\t0\t1\tGO\tchild-one\t$branch_evidence" \
  "B1\t0\t1\tDONE\t-\t$branch_evidence" > "$race_outcomes_one"
printf '%b\n' "$outcome_header" \
  "A2\t0\t1\tGO\tchild-two\t$branch_evidence" \
  "B2\t0\t1\tDONE\t-\t$branch_evidence" > "$race_outcomes_two"
"$LOOM" contingent-policy-compile --state-dir "$RACE_STATE" \
  --world race-scheduler --contingent-policy race-one --root-state root-one \
  --actions "$race_actions_one" --outcomes "$race_outcomes_one" \
  --token-budget 10 --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation race-one >/dev/null
"$LOOM" contingent-policy-compile --state-dir "$RACE_STATE" \
  --world race-scheduler --contingent-policy race-two --root-state root-two \
  --actions "$race_actions_two" --outcomes "$race_outcomes_two" \
  --token-budget 10 --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner claude --generation race-two >/dev/null
set +e
"$LOOM" contingent-policy-observe --state-dir "$RACE_STATE" \
  --world race-scheduler --contingent-policy race-one --outcome GO \
  --owner codex --generation race-one --outcome-digest "$outcome_x" \
  >"$WORK/race-one.out" 2>&1 &
race_one_pid=$!
"$LOOM" contingent-policy-observe --state-dir "$RACE_STATE" \
  --world race-scheduler --contingent-policy race-two --outcome GO \
  --owner claude --generation race-two --outcome-digest "$outcome_y" \
  >"$WORK/race-two.out" 2>&1 &
race_two_pid=$!
wait "$race_one_pid"
race_one_rc=$?
wait "$race_two_pid"
race_two_rc=$?
set -e
if [[ "$race_one_rc" -eq 0 && "$race_two_rc" -eq 1 ]]; then
  winner_policy=race-one
  winner_owner=codex
  winner_generation=race-one
  winner_outcome="$outcome_done"
  loser_policy=race-two
  loser_owner=claude
  loser_generation=race-two
  loser_outcome="$outcome_y"
  loser_root=/race/root-two
  loser_log="$WORK/race-two.out"
elif [[ "$race_one_rc" -eq 1 && "$race_two_rc" -eq 0 ]]; then
  winner_policy=race-two
  winner_owner=claude
  winner_generation=race-two
  winner_outcome="$outcome_done"
  loser_policy=race-one
  loser_owner=codex
  loser_generation=race-one
  loser_outcome="$outcome_x"
  loser_root=/race/root-one
  loser_log="$WORK/race-one.out"
else
  sed -n '1,120p' "$WORK/race-one.out" >&2
  sed -n '1,120p' "$WORK/race-two.out" >&2
  fail "concurrent successor CAS returned race-one=$race_one_rc race-two=$race_two_rc"
fi
rg -q 'epistemic-global-resource-conflict:/race/shared' "$loser_log" || \
  fail 'concurrent loser was refused by an unrelated rule'
"$LOOM" world-verify --state-dir "$RACE_STATE" \
  --world race-scheduler >/dev/null
expect_refusal concurrent-loser-current-retained \
  "epistemic-global-resource-conflict:$loser_root" \
  "$LOOM" epistemic-capability-acquire --state-dir "$RACE_STATE" \
  --world race-scheduler --capability steal-loser-root \
  --resource "$loser_root" --owner intruder --generation intruder
"$LOOM" contingent-policy-observe --state-dir "$RACE_STATE" \
  --world race-scheduler --contingent-policy "$winner_policy" --outcome DONE \
  --owner "$winner_owner" --generation "$winner_generation" \
  --outcome-digest "$winner_outcome" >/dev/null
"$LOOM" contingent-policy-observe --state-dir "$RACE_STATE" \
  --world race-scheduler --contingent-policy "$loser_policy" --outcome GO \
  --owner "$loser_owner" --generation "$loser_generation" \
  --outcome-digest "$loser_outcome" >/dev/null
"$LOOM" contingent-policy-observe --state-dir "$RACE_STATE" \
  --world race-scheduler --contingent-policy "$loser_policy" --outcome DONE \
  --owner "$loser_owner" --generation "$loser_generation" \
  --outcome-digest "$outcome_done" >/dev/null
"$LOOM" world-verify --state-dir "$RACE_STATE" \
  --world race-scheduler >/dev/null

# Verified events remain a derived Arrow projection.
"$LOOM" export-events-arrow --state-dir "$STATE_DIR" \
  --out "$WORK/contingent.arrow" > "$WORK/export.out"
rg -q 'authority=verified-derived rows=22 ' "$WORK/export.out" || {
  sed -n '1,120p' "$WORK/export.out" >&2
  fail 'Arrow projection did not contain the twenty-two verified events'
}
inspect="$($LOOM verify-events-arrow --file "$WORK/contingent.arrow")"
rg -q 'schema=loom-spectral-events-v1 rows=22 batches=1' <<< "$inspect" || \
  fail "native Arrow reader disagreed: $inspect"

# Semantic tampering is rehashed at the journal layer so each refusal proves
# replay semantics rather than merely detecting a broken hash chain.
TAMPER_STATE="$WORK/tamper-base"
"$LOOM" world-create --state-dir "$TAMPER_STATE" --world tamper-target \
  --agent codex --lane tamper-target >/dev/null
"$LOOM" knowledge-observe --state-dir "$TAMPER_STATE" --world tamper-target \
  --knowledge k --value v --error 0.01 --uncertainty bounded \
  --confidence 0.5 --provenance "$provenance" >/dev/null
"$LOOM" epistemic-claim-open --state-dir "$TAMPER_STATE" \
  --world tamper-target --claim c --knowledge k \
  --evidence "$claim_evidence" >/dev/null
"$LOOM" world-create --state-dir "$TAMPER_STATE" --world tamper-scheduler \
  --agent codex --lane tamper-scheduler >/dev/null
tamper_actions="$WORK/tamper-actions.tsv"
tamper_outcomes="$WORK/tamper-outcomes.tsv"
printf '%b\n' "$action_header" \
  "root\tA\ttamper-target\tc\tcodex\t/tamper/a\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  "next\tB\ttamper-target\tc\tkimi\t/tamper/b\t20\t20\t20\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  > "$tamper_actions"
printf '%b\n' "$outcome_header" \
  "A\t0\t1\tX\tnext\t$branch_evidence" \
  "B\t0\t1\tDONE\t-\t$branch_evidence" > "$tamper_outcomes"
"$LOOM" contingent-policy-compile --state-dir "$TAMPER_STATE" \
  --world tamper-scheduler --contingent-policy tamper-policy \
  --root-state root --actions "$tamper_actions" --outcomes "$tamper_outcomes" \
  --token-budget 10 --wall-budget 10 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner codex --generation tamper >/dev/null

for mutation in action-digest outcome-digest frontier selected-policy; do
  copy="$WORK/$mutation-tamper"
  cp -a "$TAMPER_STATE" "$copy"
  journal="$copy/loom-epistemic/worlds/tamper-scheduler/journal.tsv"
  rehash_last_event_after_payload_mutation "$journal" "$mutation"
  case "$mutation" in
    action-digest) expected='epistemic-contingent-action-set-digest-mismatch:tamper-policy' ;;
    outcome-digest) expected='epistemic-contingent-outcome-set-digest-mismatch:tamper-policy' ;;
    frontier) expected='epistemic-contingent-frontier-mismatch:tamper-policy' ;;
    selected-policy) expected='epistemic-contingent-selection-mismatch:tamper-policy' ;;
  esac
  expect_refusal "$mutation-sabotage" "$expected" \
    "$LOOM" world-verify --state-dir "$copy" --world tamper-scheduler
done

expect_refusal arrow-semantic-laundering \
  'epistemic-contingent-action-set-digest-mismatch:tamper-policy' \
  "$LOOM" export-events-arrow --state-dir "$WORK/action-digest-tamper" \
  --out "$WORK/semantic-laundered.arrow"
[[ ! -e "$WORK/semantic-laundered.arrow" ]] || \
  fail 'semantic contingent-policy sabotage produced Arrow output'

route_tamper="$WORK/route-tamper"
cp -a "$TAMPER_STATE" "$route_tamper"
"$LOOM" contingent-policy-observe --state-dir "$route_tamper" \
  --world tamper-scheduler --contingent-policy tamper-policy --outcome X \
  --owner codex --generation tamper --outcome-digest "$outcome_x" >/dev/null
route_journal="$route_tamper/loom-epistemic/worlds/tamper-scheduler/journal.tsv"
rehash_last_event_after_payload_mutation "$route_journal" route
expect_refusal route-sabotage \
  'epistemic-contingent-branch-routing-mismatch:tamper-policy/X' \
  "$LOOM" world-verify --state-dir "$route_tamper" --world tamper-scheduler

journal_tamper="$WORK/journal-tamper"
cp -a "$TAMPER_STATE" "$journal_tamper"
raw_journal="$journal_tamper/loom-epistemic/worlds/tamper-scheduler/journal.tsv"
awk -F '\t' 'BEGIN { OFS="\t" }
  NR == 2 { $6 = (substr($6, 1, 1) == "0" ? "1" : "0") substr($6, 2) }
  { print }
' "$raw_journal" > "$raw_journal.tampered"
mv "$raw_journal.tampered" "$raw_journal"
expect_refusal journal-sabotage \
  'epistemic-journal-event-digest-mismatch:seq=2' \
  "$LOOM" world-verify --state-dir "$journal_tamper" \
  --world tamper-scheduler

printf 'SOUNIO_LOOM_CONTINGENT_POLICY_GATE_PASS=true schema=loom-robust-contingent-policy-v0 frame=9011 states=3 actions=5 outcomes=6 policies=3 budgets=4 branch_conditioned=A_X_B_A_Y_C robust=min-benefit+max-burden open_loop_laundering=REFUSED nominal_partition=TOTAL future_reservation=NONE atomic_handoff=PASS concurrent_handoff=CAS_LOCKED routes=X:B+Y:C frontier_cap=REFUSED replay=RECOMPUTED action_sabotage=REFUSED outcome_sabotage=REFUSED frontier_sabotage=REFUSED selected_sabotage=REFUSED route_sabotage=REFUSED journal_sabotage=REFUSED arrow_laundering=REFUSED rows=22 runtime=OCaml+Sounio\n'
