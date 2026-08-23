#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="$ROOT_DIR/bin/sounio-coord"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-causal-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/reviewer-worktree"
STATE="$TEST_ROOT/state"

cleanup() {
  git -C "$REPO" worktree remove --force "$SECOND" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  echo "sounio-coord-causal-selftest: FAIL: $*" >&2
  exit 1
}

run_coord() {
  local cwd="$1"
  shift
  (cd "$cwd" && SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_RUNTIME_MODE=local \
    "$TOOL" "$@")
}

mkdir -p "$REPO/semantic" "$REPO/experiments" "$REPO/evidence"
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Causal Selftest'
git -C "$REPO" config user.email 'coord-causal-selftest@sounio.local'
printf 'mechanism=old\n' > "$REPO/semantic/mechanism.txt"
printf 'treatment fixture\n' > "$REPO/evidence/treatment.log"
printf 'control fixture\n' > "$REPO/evidence/control.log"
git -C "$REPO" add semantic evidence
git -C "$REPO" commit -qm 'seed causal experiment'
PRODUCER_BRANCH="$(git -C "$REPO" branch --show-current)"
git -C "$REPO" worktree add -q -b causal-reviewer "$SECOND"

run_coord "$REPO" claim --agent researcher --lane causal --ttl-seconds 600 \
  --intent 'preregister a mechanism-specific experiment' \
  --resources concept:epistemic/test gate:causal-test \
  --files 'semantic/**' 'experiments/**' 'evidence/**' >/dev/null

if run_coord "$REPO" experiment-open --agent researcher --lane causal \
  --receipt experiments/missing-falsifier.json --statement 'something happens' \
  --intervention RULE_OFF --treatment-predicate 'active refuses' \
  --control-predicate 'disabled accepts' --resource concept:epistemic/test \
  >/dev/null 2>&1; then
  fail 'experiment without a falsifier was accepted'
fi

output="$(run_coord "$REPO" experiment-open --agent researcher --lane causal \
  --id exp-causal-v1 --receipt experiments/prereg.json \
  --statement 'the provenance rule causes the refusal' \
  --falsifier 'either laundering route remains accepted with the rule active' \
  --intervention 'SOUNIO_PROVENANCE_SABOTAGE=1' \
  --treatment-predicate 'both routes refuse with E222' \
  --control-predicate 'both refusals disappear only under sabotage' \
  --resource concept:epistemic/test --resource gate:causal-test)"
grep -q '^EXPERIMENT_OPEN id=exp-causal-v1 ' <<< "$output" || \
  fail 'valid preregistration was not created'
prereg_sha="$(sed -n 's/.*prereg_sha256=\([0-9a-f]*\).*/\1/p' <<< "$output")"
[[ "$prereg_sha" =~ ^[0-9a-f]{64}$ ]] || fail 'preregistration digest is invalid'

output="$(run_coord "$REPO" experiment-status --prereg experiments/prereg.json)"
grep -q 'state=open prereg_committed=no ' <<< "$output" || \
  fail 'uncommitted preregistration state was not explicit'
git -C "$REPO" add experiments/prereg.json
git -C "$REPO" commit -qm 'preregister causal experiment'

if run_coord "$REPO" experiment-close --agent researcher --lane causal \
  --prereg experiments/prereg.json --outcome experiments/posthoc.json \
  --verdict supported --treatment treatment=PASS --control sabotage=PASS \
  --treatment-evidence evidence/treatment.log \
  --control-evidence evidence/control.log >/dev/null 2>&1; then
  fail 'experiment closed without a post-preregistration subject commit'
fi

printf 'mechanism=new\n' > "$REPO/semantic/mechanism.txt"
printf 'active: E222 E222\n' > "$REPO/evidence/treatment.log"
printf 'sabotage: refusal disappeared\n' > "$REPO/evidence/control.log"
git -C "$REPO" add semantic evidence
git -C "$REPO" commit -qm 'implement and observe causal mechanism'
subject_sha="$(git -C "$REPO" rev-parse HEAD)"

if run_coord "$REPO" experiment-close --agent researcher --lane causal \
  --prereg experiments/prereg.json --outcome experiments/control-failed.json \
  --verdict supported --treatment treatment=PASS --control sabotage=FAIL \
  --treatment-evidence evidence/treatment.log \
  --control-evidence evidence/control.log >/dev/null 2>&1; then
  fail 'supported verdict accepted a failed sabotage control'
fi

if run_coord "$REPO" experiment-close --agent researcher --lane causal \
  --prereg experiments/prereg.json --outcome experiments/false-positive.json \
  --verdict falsified --treatment treatment=PASS --control sabotage=PASS \
  --treatment-evidence evidence/treatment.log \
  --control-evidence evidence/control.log >/dev/null 2>&1; then
  fail 'falsified verdict accepted a passing treatment predicate'
fi

if run_coord "$REPO" experiment-close --agent researcher --lane causal \
  --prereg experiments/prereg.json --outcome experiments/reused-evidence.json \
  --verdict supported --treatment treatment=PASS --control sabotage=PASS \
  --treatment-evidence evidence/treatment.log \
  --control-evidence evidence/treatment.log >/dev/null 2>&1; then
  fail 'one artifact was reused as treatment and control evidence'
fi

printf 'tampered after subject commit\n' >> "$REPO/evidence/control.log"
if run_coord "$REPO" experiment-close --agent researcher --lane causal \
  --prereg experiments/prereg.json --outcome experiments/dirty-evidence.json \
  --verdict supported --treatment treatment=PASS --control sabotage=PASS \
  --treatment-evidence evidence/treatment.log \
  --control-evidence evidence/control.log >/dev/null 2>&1; then
  fail 'dirty control evidence was accepted'
fi
git -C "$REPO" show HEAD:evidence/control.log > "$REPO/evidence/control.log"

output="$(run_coord "$REPO" experiment-close --agent researcher --lane causal \
  --prereg experiments/prereg.json --outcome experiments/outcome.json \
  --verdict supported --treatment treatment=PASS --control sabotage=PASS \
  --treatment-evidence evidence/treatment.log \
  --control-evidence evidence/control.log)"
grep -q "^EXPERIMENT_CLOSED id=exp-causal-v1 verdict=supported .*subject_commit=$subject_sha$" \
  <<< "$output" || fail 'valid causal outcome was not created'
outcome_sha="$(sed -n 's/.*outcome_sha256=\([0-9a-f]*\).*/\1/p' <<< "$output")"
[[ "$outcome_sha" =~ ^[0-9a-f]{64}$ ]] || fail 'outcome digest is invalid'

if run_coord "$REPO" experiment-status --prereg experiments/prereg.json \
  --outcome experiments/outcome.json >/dev/null 2>&1; then
  fail 'uncommitted causal outcome verified as durable'
fi

request_output="$(run_coord "$SECOND" send --agent reviewer --lane integration \
  --to-agent researcher --to-lane causal --kind request \
  --message 'deliver the falsification-carrying change')"
request_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$request_output")"
[[ -n "$request_id" ]] || fail 'review request did not return a message id'

if run_coord "$REPO" handoff --agent researcher --lane causal \
  --to-agent reviewer --to-lane integration --message 'outcome is still uncommitted' \
  --commit HEAD --gate causal-test=PASS --evidence evidence/treatment.log \
  --experiment-prereg experiments/prereg.json \
  --experiment-outcome experiments/outcome.json --reply-to "$request_id" \
  >/dev/null 2>&1; then
  fail 'handoff accepted an uncommitted causal outcome'
fi

run_coord "$REPO" authorize --agent researcher --lane causal \
  --resources concept:epistemic/test gate:causal-test \
  --files semantic/mechanism.txt experiments/prereg.json evidence/control.log \
  >/dev/null || fail 'refused causal handoff released ownership'

git -C "$REPO" add experiments/outcome.json
git -C "$REPO" commit -qm 'seal causal outcome receipt'
head_sha="$(git -C "$REPO" rev-parse HEAD)"
output="$(run_coord "$REPO" experiment-status --prereg experiments/prereg.json \
  --outcome experiments/outcome.json)"
grep -q "state=supported .*subject_commit=$subject_sha$" <<< "$output" || \
  fail 'committed causal chain did not verify'

output="$(run_coord "$REPO" handoff --agent researcher --lane causal \
  --to-agent reviewer --to-lane integration \
  --message 'mechanism-specific evidence accepted' --commit HEAD \
  --gate causal-test=PASS --evidence evidence/treatment.log \
  --experiment-prereg experiments/prereg.json \
  --experiment-outcome experiments/outcome.json --reply-to "$request_id")"
handoff_id="$(sed -n 's/^HANDED_OFF .* message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$handoff_id" ]] || fail 'valid causal handoff was not published'
grep -q "commit=$head_sha .*experiment=exp-causal-v1$" <<< "$output" || \
  fail 'handoff did not report the causal experiment identity'

output="$(run_coord "$SECOND" inbox --agent reviewer --lane integration --kind handoff)"
grep -q "MESSAGE id=$handoff_id .*experiment=exp-causal-v1 " <<< "$output" || \
  fail 'recipient did not receive the causal experiment metadata'
grep -q "prereg_sha256=$prereg_sha outcome_sha256=$outcome_sha" <<< "$output" || \
  fail 'handoff did not preserve both causal receipt digests'

git -C "$SECOND" merge -q --ff-only "$PRODUCER_BRANCH"
output="$(run_coord "$SECOND" experiment-status --prereg experiments/prereg.json \
  --outcome experiments/outcome.json)"
grep -q "id=exp-causal-v1 state=supported .*subject_commit=$subject_sha$" <<< "$output" || \
  fail 'another worktree could not independently verify the causal receipt'

if run_coord "$REPO" authorize --agent researcher --lane causal \
  --resources concept:epistemic/test --files semantic/mechanism.txt \
  >/dev/null 2>&1; then
  fail 'valid causal handoff left the source claim active'
fi

run_coord "$REPO" claim --agent researcher --lane falsified --ttl-seconds 600 \
  --intent 'retain a negative causal result' \
  --resources concept:epistemic/falsified gate:causal-falsified \
  --files 'negative/**' >/dev/null
run_coord "$REPO" experiment-open --agent researcher --lane falsified \
  --id exp-falsified-v1 --receipt negative/prereg.json \
  --statement 'the candidate rule closes the remaining route' \
  --falsifier 'the route remains accepted with the candidate active' \
  --intervention 'SOUNIO_CANDIDATE_SABOTAGE=1' \
  --treatment-predicate 'the route refuses' \
  --control-predicate 'the route accepts under sabotage' \
  --resource concept:epistemic/falsified --resource gate:causal-falsified >/dev/null
git -C "$REPO" add negative/prereg.json
git -C "$REPO" commit -qm 'preregister falsified experiment'
printf 'active: route still accepted\n' > "$REPO/negative/treatment.log"
printf 'sabotage: route accepted\n' > "$REPO/negative/control.log"
git -C "$REPO" add negative/treatment.log negative/control.log
git -C "$REPO" commit -qm 'observe falsifying witness'
run_coord "$REPO" experiment-close --agent researcher --lane falsified \
  --prereg negative/prereg.json --outcome negative/outcome.json \
  --verdict falsified --treatment treatment=FAIL --control sabotage=PASS \
  --treatment-evidence negative/treatment.log \
  --control-evidence negative/control.log >/dev/null
git -C "$REPO" add negative/outcome.json
git -C "$REPO" commit -qm 'retain falsified causal outcome'
output="$(run_coord "$REPO" experiment-status --prereg negative/prereg.json \
  --outcome negative/outcome.json)"
grep -q 'id=exp-falsified-v1 state=falsified ' <<< "$output" || \
  fail 'falsified outcome was not retained as a first-class state'
if run_coord "$REPO" handoff --agent researcher --lane falsified \
  --to-agent reviewer --to-lane integration --message 'must not promote falsified result' \
  --commit HEAD --gate causal-falsified=PASS --evidence negative/treatment.log \
  --experiment-prereg negative/prereg.json \
  --experiment-outcome negative/outcome.json >/dev/null 2>&1; then
  fail 'falsified experiment was promoted as a supported causal handoff'
fi
run_coord "$REPO" authorize --agent researcher --lane falsified \
  --resources concept:epistemic/falsified --files negative/outcome.json >/dev/null || \
  fail 'refused falsified handoff released ownership'
run_coord "$REPO" release --agent researcher --lane falsified \
  --reason 'negative result retained without promotion' >/dev/null

echo 'sounio-coord-causal-selftest: PASS'
