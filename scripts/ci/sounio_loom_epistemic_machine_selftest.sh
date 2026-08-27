#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOOM="$ROOT_DIR/bin/sounio-loom"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-epistemic-machine.XXXXXX")"
STATE_DIR="$WORK/state"

export SOUNIO_COORD_RUNTIME_MODE=local
export SOUNIO_LOOM_COORD_AUTO=0

fail() {
  printf 'sounio-loom-epistemic-machine: FAIL: %s work=%s\n' "$*" "$WORK" >&2
  exit 1
}

cleanup() {
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    find "$WORK" -depth -mindepth 1 -delete 2>/dev/null || true
    rmdir "$WORK" 2>/dev/null || true
  fi
}
trap cleanup EXIT

digest() {
  printf '%s' "$1" | sha256sum | cut -d ' ' -f 1
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
    sed -n '1,160p' "$WORK/$label.err" >&2
    fail "$label was refused by an unrelated rule"
  }
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

provenance="$(digest provenance-fixture)"
evidence="$(digest evidence-fixture)"
falsifier="$(digest falsifier-fixture)"

created="$($LOOM world-create --state-dir "$STATE_DIR" --world alpha \
  --agent codex --lane epistemic-v0)"
rg -q '^LOOM_WORLD_CREATED schema=loom-epistemic-machine-v0 world=alpha ' \
  <<< "$created" || fail 'world origin receipt missing'

observed="$($LOOM knowledge-observe --state-dir "$STATE_DIR" --world alpha \
  --knowledge k-observation --value 42.0 --error 0.01 \
  --uncertainty interval-0.2 --confidence 0.91 --provenance "$provenance")"
rg -q 'knowledge=k-observation axes=5' <<< "$observed" || \
  fail 'knowledge receipt did not preserve five axes'

expect_refusal missing-axis 'epistemic-uncertainty-invalid' \
  "$LOOM" knowledge-observe --state-dir "$STATE_DIR" --world alpha \
  --knowledge k-laundered --value 42.0 --error 0.01 --uncertainty '' \
  --confidence 0.91 --provenance "$provenance"

"$LOOM" epistemic-claim-open --state-dir "$STATE_DIR" --world alpha \
  --claim c-mechanism --knowledge k-observation --evidence "$evidence" >/dev/null
"$LOOM" epistemic-claim-challenge --state-dir "$STATE_DIR" --world alpha \
  --claim c-mechanism --challenge x-falsifier --falsifier "$falsifier" >/dev/null

expect_refusal absent-claim 'epistemic-challenge-claim-missing:c-absent' \
  "$LOOM" epistemic-claim-challenge --state-dir "$STATE_DIR" --world alpha \
  --claim c-absent --challenge x-invalid --falsifier "$falsifier"

resource='/workspace/.wt/codex-1/self-hosted/check/check.sio'
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" --world alpha \
  --capability cap-alpha --resource "$resource" --owner codex \
  --generation generation-1 >/dev/null

alpha_status="$($LOOM world-status --state-dir "$STATE_DIR" --world alpha)"
alpha_head="$(sed -n 's/.* head=\([0-9a-f]\{64\}\)$/\1/p' <<< "$alpha_status")"
[[ -n "$alpha_head" ]] || fail 'could not recover alpha head'

forked="$($LOOM world-fork --state-dir "$STATE_DIR" --parent alpha \
  --child beta --agent grok --lane hostile-review \
  --hypothesis 'the mechanism is false' --parent-head "$alpha_head")"
rg -q "parent_head=$alpha_head" <<< "$forked" || \
  fail 'fork did not bind the verified parent head'

expect_refusal duplicate-capability 'epistemic-global-resource-conflict:' \
  "$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" --world beta \
  --capability cap-beta-conflict --resource "$resource" --owner grok \
  --generation generation-2

"$LOOM" epistemic-capability-release --state-dir "$STATE_DIR" --world alpha \
  --capability cap-alpha --owner codex --generation generation-1 >/dev/null
"$LOOM" epistemic-capability-acquire --state-dir "$STATE_DIR" --world beta \
  --capability cap-beta --resource "$resource" --owner grok \
  --generation generation-2 >/dev/null

wrong_head="$(printf 'a%.0s' $(seq 1 64))"
expect_refusal wrong-parent-head 'epistemic-parent-head-mismatch:' \
  "$LOOM" world-fork --state-dir "$STATE_DIR" --parent alpha --child gamma \
  --agent codex --lane counterfactual --hypothesis alternate \
  --parent-head "$wrong_head"
[[ ! -e "$STATE_DIR/loom-epistemic/worlds/gamma/journal.tsv" ]] || \
  fail 'wrong parent head created a child journal'

"$LOOM" world-fork --state-dir "$STATE_DIR" --parent alpha --child gamma \
  --agent codex --lane counterfactual --hypothesis alternate >/dev/null

worlds="$($LOOM world-list --state-dir "$STATE_DIR")"
rg -q '^loom_worlds=3$' <<< "$worlds" || fail 'world list did not recover three worldlines'

"$LOOM" export-events-arrow --state-dir "$STATE_DIR" \
  --out "$WORK/worldlines.arrow" >"$WORK/export.out"
rg -q 'authority=verified-derived rows=9 ' "$WORK/export.out" || {
  sed -n '1,120p' "$WORK/export.out" >&2
  fail 'Arrow projection did not contain the nine verified worldline events'
}
inspect="$($LOOM verify-events-arrow --file "$WORK/worldlines.arrow")"
rg -q 'schema=loom-spectral-events-v1 rows=9 batches=1' <<< "$inspect" || \
  fail "native Arrow reader disagreed: $inspect"

alpha_journal="$STATE_DIR/loom-epistemic/worlds/alpha/journal.tsv"
awk -F '\t' 'BEGIN { OFS="\t" }
  NR == 1 { $6 = (substr($6, 1, 1) == "0" ? "1" : "0") substr($6, 2) }
  { print }
' "$alpha_journal" > "$alpha_journal.tampered"
mv "$alpha_journal.tampered" "$alpha_journal"

expect_refusal journal-sabotage \
  'epistemic-journal-event-digest-mismatch:seq=1' \
  "$LOOM" world-verify --state-dir "$STATE_DIR" --world alpha
expect_refusal arrow-laundering \
  'epistemic-journal-event-digest-mismatch:seq=1' \
  "$LOOM" export-events-arrow --state-dir "$STATE_DIR" \
  --out "$WORK/laundered.arrow"
[[ ! -e "$WORK/laundered.arrow" ]] || fail 'journal sabotage produced Arrow output'

printf 'SOUNIO_LOOM_EPISTEMIC_MACHINE_GATE_PASS=true schema=%s worlds=3 events=9 axes=5 linear_capability=PASS fork_binding=PASS journal_sabotage=REFUSED arrow_laundering=REFUSED runtime=OCaml+Sounio\n' \
  'loom-epistemic-machine-v0'
