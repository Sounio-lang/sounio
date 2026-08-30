#!/usr/bin/env bash
# What the compiler answers to, and what it says it answers to, must agree.
#
# `souc --help` did not list `ontology`. The verb works -- `souc ontology resolve
# GO:0008150` returns `label=biological_process` -- and it has been absent from
# the usage block the whole time. On 2026-08-26 that omission produced two wrong
# documentation edits and one wrong premise handed to seven agents, all from the
# same reasoning: survey `--help`, conclude the command does not exist. Absence
# from a list is not absence of the thing.
#
# `install` and `search` were missing too. Three verbs, found only because
# someone ran them.
#
# WHAT THIS GATE CAN AND CANNOT PROVE
#
# The verb surface is spread across THREE layers: the bin/souc wrapper's `case`,
# the bin/madaros mapper's `case`, and the `mode == "..."` chain inside
# self-hosted/compiler/main.sio. No source scan of any one of them sees the other
# two -- which is precisely how `ontology` stayed invisible.
#
#   Direction 1, COMPLETE: every verb `--help` advertises must actually dispatch.
#     Fully checkable, and a lie in the help is a hard failure.
#
#   Direction 2, BEST EFFORT: every verb we can find dispatching must be either
#     advertised or listed as deliberately internal. The candidate set is the
#     union of the three source scans plus a checked-in list, and it CANNOT be
#     proved complete from outside a binary. This gate does not pretend
#     otherwise. A verb dispatched by a path none of the scans reads is still
#     invisible here, and that residual risk is the reason the internal list is
#     small and explicit rather than a wildcard.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "souc_help_verb_parity"

SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
require_executable "$SOUC"
# Say which binary this is about, and refuse one from another checkout. A
# SOUC_BIN exported in a developer profile silently points every invocation at
# the shared /workspace/sounio tree, so a worktree edit to bin/souc measures as
# absent no matter what you changed -- which happened while writing this gate.
echo "  souc=$SOUC"
case "$SOUC" in
  "$ROOT_DIR"/*) : ;;
  *) gate_fail "SOUC_BIN points outside this checkout ($SOUC).
       This gate compares THIS tree's bin/souc against THIS tree's dispatch, and a
       foreign binary makes the comparison meaningless. Re-run with
       \`env -u SOUC_BIN\`, or set SOUC_BIN to a path inside $ROOT_DIR." ;;
esac
INTERNAL="$ROOT_DIR/scripts/ci/fixtures/souc_internal_verbs.txt"
require_file "$INTERNAL"

HELP="$("$SOUC" --help 2>&1)"
require_nonempty "$HELP" "souc --help output"

# A verb is ABSENT when the binary answers it exactly as it answers a verb that
# cannot exist. Differential, so it holds across engines rather than keying on one
# engine's fall-through string.
_absent() {
  local v="$1" a b
  a="$(timeout 20 "$SOUC" "$v" 2>&1)"
  b="$(timeout 20 "$SOUC" __sounio_absent_probe__ 2>&1)"
  [[ "${a//$v/<V>}" == "${b//__sounio_absent_probe__/<V>}" ]]
}

# Controls first. `info` exists on every souc; a nonsense verb does not. If the
# probe cannot tell those apart it cannot answer about anything, and a gate that
# cannot discriminate must refuse rather than report a clean parity.
if _absent info;  then gate_fail "probe positive control failed: \`souc info\` reads as absent"; fi
if ! _absent __sounio_second_probe__; then gate_fail "probe negative control failed: an impossible verb reads as present"; fi

# ── candidates: three source scans, unioned with the checked-in list ──────────
mapfile -t CANDIDATES < <(
  {
    grep -oE '^[[:space:]]{2,}[a-z][a-z0-9|_-]*\)' "$ROOT_DIR/bin/souc"    2>/dev/null | tr -d ' )' | tr '|' '\n'
    grep -oE '^[[:space:]]{2,}[a-z][a-z0-9|_-]*\)' "$ROOT_DIR/bin/madaros" 2>/dev/null | tr -d ' )' | tr '|' '\n'
    grep -oE 'mode[[:space:]]*==[[:space:]]*"[a-z][^"]*"' "$ROOT_DIR/self-hosted/compiler/main.sio" 2>/dev/null \
      | sed -E 's/.*"([^"]*)"/\1/'
    grep -vE '^[[:space:]]*(#|$)' "$INTERNAL" | awk '{print $1}'
    # what the help itself advertises, so Direction 1 has its inputs
    grep -oE '^[[:space:]]+souc [a-z][a-z0-9_-]*' <<<"$HELP" | awk '{print $2}'
  } | sort -u
)
require_min_count "${#CANDIDATES[@]}" 12 "candidate verbs"

declare -a LIES=() UNDECLARED=()
dispatched=0
for v in "${CANDIDATES[@]}"; do
  [[ -z "$v" ]] && continue
  in_help=0; grep -qE "^[[:space:]]+souc $v( |$)" <<<"$HELP" && in_help=1
  is_internal=0; grep -qE "^$v([[:space:]]|$)" "$INTERNAL" && is_internal=1
  if _absent "$v"; then
    # not dispatched. Only a problem if the help advertises it.
    [[ "$in_help" -eq 1 ]] && LIES+=("$v")
    continue
  fi
  dispatched=$((dispatched + 1))
  if [[ "$in_help" -eq 0 && "$is_internal" -eq 0 ]]; then UNDECLARED+=("$v"); fi
done
require_min_count "$dispatched" 8 "verbs observed dispatching"

echo "  candidates=${#CANDIDATES[@]} dispatched=$dispatched advertised-and-dead=${#LIES[@]} dispatched-and-undeclared=${#UNDECLARED[@]}"

rc=0
if ((${#LIES[@]})); then
  echo "  FAIL: \`souc --help\` advertises a verb the binary does not dispatch:" >&2
  printf '        %s\n' "${LIES[@]}" >&2
  echo "        Either implement it or stop advertising it. A help that lies is worse" >&2
  echo "        than a help that is short." >&2
  rc=1
fi
if ((${#UNDECLARED[@]})); then
  echo "  FAIL: the binary dispatches a verb \`--help\` never mentions:" >&2
  printf '        %s\n' "${UNDECLARED[@]}" >&2
  echo "        This is the ontology case. Add it to the usage block in bin/souc, or --" >&2
  echo "        if it is deliberately internal -- add a line with its reason to" >&2
  echo "        scripts/ci/fixtures/souc_internal_verbs.txt, which is read in review." >&2
  rc=1
fi

[[ $rc -eq 0 ]] || gate_fail "souc --help and the dispatched verb surface disagree"
gate_pass "every advertised verb dispatches; every verb found dispatching is advertised or inventoried"
