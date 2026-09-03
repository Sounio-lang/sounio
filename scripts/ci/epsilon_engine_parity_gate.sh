#!/usr/bin/env bash
# Cross-engine parity for epsilon-bounded compile-fail tests.
#
# GATE_CONTRACT: v0
# GATE_ID: epsilon_engine_parity
# GATE_CLAIMS: a compile-fail test carrying an epsilon bound must be refused by BOTH engines
# GATE_ENGINE: both (Madaros default + SOUNIO_SOUC_ENGINE=lean_single)
# GATE_RESULT_ON_SKIP: fail
#
# WHY THIS EXISTS, and why it does not take a side.
#
# epsilon has two polarities in this tree. Madaros reads it as an error bound
# (self-hosted/check/epistemic.sio:595, `epsilon_subsumes` is `a <= b`), while
# lean_single and the clinical surface read it as confidence. Neither engine is
# wrong against a decision, because no decision exists yet -- see
# docs/audit/EPSILON_POLARITY_FORK_2026-08-19.md.
#
# What IS wrong under either decision is that the disagreement is SILENT. A
# patient-safety compile-fail test answering `check: OK` on the engine bin/souc
# routes to is a failure whichever way epsilon is finally defined. This gate does
# not choose the polarity. It refuses to let the divergence grow, and it prints
# every current instance by name on every run so nobody has to rediscover them.
#
# The frozen count may only SHRINK. Deciding the polarity should drive it to 0.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
FROZEN_FILE="scripts/ci/epsilon_engine_parity.frozen"
OUT="${GATE_ARTIFACT:-artifacts/gates/epsilon_engine_parity.json}"
TIMEOUT="${EPS_PARITY_TIMEOUT:-180}"

frozen=$(tr -dc '0-9' < "$FROZEN_FILE" 2>/dev/null)
[[ -n "$frozen" ]] || { echo "epsilon_engine_parity: FAIL: missing or unreadable $FROZEN_FILE" >&2; exit 1; }

# Discovery. An epsilon-bounded compile-fail test is one whose source carries an
# epsilon comparison in a type position.
mapfile -t tests < <(git ls-files 'tests/compile-fail/*.sio' 'tests/compile-fail/**/*.sio' 2>/dev/null \
  | sort -u | xargs grep -lE 'ε *(>=|>|<|=)' 2>/dev/null | sort)

# NEGATIVE CONTROL 1 -- discovery must not be empty. A gate that finds nothing
# passes for free, which is the failure mode this repo has already paid for.
if [[ "${#tests[@]}" -eq 0 ]]; then
  echo "epsilon_engine_parity: FAIL: discovered no epsilon-bounded compile-fail tests" >&2
  echo "  the corpus cannot have lost all of them; the discovery expression is broken" >&2
  exit 1
fi

refuses() { # engine file -> 0 if the engine refuses
  local engine="$1" f="$2" out
  if [[ "$engine" == lean_single ]]; then
    out=$(SOUNIO_SOUC_ENGINE=lean_single timeout "$TIMEOUT" "$SOUC" check "$f" 2>&1)
  else
    out=$(timeout "$TIMEOUT" "$SOUC" check "$f" 2>&1)
  fi
  grep -qiE 'error|failed to parse|type error' <<<"$out"
}

divergent=0; both=0; names=()
for t in "${tests[@]}"; do
  m=0; l=0
  refuses madaros     "$t" && m=1
  refuses lean_single "$t" && l=1
  if (( m == 1 && l == 1 )); then
    both=$(( both + 1 ))
  else
    divergent=$(( divergent + 1 ))
    names+=("$t")
    printf 'EPSILON_PARITY_DIVERGENT %s madaros_refuses=%d lean_refuses=%d\n' "$t" "$m" "$l"
  fi
done

# NEGATIVE CONTROL 2 -- at least one test must be refused by both engines. If
# none is, the refusal detector matched nothing and every result above is noise.
if (( both == 0 )); then
  echo "epsilon_engine_parity: FAIL: no test was refused by both engines" >&2
  echo "  with ${#tests[@]} tests discovered that means the detector is broken, not the corpus" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
status=pass; rc=0
if (( divergent > frozen )); then
  status=fail; rc=1
  echo "REFUSE: engine-divergent epsilon refusals rose ${frozen} -> ${divergent}." >&2
  echo "  A compile-fail test that one engine refuses and the other accepts is a" >&2
  echo "  silent hole whichever polarity epsilon is finally given." >&2
elif (( divergent < frozen )); then
  echo "OK: engine-divergent epsilon refusals fell ${frozen} -> ${divergent}. Lower the frozen count:"
  echo "  printf '%s\\n' ${divergent} > ${FROZEN_FILE}"
else
  echo "OK: engine-divergent epsilon refusals hold at ${frozen} (named above)."
fi

printf '%s\n' "${names[@]}" > "$(dirname "$OUT")/epsilon_engine_parity.measured.list"
cat > "$OUT" <<JSON
{
  "gate": "epsilon_engine_parity",
  "status": "${status}",
  "claims": "an epsilon-bounded compile-fail test must be refused by both engines",
  "frozen": ${frozen},
  "measured": ${divergent},
  "metrics": { "total": ${#tests[@]}, "passed": ${both}, "failed": ${divergent}, "not_run": 0 }
}
JSON
exit "$rc"
