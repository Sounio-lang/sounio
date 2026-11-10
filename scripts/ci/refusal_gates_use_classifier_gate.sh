#!/usr/bin/env bash
# refusal_gates_use_classifier_gate.sh -- meta-gate for #2425.
#
# A gate that proves "the compiler refuses X" by checking `rc != 0` (or
# `if CMD; then fail "unexpectedly compiled"; fi`) also passes when the
# compiler segfaults, times out or is OOM-killed. scripts/lib/classify_compile.sh
# tells those apart. This gate keeps every refusal gate under scripts/ci/ on it.
#
# Rule, per script. Count, on non-comment lines:
#   M = refusal-failure messages ("unexpectedly compiled", "did not reject",
#       "must refuse", "should have been refused", ...)
#   H = refusal-helper names (expect_rejection, expect_refusal, ...)
#   C = calls to sounio_expect_refusal / sounio_expect_failed_exit /
#       sounio_classify_compile
# If M + H > 0, the script must have C >= max(1, M): one classified status per
# refusal it asserts (a helper counts once, however often it is called).
#
# Escapes, both of which need a reason a reviewer can check:
#   - a line ending in `# refusal-classifier-exempt: <reason>` is not counted
#     (source-text checks, probes that characterise a known crash);
#   - a whole script listed in scripts/ci/refusal_gates_exempt.txt as
#     `<path> <reason>` is skipped (verdict probes that require rc=0 plus an
#     explicit verdict marker, so a crash cannot pass them; selftests of
#     non-compiler tools).
#
# Self-check: the gate runs its own counter on a known-bad and a known-good
# snippet first, so a regex edit that matches nothing cannot turn it green.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EXEMPT_FILE="scripts/ci/refusal_gates_exempt.txt"
SELF="scripts/ci/refusal_gates_use_classifier_gate.sh"
MSG_RE='unexpectedly (compiled|accepted|passed|succeeded|completed|typechecked)|did not (block|refuse|reject)|must (refuse|reject)|should (have been )?(refused|rejected|refuse|reject)'
HELPER_RE='\b(expect_compile_fail|compile_rejects|expect_rejection|expect_private_rejection|expect_refusal)\b'
CALL_RE='\b(sounio_expect_refusal|sounio_expect_failed_exit|sounio_classify_compile)\b'

fail() { printf 'REFUSAL_GATES_USE_CLASSIFIER_FAIL: %s\n' "$*" >&2; exit 1; }

[[ -f "$EXEMPT_FILE" ]] || fail "missing $EXEMPT_FILE"

# counts FILE -> "M H C" on stdout
counts() {
  local f="$1" body
  body="$(grep -vE '^[[:space:]]*#' "$f" | grep -v 'refusal-classifier-exempt:' || true)"
  local m h c
  m="$(grep -cE "$MSG_RE" <<<"$body" || true)"
  h="$(grep -cE "$HELPER_RE" <<<"$body" || true)"
  c="$(grep -cE "$CALL_RE" <<<"$body" || true)"
  printf '%s %s %s\n' "${m:-0}" "${h:-0}" "${c:-0}"
}

# verdict M H C -> 0 ok, 1 violation
verdict() {
  local m="$1" h="$2" c="$3" need
  (( m + h == 0 )) && return 0
  need=$(( m > 1 ? m : 1 ))
  (( c >= need ))
}

# --- self-check -------------------------------------------------------------
SC="$(mktemp -d)"
trap 'rm -rf "$SC"' EXIT
cat >"$SC/bad.sh" <<'BAD'
if bin/souc check neg.sio >log 2>&1; then
  echo "neg.sio unexpectedly compiled"; exit 1
fi
BAD
cat >"$SC/good.sh" <<'GOOD'
rc=0; bin/souc check neg.sio >log 2>&1 || rc=$?
[[ "$rc" -ne 0 ]] || { echo "neg.sio unexpectedly compiled"; exit 1; }
sounio_expect_refusal "$rc" log "" "" || exit 1
GOOD
cat >"$SC/exempt.sh" <<'EXEMPT'
grep -q forbidden src || echo "slice must refuse forbidden"  # refusal-classifier-exempt: source-text check
EXEMPT
read -r m h c < <(counts "$SC/bad.sh");    verdict "$m" "$h" "$c" && fail "self-check: unmigrated snippet passed ($m $h $c)"
read -r m h c < <(counts "$SC/good.sh");   verdict "$m" "$h" "$c" || fail "self-check: migrated snippet failed ($m $h $c)"
read -r m h c < <(counts "$SC/exempt.sh"); verdict "$m" "$h" "$c" || fail "self-check: exempt line was counted ($m $h $c)"

# --- exemption file: every entry must exist and carry a reason ---------------
declare -A EXEMPT=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  path="${line%% *}"
  reason="${line#"$path"}"
  [[ -n "${reason// /}" ]] || fail "$EXEMPT_FILE: $path has no reason"
  [[ -f "$path" ]] || fail "$EXEMPT_FILE: $path does not exist (stale entry)"
  EXEMPT["$path"]=1
done <"$EXEMPT_FILE"

# --- scan ---------------------------------------------------------------------
checked=0 exempt=0 bad=0
while IFS= read -r f; do
  [[ "$f" == "$SELF" ]] && continue
  read -r m h c < <(counts "$f")
  (( m + h == 0 )) && continue
  if [[ -n "${EXEMPT[$f]:-}" ]]; then
    exempt=$((exempt + 1))
    continue
  fi
  checked=$((checked + 1))
  if ! verdict "$m" "$h" "$c"; then
    printf 'VIOLATION %s: %s refusal message(s), %s helper reference(s), %s classifier call(s)\n' "$f" "$m" "$h" "$c" >&2
    bad=$((bad + 1))
  fi
done < <(find scripts/ci -name '*.sh' -not -path '*/fixtures/*' | LC_ALL=C sort)

(( checked > 0 )) || fail "scanned no refusal gates; the patterns no longer match anything"
(( bad == 0 )) || fail "$bad refusal gate(s) decide refusal by exit status alone; route them through scripts/lib/classify_compile.sh (sounio_expect_refusal / sounio_expect_failed_exit) or record a reason"
printf 'REFUSAL_GATES_USE_CLASSIFIER_OK checked=%d exempt=%d\n' "$checked" "$exempt"
