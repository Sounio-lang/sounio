#!/usr/bin/env bash
# Prove contextual local bindings. The historical filename is retained because
# `scope` was the first witness; `policy`, `is`, and `study` use the same contract.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${SOUNIO_MADAROS_SCOPE_CONTEXTUAL_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_MADAROS_SCOPE_CONTEXTUAL_KEEP:-0}"
WORK="${SOUNIO_MADAROS_SCOPE_CONTEXTUAL_DIR:-}"
BINDING_KIND="${SOUNIO_MADAROS_CONTEXTUAL_BINDING_KIND:-scope}"

case "$BINDING_KIND" in
  scope)
    SOURCE="$ROOT_DIR/tests/run-pass/let_scope_binding_name.sio"
    EXPECTED_BINDING='let scope = ScopeBindingPayload'
    EXPECTED_RETURN='return scope.value - 42'
    EXPECTED_MARKER='LET_SCOPE_BINDING_OK'
    ;;
  policy)
    SOURCE="$ROOT_DIR/tests/run-pass/let_policy_binding_name.sio"
    EXPECTED_BINDING='let policy = PolicyBindingPayload'
    EXPECTED_RETURN='return policy.value - 42'
    EXPECTED_MARKER='LET_POLICY_BINDING_OK'
    ;;
  is)
    SOURCE="$ROOT_DIR/tests/run-pass/let_is_binding_name.sio"
    EXPECTED_BINDING='let is = IsBindingPayload'
    EXPECTED_RETURN='return is.value - 42'
    EXPECTED_MARKER='LET_IS_BINDING_OK'
    ;;
  study)
    SOURCE="$ROOT_DIR/tests/run-pass/let_study_binding_name.sio"
    EXPECTED_BINDING='let study = StudyBindingPayload'
    EXPECTED_RETURN='return study.value - 42'
    EXPECTED_MARKER='LET_STUDY_BINDING_OK'
    ;;
  *)
    echo "[madaros-scope-contextual-binding] FAIL: unsupported contextual binding kind: $BINDING_KIND" >&2
    exit 2
    ;;
esac

fail() {
  echo "[madaros-scope-contextual-binding] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

require_raw_elf() {
  local path="$1"
  [[ -x "$path" && -s "$path" ]] || fail "Madaros input is missing, empty, or not executable: $path"
  [[ "$(head -c4 "$path" 2>/dev/null)" == $'\x7fELF' ]] || fail "Madaros input is not a raw ELF: $path"
}

if [[ "${1:-}" == '--structural-only' ]]; then
  [[ $# -eq 1 ]] || fail 'usage: madaros_scope_contextual_binding_gate.sh [--structural-only]'
  [[ -f "$SOURCE" ]] || fail "$BINDING_KIND contextual witness is missing: $SOURCE"
  grep -Fxq '//@ run-pass' "$SOURCE" || fail "$BINDING_KIND contextual witness is not a run-pass test"
  grep -Fq "$EXPECTED_BINDING" "$SOURCE" || fail "$BINDING_KIND contextual witness does not bind $BINDING_KIND"
  grep -Fq "$EXPECTED_RETURN" "$SOURCE" || fail "$BINDING_KIND contextual witness does not exercise field access in return"
  echo "[madaros-scope-contextual-binding] PASS: direct raw contextual-$BINDING_KIND witness wiring is present"
  exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: madaros_scope_contextual_binding_gate.sh [--structural-only]'

[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit Madaros ELF'
require_raw_elf "$RAW_MADAROS"
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ -f "$SOURCE" ]] || fail "$BINDING_KIND contextual witness is missing: $SOURCE"

RAW_SHA256="$(portable_sha256 "$RAW_MADAROS")"
if [[ -n "$EXPECTED_RAW_SHA256" && "$RAW_SHA256" != "$EXPECTED_RAW_SHA256" ]]; then
  fail "raw ELF SHA-256 mismatch: expected=$EXPECTED_RAW_SHA256 actual=$RAW_SHA256"
fi

if [[ -n "$WORK" ]]; then
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-scope-contextual.XXXXXX)"
fi
if [[ "$KEEP_WORK" != '1' ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

OUT="$WORK/let_${BINDING_KIND}_binding_name.elf"
COMPILE_LOG="$WORK/compile.log"
RUNTIME_LOG="$WORK/runtime.log"

set +e
(
  cd "$WORK"
  exec env \
    -u MADAROS_RAW_BIN \
    -u SOUNIO_MADAROS_BIN \
    SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    "$RAW_MADAROS" --native-v2-compile "$SOURCE" "$OUT"
) >"$COMPILE_LOG" 2>&1
COMPILE_RC=$?
set -e
[[ "$COMPILE_RC" -eq 0 ]] || {
  cat "$COMPILE_LOG" >&2
  fail "contextual $BINDING_KIND witness did not compile (rc=$COMPILE_RC)"
}

[[ -s "$OUT" ]] || fail "contextual $BINDING_KIND witness produced no output ELF"
chmod +x "$OUT"
[[ "$(head -c4 "$OUT" 2>/dev/null)" == $'\x7fELF' ]] || fail "contextual $BINDING_KIND witness output is not an ELF"

set +e
(cd "$WORK" && "$OUT") >"$RUNTIME_LOG" 2>&1
RUNTIME_RC=$?
set -e
[[ "$RUNTIME_RC" -eq 0 ]] || {
  cat "$RUNTIME_LOG" >&2
  fail "contextual $BINDING_KIND witness ELF exited rc=$RUNTIME_RC"
}
grep -Fxq "$EXPECTED_MARKER" "$RUNTIME_LOG" || {
  cat "$RUNTIME_LOG" >&2
  fail "contextual $BINDING_KIND witness lost its exact marker"
}

echo "[madaros-scope-contextual-binding] PASS: direct raw ELF preserves contextual $BINDING_KIND binding raw_sha256=$RAW_SHA256"
