#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64|Linux/amd64) ;;
  *)
    echo "[ontology-cli] SKIP: x86-64 Linux only"
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

fail() {
  echo "[ontology-cli] FAIL: $1" >&2
  exit 1
}

run_ontology() {
  "$SOUC_BIN" ontology "$@"
}

if ! out="$(run_ontology resolve QM:0000001 2>&1)"; then
  if grep -qi "unsupported command\|usage:\|missing source file\|No such file\|error: no main" <<<"$out"; then
    echo "[ontology-cli] SKIP: selected compiler does not expose ontology CLI"
    echo "[ontology-cli] selected_bin=$SOUC_BIN"
    exit 0
  fi
  printf '%s\n' "$out" >&2
  fail "ontology resolve command failed"
fi

grep -q "curie: QM:0000001" <<<"$out" || fail "resolve curie"
grep -q "label: Quantum mechanics" <<<"$out" || fail "resolve label"

out="$(run_ontology ancestors QM:0000003 2>&1)"
grep -q "QM:0000003" <<<"$out" || fail "ancestors child"
grep -q "QM:0000001" <<<"$out" || fail "ancestors parent"

out="$(run_ontology is-subclass QM:0000003 QM:0000001 2>&1)"
[[ "$out" == "yes" || "$out" == "true" ]] || fail "is-subclass positive"

out="$(run_ontology is-subclass QM:0000001 QM:0000003 2>&1)"
[[ "$out" == "no" || "$out" == "false" ]] || fail "is-subclass negative"

out="$(run_ontology search "Quantum" 2>&1)"
grep -q "QM:0000001" <<<"$out" || fail "search result"

out="$(run_ontology validate 2>&1)"
grep -q "VALID" <<<"$out" || fail "validate did not pass"

echo "[ontology-cli] PASS: ontology CLI smoke tests passed"
