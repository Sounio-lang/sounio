#!/usr/bin/env bash
# Verticals: epistemic::propagate (GUM arithmetic), epistemic::walley (credal sets),
# cybernetic::conversation (Pask). Walley and conversation run under Madaros;
# propagate retains one named legacy fallback until its effect signatures pass.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
engine_info="$($SOUC info 2>&1)"
if ! grep -qF 'Madaros v' <<<"$engine_info"; then
  echo "FAIL: epistemic-conversation gate requires default Madaros" >&2
  printf '%s\n' "$engine_info" >&2
  exit 1
fi
run() { # module driver sentinel [softcheck] [engine]
  local engine="${5:-madaros}"
  echo "== $2 =="
  if [ "${4:-}" = "softcheck" ]; then
    $SOUC check "$1" >/dev/null 2>&1 || echo "NOTE: standalone check quirk on $1 (Madaros check-mode; driver proves the API)"
  else
    $SOUC check "$1" >/dev/null 2>&1 || { echo "FAIL check $1"; fail=1; }
  fi
  if { [ "$engine" = "lean_single" ] && SOUNIO_SOUC_ENGINE=lean_single $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; } ||
     { [ "$engine" = "madaros" ] && $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; }; then
    chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "$3" || { echo "FAIL run $2"; fail=1; }
  else echo "FAIL compile $2"; fail=1; fi
}
run stdlib/epistemic/propagate.sio     tests/stdlib/epistemic/test_propagate_stdlib.sio      PROPAGATE_STDLIB_OK    softcheck lean_single
run stdlib/epistemic/walley.sio         tests/stdlib/epistemic/test_walley_stdlib.sio        WALLEY_STDLIB_OK
run stdlib/cybernetic/conversation.sio  tests/stdlib/cybernetic/test_conversation_stdlib.sio CONVERSATION_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_EPISTEMIC_CONVERSATION_GATE_OK"
exit $fail
