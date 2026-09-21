#!/usr/bin/env bash
# The sabotage gate's own control matrix, versioned.
#
# It lived in a PR description, where it ran once and then never again. A
# control that is not executed is a control that does not exist -- which is the
# gate's own thesis, and the third time this week it was violated by the thing
# arguing it (the gate landed unnamed, then named-but-inert, then with an
# unversioned self-test).
#
# Every cell must produce a DIFFERENT verdict. The first version of this harness
# returned "died at run" in all four cells because the fake compiler exited 0
# without writing its output, and the gate read "cannot run" as "died". A matrix
# whose cells agree measures nothing.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "witness_sabotage_gate_selftest"

W="$(mktemp -d /tmp/sab-selftest.XXXXXX)"; trap 'rm -rf "$W"' EXIT
mkdir -p "$W/scripts/ci" "$W/scripts/lib" "$W/tests/run-pass"
cp "$ROOT_DIR/scripts/ci/witness_declares_its_sabotage_gate.sh" "$W/scripts/ci/"
cp "$ROOT_DIR/scripts/lib/gate_assert.sh" "$W/scripts/lib/"

# A synthetic compiler. Always writes an executable unless told to fail, so
# "could not build" never masquerades as "died".
cat > "$W/fakemad" <<'EOF'
#!/usr/bin/env bash
if [ "$1" = "build" ]; then
  if [ "${FAKE_BUILD_FAILS:-0}" = "1" ] && [ -n "${SOUNIO_WIDE_MUL_SABOTAGE:-}" ]; then
    echo "${FAKE_BUILD_MSG:-error[E999]: refused under sabotage}" >&2
    exit "${FAKE_BUILD_RC:-1}"
  fi
  printf '#!/bin/sh\nexec "%s"\n' "$FAKE_RUN" > "$3"; chmod +x "$3"; exit 0
fi
exit 0
EOF
chmod +x "$W/fakemad"

total=0; passed=0; failed=0
cell() {  # name | expected substring | witness-header | sabotaged-body | extra env
  local name="$1" want="$2" header="$3" body="$4"; shift 4
  total=$((total + 1))
  printf '%s\nfn main() {}\n' "$header" > "$W/tests/run-pass/a.sio"
  printf '//@ run-pass\nfn main() {}\n' > "$W/tests/run-pass/b.sio"
  # A body prefixed ALWAYS: runs whether or not the sabotage is set -- the only
  # way to make the CLEAN control fail, which is what "unjudgeable" means.
  if [[ "$body" == ALWAYS:* ]]; then
    printf '#!/usr/bin/env bash\n%s\nexit 0\n' "${body#ALWAYS:}" > "$W/run_x"
  else
    printf '#!/usr/bin/env bash\nif [ -n "${SOUNIO_WIDE_MUL_SABOTAGE:-}" ]; then\n%s\nfi\nexit 0\n' "$body" > "$W/run_x"
  fi
  chmod +x "$W/run_x"
  local out
  out="$(cd "$W" && env FAKE_RUN="$W/run_x" SOUNIO_WITNESS_SABOTAGE_MADAROS="$W/fakemad" "$@" \
        bash scripts/ci/witness_declares_its_sabotage_gate.sh 2>&1)"
  if grep -qF "$want" <<<"$out"; then
    printf '  ok    %-26s -> %s\n' "$name" "$want"; passed=$((passed + 1))
  else
    printf '  FAIL  %-26s expected %s\n' "$name" "$want"
    sed 's/^/          /' <<<"$out" | head -8
    failed=$((failed + 1))
  fi
}

H='//@ run-pass\n//@ sabotage: wide-mul'
cell "clean run-fail"      "died at run under wide-mul"   "$(printf "$H")" 'exit 1'
cell "survives sabotage"   "SURVIVED"                     "$(printf "$H")" 'exit 0'
cell "dies by signal"      "died by signal 11"            "$(printf "$H")" 'kill -SEGV $$; sleep 5'
cell "times out"           "TIMED OUT"                    "$(printf "$H")" 'exit 124'
cell "refused at compile"  "refused at compile"           "$(printf '%s\n//@ sabotage-expect: compile-refused' "$(printf "$H")")" 'exit 1' FAKE_BUILD_FAILS=1
cell "wrong death class"   "MISATTRIBUTED"                "$(printf '%s\n//@ sabotage-expect: compile-refused' "$(printf "$H")")" 'exit 1'
cell "wrong diagnostic"    "MISATTRIBUTED"                "$(printf '%s\n//@ sabotage-error-pattern: E242' "$(printf "$H")")" 'exit 1' FAKE_BUILD_FAILS=1
cell "unjudgeable at ceiling" "NOT JUDGED"                "$(printf "$H")" 'ALWAYS:exit 1' SOUNIO_WITNESS_UNJUDGEABLE_CEILING=1
cell "unjudgeable over ceiling" "unjudgeable witnesses rose"  "$(printf "$H")" 'ALWAYS:exit 1' SOUNIO_WITNESS_UNJUDGEABLE_CEILING=0

echo "witness_sabotage_gate_selftest: total=$total passed=$passed failed=$failed"
if [[ $failed -ne 0 ]]; then
  gate_fail "$failed control cell(s) did not produce their distinct verdict"
fi
gate_pass "$passed/$total cells each produced a different, expected verdict"
exit 0
