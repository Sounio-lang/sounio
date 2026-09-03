#!/usr/bin/env bash
# Selftest for scripts/lib/gate_assert.sh ELF / engine / rc helpers.
#
# A helper that is never shown to fail is not a helper. This gate breaks
# the instrument lies on purpose:
#   1. a non-ELF artefact (the `-o` swallow class)
#   2. a compile log that is lean_single while --version would say Madaros
#   3. an rc read through a pipe (empty) vs a file-backed capture
#   4. a skip for a missing tool that used to exit 0 (cursor-2 skip-vacuous)
#
# Two fixture gates (the positive control): a skip must print SKIPPED and
# leave rc!=0; a compile-to-void must refuse. If this helper reports OK
# on either bad case, it is not a helper.
#
# See docs/audit/GATE_INSTRUMENT_CENSUS_2026-08-18.md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "gate_assert_instrument_selftest"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/gate-assert-instrument.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# --- require_elf ---
if ( require_elf "$TMP/missing" "missing" ) >/dev/null 2>&1; then
  gate_fail "require_elf accepted a missing path"
fi
: > "$TMP/empty"
if ( require_elf "$TMP/empty" "empty" ) >/dev/null 2>&1; then
  gate_fail "require_elf accepted an empty file"
fi
printf 'not an elf\n' > "$TMP/dash-o"
if ( require_elf "$TMP/dash-o" "text" ) >/dev/null 2>&1; then
  gate_fail "require_elf accepted a non-ELF file (the -o swallow class)"
fi
# minimal ELF magic, rest padding
printf '\x7fELF' > "$TMP/real.elf"
printf '\0%.0s' {1..12} >> "$TMP/real.elf"
require_elf "$TMP/real.elf" "fixture ELF"
echo "require_elf: missing/empty/text refuse, 7fELF accepts"

# --- classify_compile_log: must NOT trust --version ---
cat > "$TMP/madaros.log" <<'EOF'
Madaros v0.80.0 -- the Sounio self-hosted compiler
the bare highland that does not negotiate with ill-formed code -- Sfakia, Crete
Compilation successful!
   Output: /tmp/w2.elf
EOF
cat > "$TMP/lean.log" <<'EOF'
source: t.sio 24 bytes
total source: 10120 bytes
tokens: 2401
elf: -o 36924 bytes (bss=1048984)
EOF
[[ "$(classify_compile_log "$TMP/madaros.log")" == "madaros" ]] \
  || gate_fail "classify_compile_log missed Madaros banner"
[[ "$(classify_compile_log "$TMP/lean.log")" == "lean_single" ]] \
  || gate_fail "classify_compile_log missed lean_single log (the -o 36924 class)"
require_compile_engine "$TMP/madaros.log" madaros
if ( require_compile_engine "$TMP/lean.log" madaros ) >/dev/null 2>&1; then
  gate_fail "require_compile_engine accepted lean_single as madaros"
fi
echo "classify_compile_log: madaros vs lean_single (log, not --version)"

# --- gate_capture_rc vs pipe ---
gate_capture_rc "$TMP/rc17" -- bash -c 'exit 17'
require_rc_file "$TMP/rc17" 17
# the E230 round-3 lie: grep a stream that never contained run_rc=
pipe_rc="$(tail -25 /dev/null | grep -E '^run_rc=' | tail -1 | awk -F= '{print $2}' || true)"
[[ -z "$pipe_rc" ]] || gate_fail "pipe extraction unexpectedly produced: $pipe_rc"
if ( require_rc_file "$TMP/missing.rc" ) >/dev/null 2>&1; then
  gate_fail "require_rc_file accepted a missing file"
fi
echo "gate_capture_rc: file-backed 17; pipe extraction empty"

# --- fixture 1: a gate that skips must print SKIPPED and not exit 0 ---
cat > "$TMP/skip_gate.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "skip_fixture"
require_tool definitely-not-installed-ptxas-xyz "no ptxas"
gate_pass "reached pass — skip was silent"
EOF
set +e
bash "$TMP/skip_gate.sh" > "$TMP/skip.out" 2> "$TMP/skip.err"
skip_rc=$?
set -e
if [[ "$skip_rc" -eq 0 ]]; then
  gate_fail "skip fixture exited 0 — that is the skip-vacuous green (cursor-2 measured=no)"
fi
if ! grep -qE 'SKIPPED' "$TMP/skip.out" "$TMP/skip.err"; then
  gate_fail "skip fixture rc=$skip_rc but did not print SKIPPED"
fi
if grep -q 'measured=yes' "$TMP/skip.out" "$TMP/skip.err"; then
  gate_fail "skip fixture claimed measured=yes"
fi
grep -q 'measured=no' "$TMP/skip.out" "$TMP/skip.err" \
  || gate_fail "skip fixture did not print measured=no"
echo "skip fixture: rc=$skip_rc SKIPPED measured=no (not 0)"

# --- fixture 2: compile-to-void must refuse ---
cat > "$TMP/void_gate.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "void_fixture"
# Fake a successful compile that wrote nothing.
: > "$TMP/void.elf"
require_elf "$TMP/void.elf" "compile artefact"
gate_pass "accepted empty compile"
EOF
set +e
bash "$TMP/void_gate.sh" > "$TMP/void.out" 2> "$TMP/void.err"
void_rc=$?
set -e
if [[ "$void_rc" -eq 0 ]]; then
  gate_fail "void-compile fixture exited 0 — empty artefact read as success"
fi
if grep -qE '_OK' "$TMP/void.out"; then
  gate_fail "void-compile fixture printed OK"
fi
echo "void-compile fixture: rc=$void_rc refused empty artefact"

gate_pass "ELF magic, compile-log engine, file-backed rc, skip!=0"
echo "GATE_ASSERT_INSTRUMENT_SELFTEST_OK"
