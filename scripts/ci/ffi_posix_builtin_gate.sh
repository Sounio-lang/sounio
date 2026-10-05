#!/usr/bin/env bash
# P0-F POSIX extern "C" builtin gate — the durable artefact of the P0-F feature.
#
# Two things, both mandatory:
#   1. REFRAME arms — proves the E250 fail-closed guard still holds for names
#      NOT on the allowlist (so the reframe that reclassified P0-F stays true).
#      Includes the refutation branch: if an unimplemented extern ever COMPILES
#      and returns a fabricated 0 again, this gate FAILS.
#   2. Per-name EXECUTION witnesses — proves each allowlisted name actually RUNS
#      (real pid, real memory round trip, real exit/abort status, real system
#      side effect + wait status), not merely that it stopped raising E250.
#      Adding a name to name_is_native_backend_builtin turns off its E250 guard,
#      so "no longer E250" and "correct" are different claims; this gate asserts
#      the second.
#
# Self-attesting: records the exact commit under test and FAILS if it cannot.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# --- commit attestation (fixes the earlier commit=UNKNOWN provenance gap) ---
COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
[[ -z "$COMMIT" && -f "$ROOT_DIR/.p0f_commit" ]] && COMMIT="$(cat "$ROOT_DIR/.p0f_commit")"
[[ -z "$COMMIT" ]] && COMMIT="${P0F_COMMIT:-}"
if [[ -z "$COMMIT" ]]; then
  echo "FFI_POSIX_GATE_FAIL: cannot attest commit (no git, no .p0f_commit, no P0F_COMMIT)"; exit 1
fi

export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
export TMPDIR="${TMPDIR:-/tmp}"
W="$ROOT_DIR/tests/ffi_posix"
FAILED=0

# SOUC: a Madaros built from THIS source. Build from source unless a caller
# provides one it vouches for (never the stale checked-in ELF by default).
SOUC="${SOUNIO_GATE_SOUC:-}"
if [[ -z "$SOUC" ]]; then
  bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros > "$TMPDIR/ffi-posix-build.log" 2>&1 \
    || { echo "FFI_POSIX_GATE_FAIL: build failed"; tail -20 "$TMPDIR/ffi-posix-build.log"; exit 1; }
  SOUC="$ROOT_DIR/bin/souc"
fi

echo "=== FFI POSIX BUILTIN GATE ==="
echo "commit=$COMMIT  host=$(hostname)  souc=$("$SOUC" --version 2>&1 | head -1)"

pass(){ echo "  PASS  $1"; }
fail(){ echo "  FAIL  $1"; FAILED=1; }

# STRUCTURAL arm (the func_ref-path guard). A per-name EXECUTION witness proves
# only the path it happens to exercise; it says nothing about the func_ref
# authority the mirror checks. So assert the #1622 mirror directly: every name
# the checker allowlists (removes E250 for) must appear in the backend authority
# native_v2_builtin_id_for_func_ref under the SAME spelling. This is exactly the
# invariant a name like `free` vs recogniser `free_extern` violated silently —
# a byte-match that resolves at runtime while the declared authority disagrees.
echo "--- STRUCTURAL: checker allowlist <-> backend func_ref authority mirror ---"
if bash "$ROOT_DIR/scripts/ci/extern_builtin_mirror_gate.sh" > "$TMPDIR/ffi-mirror.log" 2>&1; then
  pass "extern-builtin mirror: checker and backend agree (no drift)"
else
  fail "extern-builtin mirror drift: $(grep -E '^[-+]' "$TMPDIR/ffi-mirror.log" | tr '\n' ' ')"
fi
run(){ timeout 120 "$SOUC" run "$1" 2>&1; }
check(){ timeout 120 "$SOUC" check "$1" 2>&1; }
compile(){ timeout 120 "$SOUC" compile "$1" -o "$2" 2>&1; }

echo "--- REFRAME arms (E250 guard must still hold; refutation branch live) ---"
# Control A: an implemented extern must execute (path is exercised).
run "$W/arm_control_implemented.sio" | grep -q 'CONTROL_A implemented-extern OK' && pass "arm_A implemented-extern executes" || fail "arm_A"
# Control B: the zero-detector must fire on a genuine zero (positive control).
run "$W/arm_control_plain_zero.sio" | grep -q 'CONTROL_B ZERO_OBSERVED' && pass "arm_B zero-detector fires" || fail "arm_B"
# Control C: undeclared plain call is E137, not E250 (extern-specificity).
CU="$(check "$W/arm_control_undeclared.sio")"
{ echo "$CU" | grep -q 'E137' && ! echo "$CU" | grep -q 'E250'; } && pass "arm_C undeclared->E137" || fail "arm_C"
# CLAIM / refutation: a garbage unimplemented extern MUST be refused (E250) and
# MUST NOT compile-and-return-0. If it ever fabricates a zero, this fails.
UC="$(check "$W/arm_claim_unimplemented.sio")"
UR="$(run "$W/arm_claim_unimplemented.sio")"
if echo "$UR" | grep -q 'CLAIM fabricated-zero v=0'; then
  fail "arm_CLAIM REGRESSION: unimplemented extern fabricated a zero"
elif echo "$UC" | grep -q 'E250'; then
  pass "arm_CLAIM unimplemented->E250 (fail-closed intact)"
else
  fail "arm_CLAIM inconclusive (no E250, no fabricated zero)"
fi

echo "--- per-name EXECUTION witnesses (must actually run, not just no-E250) ---"
# getpid / getppid: nonzero, distinct, AND getpid() == the process's real OS pid.
# Launch the ELF in the background so $! is its actual kernel-assigned pid, then
# assert the value getpid() printed equals it — the direct "equal to the real
# pid" check, not just nonzero.
compile "$W/wf_pid.sio" "$TMPDIR/wf_pid.elf" >/dev/null 2>&1
"$TMPDIR/wf_pid.elf" > "$TMPDIR/wf_pid.out" 2>&1 &
REALPID=$!
wait "$REALPID" 2>/dev/null
PIDOUT="$(cat "$TMPDIR/wf_pid.out")"
GP="$(echo "$PIDOUT" | grep -oE 'GETPID=[0-9]+' | head -1 | cut -d= -f2)"
if echo "$PIDOUT" | grep -q 'WF_PID PASS' && [[ "$GP" == "$REALPID" ]]; then
  pass "getpid==real pid ($GP) & getppid distinct"
else
  fail "getpid/getppid (printed=$GP real=$REALPID)"
fi

# malloc/free: write-then-read round trip through the returned pointer.
run "$W/wf_malloc_free.sio" | grep -q 'WF_MALLOC PASS' && pass "malloc/free write-read roundtrip" || fail "malloc/free"

# exit(7): observed process exit status == 7, UNREACHABLE line absent.
compile "$W/wf_exit.sio" "$TMPDIR/wf_exit.elf" >/dev/null 2>&1
EOUT="$("$TMPDIR/wf_exit.elf" 2>&1)"; ERC=$?
{ [[ "$ERC" -eq 7 ]] && ! echo "$EOUT" | grep -q 'UNREACHABLE'; } && pass "exit(7) -> status 7" || fail "exit (status=$ERC)"

# abort(): observed exit status 134 (128+SIGABRT), UNREACHABLE absent.
compile "$W/wf_abort.sio" "$TMPDIR/wf_abort.elf" >/dev/null 2>&1
AOUT="$("$TMPDIR/wf_abort.elf" 2>&1)"; ARC=$?
{ [[ "$ARC" -eq 134 ]] && ! echo "$AOUT" | grep -q 'UNREACHABLE'; } && pass "abort() -> status 134" || fail "abort (status=$ARC)"

# exit(1) reached transitively through a stdlib constructor: sed_ker_lz_gen(k)
# with k out of [0,3] must hard-refuse via process_exit(1), not fabricate a
# zero. Observed exit status 1, UNREACHABLE absent. (Fail-closed guard for
# stdlib/algebra/sedenion_kernel.sio; the zero it used to return trivially
# passed the kernel predicate while being no generator.)
compile "$W/sed_ker_lz_gen_refuse.sio" "$TMPDIR/skg_refuse.elf" >/dev/null 2>&1
chmod +x "$TMPDIR/skg_refuse.elf" 2>/dev/null || true
KOUT="$("$TMPDIR/skg_refuse.elf" 2>&1)"; KRC=$?
{ [[ "$KRC" -eq 1 ]] && echo "$KOUT" | grep -q 'SKG before' && ! echo "$KOUT" | grep -q 'UNREACHABLE'; } \
  && pass "sed_ker_lz_gen(99) -> process_exit(1) (status 1, UNREACHABLE absent)" \
  || fail "sed_ker_lz_gen refuse (status=$KRC)"

# system(): observable side effect (sentinel file) AND real wait status (768).
rm -f "$TMPDIR/p0f_system_sentinel"
SOUT="$(run "$W/wf_system.sio")"
{ echo "$SOUT" | grep -q 'WF_SYSTEM PASS' && [[ -f "$TMPDIR/p0f_system_sentinel" ]]; } \
  && pass "system() side-effect + wait status 768" || fail "system"
rm -f "$TMPDIR/p0f_system_sentinel"

echo "=== VERDICT ==="
if [[ "$FAILED" -eq 0 ]]; then
  echo "FFI_POSIX_GATE_OK commit=$COMMIT — all reframe arms hold and all 7 allowlisted names EXECUTE"
  exit 0
else
  echo "FFI_POSIX_GATE_FAIL commit=$COMMIT — see FAIL lines above"
  exit 1
fi
