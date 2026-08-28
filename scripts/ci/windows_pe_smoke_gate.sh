#!/usr/bin/env bash
# windows_pe_smoke_gate.sh
#
# Validates Linux -> Windows cross-compilation of the Sounio native backend.
# Compiles small .sio programs to x86_64-windows PE32+ executables, checks the
# binary structure, and -- when wine is available -- executes them and asserts
# exit code / stdout.
#
# Structural checks always run. Execution checks run only if `wine` is on PATH;
# otherwise they are SKIPPED (not failed) so the gate is useful in CI lanes
# without a Windows runtime.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SOUC="${SOUC:-./bin/souc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export WINEDEBUG="${WINEDEBUG:--all}"

fail() { echo "FAIL: $*" >&2; exit 1; }
have_wine() { command -v wine >/dev/null 2>&1; }

echo "== Windows PE smoke gate =="
echo "souc: $SOUC"
have_wine && echo "wine: $(wine --version 2>/dev/null)" || echo "wine: absent (execution checks will SKIP)"

# --- Case 1: exit code ----------------------------------------------------
cat > "$WORK/exit42.sio" <<'EOF'
fn main() -> i64 {
    return 42
}
EOF
"$SOUC" compile "$WORK/exit42.sio" -o "$WORK/exit42.exe" --target x86_64-windows >/dev/null
file "$WORK/exit42.exe" | grep -q "PE32+ executable" || fail "exit42.exe is not a PE32+ executable"
echo "PASS: exit42.exe is PE32+"

# --- Case 2: stdout -------------------------------------------------------
cat > "$WORK/print.sio" <<'EOF'
fn main() -> i64 {
    print("hello from windows\n")
    return 0
}
EOF
"$SOUC" compile "$WORK/print.sio" -o "$WORK/print.exe" --target x86_64-windows >/dev/null
file "$WORK/print.exe" | grep -q "PE32+ executable" || fail "print.exe is not a PE32+ executable"
strings "$WORK/print.exe" | grep -q "kernel32.dll" || fail "print.exe missing kernel32.dll import"
echo "PASS: print.exe is PE32+ and imports kernel32.dll"

# --- Case 3: file I/O round-trip -----------------------------------------
cat > "$WORK/fileio.sio" <<'EOF'
fn main() -> i64 {
    let buf = read_file("__IN__")
    write_file("__OUT__", buf, str_len(buf))
    print("done\n")
    return 0
}
EOF
# Inject absolute paths (the program embeds string literals at compile time).
sed -i "s#__IN__#$WORK/in.txt#; s#__OUT__#$WORK/out.txt#" "$WORK/fileio.sio"
printf 'roundtrip-payload-xyz' > "$WORK/in.txt"
"$SOUC" compile "$WORK/fileio.sio" -o "$WORK/fileio.exe" --target x86_64-windows >/dev/null
file "$WORK/fileio.exe" | grep -q "PE32+ executable" || fail "fileio.exe is not a PE32+ executable"
echo "PASS: fileio.exe is PE32+"

# --- Execution checks (wine) ---------------------------------------------
if have_wine; then
    # Use a dedicated prefix and PRE-CREATE its directory. If wine has to
    # auto-create a non-existent WINEPREFIX, it poisons exit-code propagation
    # for the whole session — the launcher then returns 1 for *every* program,
    # even `fn main(){return 7}` (root-caused 2026-05-21; see windows_native_souc.md).
    # A pre-existing prefix dir makes exit codes correct from the first call.
    export WINEPREFIX="$WORK/wineprefix"
    mkdir -p "$WINEPREFIX"
    wine cmd /c exit 0 >/dev/null 2>&1 || true   # initialize prefix synchronously

    set +e
    wine "$WORK/exit42.exe" >/dev/null 2>&1
    rc=$?
    set -e
    [ "$rc" -eq 42 ] || fail "exit42.exe returned $rc under wine, expected 42"
    echo "PASS: exit42.exe exits 42 under wine"

    out="$(wine "$WORK/print.exe" 2>/dev/null)"
    [ "$out" = "hello from windows" ] || fail "print.exe stdout was '$out', expected 'hello from windows'"
    echo "PASS: print.exe prints 'hello from windows' under wine"

    rm -f "$WORK/out.txt"
    wine "$WORK/fileio.exe" >/dev/null 2>&1
    [ -f "$WORK/out.txt" ] || fail "fileio.exe did not produce output file under wine"
    got="$(cat "$WORK/out.txt")"
    [ "$got" = "roundtrip-payload-xyz" ] || fail "fileio.exe round-trip got '$got', expected 'roundtrip-payload-xyz'"
    echo "PASS: fileio.exe round-trips a file under wine"
else
    echo "SKIP: wine absent -- execution checks not run"
fi

echo "== Windows PE smoke gate: OK =="
