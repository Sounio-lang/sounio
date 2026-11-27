#!/usr/bin/env python3
# pkg_dispatch_usage_gate.py — the pkg/install CLI usage errors must exit
# non-zero. A usage error that prints a diagnostic and then returns exits 0:
# the silent-success class the wrapper-hardening campaign exists to refuse
# (#3378, #3400, #3514).
#
# WHY.
#
# `run_pkg_dispatch` and `run_pkg_install_shorthand` in
# self-hosted/compiler/main.sio answered the three pkg/install usage errors by
# printing the diagnostic and then RETURNING, so the process exited 0 on a
# command that did nothing:
#
#   $ souc pkg              # "souc pkg: missing subcommand"        -> rc=0
#   $ souc pkg nosuchverb   # "souc pkg: unknown subcommand: ..."   -> rc=0
#   $ souc install          # "souc install: missing package name"  -> rc=0
#
# A shell script that runs `souc pkg $sub` against a typo would then report
# success on a command that produced nothing. The file's working idiom for a
# failing command is `panic` (rc=1), used by run_check_mode and the
# ir-dump/ir-roundtrip modes; `exit_process()` is a documented no-op under the
# lean_single seed (E200), which is why it was never an option here.
#
# A guard that refuses everything would pass a one-sided test, so both
# directions are driven: the three refusals must fire (non-zero AND name the
# problem), and the honest paths (pkg help / self-test / info, and a versioned
# install) must still exit 0. The registry stubs (`pkg info`, `install
# <name>@<version>`) are deliberate R0-R2 no-ops and stay rc=0.
#
# HOW. `bin/souc` is driven with the compiler under test pinned through
# MADAROS_RAW_BIN, so the gate measures exactly one build (a current-source
# Madaros) rather than whatever the tree happens to resolve. Each verb runs in
# a throwaway working directory so no stray sounio.toml, package cache or
# locally named file can change the answer.
#
# Usage:
#   MADAROS_RAW_BIN=/path/to/madaros python3 scripts/ci/pkg_dispatch_usage_gate.py
#
# Exit 0 only when every case behaves as stated; the failure lines name the
# case, the expectation and what was observed.
import os
import subprocess
import sys
import tempfile

GATE = "pkg_dispatch_usage_gate"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOUC = os.path.join(ROOT, "bin", "souc")

# (args, diagnostic the refusal must carry). All three are usage errors: the
# process must exit non-zero and say why.
REFUSALS = (
    (["pkg"], "missing subcommand"),
    (["pkg", "nosuchverb"], "unknown subcommand"),
    (["install"], "missing package name"),
)

# The honest paths: these must keep exiting 0. pkg self-test / info and a
# versioned install are deliberate registry stubs, not usage errors.
CONTROLS = (
    ["pkg", "help"],
    ["pkg", "self-test"],
    ["pkg", "info"],
    ["install", "example@1.0.0"],
)

failures = []
cases = 0
refusals = 0


def record(name, ok, detail=""):
    global cases
    cases += 1
    if ok:
        print(f"  ok   {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name} — {detail}", file=sys.stderr)


def gate_env(raw):
    env = dict(os.environ)
    # Pin the compiler under test and remove every other routing knob so the
    # invocation cannot silently fall back to the committed (pre-fix) ELF.
    for var in ("SOUC_BIN", "SOUNIO_SOUC_BIN", "SOUNIO_MADAROS_BIN", "SOUNIO_SOUC_ENGINE"):
        env.pop(var, None)
    env["MADAROS_RAW_BIN"] = raw
    env["SOUNIO_STDLIB_PATH"] = os.path.join(ROOT, "stdlib")
    return env


def run_invoke(raw, args, cwd):
    proc = subprocess.run(
        [SOUC, *args],
        cwd=cwd,
        env=gate_env(raw),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    return proc.returncode, out


def refusal_case(raw, work, args, needle):
    global refusals
    name = " ".join(args)
    rc, out = run_invoke(raw, args, work)
    refusals += 1
    record(f"{name}: refused non-zero", rc != 0, f"rc=0 (silent success) out={out[-200:]!r}")
    record(f"{name}: names the refusal", needle in out, f"output lacks {needle!r}: {out[-200:]!r}")


def control_case(raw, work, args):
    name = " ".join(args)
    rc, out = run_invoke(raw, args, work)
    record(f"{name}: rc=0", rc == 0, f"rc={rc}: {out[-200:]!r}")


def main():
    raw = os.environ.get("MADAROS_RAW_BIN", "").strip()
    if not raw:
        print(
            "GATE FAIL: MADAROS_RAW_BIN is unset — pin the compiler under test "
            "(a current-source Madaros), not the committed prebuilt",
            file=sys.stderr,
        )
        return 2
    if not (os.path.isfile(raw) and os.access(raw, os.X_OK)):
        print(f"GATE FAIL: MADAROS_RAW_BIN is not an executable: {raw}", file=sys.stderr)
        return 2
    if not (os.path.isfile(SOUC) and os.access(SOUC, os.X_OK)):
        print(f"GATE FAIL: {SOUC} is missing — the dispatcher under test does not exist", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="pkg-usage-gate-") as work:
        print("[pkg-usage] usage errors must exit non-zero:")
        for args, needle in REFUSALS:
            refusal_case(raw, work, args, needle)

        print("[pkg-usage] honest paths must exit 0:")
        for args in CONTROLS:
            control_case(raw, work, args)

    expected = len(REFUSALS) + len(CONTROLS)
    if cases < expected:
        print(f"GATE FAIL: only {cases} cases ran of {expected} — a case silently emptied", file=sys.stderr)
        return 1
    if failures:
        print(file=sys.stderr)
        print(f"PKG_DISPATCH_USAGE_GATE: {len(failures)} case(s) wrong of {cases}.", file=sys.stderr)
        print("  A pkg/install usage error must exit non-zero and name the problem;", file=sys.stderr)
        print("  the honest paths (help/self-test/info, versioned install) must exit 0.", file=sys.stderr)
        return 1
    print(f"PKG_DISPATCH_USAGE_GATE_OK: {cases} cases, {refusals} of them refusals, each behaved as stated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
