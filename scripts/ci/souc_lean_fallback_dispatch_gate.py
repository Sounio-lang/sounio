#!/usr/bin/env python3
# souc_lean_fallback_dispatch_gate.py — the lean_single paths must dispatch the
# public verbs honestly: preserve the caller's source bytes, produce the
# artifact that was asked for, and refuse the ones lean_single cannot produce.
#
# WHY.
#
# bin/souc's default engine is Madaros; when no Madaros ELF resolves it falls
# back to the legacy lean_single ELF, and SOUNIO_SOUC_ENGINE=lean_single forces
# the same engine. lean_single speaks a raw `SRC OUT` CLI and emits a native
# x86-64 ELF only. Two dispatch defects lived in those paths:
#
#   1. The fallback exec'd the raw ELF with the verb arguments UNTRANSLATED, so
#      `souc run prog.sio` compiled a source literally named `run` and wrote the
#      output ELF over prog.sio when such a file existed in the working
#      directory — exit 0, the caller's source replaced by a binary (PR #3400).
#   2. The compile|build translation parsed only `-o` and silently dropped every
#      other option, so `souc build prog.sio --backend gpu -o OUT` exited 0 with
#      a NATIVE ELF under OUT where the public `souc --help` contract promises a
#      GPU (PTX) artifact — the wrong artifact type reporting success (PR #3400
#      review handoff, 2026-11-01).
#   3. The same loop's `*) shift` arm also dropped `--target`/`--target=...`
#      and the short `-t` alias, so `souc build prog.sio --target wasm32 -o
#      OUT` (a docs/FAQ.md form) or the FPGA flow's `souc compile ... -t
#      fpga-u250 -o OUT.cpp` exited 0 with a native ELF under OUT (PR #3400
#      review, 2026-11-02).
#      lean_single's raw CLI accepts even bogus targets and still emits a
#      native ELF, so the wrapper refuses every target except the native
#      x86_64-linux it actually produces — it does NOT pass selectors through.
#
# Both defects are silent successes: nothing complained, and the measurement or
# the filesystem then told a lie. This gate exists because no committed test
# exercised the AUTOMATIC fallback at all — every existing gate either forces
# the engine or pins SOUNIO_TEST_SOUC_BIN — so a regression here merged green.
#
# HOW. The automatic fallback is driven in a Madaros-less sandbox: a throwaway
# tree holding only bin/souc and the committed lean_single ELF, so no Madaros
# override, host build or committed prebuilt can resolve and the fallback is
# the only path left. Each verb is exercised from a working directory that
# contains an adversarial decoy file named after the verb (`run`, `check`,
# `compile`, `build`): the dispatch must translate the verb, not mistake it for
# a source path. Source and decoy bytes are hashed before and after; the
# compiled artifact must exist, carry the ELF magic, and run.
#
# Both directions are driven, as in compiler_override_fail_closed_gate.sh: the
# refusals must fire (non-native backends, --gpu-target, a missing --backend
# value — and no output file may be written), and the honest configurations
# (no backend, --backend native, run/check/compile/build, the default Madaros
# path) must still work — a guard that rejects everything would pass a
# one-sided test.
#
# Exit 0 only when every case behaves as stated; the failure lines name the
# case, the expectation and what was observed.
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile

GATE = "souc_lean_fallback_dispatch_gate"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOUC = os.path.join(ROOT, "bin", "souc")
LEAN = os.path.join(ROOT, "bin", "souc-lean-single-x86_64")
MATERIALIZER = os.path.join(ROOT, "scripts", "lib", "materialize_madaros_prebuilt.sh")
MARKER = "251"
FIXTURE = (
    "fn main() -> i64 with IO {\n"
    '    print_int(251)\n'
    '    print_char(10)\n'
    "    0\n"
    "}\n"
)

failures = []
cases = 0
refusals = 0


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def is_elf(path):
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as fh:
        return fh.read(4) == b"\x7fELF"


def record(name, ok, detail=""):
    global cases
    cases += 1
    if ok:
        print(f"  ok   {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name} — {detail}", file=sys.stderr)


def scrubbed_env(extra=None):
    env = dict(os.environ)
    for var in (
        "SOUC_BIN",
        "SOUNIO_SOUC_BIN",
        "MADAROS_RAW_BIN",
        "SOUNIO_MADAROS_BIN",
        "SOUNIO_REQUIRE_COMMITTED_MADAROS",
        # An inherited SOUNIO_SOUC_ENGINE=lean_single would silently turn every
        # purported automatic-fallback case into a forced-engine case and then
        # fail the Madaros controls even when the dispatch is correct (PR #3400
        # review, 2026-11-02). The forced cases below restore it explicitly
        # through extra_env, so the automatic cases are automatic regardless of
        # the environment this gate is invoked from.
        "SOUNIO_SOUC_ENGINE",
    ):
        env.pop(var, None)
    env["SOUNIO_STDLIB_PATH"] = os.path.join(ROOT, "stdlib")
    if extra:
        env.update(extra)
    return env


def run_invoke(souc, args, cwd, extra_env=None):
    proc = subprocess.run(
        [souc, *args],
        cwd=cwd,
        env=scrubbed_env(extra_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")


def write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def verb_case(label, souc, work, name, verb, extra_env=None):
    """Adversarial dispatch: a decoy file named after the verb sits in the cwd.

    The verb must be translated (lean_single's raw SRC OUT form), the source
    must survive byte-identical, and the decoy must be left untouched."""
    case_dir = os.path.join(work, name)
    os.makedirs(case_dir)
    src = os.path.join(case_dir, "prog.sio")
    decoy = os.path.join(case_dir, verb)
    write(src, FIXTURE)
    write(decoy, "decoy: not a sounio source\n")
    src_before, decoy_before = sha256(src), sha256(decoy)

    if verb in ("compile", "build"):
        out = os.path.join(case_dir, "out.elf")
        rc, stdout, stderr = run_invoke(souc, [verb, "prog.sio", "-o", "out.elf"], case_dir, extra_env)
        record(f"{label}: {verb} rc=0", rc == 0, f"rc={rc}: {stderr[-300:]}")
        record(f"{label}: {verb} wrote a native ELF", is_elf(out), f"missing or not an ELF: {out}")
        if is_elf(out):
            os.chmod(out, os.stat(out).st_mode | stat.S_IXUSR)
            proc = subprocess.run([out], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
            got = proc.stdout.decode("utf-8", "replace").strip()
            record(f"{label}: {verb} artifact runs", proc.returncode == 0 and MARKER in got, f"rc={proc.returncode} out={got!r}")
    elif verb == "run":
        rc, stdout, stderr = run_invoke(souc, ["run", "prog.sio"], case_dir, extra_env)
        record(f"{label}: run rc=0 with program output", rc == 0 and MARKER in stdout, f"rc={rc} stdout={stdout[-200:]!r} stderr={stderr[-200:]!r}")
    else:  # check
        rc, stdout, stderr = run_invoke(souc, ["check", "prog.sio"], case_dir, extra_env)
        record(f"{label}: check rc=0", rc == 0, f"rc={rc}: {stderr[-300:]}")

    record(f"{label}: {verb} preserved the source", sha256(src) == src_before, "prog.sio bytes changed")
    record(f"{label}: {verb} left the decoy untouched", sha256(decoy) == decoy_before, f"decoy '{verb}' bytes changed")


def refusal_case(label, souc, work, name, args, must_not_exist, needle, extra_env=None):
    """An unsupported backend request must fail nonzero, say why, and write nothing."""
    global refusals
    case_dir = os.path.join(work, name)
    os.makedirs(case_dir)
    write(os.path.join(case_dir, "prog.sio"), FIXTURE)
    rc, stdout, stderr = run_invoke(souc, args, case_dir, extra_env)
    refusals += 1
    record(f"{label}: {name} refused nonzero", rc != 0, f"rc=0 (silent success) stdout={stdout[-150:]!r}")
    record(
        f"{label}: {name} names the refusal",
        "E-SRB-BACKEND-001" in stderr and needle in stderr,
        f"stderr lacks E-SRB-BACKEND-001/{needle!r}: {stderr[-300:]!r}",
    )
    absent = not os.path.exists(os.path.join(case_dir, must_not_exist))
    record(f"{label}: {name} wrote no misleading artifact", absent, f"{must_not_exist} exists (is_elf={is_elf(os.path.join(case_dir, must_not_exist))})")


def acceptance_case(label, souc, work, name, extra_args, extra_env=None):
    """--backend native is truthful for lean_single: it must still compile."""
    case_dir = os.path.join(work, name)
    os.makedirs(case_dir)
    write(os.path.join(case_dir, "prog.sio"), FIXTURE)
    rc, stdout, stderr = run_invoke(souc, ["build", "prog.sio", *extra_args, "-o", "out.elf"], case_dir, extra_env)
    out = os.path.join(case_dir, "out.elf")
    record(f"{label}: {name} rc=0", rc == 0, f"rc={rc}: {stderr[-300:]}")
    record(f"{label}: {name} produced a native ELF", is_elf(out), "out.elf missing or not an ELF")


def main():
    for path, why in ((SOUC, "bin/souc is missing"), (LEAN, "the committed lean_single ELF is missing")):
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            print(f"GATE FAIL: {why} — the dispatch under test does not exist", file=sys.stderr)
            return 1

    with tempfile.TemporaryDirectory(prefix="souc-lean-fallback-gate-") as work:
        # A Madaros-less tree: the automatic fallback is the only path left.
        fake_bin = os.path.join(work, "tree", "bin")
        os.makedirs(fake_bin)
        fake_souc = os.path.join(fake_bin, "souc")
        shutil.copy2(SOUC, fake_souc)
        shutil.copy2(LEAN, os.path.join(fake_bin, "souc-lean-single-x86_64"))

        print("[fallback-dispatch] automatic fallback, adversarial cwd:")
        for verb in ("run", "check", "compile", "build"):
            verb_case("fallback", fake_souc, work, f"auto-{verb}", verb)

        print("[fallback-dispatch] automatic fallback, backend honesty:")
        refusal_case("fallback", fake_souc, work, "backend-gpu", ["build", "prog.sio", "--backend", "gpu", "-o", "gpu.ptx"], "gpu.ptx", "--backend gpu")
        refusal_case("fallback", fake_souc, work, "backend-equals-gpu", ["build", "prog.sio", "--backend=gpu", "-o", "gpu.ptx"], "gpu.ptx", "--backend gpu")
        refusal_case("fallback", fake_souc, work, "backend-llvm", ["compile", "prog.sio", "--backend", "llvm", "-o", "x.out"], "x.out", "--backend llvm")
        refusal_case("fallback", fake_souc, work, "gpu-target", ["build", "prog.sio", "--gpu-target", "dgx-sm121", "-o", "g.ptx"], "g.ptx", "--gpu-target")
        refusal_case("fallback", fake_souc, work, "backend-missing-value", ["build", "prog.sio", "--backend"], "prog.sio.x", "--backend")
        refusal_case("fallback", fake_souc, work, "target-wasm32", ["build", "prog.sio", "--target", "wasm32", "-o", "out.wasm"], "out.wasm", "--target wasm32")
        refusal_case("fallback", fake_souc, work, "target-equals-aarch64", ["build", "prog.sio", "--target=aarch64-linux", "-o", "a64.elf"], "a64.elf", "--target aarch64-linux")
        refusal_case("fallback", fake_souc, work, "target-missing-value", ["build", "prog.sio", "--target"], "prog.sio.x", "--target")
        refusal_case("fallback", fake_souc, work, "target-short-fpga", ["build", "prog.sio", "-t", "fpga-u250", "-o", "san.cpp"], "san.cpp", "--target fpga-u250")
        acceptance_case("fallback", fake_souc, work, "backend-native", ["--backend", "native"])
        acceptance_case("fallback", fake_souc, work, "backend-native-equals", ["--backend=native"])
        acceptance_case("fallback", fake_souc, work, "target-native", ["--target", "x86_64-linux"])
        acceptance_case("fallback", fake_souc, work, "target-native-equals", ["--target=x86_64-linux"])
        acceptance_case("fallback", fake_souc, work, "target-native-short", ["-t", "x86_64-linux"])

        print("[fallback-dispatch] forced legacy engine:")
        forced = {"SOUNIO_SOUC_ENGINE": "lean_single"}
        verb_case("forced", fake_souc, work, "forced-run", "run", forced)
        verb_case("forced", fake_souc, work, "forced-build", "build", forced)
        refusal_case("forced", fake_souc, work, "forced-backend-gpu", ["build", "prog.sio", "--backend", "gpu", "-o", "gpu.ptx"], "gpu.ptx", "--backend gpu", forced)
        refusal_case("forced", fake_souc, work, "forced-target-hls", ["build", "prog.sio", "--target", "hls", "-o", "out.cpp"], "out.cpp", "--target hls", forced)
        acceptance_case("forced", fake_souc, work, "forced-backend-native", ["--backend", "native"], forced)
        acceptance_case("forced", fake_souc, work, "forced-target-native", ["--target", "x86_64-linux"], forced)

    # The default Madaros path must be untouched by the lean-side refusal.
    print("[fallback-dispatch] default Madaros path unaffected:")
    mat = subprocess.run(["bash", MATERIALIZER], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=False)
    if mat.returncode != 0:
        print("GATE FAIL: the committed madaros prebuilt does not materialize", file=sys.stderr)
        return 1
    rc, stdout, stderr = run_invoke(SOUC, ["--version"], ROOT)
    record("madaros: --version works", rc == 0 and "madaros" in stdout, f"rc={rc} stdout={stdout[-200:]!r}")
    with tempfile.TemporaryDirectory(prefix="souc-lean-fallback-gate-mad-") as mwork:
        write(os.path.join(mwork, "prog.sio"), FIXTURE)
        rc, stdout, stderr = run_invoke(SOUC, ["build", "prog.sio", "--backend", "native", "-o", "m.elf"], mwork)
        record("madaros: build --backend native works", rc == 0 and is_elf(os.path.join(mwork, "m.elf")), f"rc={rc}: {stderr[-300:]}")

    if cases < 30:
        print(f"GATE FAIL: only {cases} cases ran — an extraction silently emptied and the gate would pass on noise", file=sys.stderr)
        return 1
    if failures:
        print(file=sys.stderr)
        print(f"SOUC_LEAN_FALLBACK_DISPATCH_GATE: {len(failures)} case(s) wrong of {cases}.", file=sys.stderr)
        print("  The lean_single paths must translate the public verbs, preserve the caller's", file=sys.stderr)
        print("  source bytes, and refuse the artifacts lean_single cannot produce — never", file=sys.stderr)
        print("  exit 0 on a silently wrong dispatch.", file=sys.stderr)
        return 1
    print(f"SOUC_LEAN_FALLBACK_DISPATCH_GATE_OK: {cases} cases, {refusals} of them refusals, each behaved as stated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
