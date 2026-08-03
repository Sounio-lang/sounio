#!/usr/bin/env python3
"""Self-falsifying compilation, rung R2 — verdict-token binding.

Spec: docs/research/self_falsifying_compilation_line_r2_2026-07-26.md

R0 established that exit-code gating binds a build to a *computation*, not to a
*proposition*: a gate that exits 0 says the check ran, not that the check
establishes what the claim declares. R2 adds verdict-token binding to the
compiler — a claim may declare `verdict_token`, and the token the gate actually
emits must equal it or the claim is falsified before codegen.

This closes the DRIFT class only. It cannot close shared misinterpretation, and
R0 §3's proposition says why; the fixtures below are built so the distinction is
visible rather than asserted.

Clauses (static; behaviour is proved by the gate's compile arm):

  T1_EXECUTOR_SURFACE  the compiler implements the field, the capture, the
                       extraction and both failure outcomes
  T2_FIXTURES          three probes discriminate match / drift / absent
  T3_NO_SHELL_STRING   capture uses open+dup2, not a shell redirect, so the
                       mechanism's no-command-injection property survives
  T4_REACH             how much of the corpus could be token-bound at all

Pure Python 3. No third-party dependencies.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

EXECUTOR = "self-hosted/compiler/claim_executor.sio"
FIXTURE_DIR = "scripts/ci/fixtures"

# Behaviour receipt written by the gate's compile arm. Lives under artifacts/
# (untracked) so producing it never dirties the working tree — the hermeticity
# rule R1 established. Absence means UNPROVEN, which is the safe direction.
RECEIPT = "artifacts/self_falsifying_r2_receipt.txt"

PROBES = [
    # (source fixture, gate fixture, declared token, expected outcome marker)
    ("self_falsifying_token_pass.sio", "self_falsifying_token_match.sh",
     "TOKEN_ALPHA", "CLAIM_PASS"),
    ("self_falsifying_token_drift.sio", "self_falsifying_token_drift.sh",
     "TOKEN_ALPHA", "CLAIM_TOKEN_MISMATCH"),
    ("self_falsifying_token_absent.sio", "self_falsifying_token_absent.sh",
     "TOKEN_ALPHA", "CLAIM_TOKEN_ABSENT"),
]


def read(rel: str) -> str:
    try:
        return (REPO / rel).read_text(errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------- T1


def clause_t1() -> bool:
    src = read(EXECUTOR)
    if not src:
        print(f"T1_EXECUTOR_SURFACE FAIL — {EXECUTOR} unreadable")
        return False

    required = [
        ('"verdict_token"', "the claim field is recognised"),
        ("CLAIM_GATE_TOKEN_MISMATCH", "mismatch outcome defined"),
        ("CLAIM_GATE_TOKEN_ABSENT", "absent outcome defined"),
        ("ce_extract_verdict_token", "token extraction implemented"),
        ("ce_capture_path", "per-process capture path"),
        ("CLAIM_TOKEN_MISMATCH ", "mismatch is reported"),
        ("CLAIM_TOKEN_ABSENT ", "absent is reported"),
    ]
    ok = True
    for needle, why in required:
        if needle not in src:
            print(f"  MISSING {needle!r} — {why}")
            ok = False

    # Both new outcomes must count as failures, or the binding is decorative.
    # The outcome codes are integer literals (4 = MISMATCH, 5 = ABSENT) and the
    # reporting chain branches on `decided`, never on `outcome` — see the R2
    # spec §3.2 for why. Match the shape that is actually compiled.
    #
    # MAINTENANCE: these patterns pin the implementation's magic numbers.
    # If the outcome codes are ever restored to named module constants (a
    # reasonable thing to do after a compiler fix), update the patterns here —
    # a failure would then be a stale contract, NOT a regression in the guard.
    #
    # The note above anticipated the CODES changing and pinned the decision
    # VARIABLE instead, as `decided == 4`. R17 then added witness binding and
    # renamed the branch variable to `settled`; R20 added provenance and renamed
    # it to `final_out`. Behaviour was identical throughout — T5's receipt
    # confirms it on a real compiler — but this clause went red on the spelling.
    # The variable name is an implementation detail those rungs were entitled to
    # change; what this clause means to assert is that each outcome code
    # increments the failure count. Matched that way now, so the next rename
    # does not fail it again.
    mismatch_block = re.search(
        r"\b\w+ == 4\b.*?failed = failed \+ 1", src, re.DOTALL)
    absent_block = re.search(
        r"\b\w+ == 5\b.*?failed = failed \+ 1", src, re.DOTALL)
    if not mismatch_block:
        print("  token mismatch does not increment the failure count")
        ok = False
    if not absent_block:
        print("  token absent does not increment the failure count")
        ok = False

    print(f"T1_EXECUTOR_SURFACE {'PASS' if ok else 'FAIL'} — "
          f"verdict_token field, capture, extraction, and both outcomes present "
          f"and counted as failures")
    return ok


# ---------------------------------------------------------------- T2


def clause_t2() -> bool:
    ok = True
    for sio, sh, token, expect in PROBES:
        s = read(f"{FIXTURE_DIR}/{sio}")
        g = read(f"{FIXTURE_DIR}/{sh}")
        if not s or not g:
            print(f"  missing probe pair: {sio} / {sh}")
            ok = False
            continue
        if f'verdict_token = "{token}"' not in s:
            print(f"  {sio}: does not declare verdict_token = \"{token}\"")
            ok = False
        if f"{FIXTURE_DIR}/{sh}" not in s:
            print(f"  {sio}: not bound to {sh}")
            ok = False
        # Every probe gate must exit 0 — otherwise it would be caught by plain
        # exit-code gating and would prove nothing about token binding.
        if "exit 0" not in g:
            print(f"  {sh}: does not exit 0, so it cannot isolate token binding")
            ok = False

    # The drift gate must emit a DIFFERENT token; the absent gate none.
    drift = read(f"{FIXTURE_DIR}/self_falsifying_token_drift.sh")
    absent = read(f"{FIXTURE_DIR}/self_falsifying_token_absent.sh")
    match = read(f"{FIXTURE_DIR}/self_falsifying_token_match.sh")
    if "TOKEN_ALPHA" not in match:
        print("  match gate does not emit TOKEN_ALPHA")
        ok = False
    if "TOKEN_ALPHA" in drift or "_VERDICT" not in drift:
        print("  drift gate must emit a verdict token that is NOT TOKEN_ALPHA")
        ok = False
    if "_VERDICT" in absent:
        print("  absent gate must emit no verdict token at all")
        ok = False

    print(f"T2_FIXTURES {'PASS' if ok else 'FAIL'} — {len(PROBES)} probes "
          f"discriminate match / drift / absent, all exiting 0")
    return ok


# ---------------------------------------------------------------- T3


def clause_t3() -> bool:
    """The capture must not reintroduce a shell command string.

    The mechanism's stated sandbox property is that argv is fixed and no
    command line is interpolated. Capturing output via `bash -c "gate > file"`
    would trade that away; open+dup2 in the child keeps it.
    """
    src = read(EXECUTOR)
    ok = True

    if "syscall6(33," not in src:
        print("  no dup2 call found — capture may not be redirect-based")
        ok = False
    if 'CE_ARGV[1] = gate_path as i64' not in src:
        print("  argv no longer passes the gate path as a fixed slot")
        ok = False
    for bad in ['"-c"', "' -c '", "> \" +", '" > "']:
        if bad in src:
            print(f"  shell redirect/command string found: {bad}")
            ok = False

    print(f"T3_NO_SHELL_STRING {'PASS' if ok else 'FAIL'} — "
          f"capture is open+dup2 with fixed argv; no command interpolation")
    return ok


# ---------------------------------------------------------------- T4


def clause_t4() -> tuple[bool, dict]:
    """How much of the corpus could be token-bound at all?

    Deliberately reported as a metric and NOT folded into the verdict token:
    embedding a moving count in a token is the sub-token failure mode this line
    documents (R0 §5's amendment).
    """
    specs = sorted((REPO / "docs/research").glob("*.md"))
    status_re = re.compile(r"^\*\*Status:\*\*", re.MULTILINE)
    token_re = re.compile(r"^\*\*Status:\*\* `[^`]*` — `([A-Za-z0-9_]+)`", re.MULTILINE)

    total = len(specs)
    with_status = 0
    with_token = 0
    for p in specs:
        try:
            t = p.read_text(errors="replace")
        except OSError:
            continue
        if status_re.search(t):
            with_status += 1
        if token_re.search(t):
            with_token += 1

    stats = {"specs": total, "with_status": with_status, "with_token": with_token}
    pct_all = (100.0 * with_token / total) if total else 0.0
    pct_status = (100.0 * with_token / with_status) if with_status else 0.0
    print(f"T4_REACH {with_token}/{total} specs declare a parseable verdict token "
          f"({pct_all:.1f}% of all specs; {pct_status:.1f}% of the "
          f"{with_status} that carry a Status line)")
    print(f"T4_REACH   the wide denominator is the honest one: "
          f"{total - with_status} specs have no Status line at all and need the "
          f"convention introduced before a token can be bound")
    ok = total > 0
    print(f"T4_REACH {'PASS' if ok else 'FAIL'} — measured")
    return ok, stats


# ---------------------------------------------------------------- T5


def clause_t5() -> bool:
    """Source surface is not behaviour — require a receipt that it ran.

    This clause exists because of a mistake made building this very rung: T1
    passed (the field, the capture and both outcomes were all present in the
    source) while the compiler built from that source SIGSEGV'd on every claim,
    including claims that used none of the new machinery. A contract that
    certifies "IMPLEMENTED" from source text alone repeats, inside this line's
    own tooling, exactly the error the line studies: checking the computation
    instead of the proposition.

    So: no receipt from an actual compile-arm run, no claim of implementation.
    The receipt is bound to the executor's content hash, so editing the
    executor invalidates it.
    """
    src_path = REPO / EXECUTOR
    try:
        digest = hashlib.sha256(src_path.read_bytes()).hexdigest()
    except OSError:
        print(f"T5_BEHAVIOUR_RECEIPT FAIL — cannot hash {EXECUTOR}")
        return False

    rpath = REPO / RECEIPT
    if not rpath.exists():
        print(f"T5_BEHAVIOUR_RECEIPT FAIL — no receipt at {RECEIPT}. "
              f"Token binding is UNPROVEN until the compile arm has run: "
              f"SFCL_R2_RUN_COMPILE=1 bash scripts/ci/"
              f"self_falsifying_compilation_line_r2_gate.sh")
        return False

    fields: dict[str, str] = {}
    for line in rpath.read_text(errors="replace").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            fields[k.strip()] = v.strip()

    ok = True
    if fields.get("executor_sha256") != digest:
        print(f"T5_BEHAVIOUR_RECEIPT FAIL — receipt is stale: it was produced "
              f"from a different {EXECUTOR}. Re-run the compile arm.")
        ok = False
    for probe in ("D1", "D2", "D3", "D4"):
        if fields.get(probe) != "PASS":
            print(f"  receipt does not record {probe}=PASS "
                  f"(got {fields.get(probe)!r})")
            ok = False

    if ok:
        print(f"T5_BEHAVIOUR_RECEIPT PASS — D1..D4 observed on a compiler built "
              f"from this exact executor source ({digest[:12]})")
    return ok


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R2 — verdict-token binding")
    print("=" * 72)

    t1 = clause_t1()
    print()
    t2 = clause_t2()
    print()
    t3 = clause_t3()
    print()
    t4, stats = clause_t4()
    print()
    t5 = clause_t5()
    print()

    print("=" * 72)
    if not (t1 and t2 and t3 and t4 and t5):
        print("SELF_FALSIFYING_R2_VERDICT INCOMPLETE")
        return 1

    # The token states what was built and what it can reach, never a count:
    # counts move, and a token carrying one drifts without the claim changing.
    token = "TOKEN_BINDING_IMPLEMENTED__CATCHES_DRIFT_NOT_MISINTERPRETATION"
    print(f"  executor surface : verdict_token field + capture + extraction")
    print(f"  probes           : match / drift / absent, all exiting 0")
    print(f"  reach            : {stats['with_token']}/{stats['specs']} specs token-bearing")
    print(f"SELF_FALSIFYING_R2_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
