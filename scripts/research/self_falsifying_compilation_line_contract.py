#!/usr/bin/env python3
"""Self-falsifying compilation — research-line opening contract.

Spec: docs/research/self_falsifying_compilation_line_2026-07-26.md

This contract reproduces the audit that opens the line. It measures, from the
repository itself, three things:

  S1_SUBSTRATE_SURFACE  the --verify-claims machinery exists in compiler source
  S2_CORPUS_GAP         how much of the repo's empirical surface is bound to
                        source-level claims (answer: essentially none)
  S3_BINDING_GAP        how many CI gates are named by any claim
  S4_RETROSPECTIVE      for the audited historical corrections: did spec and
                        harness AGREE on a verdict token both before and after
                        the correction?

S4 is the clause that decides whether a retrospective evaluation of the
self-falsifying compiler is meaningful. If spec and harness agreed on a token
both before and after each correction, then neither exit-code gating nor
verdict-token matching would have gone red at the moment the claim was wrong:
the error lived in the proposition that both sides shared.

Pure Python 3 + git. No third-party dependencies.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Historical corrections audited when the line was opened. Each entry names the
# correction commit and the spec/harness pair it rewrote. The "before" state is
# always the correction's first parent.
AUDITED_CORRECTIONS = [
    (
        "daa0635d0",
        "ord-3 module overclaim deflated (2*V3 is CD-doubling, not a fingerprint)",
        "docs/research/functor_f_ord3_module_decomp_spec_2026-07-26.md",
        "scripts/research/functor_f_ord3_module_decomp_contract.py",
    ),
    (
        "ec579a24c",
        "E6 bridge corrected (phi IS the cubic cross-term)",
        "docs/research/functor_f_e6_albert_shadow_spec_2026-07-25.md",
        "scripts/research/functor_f_e6_albert_shadow_contract.py",
    ),
    (
        "eb38e9ce5",
        "ord-3 symmetry-fill group id corrected (2^3:PSL(2,7)/192, not S4/24)",
        "docs/research/functor_f_ord3_symmetry_fill_spec_2026-07-25.md",
        "scripts/research/functor_f_ord3_symmetry_fill_contract.py",
    ),
]

# Spec header form: **Status:** `EXECUTABLE` — `SOME_VERDICT_TOKEN`
# Tokens are conventionally SHOUTY_SNAKE but not strictly: ORD3_MODULE_IS_2xV3
# carries a lowercase 'x', so the character class must admit mixed case.
SPEC_STATUS_RE = re.compile(r"\*\*Status:\*\*\s*`[^`]*`\s*[—-]+\s*`([A-Za-z0-9_]+)`")
# Harness verdict emission: any quoted string containing "<PREFIX>_VERDICT <TOKEN>"
HARNESS_VERDICT_RE = re.compile(r"[A-Z0-9_]*_VERDICT\s+([A-Za-z0-9_]+)")


def _git_run(args: tuple[str, ...]) -> tuple[int, str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.returncode, out.stdout
    except OSError:
        return 127, ""


def git(*args: str) -> str:
    """Run git in the repo, returning stdout ('' on failure)."""
    rc, out = _git_run(args)
    return out if rc == 0 else ""


def git_ok(*args: str) -> bool:
    """Run git for its exit status only (stdout may legitimately be empty)."""
    rc, _ = _git_run(args)
    return rc == 0


def tracked(pattern: str) -> list[str]:
    return [p for p in git("ls-files", pattern).splitlines() if p]


def spec_token(blob: str) -> str | None:
    m = SPEC_STATUS_RE.search(blob)
    return m.group(1) if m else None


def harness_tokens(blob: str) -> set[str]:
    """Verdict tokens the harness can emit, minus obvious placeholders."""
    toks = set(HARNESS_VERDICT_RE.findall(blob))
    return {t for t in toks if t not in {"INCOMPLETE", "FAIL", "FAILED", "ERROR"}}


# ---------------------------------------------------------------- S1


def clause_s1() -> bool:
    """The --verify-claims substrate exists in compiler source."""
    required = [
        ("self-hosted/compiler/claim_executor.sio", "module claim_executor"),
        ("self-hosted/compiler/main.sio", "claim_executor_verify"),
        ("self-hosted/compiler/main.sio", "--verify-claims"),
        ("self-hosted/parser/ast.sio", "ast_claim_slot_field"),
        ("scripts/ci/self_falsifying_compiler_gate.sh", "F5_FAIL_BLOCKS"),
    ]
    ok = True
    for rel, needle in required:
        path = REPO / rel
        if not path.exists() or needle not in path.read_text(errors="replace"):
            print(f"  MISSING {needle!r} in {rel}")
            ok = False
    print(f"S1_SUBSTRATE_SURFACE {'PASS' if ok else 'FAIL'} "
          f"— claim executor, --verify-claims flag, registry accessors, gate")
    return ok


# ---------------------------------------------------------------- S2


def clause_s2() -> tuple[bool, dict]:
    """Measure the corpus gap: native claims vs the repo's empirical surface."""
    claim_files: dict[str, int] = {}
    for rel in tracked("*.sio"):
        try:
            text = (REPO / rel).read_text(errors="replace")
        except OSError:
            continue
        n = len(re.findall(r"^claim\s+\w+\s*\{", text, re.MULTILINE))
        if n:
            claim_files[rel] = n

    def is_toy(rel: str) -> bool:
        return rel.startswith(("tests/", "scripts/ci/fixtures/"))

    production = {r: n for r, n in claim_files.items() if not is_toy(r)}
    toy = {r: n for r, n in claim_files.items() if is_toy(r)}

    gates = [p for p in tracked("scripts/ci/*.sh") if "gate" in Path(p).name]
    contracts = [p for p in tracked("scripts/research/*.py") if "contract" in Path(p).name]

    stats = {
        "native_claim_files": len(claim_files),
        "native_claims": sum(claim_files.values()),
        "production_claim_files": len(production),
        "production_claims": sum(production.values()),
        "toy_claim_files": len(toy),
        "gates": len(gates),
        "contracts": len(contracts),
        "gate_paths": set(gates),
    }

    print(f"S2_CORPUS_GAP native claim blocks: {stats['native_claims']} "
          f"in {stats['native_claim_files']} files "
          f"({stats['toy_claim_files']} tests/fixtures, "
          f"{stats['production_claim_files']} production)")
    print(f"S2_CORPUS_GAP empirical surface: {stats['gates']} CI gates, "
          f"{stats['contracts']} research contracts")
    for rel in sorted(claim_files):
        print(f"    {'toy ' if is_toy(rel) else 'PROD'} {claim_files[rel]:2d}  {rel}")

    # The clause records the gap; it PASSES when the measurement is coherent
    # (there is an empirical surface to bind, and it is measurable).
    ok = stats["gates"] > 0 and stats["contracts"] > 0
    verdict = "UNBOUND" if stats["production_claims"] == 0 else "PARTIALLY_BOUND"
    print(f"S2_CORPUS_GAP {'PASS' if ok else 'FAIL'} — corpus is {verdict}")
    return ok, stats


# ---------------------------------------------------------------- S3


def clause_s3(stats: dict) -> bool:
    """How many CI gates are named by any claim (native or comment-form)?"""
    named: set[str] = set()
    for rel in tracked("*.sio"):
        try:
            text = (REPO / rel).read_text(errors="replace")
        except OSError:
            continue
        if "claim" not in text and "@claim" not in text:
            continue
        for m in re.finditer(r"(scripts/ci/[\w./-]+\.sh)", text):
            named.add(m.group(1))

    # Count only real CI gates: exclude fixtures (the mechanism's own test
    # scaffolding) and any referenced script that is not in the gate set, so
    # numerator and denominator range over the same population.
    real = {g for g in named if "/fixtures/" not in g} & stats["gate_paths"]
    total = stats["gates"]
    pct = (100.0 * len(real) / total) if total else 0.0
    print(f"S3_BINDING_GAP {len(real)}/{total} CI gates named by a claim ({pct:.1f}%)")
    for g in sorted(real):
        print(f"    bound: {g}")
    ok = total > 0
    print(f"S3_BINDING_GAP {'PASS' if ok else 'FAIL'} — measured")
    return ok, len(real)


# ---------------------------------------------------------------- S4


def clause_s4() -> tuple[bool, list[dict]]:
    """Would a claim gate have gone red at the historical corrections?

    For each audited correction we compare, before and after, the verdict token
    declared in the spec against the verdict tokens the harness can emit. If
    they agree on BOTH sides, the correction changed claim and check in
    lockstep — no exit-code gate and no token-matching gate would have fired.
    """
    if not git_ok("rev-parse", "--git-dir"):
        print("S4_RETROSPECTIVE SKIP — not a git repository")
        return True, []

    rows = []
    unreachable = []
    for sha, what, spec_path, harness_path in AUDITED_CORRECTIONS:
        if not git_ok("cat-file", "-e", f"{sha}^{{commit}}"):
            print(f"  {sha} UNREACHABLE — commit not in this clone/branch")
            unreachable.append(sha)
            continue

        before_spec = git("show", f"{sha}^:{spec_path}")
        after_spec = git("show", f"{sha}:{spec_path}")
        before_h = git("show", f"{sha}^:{harness_path}")
        after_h = git("show", f"{sha}:{harness_path}")

        bt, at = spec_token(before_spec), spec_token(after_spec)
        bh, ah = harness_tokens(before_h), harness_tokens(after_h)

        agreed_before = bool(bt) and bt in bh
        agreed_after = bool(at) and at in ah
        token_changed = bool(bt) and bool(at) and bt != at

        # Did the correction touch any CI gate script?
        touched = git("show", "--name-only", "--format=", sha).splitlines()
        gate_touched = any("scripts/ci/" in p and p.endswith(".sh") for p in touched)

        # The question is only ever about the BEFORE state: at the parent
        # commit the claim was wrong, so would any compile-time claim gate have
        # gone red *then*? Exit-code gating fires only if the harness failed;
        # token matching fires only if spec and harness disagreed. If they
        # agreed at the parent and no CI gate script changed, neither fires.
        # Whether the token later changed is a descriptor of the error's depth,
        # not part of the predicate:
        #   token_changed=True  -> the headline proposition was wrong
        #   token_changed=False -> the error sat BELOW the token's resolution
        silent = agreed_before and not gate_touched
        depth = "headline" if token_changed else "below-token-resolution"

        rows.append({
            "sha": sha,
            "what": what,
            "before_token": bt,
            "after_token": at,
            "agreed_before": agreed_before,
            "agreed_after": agreed_after,
            "token_changed": token_changed,
            "gate_touched": gate_touched,
            "silent": silent,
            "depth": depth,
            "n_before_tokens": len(bh),
            "exact": len(bh) == 1,
        })

        # harness_tokens() collects every token the harness *could* emit, so a
        # membership test over-approximates "the harness emitted this". Report
        # the count: when exactly one non-placeholder token is reachable, the
        # membership test is exact, not an approximation.
        exact_note = "exact" if len(bh) == 1 else f"OVER-APPROX ({len(bh)} tokens emittable)"
        print(f"  {sha}  {what}")
        print(f"      before: spec={bt} harness_agrees={agreed_before} [{exact_note}]")
        print(f"      after : spec={at} harness_agrees={agreed_after}")
        print(f"      token changed={token_changed} ({depth})  ci-gate touched={gate_touched}")
        print(f"      => {'SILENT — no claim gate would have fired' if silent else 'gate-visible'}")

    silent_n = sum(1 for r in rows if r["silent"])
    headline_n = sum(1 for r in rows if r["silent"] and r["depth"] == "headline")
    below_n = silent_n - headline_n
    n_exact = sum(1 for r in rows if r["exact"])

    if rows:
        print(f"S4_RETROSPECTIVE {silent_n}/{len(rows)} audited corrections were SILENT "
              f"— at the commit where the claim was false, spec and harness agreed "
              f"and no CI gate changed, so no claim gate would have fired")
        print(f"S4_RETROSPECTIVE   of those: {headline_n} wrong at the headline token "
              f"(claim and check misinterpreted together), "
              f"{below_n} wrong below the token's resolution")
        print(f"S4_RETROSPECTIVE   token-agreement test was exact (single emittable "
              f"token) in {n_exact}/{len(rows)} cases")

    # The clause records outcomes rather than requiring a particular one, but it
    # MUST fail when it cannot measure: the audited commits are branch-local, so
    # in a fresh clone or after a rebase this clause would otherwise degrade to
    # a vacuous PASS while reporting nothing. An unmeasured clause is a failure.
    ok = not unreachable
    if unreachable:
        print(f"S4_RETROSPECTIVE FAIL — {len(unreachable)}/{len(AUDITED_CORRECTIONS)} "
              f"audited commits unreachable ({', '.join(unreachable)}); the clause "
              f"cannot be evaluated in this clone. These commits are branch-local "
              f"to the functor-F research lane and are not on main.")
    else:
        print(f"S4_RETROSPECTIVE PASS — all {len(rows)} audited commits measured")
    return ok, rows


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION — research-line opening audit")
    print("=" * 72)

    s1 = clause_s1()
    print()
    s2, stats = clause_s2()
    print()
    s3, bound_gates = clause_s3(stats)
    print()
    s4, rows = clause_s4()
    print()

    print("=" * 72)
    all_pass = s1 and s2 and s3 and s4
    if not all_pass:
        print("SELF_FALSIFYING_LINE_VERDICT INCOMPLETE")
        return 1

    unbound = stats["production_claims"] == 0
    silent_all = bool(rows) and all(r["silent"] for r in rows)

    if unbound and silent_all:
        token = "SUBSTRATE_LIVE__CORPUS_UNBOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE"
    elif unbound:
        token = "SUBSTRATE_LIVE__CORPUS_UNBOUND__RETROSPECTIVE_MIXED"
    elif silent_all:
        token = "SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE"
    else:
        token = "SUBSTRATE_LIVE__CORPUS_BOUND__RETROSPECTIVE_MIXED"

    print(f"  substrate surface : present")
    print(f"  production claims : {stats['production_claims']} "
          f"(vs {stats['gates']} gates, {stats['contracts']} contracts)")
    print(f"  gates bound       : {bound_gates}/{stats['gates']}")
    print(f"  silent corrections: {sum(1 for r in rows if r['silent'])}/{len(rows)}")
    print(f"SELF_FALSIFYING_LINE_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
