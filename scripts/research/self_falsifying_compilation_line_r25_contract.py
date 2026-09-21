#!/usr/bin/env python3
"""Self-falsifying compilation, rung R25 — research authority is path-default historical.

Spec: docs/research/self_falsifying_compilation_line_r25_2026-07-31.md

R22: last_validated is a date literal.
R23: validated_by is path ownership under a validation name.
This rung is the third field in the same docs:meta block that looks like a
judgment: authority.

For docs/research/, authority is not measured. It is:

    ACTIVE_RESEARCH_DOCS.has(relPath) ? 'repo_only' : 'historical'

ACTIVE_RESEARCH_DOCS is a hand-edited Set in the generator -- four paths as of
2026-08-15 (rna_cayley_dickson_confirmatory_preregistration_2026-08-09.md was
whitelisted then, see docs/audit/BRANCH_AUDIT_2026-08-15.md; it was three
paths from 2026-07-31 through 2026-08-15, which is why this rung's clause ID
still says THREE -- see WHITELIST_SIZE below). Everything else under
docs/research/ is stamped historical, and check_docs_registry.mjs enforces both
the authority field and the auto-inserted "Docs status: historical" note that
says the page is preserved for lineage.

So a research finding written today as EXECUTABLE is green only when it claims
to be historical lineage — unless someone hand-edits the whitelist.

CLAUSES:
  V1_WHITELIST_IS_THREE           ACTIVE_RESEARCH_DOCS is a Set of
                                  WHITELIST_SIZE path literals in the
                                  generator (clause ID kept for rung
                                  continuity even though the count grew).
  V2_DEFAULT_IS_HISTORICAL        the path rule is ternary on that Set; research
                                  not in the Set is historical.
  V3_CORPUS_IS_LINEAGE_DEFAULT    census of research authority + status notes:
                                  nearly all historical with the lineage note.
  V4_GATE_REJECTS_CURRENT         hermetic farm: green unmodified; one historical
                                  research page given authority: repo_only →
                                  checker rejects (expected "historical").

WHAT THIS DOES NOT MEASURE. Whether historical is a useful label for some pages.
Whether the three whitelist entries deserve repo_only. Only: the field is not a
measurement of currency, and the gate enforces the path default.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/docs/governance_registry.mjs"
CHECKER = ROOT / "scripts/docs/check_docs_registry.mjs"
REGISTRY = ROOT / "docs/governance/topic-registry.v1.json"
SYNC = "scripts/docs/sync_governance_metadata.mjs"
# Fingerprint of ACTIVE_RESEARCH_DOCS's size. Was 3 from this rung's
# writing (2026-07-31) through 2026-08-15; grew to 4 when
# rna_cayley_dickson_confirmatory_preregistration_2026-08-09.md was
# whitelisted (docs/audit/BRANCH_AUDIT_2026-08-15.md). Bump this
# deliberately when the whitelist legitimately changes size -- do not
# let this constant silently drift out of sync with governance_registry.mjs.
WHITELIST_SIZE = 4

SUBJECT = "docs/research/self_falsifying_compilation_line_r24_2026-07-31.md"
FARM_COPY = ["docs", "examples", "paper", "spec", "README.md"]
FARM_WEBSITE_COPY = "website/src/content"

META_RE = re.compile(r"^<!-- docs:meta\n([\s\S]*?)\n-->", re.M)
AUTH_RE = re.compile(r"^authority:\s*(.+)$", re.M)
ACTIVE_SET_RE = re.compile(
    r"const\s+ACTIVE_RESEARCH_DOCS\s*=\s*new\s+Set\(\[([\s\S]*?)\]\)",
    re.M,
)
RESEARCH_RULE_RE = re.compile(
    r"if\s*\(\s*relPath\.startsWith\(\s*'docs/research/'\s*\)\s*\)\s*\{"
    r"[\s\S]*?authority:\s*ACTIVE_RESEARCH_DOCS\.has\(relPath\)\s*\?\s*"
    r"'([^']+)'\s*:\s*'([^']+)'",
    re.M,
)


def declared_authority(path: Path) -> str:
    text = path.read_text(errors="replace")
    block = META_RE.search(text)
    body = block.group(1) if block else ""
    field = AUTH_RE.search(body)
    return field.group(1).strip() if field else "<absent>"


def clause_v1() -> tuple[bool, set[str]]:
    src = GENERATOR.read_text()
    m = ACTIVE_SET_RE.search(src)
    if not m:
        print("V1 ACTIVE_RESEARCH_DOCS Set not found")
        print("V1_WHITELIST_IS_THREE FAIL")
        print()
        return False, set()
    paths = set(re.findall(r"'([^']+)'", m.group(1)))
    # line number
    line_n = 0
    for n, line in enumerate(src.splitlines(), start=1):
        if "ACTIVE_RESEARCH_DOCS" in line and "new Set" in line:
            line_n = n
            break
    print("V1 ACTIVE_RESEARCH_DOCS whitelist:")
    print(f"    governance_registry.mjs:{line_n}  Set size={len(paths)}")
    for p in sorted(paths):
        print(f"        {p}")
    ok = len(paths) == WHITELIST_SIZE and all(p.startswith("docs/research/") for p in paths)
    print(f"V1_WHITELIST_IS_THREE {'PASS' if ok else 'FAIL'}")
    print()
    return ok, paths


def clause_v2() -> bool:
    src = GENERATOR.read_text()
    m = RESEARCH_RULE_RE.search(src)
    if not m:
        print("V2 research path authority rule not found")
        print("V2_DEFAULT_IS_HISTORICAL FAIL")
        print()
        return False
    when_active, when_default = m.group(1), m.group(2)
    rule_line = 0
    for n, line in enumerate(src.splitlines(), start=1):
        if "docs/research/" in line and "startsWith" in line:
            rule_line = n
            break
    print("V2 path rule for research authority:")
    print(f"    governance_registry.mjs:{rule_line}")
    print(f"    ACTIVE_RESEARCH_DOCS.has(relPath) ? {when_active!r} : {when_default!r}")
    ok = when_active == "repo_only" and when_default == "historical"
    print(f"V2_DEFAULT_IS_HISTORICAL {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_v3(whitelist: set[str]) -> bool:
    registry = json.loads(REGISTRY.read_text())
    research = [
        t
        for t in registry["topics"]
        if (t.get("repo_doc_path") or "").startswith("docs/research/")
    ]
    auth_census: dict[str, int] = {}
    note_hist = 0
    note_missing = 0
    whitelist_repo_only = 0
    non_whitelist_historical = 0
    for t in research:
        rel = t["repo_doc_path"]
        a = t.get("authority") or "<absent>"
        auth_census[a] = auth_census.get(a, 0) + 1
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        if a == "historical":
            if "Docs status: `historical`" in text:
                note_hist += 1
            else:
                note_missing += 1
        if rel in whitelist and a == "repo_only":
            whitelist_repo_only += 1
        if rel not in whitelist and a == "historical":
            non_whitelist_historical += 1

    print(f"V3 research corpus: n={len(research)}")
    print(f"    authority census: {auth_census}")
    print(f"    historical docs with lineage status note: {note_hist}")
    print(f"    historical docs missing status note: {note_missing}")
    print(f"    whitelist paths that are repo_only: {whitelist_repo_only}/{len(whitelist)}")
    print(
        f"    non-whitelist paths that are historical: "
        f"{non_whitelist_historical}/{len(research) - len(whitelist)}"
    )
    # Nearly all research is historical; lineage note is present; whitelist is tiny.
    hist = auth_census.get("historical", 0)
    ok = (
        len(research) > 100
        and hist >= len(research) - 5  # allow dual/repo_only few
        and note_missing == 0
        and note_hist == hist
        and len(whitelist) == WHITELIST_SIZE
        and non_whitelist_historical >= len(research) - 5
    )
    print(f"V3_CORPUS_IS_LINEAGE_DEFAULT {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def run_checker(cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ["node", str(CHECKER)], cwd=str(cwd), capture_output=True, text=True
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def clause_v4() -> bool:
    subject = ROOT / SUBJECT
    if not subject.is_file():
        print(f"V4 subject absent: {SUBJECT}")
        print("V4_GATE_REJECTS_CURRENT FAIL")
        print()
        return False
    cur = declared_authority(subject)
    if cur != "historical":
        print(f"V4 subject authority is {cur!r}, need historical for the control")
        print("V4_GATE_REJECTS_CURRENT FAIL")
        print()
        return False

    witnessed = [
        ROOT / "docs/governance/topic-registry.v1.json",
        ROOT / "docs/governance/DOCS_AUTHORITY_MATRIX.md",
        ROOT / "docs/governance/DOCS_ACCEPTANCE_REPORT.md",
        subject,
    ]
    before = {
        p: (p.stat().st_mtime_ns, p.stat().st_size) for p in witnessed if p.is_file()
    }

    farm = Path(tempfile.mkdtemp(prefix="sfcl-r25-farm."))
    try:
        copied = [p for p in FARM_COPY if (ROOT / p).exists()]
        cp = subprocess.run(
            ["cp", "-a", *copied, str(farm)], cwd=str(ROOT), capture_output=True, text=True
        )
        if cp.returncode != 0:
            print(f"V4 farm could not be built: {cp.stderr.strip()[:200]}")
            print("V4_GATE_REJECTS_CURRENT FAIL")
            print()
            return False
        web = ROOT / "website"
        if web.is_dir():
            (farm / "website/src").mkdir(parents=True)
            subprocess.run(
                ["cp", "-a", FARM_WEBSITE_COPY, str(farm / "website/src")],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            for entry in sorted(os.listdir(web)):
                if entry != "src":
                    os.symlink(web / entry, farm / "website" / entry)
            if (web / "src").is_dir():
                for entry in sorted(os.listdir(web / "src")):
                    if entry != "content":
                        os.symlink(web / "src" / entry, farm / "website/src" / entry)
        skip = {".git", "website", *FARM_COPY}
        for entry in sorted(os.listdir(ROOT)):
            if entry in skip:
                continue
            os.symlink(ROOT / entry, farm / entry)

        sync = subprocess.run(
            ["node", str(ROOT / SYNC)], cwd=str(farm), capture_output=True, text=True
        )
        if sync.returncode != 0:
            print(f"V4 farm sync failed: {sync.stderr.strip()[:200]}")
            print("V4_GATE_REJECTS_CURRENT FAIL")
            print()
            return False
        print(f"V4 farm synced to consistency: {sync.stdout.strip()[:90]}")

        rc_clean, _ = run_checker(farm)
        clean_green = rc_clean == 0
        print(
            f"V4 negative control -- farm unmodified: checker rc={rc_clean}"
            f" ({'green' if clean_green else 'RED'})"
        )

        # Positive: claim the page is current (repo_only) without whitelist membership.
        target = farm / SUBJECT
        text = target.read_text()
        patched, n_sub = re.subn(
            r"^authority: historical$",
            "authority: repo_only",
            text,
            count=1,
            flags=re.M,
        )
        if n_sub != 1:
            print("V4 could not patch the subject's authority")
            print("V4_GATE_REJECTS_CURRENT FAIL")
            print()
            return False
        target.unlink()
        target.write_text(patched)

        rc_true, out_true = run_checker(farm)
        expected = (
            f'{SUBJECT} metadata mismatch for authority: expected "historical"'
        )
        rejected = rc_true != 0 and expected in out_true
        print(f"    {SUBJECT}")
        print(f"    path default (enforced)  historical")
        print(f"    claimed as current       repo_only")
        print(
            f"V4 positive control -- claim current: checker rc={rc_true}"
            f" ({'REJECTED' if rejected else 'accepted'})"
        )
        if rejected:
            for line in out_true.splitlines():
                if "authority" in line and SUBJECT in line:
                    print(f"    {line.strip()}")
                    break

        after = {
            p: (p.stat().st_mtime_ns, p.stat().st_size) for p in witnessed if p.is_file()
        }
        hermetic_n = 0
        hermetic_ok = True
        for p, b in before.items():
            a = after.get(p)
            if a == b:
                hermetic_n += 1
            else:
                hermetic_ok = False
                print(f"    HERMETIC BREACH: {p}")
        print(f"V4 hermetic: {hermetic_n} working-tree files unchanged")

        ok = clean_green and rejected and hermetic_ok
        print(f"V4_GATE_REJECTS_CURRENT {'PASS' if ok else 'FAIL'}")
        print()
        return ok
    finally:
        subprocess.run(["rm", "-rf", str(farm)], check=False)


def main() -> int:
    if not GENERATOR.is_file() or not CHECKER.is_file():
        print("generator or checker absent")
        return 1
    if which("node") is None:
        print("node absent")
        return 1

    print("Self-falsifying compilation R25 — research authority is path-default historical")
    print(f"ROOT={ROOT}")
    print()

    ok1, whitelist = clause_v1()
    ok2 = clause_v2()
    ok3 = clause_v3(whitelist)
    ok4 = clause_v4()

    all_ok = ok1 and ok2 and ok3 and ok4
    verdict = "RESEARCH_AUTHORITY_IS_PATH_DEFAULT_HISTORICAL__GATE_REJECTS_CURRENT"
    print(f"SELF_FALSIFYING_R25_VERDICT {verdict}")
    print(f"OVERALL {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
