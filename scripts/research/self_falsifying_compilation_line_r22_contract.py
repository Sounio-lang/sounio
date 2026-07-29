#!/usr/bin/env python3
"""Self-falsifying compilation, rung R22 — the gate that certifies a literal.

Spec: docs/research/self_falsifying_compilation_line_r22_2026-07-29.md

This line studies checks that guard nothing. R1 found claims bound to no gate;
R5 found a gate nobody ran; R20 found an oracle never committed. This rung
finds the inverse and the worst-behaved of the family: a gate that runs on
every push, is green, and certifies a **string literal**.

Every governed repository document carries a `docs:meta` block with a field
named `last_validated`. The field has the form of a measurement. It is a
constant in the generator:

    scripts/docs/governance_registry.mjs:649   'last_validated: 2026-03-07',
    scripts/docs/governance_registry.mjs:730   last_validated: '2026-03-07',

Neither site reads the topic, the filesystem, git, or any gate result. And
`scripts/docs/check_docs_registry.mjs` -- wired into CI at
.github/workflows/ci.yml -- compares every document's field against that
constant (check_docs_registry.mjs:126-131, called at :159 and :172). So the
enforcement runs the wrong way round: a document that records the date it was
actually validated is a **gate failure**, and the gate is green exactly when
the field carries no information.

CLAUSES:
  V1_VALUE_IS_A_LITERAL         the date is a quoted literal at both sites, and
                                metadataFieldsForTopic returns the same value
                                for every topic in the registry -- topic
                                independence measured, not argued.
  V2_ONE_DATE_FOR_EVERY_DOC     census of the declared field over every
                                governed repo doc: one distinct value.
  V3_DATE_PRECEDES_THE_REPO     no commit in this repository is older than the
                                declared date. The whole corpus claims a
                                validation that predates its own history.
  V4_GATE_REJECTS_THE_TRUE_DATE end-to-end, hermetic: in a hardlink farm (the
                                working tree is never touched) the real checker
                                passes on the corpus as-is, and fails when one
                                document is given the date git says it was
                                added. Both arms, so the instrument has a
                                positive AND a negative control.

WHAT THIS DOES NOT MEASURE. Whether the documents were in fact validated on
some date. The point is narrower and harder: the field cannot answer that
question, and the gate that reads it cannot notice.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/docs/governance_registry.mjs"
CHECKER = ROOT / "scripts/docs/check_docs_registry.mjs"
REGISTRY = ROOT / "docs/governance/topic-registry.v1.json"

# The subject of the positive control. Fixed rather than discovered so the
# receipt in the spec stays reproducible.
SUBJECT = "docs/research/self_falsifying_compilation_line_r21_2026-07-28.md"

# The four trees the checker WALKS need real directories, so they are copied
# with hardlinks (same bytes, no data movement). Everything else it merely
# resolves -- link targets, related artifacts -- so a symlink is enough, and
# that is the difference between a 0.4 s farm and a 130 s one.
FARM_WALKED = ["docs", "examples", "paper", "website", "spec"]

META_RE = re.compile(r"^<!-- docs:meta\n([\s\S]*?)\n-->", re.M)
FIELD_RE = re.compile(r"^last_validated:\s*(.+)$", re.M)
SITE_RE = re.compile(r"^\s*'?last_validated'?\s*:\s*'?([0-9]{4}-[0-9]{2}-[0-9]{2})'?",
                     re.M)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def declared_field(path: Path) -> str:
    text = path.read_text(errors="replace")
    block = META_RE.search(text)
    body = block.group(1) if block else ""
    field = FIELD_RE.search(body)
    return field.group(1).strip() if field else "<absent>"


def clause_v1() -> tuple[bool, str]:
    src = GENERATOR.read_text()
    sites = []
    for n, line in enumerate(src.splitlines(), start=1):
        m = SITE_RE.match(line)
        if m:
            sites.append((n, m.group(1)))
    print("V1 the two sites that write the field:")
    for n, value in sites:
        print(f"    governance_registry.mjs:{n}  -> {value!r}  (literal)")
    literals = {value for _, value in sites}
    ok = len(sites) == 2 and len(literals) == 1

    # Topic independence, measured: ask the generator itself.
    script = (
        "import(process.argv[1]).then(async (m) => {"
        "  const r = await m.buildGovernedTopicRegistry(process.argv[2]);"
        "  const vals = new Set(r.topics.map((t) => "
        "    m.metadataFieldsForTopic(t).last_validated));"
        "  console.log(JSON.stringify({topics: r.topics.length,"
        "    distinct: [...vals]}));"
        "});"
    )
    proc = subprocess.run(["node", "-e", script, str(GENERATOR), str(ROOT)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    generator would not run: {proc.stderr.strip()[:200]}")
        return False, ""
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"    metadataFieldsForTopic over {seen['topics']} topics "
          f"-> distinct values {seen['distinct']}")
    print("    The function takes a topic and ignores it. No input reaches"
          " the field.")
    ok = ok and len(seen["distinct"]) == 1 and seen["distinct"][0] in literals
    print(f"V1_VALUE_IS_A_LITERAL {'PASS' if ok else 'FAIL'}")
    print()
    return ok, sites[0][1] if sites else ""


def clause_v2(literal: str) -> bool:
    registry = json.loads(REGISTRY.read_text())
    docs = [t["repo_doc_path"] for t in registry["topics"]
            if t.get("repo_doc_path")]
    census: dict[str, int] = {}
    unreadable = 0
    for rel in docs:
        path = ROOT / rel
        if not path.is_file():
            unreadable += 1
            continue
        value = declared_field(path)
        census[value] = census.get(value, 0) + 1
    print(f"V2 census over {len(docs)} governed repo docs "
          f"({unreadable} unreadable):")
    for value, count in sorted(census.items(), key=lambda kv: -kv[1]):
        print(f"    {value!r}: {count}")
    ok = len(census) == 1 and literal in census and unreadable == 0
    print(f"V2_ONE_DATE_FOR_EVERY_DOC {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_v3(literal: str) -> bool:
    older = git("log", f"--before={literal}", "--oneline")
    n_older = len([ln for ln in older.splitlines() if ln.strip()])
    first = git("log", "--reverse", "--format=%ad", "--date=short")
    first_date = first.splitlines()[0].strip() if first else ""
    print(f"V3 declared validation date: {literal}")
    print(f"    commits in this repository older than it: {n_older}")
    print(f"    oldest commit in this repository:         {first_date}")
    delta = ""
    if first_date:
        d0 = date.fromisoformat(literal)
        d1 = date.fromisoformat(first_date)
        delta = (d1 - d0).days
        print(f"    the corpus declares a validation {delta} days before the"
              " repository's own history begins")
    ok = n_older == 0 and bool(first_date) and isinstance(delta, int) and delta > 0
    print(f"V3_DATE_PRECEDES_THE_REPO {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def run_checker(cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(["node", str(CHECKER)], cwd=str(cwd),
                          capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)


def clause_v4(literal: str) -> bool:
    subject = ROOT / SUBJECT
    if not subject.is_file():
        print(f"V4 subject absent: {SUBJECT}")
        print("V4_GATE_REJECTS_THE_TRUE_DATE FAIL")
        print()
        return False

    added = git("log", "--diff-filter=A", "--format=%ad", "--date=short",
                "--", SUBJECT)
    true_date = added.splitlines()[-1].strip() if added else ""
    if not true_date or true_date == literal:
        print(f"V4 no usable git addition date for {SUBJECT} "
              f"(got {true_date!r})")
        print("V4_GATE_REJECTS_THE_TRUE_DATE FAIL")
        print()
        return False

    farm = Path(tempfile.mkdtemp(prefix="sfcl-r22-farm."))
    try:
        # Hardlink farm: same bytes, different tree. The working tree is read
        # but never written -- this line's own hermeticity rule (R1 B5).
        walked = [p for p in FARM_WALKED if (ROOT / p).exists()]
        cp = subprocess.run(["cp", "-al", *walked, str(farm)], cwd=str(ROOT),
                            capture_output=True, text=True)
        if cp.returncode != 0:
            print(f"V4 farm could not be built: {cp.stderr.strip()[:200]}")
            print("V4_GATE_REJECTS_THE_TRUE_DATE FAIL")
            print()
            return False
        for entry in sorted(os.listdir(ROOT)):
            if entry == ".git" or entry in FARM_WALKED:
                continue
            os.symlink(ROOT / entry, farm / entry)

        # NEGATIVE CONTROL. An instrument that fails on everything measures
        # nothing, so the farm must first reproduce the green result.
        rc_clean, out_clean = run_checker(farm)
        clean_green = rc_clean == 0
        print(f"V4 negative control -- farm unmodified: checker rc={rc_clean}"
              f" ({'green' if clean_green else 'RED'})")

        # POSITIVE CONTROL. Give one document the date git says it was added.
        target = farm / SUBJECT
        text = target.read_text()
        patched, n_sub = re.subn(rf"^last_validated: {re.escape(literal)}$",
                                 f"last_validated: {true_date}", text,
                                 count=1, flags=re.M)
        if n_sub != 1:
            print("V4 could not patch the subject's field")
            print("V4_GATE_REJECTS_THE_TRUE_DATE FAIL")
            print()
            return False
        target.unlink()  # break the hardlink before writing
        target.write_text(patched)

        rc_true, out_true = run_checker(farm)
        expected = (f'{SUBJECT} metadata mismatch for last_validated: '
                    f'expected "{literal}"')
        rejected = rc_true != 0 and expected in out_true
        print(f"    {SUBJECT}")
        print(f"    git says it was added   {true_date}")
        print(f"    the generator asserts   {literal}")
        print(f"V4 positive control -- truthful date: checker rc={rc_true}"
              f" ({'REJECTED' if rejected else 'accepted'})")
        if rejected:
            for line in out_true.splitlines():
                if "last_validated" in line:
                    print(f"      {line.strip()}")
        ok = clean_green and rejected
    finally:
        shutil.rmtree(farm, ignore_errors=True)

    print(f"V4_GATE_REJECTS_THE_TRUE_DATE {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main() -> int:
    print("Self-falsifying compilation R22 -- the gate that certifies a literal")
    print("=" * 72)
    print()

    ok1, literal = clause_v1()
    if not literal:
        print("SELF_FALSIFYING_R22_VERDICT INCONCLUSIVE")
        return 1
    ok2 = clause_v2(literal)
    ok3 = clause_v3(literal)
    ok4 = clause_v4(literal)

    ok = ok1 and ok2 and ok3 and ok4
    verdict = ("VALIDATION_DATE_IS_A_LITERAL__GATE_REJECTS_THE_TRUE_DATE"
               if ok else "INCONCLUSIVE")
    print("-" * 72)
    print("A field shaped like a measurement is a constant in the generator,")
    print("and the check wired into CI enforces the constant. The corpus is")
    print("uniform because uniformity is what passes: a document recording")
    print("when it was really validated turns the gate red. The gate is green")
    print("exactly when the field it guards carries no information.")
    print()
    print(f"SELF_FALSIFYING_R22_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
