#!/usr/bin/env python3
"""Self-falsifying compilation, rung R23 — validated_by is path ownership.

Spec: docs/research/self_falsifying_compilation_line_r23_2026-07-30.md

R22 found that last_validated is a date literal the gate enforces. This rung
is the sibling field in the same docs:meta block: validated_by.

The field is named as if it recorded who validated the document. The generator
fills it from topic.owner_agent, and for every path under docs/research/ the
owner is the literal string 'A6' — path-prefix inference, not a review record.
check_docs_registry.mjs enforces equality field-by-field. So a document that
records a different validator is a gate failure, and the gate is green exactly
when validated_by equals the directory's owner label.

CLAUSES:
  V1_FIELD_EQUALS_OWNER_AGENT     both generator sites set validated_by from
                                  topic.owner_agent; measured by reading source
                                  and calling metadataFieldsForTopic.
  V2_PATH_PREFIX_OWNS_RESEARCH    docs/research/ hardcodes owner_agent 'A6';
                                  every research document declares A6.
  V3_CORPUS_IS_PATH_OWNERSHIP     over every governed repo doc, declared
                                  validated_by equals registry owner_agent
                                  (zero mismatches) — the field never records
                                  a validator other than the path owner.
  V4_GATE_REJECTS_TRUE_VALIDATOR  hermetic farm (synced): unmodified green;
                                  one document given a non-owner validator →
                                  checker rejects with validated_by mismatch.

WHAT THIS DOES NOT MEASURE. Whether A6 (or anyone) actually reviewed a page.
Whether ownership labels are useful. Only: the field named validated_by cannot
answer a validation question, and the gate that reads it cannot notice.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/docs/governance_registry.mjs"
CHECKER = ROOT / "scripts/docs/check_docs_registry.mjs"
REGISTRY = ROOT / "docs/governance/topic-registry.v1.json"
SYNC = "scripts/docs/sync_governance_metadata.mjs"

# Subject for the positive control: an established research page (not this rung).
SUBJECT = "docs/research/self_falsifying_compilation_line_r22_2026-07-29.md"
# A validator string that is not any path-owner label used in the generator.
TRUE_VALIDATOR = "human"

FARM_COPY = ["docs", "examples", "paper", "spec", "README.md"]
FARM_WEBSITE_COPY = "website/src/content"

META_RE = re.compile(r"^<!-- docs:meta\n([\s\S]*?)\n-->", re.M)
VB_RE = re.compile(r"^validated_by:\s*(.+)$", re.M)
RESEARCH_RULE_RE = re.compile(
    r"if\s*\(\s*relPath\.startsWith\(\s*'docs/research/'\s*\)\s*\)\s*\{"
    r"[\s\S]*?owner_agent:\s*'([^']+)'",
    re.M,
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    ).stdout.strip()


def declared_validated_by(path: Path) -> str:
    text = path.read_text(errors="replace")
    block = META_RE.search(text)
    body = block.group(1) if block else ""
    field = VB_RE.search(body)
    return field.group(1).strip() if field else "<absent>"


def clause_v1() -> bool:
    src = GENERATOR.read_text()
    sites = []
    for n, line in enumerate(src.splitlines(), start=1):
        # formatRepoMetadataBlock: `validated_by: ${topic.owner_agent}`
        # metadataFieldsForTopic: validated_by: topic.owner_agent,
        if "validated_by" in line and "owner_agent" in line:
            sites.append((n, line.strip()))
    print("V1 the sites that write validated_by from owner_agent:")
    for n, line in sites:
        print(f"    governance_registry.mjs:{n}  -> {line}")
    ok = len(sites) >= 2

    script = (
        "import(process.argv[1]).then(async (m) => {"
        "  const r = await m.buildGovernedTopicRegistry(process.argv[2]);"
        "  let match = 0, mismatch = 0;"
        "  for (const t of r.topics) {"
        "    const f = m.metadataFieldsForTopic(t);"
        "    if (f.validated_by === t.owner_agent) match++; else mismatch++;"
        "  }"
        "  console.log(JSON.stringify({topics: r.topics.length, match, mismatch}));"
        "});"
    )
    proc = subprocess.run(
        ["node", "-e", script, str(GENERATOR), str(ROOT)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"    generator would not run: {proc.stderr.strip()[:200]}")
        print("V1_FIELD_EQUALS_OWNER_AGENT FAIL")
        print()
        return False
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    print(
        f"    metadataFieldsForTopic over {seen['topics']} topics: "
        f"match={seen['match']} mismatch={seen['mismatch']}"
    )
    print("    The field is filled from owner_agent. No validation input reaches it.")
    ok = ok and seen["mismatch"] == 0 and seen["match"] == seen["topics"] and seen["topics"] > 0
    print(f"V1_FIELD_EQUALS_OWNER_AGENT {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_v2() -> tuple[bool, str]:
    src = GENERATOR.read_text()
    m = RESEARCH_RULE_RE.search(src)
    path_owner = m.group(1) if m else ""
    # Line number of the research path rule for the receipt.
    rule_line = 0
    for n, line in enumerate(src.splitlines(), start=1):
        if "docs/research/" in line and "startsWith" in line:
            rule_line = n
            break
    print("V2 path-prefix rule for research:")
    print(f"    governance_registry.mjs:{rule_line}  docs/research/ -> owner_agent {path_owner!r}")

    registry = json.loads(REGISTRY.read_text())
    research = [
        t
        for t in registry["topics"]
        if (t.get("repo_doc_path") or "").startswith("docs/research/")
    ]
    owners: dict[str, int] = {}
    for t in research:
        o = t.get("owner_agent") or "<absent>"
        owners[o] = owners.get(o, 0) + 1

    # Census declared field on disk
    census: dict[str, int] = {}
    unreadable = 0
    for t in research:
        rel = t["repo_doc_path"]
        path = ROOT / rel
        if not path.is_file():
            unreadable += 1
            continue
        v = declared_validated_by(path)
        census[v] = census.get(v, 0) + 1

    print(f"    registry research topics: {len(research)}")
    print(f"    owner_agent distribution: {owners}")
    print(f"    declared validated_by census ({unreadable} unreadable):")
    for value, count in sorted(census.items(), key=lambda kv: -kv[1]):
        print(f"        {value!r}: {count}")

    ok = (
        path_owner == "A6"
        and len(owners) == 1
        and owners.get("A6", 0) == len(research)
        and len(census) == 1
        and census.get("A6", 0) == len(research) - unreadable
        and unreadable == 0
        and len(research) > 0
    )
    print(f"V2_PATH_PREFIX_OWNS_RESEARCH {'PASS' if ok else 'FAIL'}")
    print()
    return ok, path_owner


def clause_v3() -> bool:
    registry = json.loads(REGISTRY.read_text())
    match = 0
    mismatch = 0
    samples: list[tuple[str, str, str]] = []
    for t in registry["topics"]:
        rel = t.get("repo_doc_path")
        if not rel:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        declared = declared_validated_by(path)
        owner = t.get("owner_agent") or "<absent>"
        if declared == owner:
            match += 1
        else:
            mismatch += 1
            if len(samples) < 5:
                samples.append((rel, declared, owner))

    print(f"V3 corpus: declared validated_by vs registry owner_agent")
    print(f"    match={match} mismatch={mismatch}")
    for rel, declared, owner in samples:
        print(f"    mismatch {rel}: declared={declared!r} owner={owner!r}")
    print("    The field never records a validator other than the path owner.")
    ok = mismatch == 0 and match > 0
    print(f"V3_CORPUS_IS_PATH_OWNERSHIP {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def run_checker(cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ["node", str(CHECKER)], cwd=str(cwd), capture_output=True, text=True
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def clause_v4(path_owner: str) -> bool:
    subject = ROOT / SUBJECT
    if not subject.is_file():
        print(f"V4 subject absent: {SUBJECT}")
        print("V4_GATE_REJECTS_TRUE_VALIDATOR FAIL")
        print()
        return False

    current = declared_validated_by(subject)
    if current != path_owner:
        print(f"V4 subject validated_by is {current!r}, expected path owner {path_owner!r}")
        print("V4_GATE_REJECTS_TRUE_VALIDATOR FAIL")
        print()
        return False
    if TRUE_VALIDATOR == path_owner:
        print("V4 TRUE_VALIDATOR collides with path owner; pick another string")
        print("V4_GATE_REJECTS_TRUE_VALIDATOR FAIL")
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

    farm = Path(tempfile.mkdtemp(prefix="sfcl-r23-farm."))
    try:
        copied = [p for p in FARM_COPY if (ROOT / p).exists()]
        cp = subprocess.run(
            ["cp", "-a", *copied, str(farm)], cwd=str(ROOT), capture_output=True, text=True
        )
        if cp.returncode != 0:
            print(f"V4 farm could not be built: {cp.stderr.strip()[:200]}")
            print("V4_GATE_REJECTS_TRUE_VALIDATOR FAIL")
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
            print("V4_GATE_REJECTS_TRUE_VALIDATOR FAIL")
            print()
            return False
        print(f"V4 farm synced to consistency: {sync.stdout.strip()[:90]}")

        rc_clean, _out_clean = run_checker(farm)
        clean_green = rc_clean == 0
        print(
            f"V4 negative control -- farm unmodified: checker rc={rc_clean}"
            f" ({'green' if clean_green else 'RED'})"
        )

        target = farm / SUBJECT
        text = target.read_text()
        patched, n_sub = re.subn(
            rf"^validated_by: {re.escape(path_owner)}$",
            f"validated_by: {TRUE_VALIDATOR}",
            text,
            count=1,
            flags=re.M,
        )
        if n_sub != 1:
            print("V4 could not patch the subject's validated_by")
            print("V4_GATE_REJECTS_TRUE_VALIDATOR FAIL")
            print()
            return False
        target.unlink()
        target.write_text(patched)

        rc_true, out_true = run_checker(farm)
        expected = (
            f'{SUBJECT} metadata mismatch for validated_by: '
            f'expected "{path_owner}"'
        )
        rejected = rc_true != 0 and expected in out_true
        print(f"    {SUBJECT}")
        print(f"    path owner (enforced)   {path_owner}")
        print(f"    truthful validator put  {TRUE_VALIDATOR}")
        print(
            f"V4 positive control -- truthful validator: checker rc={rc_true}"
            f" ({'REJECTED' if rejected else 'accepted'})"
        )
        if rejected:
            for line in out_true.splitlines():
                if "validated_by" in line:
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
        print(f"V4_GATE_REJECTS_TRUE_VALIDATOR {'PASS' if ok else 'FAIL'}")
        print()
        return ok
    finally:
        # farm may contain symlinks into ROOT — only unlink farm root via rm -rf
        subprocess.run(["rm", "-rf", str(farm)], check=False)


def main() -> int:
    if not GENERATOR.is_file() or not CHECKER.is_file():
        print("generator or checker absent")
        return 1
    if not shutil_which("node"):
        print("node absent")
        return 1

    print("Self-falsifying compilation R23 — validated_by is path ownership")
    print(f"ROOT={ROOT}")
    print()

    ok1 = clause_v1()
    ok2, path_owner = clause_v2()
    ok3 = clause_v3()
    ok4 = clause_v4(path_owner if path_owner else "A6")

    all_ok = ok1 and ok2 and ok3 and ok4
    verdict = "VALIDATED_BY_IS_PATH_OWNERSHIP__GATE_REJECTS_TRUE_VALIDATOR"
    print(f"SELF_FALSIFYING_R23_VERDICT {verdict}")
    print(f"OVERALL {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def shutil_which(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


if __name__ == "__main__":
    sys.exit(main())
