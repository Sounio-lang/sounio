#!/usr/bin/env python3
"""Self-falsifying compilation, rung R22 (inverted 2026-08-16) — the guard that
replaced the gate that certified a literal.

Spec: docs/research/self_falsifying_compilation_line_r22_2026-07-29.md

ORIGINAL FINDING (2026-07-29, preserved in the spec): every governed document
carried a `last_validated` field shaped like a measurement that was in fact a
string literal in the generator ('2026-03-07', two sites), declared identically
by every document, older than the repository's first commit — and the CI
checker ENFORCED the constant, so a document recording the date it was really
validated was a gate failure. The gate was green exactly when the field it
guarded carried no information.

CLOSURE (#1752, 2026-08-16): the provenance pair is now preserve-per-document.
`preservedProvenance` keeps an existing well-formed record (a real YYYY-MM-DD
date and, when present, a non-empty validator); a missing or malformed record
still gets the generator's defaults. The checker stopped enforcing constant
equality on the pair and enforces shape instead — a real date, a non-empty
validator — while the structural four (topic_id, authority, audience,
source_of_truth) remain registry-authoritative.

This contract is the INVERTED instrument. Where the original demonstrated
that the truth was rejected, this one guards that the truth is accepted and
the surrounding contract still bites:

CLAUSES:
  V1_GENERATOR_PRESERVES_PROVENANCE  the default literal still exists at two
                                sites (a headerless doc still gets stamped),
                                but it is now a FALLBACK: measured by calling
                                preservedProvenance and metadataFieldsForTopic
                                with crafted records. The field has an input.
  V2_CORPUS_PAIR_IS_WELL_FORMED  census over every governed repo doc: every
                                declared last_validated is a YYYY-MM-DD date
                                and every validated_by is non-empty. Uniformity
                                is NO LONGER required — that was the defect.
  V3_STRUCTURE_STAYS_REGISTRY_BOUND  the inversion loosened only the pair:
                                structural fields still come from the topic,
                                and a malformed date still falls back to the
                                defaults. Measured, not argued.
  V4_TRUTHFUL_DATE_IS_ACCEPTED  end-to-end, hermetic, synced farm with five
                                arms: unmodified corpus stays green; a
                                git-true date is ACCEPTED; it SURVIVES a
                                re-sync (the exact regression #1752 fixed);
                                a malformed date is still REJECTED; a forged
                                structural field is still REJECTED. An
                                instrument with one arm measures nothing.

WHAT THIS DOES NOT MEASURE. Whether the documents were in fact validated on
the dates they declare. Preserving a record is not auditing it; this guard
keeps the field able to carry information, not verifies the information.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/docs/governance_registry.mjs"
CHECKER = ROOT / "scripts/docs/check_docs_registry.mjs"
REGISTRY = ROOT / "docs/governance/topic-registry.v1.json"

# The subject of the truthful-date control. Fixed rather than discovered so the
# receipt in the spec stays reproducible. R21's spec: authored 2026-07-28, long
# after the 2026-03-07 placeholder, so git's addition date differs from the
# default — the two arms cannot accidentally agree.
SUBJECT = "docs/research/self_falsifying_compilation_line_r21_2026-07-28.md"
MALFORMED_DATE = "2026-13-45"  # well-shaped month/day? no — ISO-invalid on purpose

# The farm is SYNCED before it is measured (see clause_v4), so every tree the
# sync can write must be a REAL copy — a hardlink would write through to the
# working tree, and sync_governance_metadata.mjs writes the three governance
# artifacts unconditionally. Everything else the checker merely resolves —
# link targets, related artifacts — so a symlink is enough. Real copy of these
# is 2.6 s; a whole-tree hardlink copy of 29 GB is 2 min 09 s.
FARM_COPY = ["docs", "examples", "paper", "spec", "README.md"]
FARM_WEBSITE_COPY = "website/src/content"
SYNC = "scripts/docs/sync_governance_metadata.mjs"

META_RE = re.compile(r"^<!-- docs:meta\n([\s\S]*?)\n-->", re.M)
FIELD_RE = re.compile(r"^last_validated:\s*(.+)$", re.M)
VB_FIELD_RE = re.compile(r"^validated_by:\s*(.*)$", re.M)
SITE_RE = re.compile(r"^\s*'?last_validated'?\s*:\s*'?([0-9]{4}-[0-9]{2}-[0-9]{2})'?",
                     re.M)
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def declared_field(path: Path, regex: re.Pattern) -> str:
    text = path.read_text(errors="replace")
    block = META_RE.search(text)
    body = block.group(1) if block else ""
    field = regex.search(body)
    return field.group(1).strip() if field else "<absent>"


def node_eval(script: str) -> dict | None:
    proc = subprocess.run(["node", "-e", script, str(GENERATOR), str(ROOT)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    generator would not run: {proc.stderr.strip()[:200]}")
        return None
    return json.loads(proc.stdout.strip().splitlines()[-1])


def clause_v1() -> tuple[bool, str]:
    """The default literal exists (fallback for headerless docs) and is
    bypassed whenever the document carries a well-formed record."""
    src = GENERATOR.read_text()
    sites = []
    for n, line in enumerate(src.splitlines(), start=1):
        m = SITE_RE.match(line)
        if m:
            sites.append((n, m.group(1)))
    print("V1 the default (fallback) sites that stamp a headerless doc:")
    for n, value in sites:
        print(f"    governance_registry.mjs:{n}  -> {value!r}  (default)")
    preserve_def = next((n for n, line in enumerate(src.splitlines(), start=1)
                         if "export function preservedProvenance" in line), 0)
    print(f"    governance_registry.mjs:{preserve_def}  -> export function "
          "preservedProvenance  (the preserve rule)")
    defaults = {value for _, value in sites}
    structural_ok = len(sites) == 2 and len(defaults) == 1 and preserve_def > 0

    # The inverse of the original topic-independence measurement: the field now
    # has an input. Ask the generator itself, over crafted records.
    script = (
        "import(process.argv[1]).then(async (m) => {"
        "  const r = await m.buildGovernedTopicRegistry(process.argv[2]);"
        "  const t = r.topics.find((x) => x.repo_doc_path"
        "    && x.repo_doc_path.startsWith('docs/research/'));"
        "  const good = m.preservedProvenance("
        "    {last_validated: '2026-08-16', validated_by: 'cursor-agent'});"
        "  const badDate = m.preservedProvenance("
        "    {last_validated: '2026-13-45', validated_by: 'cursor-agent'});"
        "  const noValidator = m.preservedProvenance({last_validated: '2026-08-16'});"
        "  const withRecord = m.metadataFieldsForTopic(t,"
        "    {last_validated: '2026-08-16', validated_by: 'cursor-agent'});"
        "  const withoutRecord = m.metadataFieldsForTopic(t);"
        "  console.log(JSON.stringify({"
        "    good, badDate, noValidator, withRecord, withoutRecord,"
        "    topic: t.topic_id, defaultDate: withoutRecord.last_validated,"
        "    defaultValidator: withoutRecord.validated_by}));"
        "});"
    )
    seen = node_eval(script)
    if seen is None:
        print("V1_GENERATOR_PRESERVES_PROVENANCE FAIL")
        print()
        return False, ""
    print(f"    preservedProvenance(good record)      -> {seen['good']}")
    print(f"    preservedProvenance(malformed date)   -> {seen['badDate']}")
    print(f"    preservedProvenance(no validator)     -> {seen['noValidator']}")
    print(f"    topic {seen['topic']}")
    print(f"      with its own record: last_validated="
          f"{seen['withRecord']['last_validated']!r} "
          f"validated_by={seen['withRecord']['validated_by']!r}")
    print(f"      with no record:      last_validated="
          f"{seen['withoutRecord']['last_validated']!r} "
          f"validated_by={seen['withoutRecord']['validated_by']!r}")
    ok = (
        structural_ok
        and seen["good"] == {"last_validated": "2026-08-16",
                             "validated_by": "cursor-agent"}
        and seen["badDate"] == {}
        and seen["noValidator"] == {"last_validated": "2026-08-16"}
        and seen["withRecord"]["last_validated"] == "2026-08-16"
        and seen["withRecord"]["validated_by"] == "cursor-agent"
        and seen["withoutRecord"]["last_validated"] == seen["defaultDate"]
        and seen["withoutRecord"]["validated_by"] == seen["defaultValidator"]
        and seen["withRecord"]["last_validated"] != seen["defaultDate"]
    )
    print("    The field has an input now: the document's own record, when"
          " well-formed; the literal only when there is none.")
    print(f"V1_GENERATOR_PRESERVES_PROVENANCE {'PASS' if ok else 'FAIL'}")
    print()
    return ok, (next(iter(defaults)) if len(defaults) == 1 else "")


def clause_v2() -> bool:
    """Shape census over the governed corpus. Uniformity is no longer the
    contract — it was the defect. Well-formedness is."""
    registry = json.loads(REGISTRY.read_text())
    docs = [t["repo_doc_path"] for t in registry["topics"]
            if t.get("repo_doc_path")]
    dates: dict[str, int] = {}
    validators: dict[str, int] = {}
    unreadable = 0
    malformed: list[tuple[str, str, str]] = []
    for rel in docs:
        path = ROOT / rel
        if not path.is_file():
            unreadable += 1
            continue
        date = declared_field(path, FIELD_RE)
        validator = declared_field(path, VB_FIELD_RE)
        dates[date] = dates.get(date, 0) + 1
        validators[validator] = validators.get(validator, 0) + 1
        if not ISO_RE.match(date) or not validator:
            malformed.append((rel, date, validator))
    print(f"V2 shape census over {len(docs)} governed repo docs "
          f"({unreadable} unreadable):")
    print(f"    distinct declared dates: {len(dates)} "
          f"(top: {sorted(dates.items(), key=lambda kv: -kv[1])[:3]})")
    print(f"    distinct declared validators: {len(validators)} "
          f"(top: {sorted(validators.items(), key=lambda kv: -kv[1])[:3]})")
    for rel, date, validator in malformed[:5]:
        print(f"    malformed: {rel}: last_validated={date!r} "
              f"validated_by={validator!r}")
    ok = unreadable == 0 and not malformed and len(docs) > 0
    print("    Every declared record is a real date and a non-empty validator."
          " Uniformity is no longer required — that was the defect.")
    print(f"V2_CORPUS_PAIR_IS_WELL_FORMED {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_v3() -> bool:
    """The inversion loosened ONLY the pair. Structure stays registry-bound,
    and a malformed date still falls back to the defaults."""
    script = (
        "import(process.argv[1]).then(async (m) => {"
        "  const r = await m.buildGovernedTopicRegistry(process.argv[2]);"
        "  const t = r.topics.find((x) => x.repo_doc_path"
        "    && x.repo_doc_path.startsWith('docs/research/'));"
        "  const forged = m.metadataFieldsForTopic(t, {"
        "    topic_id: 'forged.topic', authority: 'website_only',"
        "    audience: 'forged', last_validated: '2026-08-16',"
        "    validated_by: 'someone-else'});"
        "  const malformed = m.metadataFieldsForTopic(t, {"
        "    last_validated: '2026-13-45', validated_by: 'someone-else'});"
        "  console.log(JSON.stringify({"
        "    structural: {topic_id: forged.topic_id == t.topic_id,"
        "      authority: forged.authority == t.authority,"
        "      audience: forged.audience == t.audience,"
        "      source_of_truth: forged.source_of_truth"
        "        == m.topicSourceOfTruth(t.topic_id)},"
        "    forged_provenance_kept: forged.last_validated == '2026-08-16'"
        "      && forged.validated_by == 'someone-else',"
        "    malformed_falls_back: malformed.last_validated"
        "      == m.metadataFieldsForTopic(t).last_validated"
        "      && malformed.validated_by"
        "      == m.metadataFieldsForTopic(t).validated_by}));"
        "});"
    )
    seen = node_eval(script)
    if seen is None:
        print("V3_STRUCTURE_STAYS_REGISTRY_BOUND FAIL")
        print()
        return False
    print("V3 boundary of the inversion, measured on a real topic:")
    print(f"    structural fields from topic, not doc: {seen['structural']}")
    print(f"    provenance adopted from doc:            "
          f"{seen['forged_provenance_kept']}")
    print(f"    malformed date falls back to defaults:  "
          f"{seen['malformed_falls_back']}")
    ok = (all(seen["structural"].values()) and seen["forged_provenance_kept"]
          and seen["malformed_falls_back"])
    print("V3_STRUCTURE_STAYS_REGISTRY_BOUND "
          f"{'PASS' if ok else 'FAIL'}")
    print()
    return ok


def run_checker(cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(["node", str(CHECKER)], cwd=str(cwd),
                          capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)


def build_farm() -> Path | None:
    farm = Path(tempfile.mkdtemp(prefix="sfcl-r22-farm."))
    copied = [p for p in FARM_COPY if (ROOT / p).exists()]
    cp = subprocess.run(["cp", "-a", *copied, str(farm)], cwd=str(ROOT),
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print(f"V4 farm could not be built: {cp.stderr.strip()[:200]}")
        shutil.rmtree(farm, ignore_errors=True)
        return None
    web = ROOT / "website"
    if web.is_dir():
        (farm / "website/src").mkdir(parents=True)
        subprocess.run(["cp", "-a", FARM_WEBSITE_COPY,
                        str(farm / "website/src")], cwd=str(ROOT),
                       capture_output=True, text=True)
        for entry in sorted(os.listdir(web)):
            if entry != "src":
                os.symlink(web / entry, farm / "website" / entry)
        if (web / "src").is_dir():
            for entry in sorted(os.listdir(web / "src")):
                if entry != "content":
                    os.symlink(web / "src" / entry,
                               farm / "website/src" / entry)
    skip = {".git", "website", *FARM_COPY}
    for entry in sorted(os.listdir(ROOT)):
        if entry in skip:
            continue
        os.symlink(ROOT / entry, farm / entry)
    return farm


def clause_v4(default_date: str) -> bool:
    subject = ROOT / SUBJECT
    if not subject.is_file():
        print(f"V4 subject absent: {SUBJECT}")
        print("V4_TRUTHFUL_DATE_IS_ACCEPTED FAIL")
        print()
        return False

    added = git("log", "--diff-filter=A", "--format=%ad", "--date=short",
                "--", SUBJECT)
    true_date = added.splitlines()[-1].strip() if added else ""
    if not true_date or true_date == default_date:
        print(f"V4 no usable git addition date for {SUBJECT} "
              f"(got {true_date!r})")
        print("V4_TRUTHFUL_DATE_IS_ACCEPTED FAIL")
        print()
        return False

    witnessed = [ROOT / "docs/governance/topic-registry.v1.json",
                 ROOT / "docs/governance/DOCS_AUTHORITY_MATRIX.md",
                 ROOT / "docs/governance/DOCS_ACCEPTANCE_REPORT.md",
                 subject]
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size)
              for p in witnessed if p.is_file()}

    farm = build_farm()
    if farm is None:
        print("V4_TRUTHFUL_DATE_IS_ACCEPTED FAIL")
        print()
        return False
    try:
        # SYNC THE FARM BEFORE MEASURING IT. Without this the negative control
        # inherits the repository's registry staleness, and this rung reports
        # the docs-registry gate's news instead of its own. Staleness is that
        # gate's job; this clause asks one question and must be answerable
        # independently of it.
        sync = subprocess.run(["node", str(ROOT / SYNC)], cwd=str(farm),
                              capture_output=True, text=True)
        if sync.returncode != 0:
            print(f"V4 farm sync failed: {sync.stderr.strip()[:200]}")
            print("V4_TRUTHFUL_DATE_IS_ACCEPTED FAIL")
            print()
            return False
        print(f"V4 farm synced to consistency: {sync.stdout.strip()[:90]}")

        target = farm / SUBJECT
        pristine = target.read_text()

        def reset_subject() -> None:
            target.unlink()  # break any link before writing
            target.write_text(pristine)

        # NEGATIVE CONTROL. An instrument that fails on everything measures
        # nothing, so the farm must first reproduce the green result.
        rc_clean, _ = run_checker(farm)
        clean_green = rc_clean == 0
        print(f"V4 negative control -- farm unmodified: checker rc={rc_clean}"
              f" ({'green' if clean_green else 'RED'})")

        # ARM 1 — THE TRUTH IS ACCEPTED. This is the sentence the original
        # rung could not say. Give the subject the date git says it was added.
        patched, n_sub = re.subn(rf"^last_validated: {re.escape(default_date)}$",
                                 f"last_validated: {true_date}", pristine,
                                 count=1, flags=re.M)
        if n_sub != 1:
            print("V4 could not patch the subject's field")
            print("V4_TRUTHFUL_DATE_IS_ACCEPTED FAIL")
            print()
            return False
        target.unlink()
        target.write_text(patched)
        rc_true, out_true = run_checker(farm)
        accepted = rc_true == 0
        print(f"    {SUBJECT}")
        print(f"    git says it was added   {true_date}")
        print(f"    the document now says   {true_date}")
        print(f"V4 truthful-date control -- git-true date: checker rc={rc_true}"
              f" ({'accepted' if accepted else 'REJECTED'})")
        if not accepted:
            for line in out_true.splitlines():
                if "last_validated" in line:
                    print(f"      {line.strip()}")

        # ARM 2 — THE TRUTH SURVIVES THE SYNC. The regression #1752 fixed was
        # the sync STAMPING the placeholder over a real record. Re-run the
        # sync in the farm and require the true date to still be there.
        survived = False
        sync2 = subprocess.run(["node", str(ROOT / SYNC)], cwd=str(farm),
                               capture_output=True, text=True)
        if sync2.returncode == 0:
            after_date = declared_field(target, FIELD_RE)
            survived = after_date == true_date
            print(f"V4 preserve control -- survives the sync: "
                  f"last_validated={after_date!r}"
                  f"{'  (preserved)' if survived else '  (STAMPED OVER)'}")
        else:
            print(f"V4 preserve control -- re-sync failed: "
                  f"{sync2.stderr.strip()[:120]}")

        # ARM 3 — MALFORMED IS STILL REJECTED. Accepting the truth must not
        # have bought a vacuous checker: a malformed date still fails, and the
        # failure is attributed to the field under study.
        reset_subject()
        bad, n_sub = re.subn(
            rf"^last_validated: {re.escape(default_date)}$",
            f"last_validated: {MALFORMED_DATE}", pristine,
            count=1, flags=re.M)
        if n_sub != 1:
            print("V4 could not patch the subject's field (malformed arm)")
            print("V4_TRUTHFUL_DATE_IS_ACCEPTED FAIL")
            print()
            return False
        target.unlink()
        target.write_text(bad)
        rc_bad, out_bad = run_checker(farm)
        bad_rejected = rc_bad != 0 and "expected a YYYY-MM-DD date" in out_bad
        print(f"V4 malformed-date control -- {MALFORMED_DATE!r}: "
              f"checker rc={rc_bad}"
              f" ({'REJECTED' if rc_bad != 0 else 'accepted'})")
        if rc_bad != 0:
            for line in out_bad.splitlines():
                if "last_validated" in line:
                    print(f"      {line.strip()}")

        # ARM 4 — STRUCTURE STILL BITES. Provenance is preserve-per-document;
        # nothing else is. A forged structural field must still fail, which is
        # what separates this guard from a gate weakened into passing.
        reset_subject()
        forged_topic = next(
            line for line in pristine.splitlines()
            if line.startswith("topic_id: "))
        forged, _ = re.subn(re.escape(forged_topic), "topic_id: forged.topic",
                            pristine, count=1)
        target.unlink()
        target.write_text(forged)
        rc_forged, out_forged = run_checker(farm)
        forged_rejected = (rc_forged != 0
                           and "metadata mismatch for topic_id" in out_forged)
        print(f"V4 structural control -- forged topic_id: "
              f"checker rc={rc_forged}"
              f" ({'REJECTED' if rc_forged != 0 else 'accepted'})")
        if rc_forged != 0:
            for line in out_forged.splitlines():
                if "topic_id" in line:
                    print(f"      {line.strip()}")

        reset_subject()
        ok = (clean_green and accepted and survived and bad_rejected
              and forged_rejected)
    finally:
        shutil.rmtree(farm, ignore_errors=True)

    touched = [p for p, sig in before.items()
               if not p.is_file() or (p.stat().st_mtime_ns,
                                      p.stat().st_size) != sig]
    if touched:
        for p in touched:
            print(f"    HERMETICITY BREACH: {p.relative_to(ROOT)} was written")
        ok = False
    else:
        print(f"V4 hermetic: {len(before)} working-tree files unchanged"
              " across the farm syncs")

    print(f"V4_TRUTHFUL_DATE_IS_ACCEPTED {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main() -> int:
    print("Self-falsifying compilation R22 (inverted) -- the guard that"
          " replaced the gate that certified a literal")
    print("=" * 72)
    print()

    ok1, default_date = clause_v1()
    if not default_date:
        print("SELF_FALSIFYING_R22_VERDICT INCONCLUSIVE")
        return 1
    ok2 = clause_v2()
    ok3 = clause_v3()
    ok4 = clause_v4(default_date)

    ok = ok1 and ok2 and ok3 and ok4
    verdict = ("PROVENANCE_IS_PRESERVED__GATE_ACCEPTS_THE_TRUE_DATE"
               if ok else "INCONCLUSIVE")
    print("-" * 72)
    print("The original rung proved the field could not carry information:")
    print("a truthful date was a gate failure, and uniformity was what")
    print("passed. Closed 2026-08-16 by #1752: the record is preserved per")
    print("document, the checker enforces its shape, and structure stays")
    print("registry-bound. This inverted instrument guards the fixed")
    print("property in both directions -- the truth accepted, malformed")
    print("and forged values still rejected.")
    print()
    print(f"SELF_FALSIFYING_R22_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
