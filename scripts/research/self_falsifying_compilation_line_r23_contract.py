#!/usr/bin/env python3
"""Self-falsifying compilation, rung R23 (inverted 2026-08-16) — the guard that
replaced the ownership label misnamed as validation.

Spec: docs/research/self_falsifying_compilation_line_r23_2026-07-30.md

ORIGINAL FINDING (2026-07-30, preserved in the spec): `validated_by` was
filled from `topic.owner_agent`, and for every path under `docs/research/`
the owner was the path-prefix literal 'A6' — ownership inference wearing a
validation name. The CI checker enforced equality, so a document recording a
different validator was a gate failure, and the field answered a directory
question under a validation name.

CLOSURE (#1752, 2026-08-16, sibling of R22's): the provenance pair is
preserve-per-document. `preservedProvenance` keeps an existing well-formed
validator (and a real date); a missing, empty, or malformed record still gets
the generator's defaults. The checker enforces shape instead of equality —
a non-empty validator — while the structural four stay registry-authoritative.

This contract is the INVERTED instrument, guarding the fixed property in both
directions:

CLAUSES:
  V1_GENERATOR_PRESERVES_VALIDATOR  the owner_agent default still exists (a
                                headerless doc still gets stamped), but it is
                                now a FALLBACK: a document's own non-empty
                                validator wins. Measured by calling
                                preservedProvenance and metadataFieldsForTopic
                                with crafted records.
  V2_CORPUS_VALIDATORS_WELL_FORMED  census over every governed repo doc: every
                                declared validated_by is non-empty. Owner-label
                                equality is NO LONGER required — that was the
                                defect.
  V3_STRUCTURE_STAYS_REGISTRY_BOUND  the inversion loosened only the pair:
                                structural fields still come from the topic,
                                and an empty validator still falls back to the
                                default. Measured, not argued.
  V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED  end-to-end, hermetic, synced farm with
                                five arms: unmodified corpus stays green; a
                                non-owner validator is ACCEPTED; it SURVIVES a
                                re-sync (the exact regression #1752 fixed); an
                                empty validator is still REJECTED; a forged
                                structural field is still REJECTED.

WHAT THIS DOES NOT MEASURE. Whether the named validator actually reviewed the
document. Preserving a name is not auditing it; this guard keeps the field
able to carry the record, not verifies the record.
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

# Subject for the truthful-validator control: R22's spec (not this rung),
# a research page whose owner label is the path-prefix default.
SUBJECT = "docs/research/self_falsifying_compilation_line_r22_2026-07-29.md"
# A validator string that is not any path-owner label used in the generator.
TRUE_VALIDATOR = "human"

FARM_COPY = ["docs", "examples", "paper", "spec", "README.md"]
FARM_WEBSITE_COPY = "website/src/content"
SYNC = "scripts/docs/sync_governance_metadata.mjs"

META_RE = re.compile(r"^<!-- docs:meta\n([\s\S]*?)\n-->", re.M)
VB_RE = re.compile(r"^validated_by:\s*(.*)$", re.M)


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


def node_eval(script: str) -> dict | None:
    proc = subprocess.run(
        ["node", "-e", script, str(GENERATOR), str(ROOT)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"    generator would not run: {proc.stderr.strip()[:200]}")
        return None
    return json.loads(proc.stdout.strip().splitlines()[-1])


def clause_v1() -> tuple[bool, str]:
    """The owner_agent default still exists (fallback) and is bypassed when
    the document carries a non-empty validator of its own."""
    src = GENERATOR.read_text()
    sites = []
    for n, line in enumerate(src.splitlines(), start=1):
        # formatRepoMetadataBlock: `validated_by: ${topic.owner_agent}`
        # metadataFieldsForTopic: validated_by: topic.owner_agent,
        if "validated_by" in line and "owner_agent" in line:
            sites.append((n, line.strip()))
    print("V1 the default (fallback) sites that fill validated_by from owner_agent:")
    for n, line in sites:
        print(f"    governance_registry.mjs:{n}  -> {line}")
    preserve_def = next(
        (n for n, line in enumerate(src.splitlines(), start=1)
         if "export function preservedProvenance" in line), 0)
    print(f"    governance_registry.mjs:{preserve_def}  -> export function "
          "preservedProvenance  (the preserve rule)")
    structural_ok = len(sites) >= 2 and preserve_def > 0

    # The inverse of the original owner-equality measurement: the field now
    # has an input. Ask the generator itself, over crafted records.
    script = (
        "import(process.argv[1]).then(async (m) => {"
        "  const r = await m.buildGovernedTopicRegistry(process.argv[2]);"
        "  const t = r.topics.find((x) => x.repo_doc_path"
        "    && x.repo_doc_path.startsWith('docs/research/'));"
        "  const good = m.preservedProvenance("
        "    {last_validated: '2026-08-16', validated_by: 'human'});"
        "  const empty = m.preservedProvenance("
        "    {last_validated: '2026-08-16', validated_by: '   '});"
        "  const absent = m.preservedProvenance({last_validated: '2026-08-16'});"
        "  const withRecord = m.metadataFieldsForTopic(t,"
        "    {last_validated: '2026-08-16', validated_by: 'human'});"
        "  const withoutRecord = m.metadataFieldsForTopic(t);"
        "  console.log(JSON.stringify({"
        "    good, empty, absent, withRecord, withoutRecord,"
        "    topic: t.topic_id, owner: t.owner_agent}));"
        "});"
    )
    seen = node_eval(script)
    if seen is None:
        print("V1_GENERATOR_PRESERVES_VALIDATOR FAIL")
        print()
        return False, ""
    print(f"    preservedProvenance(non-owner validator) -> {seen['good']}")
    print(f"    preservedProvenance(blank validator)      -> {seen['empty']}")
    print(f"    preservedProvenance(no validator)         -> {seen['absent']}")
    print(f"    topic {seen['topic']} (registry owner {seen['owner']!r})")
    print(f"      with its own record: validated_by={seen['withRecord']['validated_by']!r}")
    print(f"      with no record:      validated_by={seen['withoutRecord']['validated_by']!r}")
    ok = (
        structural_ok
        and seen["good"] == {"last_validated": "2026-08-16",
                             "validated_by": "human"}
        and seen["empty"] == {"last_validated": "2026-08-16"}
        and seen["absent"] == {"last_validated": "2026-08-16"}
        and seen["withRecord"]["validated_by"] == TRUE_VALIDATOR
        and seen["withoutRecord"]["validated_by"] == seen["owner"]
        and seen["withRecord"]["validated_by"] != seen["owner"]
    )
    print("    The field has an input now: the document's own validator, when"
          " non-empty; the owner label only when there is none.")
    print(f"V1_GENERATOR_PRESERVES_VALIDATOR {'PASS' if ok else 'FAIL'}")
    print()
    return ok, (seen["owner"] if seen else "")


def clause_v2() -> bool:
    """Shape census. Owner-label equality is no longer the contract — it was
    the defect. Non-emptiness is."""
    registry = json.loads(REGISTRY.read_text())
    docs = [t["repo_doc_path"] for t in registry["topics"]
            if t.get("repo_doc_path")]
    census: dict[str, int] = {}
    unreadable = 0
    empty = []
    for rel in docs:
        path = ROOT / rel
        if not path.is_file():
            unreadable += 1
            continue
        v = declared_validated_by(path)
        census[v] = census.get(v, 0) + 1
        if not v:
            empty.append(rel)
    research = [v for rel, v in
                ((rel, declared_validated_by(ROOT / rel)) for rel in docs
                 if rel.startswith("docs/research/") and (ROOT / rel).is_file())]
    print(f"V2 shape census over {len(docs)} governed repo docs "
          f"({unreadable} unreadable):")
    print(f"    distinct declared validators: {len(census)} "
          f"(top: {sorted(census.items(), key=lambda kv: -kv[1])[:3]})")
    print(f"    docs/research/ pages declaring a non-owner validator: "
          f"{sum(1 for v in research if v not in ('A6', '<absent>'))} "
          f"of {len(research)}")
    for rel in empty[:5]:
        print(f"    empty validator: {rel}")
    ok = unreadable == 0 and not empty and len(docs) > 0
    print("    Every declared validator is non-empty. Owner-label equality is"
          " no longer required — that was the defect.")
    print(f"V2_CORPUS_VALIDATORS_WELL_FORMED {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_v3() -> bool:
    """The inversion loosened ONLY the pair. Structure stays registry-bound,
    and an empty validator still falls back to the default."""
    script = (
        "import(process.argv[1]).then(async (m) => {"
        "  const r = await m.buildGovernedTopicRegistry(process.argv[2]);"
        "  const t = r.topics.find((x) => x.repo_doc_path"
        "    && x.repo_doc_path.startsWith('docs/research/'));"
        "  const forged = m.metadataFieldsForTopic(t, {"
        "    topic_id: 'forged.topic', authority: 'website_only',"
        "    audience: 'forged', last_validated: '2026-08-16',"
        "    validated_by: 'human'});"
        "  const blank = m.metadataFieldsForTopic(t,"
        "    {last_validated: '2026-08-16', validated_by: ''});"
        "  const d = m.metadataFieldsForTopic(t);"
        "  console.log(JSON.stringify({"
        "    structural: {topic_id: forged.topic_id == t.topic_id,"
        "      authority: forged.authority == t.authority,"
        "      audience: forged.audience == t.audience,"
        "      source_of_truth: forged.source_of_truth"
        "        == m.topicSourceOfTruth(t.topic_id)},"
        "    forged_validator_kept: forged.validated_by == 'human',"
        "    blank_falls_back: blank.validated_by == d.validated_by,"
        "    blank_date_kept: blank.last_validated == '2026-08-16'}));"
        "});"
    )
    seen = node_eval(script)
    if seen is None:
        print("V3_STRUCTURE_STAYS_REGISTRY_BOUND FAIL")
        print()
        return False
    print("V3 boundary of the inversion, measured on a real topic:")
    print(f"    structural fields from topic, not doc:  {seen['structural']}")
    print(f"    validator adopted from doc:              "
          f"{seen['forged_validator_kept']}")
    print(f"    blank validator falls back to owner:     "
          f"{seen['blank_falls_back']}")
    print(f"    well-formed date kept alongside blank:   "
          f"{seen['blank_date_kept']}")
    ok = (all(seen["structural"].values()) and seen["forged_validator_kept"]
          and seen["blank_falls_back"] and seen["blank_date_kept"])
    print(f"V3_STRUCTURE_STAYS_REGISTRY_BOUND {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def run_checker(cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ["node", str(CHECKER)], cwd=str(cwd), capture_output=True, text=True
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def build_farm() -> Path | None:
    farm = Path(tempfile.mkdtemp(prefix="sfcl-r23-farm."))
    copied = [p for p in FARM_COPY if (ROOT / p).exists()]
    cp = subprocess.run(
        ["cp", "-a", *copied, str(farm)], cwd=str(ROOT), capture_output=True, text=True
    )
    if cp.returncode != 0:
        print(f"V4 farm could not be built: {cp.stderr.strip()[:200]}")
        shutil.rmtree(farm, ignore_errors=True)
        return None
    web = ROOT / "website"
    if web.is_dir():
        (farm / "website/src").mkdir(parents=True)
        subprocess.run(
            ["cp", "-a", FARM_WEBSITE_COPY, str(farm / "website/src")],
            cwd=str(ROOT), capture_output=True, text=True,
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
    return farm


def clause_v4(owner: str) -> bool:
    subject = ROOT / SUBJECT
    if not subject.is_file():
        print(f"V4 subject absent: {SUBJECT}")
        print("V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED FAIL")
        print()
        return False

    current = declared_validated_by(subject)
    if current != owner:
        print(f"V4 subject validated_by is {current!r}, expected owner default {owner!r}")
        print("V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED FAIL")
        print()
        return False
    if TRUE_VALIDATOR == owner:
        print("V4 TRUE_VALIDATOR collides with the owner default; pick another string")
        print("V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED FAIL")
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

    farm = build_farm()
    if farm is None:
        print("V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED FAIL")
        print()
        return False
    try:
        # SYNC THE FARM BEFORE MEASURING IT, as in R22: staleness is the
        # docs-registry gate's news, not this one's.
        sync = subprocess.run(
            ["node", str(ROOT / SYNC)], cwd=str(farm), capture_output=True, text=True
        )
        if sync.returncode != 0:
            print(f"V4 farm sync failed: {sync.stderr.strip()[:200]}")
            print("V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED FAIL")
            print()
            return False
        print(f"V4 farm synced to consistency: {sync.stdout.strip()[:90]}")

        target = farm / SUBJECT
        pristine = target.read_text()

        def reset_subject() -> None:
            target.unlink()  # break any link before writing
            target.write_text(pristine)

        # NEGATIVE CONTROL. The farm must first reproduce the green result.
        rc_clean, _ = run_checker(farm)
        clean_green = rc_clean == 0
        print(
            f"V4 negative control -- farm unmodified: checker rc={rc_clean}"
            f" ({'green' if clean_green else 'RED'})"
        )

        # ARM 1 — THE TRUTH IS ACCEPTED. The original rung's positive control
        # was this exact patch failing; it must now pass.
        patched, n_sub = re.subn(
            rf"^validated_by: {re.escape(owner)}$",
            f"validated_by: {TRUE_VALIDATOR}",
            pristine, count=1, flags=re.M,
        )
        if n_sub != 1:
            print("V4 could not patch the subject's validated_by")
            print("V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED FAIL")
            print()
            return False
        target.unlink()
        target.write_text(patched)
        rc_true, out_true = run_checker(farm)
        accepted = rc_true == 0
        print(f"    {SUBJECT}")
        print(f"    owner label (default)   {owner}")
        print(f"    truthful validator put  {TRUE_VALIDATOR}")
        print(
            f"V4 truthful-validator control -- non-owner name: checker rc={rc_true}"
            f" ({'accepted' if accepted else 'REJECTED'})"
        )
        if not accepted:
            for line in out_true.splitlines():
                if "validated_by" in line:
                    print(f"      {line.strip()}")

        # ARM 2 — THE TRUTH SURVIVES THE SYNC. The regression #1752 fixed was
        # the sync STAMPING the owner label over a real record.
        survived = False
        sync2 = subprocess.run(
            ["node", str(ROOT / SYNC)], cwd=str(farm), capture_output=True, text=True
        )
        if sync2.returncode == 0:
            after_v = declared_validated_by(target)
            survived = after_v == TRUE_VALIDATOR
            print(f"V4 preserve control -- survives the sync: "
                  f"validated_by={after_v!r}"
                  f"{'  (preserved)' if survived else '  (STAMPED OVER)'}")
        else:
            print(f"V4 preserve control -- re-sync failed: "
                  f"{sync2.stderr.strip()[:120]}")

        # ARM 3 — EMPTY IS STILL REJECTED. Accepting the truth must not have
        # bought a vacuous checker: a blanked validator still fails, and the
        # failure is attributed to the field under study.
        reset_subject()
        blanked, n_sub = re.subn(
            rf"^validated_by: {re.escape(owner)}$",
            "validated_by: ",
            pristine, count=1, flags=re.M,
        )
        if n_sub != 1:
            print("V4 could not patch the subject's validated_by (empty arm)")
            print("V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED FAIL")
            print()
            return False
        target.unlink()
        target.write_text(blanked)
        rc_blank, out_blank = run_checker(farm)
        blank_rejected = (rc_blank != 0
                          and "expected a non-empty validator" in out_blank)
        print(f"V4 empty-validator control -- blanked field: checker rc={rc_blank}"
              f" ({'REJECTED' if rc_blank != 0 else 'accepted'})")
        if rc_blank != 0:
            for line in out_blank.splitlines():
                if "validated_by" in line:
                    print(f"      {line.strip()}")

        # ARM 4 — STRUCTURE STILL BITES. A forged structural field must still
        # fail, separating this guard from a gate weakened into passing.
        reset_subject()
        forged_topic = next(
            line for line in pristine.splitlines()
            if line.startswith("topic_id: ")
        )
        forged, _ = re.subn(re.escape(forged_topic), "topic_id: forged.topic",
                            pristine, count=1)
        target.unlink()
        target.write_text(forged)
        rc_forged, out_forged = run_checker(farm)
        forged_rejected = (rc_forged != 0
                           and "metadata mismatch for topic_id" in out_forged)
        print(f"V4 structural control -- forged topic_id: checker rc={rc_forged}"
              f" ({'REJECTED' if rc_forged != 0 else 'accepted'})")
        if rc_forged != 0:
            for line in out_forged.splitlines():
                if "topic_id" in line:
                    print(f"      {line.strip()}")

        reset_subject()
        ok = (clean_green and accepted and survived and blank_rejected
              and forged_rejected)
    finally:
        # farm may contain symlinks into ROOT — only unlink farm root via rm -rf
        subprocess.run(["rm", "-rf", str(farm)], check=False)

    touched = [p for p, sig in before.items()
               if not p.is_file() or (p.stat().st_mtime_ns,
                                      p.stat().st_size) != sig]
    hermetic_ok = not touched
    for p in touched:
        print(f"    HERMETIC BREACH: {p.relative_to(ROOT)}")
    if hermetic_ok:
        print(f"V4 hermetic: {len(before)} working-tree files unchanged")

    print(f"V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED {'PASS' if ok and hermetic_ok else 'FAIL'}")
    print()
    return ok and hermetic_ok


def main() -> int:
    if not GENERATOR.is_file() or not CHECKER.is_file():
        print("generator or checker absent")
        return 1
    if not shutil.which("node"):
        print("node absent")
        return 1

    print("Self-falsifying compilation R23 (inverted) — the guard that"
          " replaced the ownership label misnamed as validation")
    print(f"ROOT={ROOT}")
    print()

    ok1, owner = clause_v1()
    if not owner:
        print("SELF_FALSIFYING_R23_VERDICT INCONCLUSIVE")
        return 1
    ok2 = clause_v2()
    ok3 = clause_v3()
    ok4 = clause_v4(owner)

    all_ok = ok1 and ok2 and ok3 and ok4
    verdict = "VALIDATOR_IS_PRESERVED__GATE_ACCEPTS_THE_TRUE_VALIDATOR" if all_ok else "INCONCLUSIVE"
    print(f"SELF_FALSIFYING_R23_VERDICT {verdict}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
