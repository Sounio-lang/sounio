#!/usr/bin/env python3
"""Verify archived bytes/source lineage. Does not rerun or grant runtime acceptance."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique)

def safe_path(root, name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("unsafe artifact path")
    rel = PurePosixPath(name)
    if rel.is_absolute() or not rel.parts or any(p in ("..", ".") for p in rel.parts) or str(rel) != name:
        raise ValueError("unsafe artifact path: " + name)
    target = root
    for part in rel.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("symlink artifact: " + name)
    return target

def verify_archive(archive, repo, recovery_file=None):
    archive = Path(archive).resolve()
    repo = Path(repo).resolve()
    summary = read_json(archive / "archive-summary.json")
    if summary.get("schema") != 1:
        raise ValueError("unsupported archive schema")
    commit = summary.get("source_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid source commit")
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
                   check=True, capture_output=True)
    inventory = {}
    for item in summary["archive_files"]:
        name = item["path"]
        path = safe_path(archive, name)
        if name in inventory:
            raise ValueError("duplicate artifact: " + name)
        if not path.is_file():
            raise ValueError("missing artifact: " + name)
        raw = path.read_bytes()
        if type(item["bytes"]) is not int or len(raw) != item["bytes"] or digest(raw) != item["sha256"]:
            raise ValueError("artifact digest/size mismatch: " + name)
        inventory[name] = item
    if not inventory:
        raise ValueError("empty archive")
    actual = {str(p.relative_to(archive)) for p in archive.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(inventory) | {"archive-summary.json"}:
        raise ValueError("unlisted archive files")
    def bound(name):
        safe_path(archive, name)
        if name not in inventory:
            raise ValueError("unbound required artifact: " + name)
        return archive / name
    journal = [json.loads(line) for line in bound("journal.jsonl").read_text().splitlines()]
    if not journal:
        raise ValueError("empty journal")
    for event in journal:
        name = event["artifact"]
        if digest(bound(name).read_bytes()) != event["sha256"]:
            raise ValueError("journal digest mismatch: " + name)
    manifest = read_json(bound("manifest.json"))
    sources = manifest["code_dependencies"]
    if not sources:
        raise ValueError("empty source dependency manifest")
    recovery_file = recovery_file or Path(__file__).with_name("source-recoveries.json")
    recovery = read_json(Path(recovery_file))
    if recovery.get("schema") != 1 or recovery.get("archive_source_commit") != commit:
        raise ValueError("source recovery identity")
    recovered = set()
    for name, expected in sources.items():
        safe_path(repo / "tools/pireus/continuity", name)
        original = subprocess.run(["git", "-C", str(repo), "show",
            commit + ":tools/pireus/continuity/" + name], capture_output=True)
        if original.returncode == 0:
            raw = original.stdout
        else:
            entry = recovery["sources"].get(name)
            if not entry or entry.get("sha256") != expected:
                raise ValueError("unpreserved historical source: " + name)
            recovered_commit = entry.get("recovery_commit", "")
            if not re.fullmatch(r"[0-9a-f]{40}", recovered_commit):
                raise ValueError("invalid recovery commit")
            subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                            recovered_commit, "HEAD"], check=True, capture_output=True)
            raw = subprocess.check_output(["git", "-C", str(repo), "show",
                recovered_commit + ":tools/pireus/continuity/" + name], stderr=subprocess.PIPE)
            recovered.add(name)
        if digest(raw) != expected:
            raise ValueError("historical source digest mismatch: " + name)
    if recovered != set(recovery["sources"]):
        raise ValueError("unused source recovery")
    count = manifest["budget"]
    if type(count) is not int or count < 1 or summary["generations"] != count:
        raise ValueError("generation count mismatch")
    tokens = 0
    for index in range(count):
        prefix = "%03d" % index
        raw = bound(prefix + ".token.response.json").read_bytes()
        for rank in (0, 1):
            if bound("worker/rank-%d-%s.json" % (rank, prefix)).read_bytes() != raw:
                raise ValueError("paired response mismatch")
        response = read_json(bound(prefix + ".token.response.json"))
        tokens += len(response["output_ids"])
        decoded = read_json(bound(prefix + ".decoded.json"))
        if (decoded["token_response_sha256"] != digest(raw)
            or bound(prefix + ".proposal.json").read_bytes() != decoded["text"].encode()):
            raise ValueError("decoded text changed")
    if summary["generated_tokens"] != tokens:
        raise ValueError("token count mismatch")
    benchmark = read_json(bound("benchmark-report.json"))
    if (summary["gain_eligible"] != benchmark["gain_eligible"]
        or summary["claim_ready"] is not benchmark["claim_ready"]):
        raise ValueError("summary disagrees with archived decision")
    return dict(schema=1, gate="PIREUS_ARCHIVE_CUSTODY_PASS",
                scope="archive-integrity-and-historical-source-provenance",
                source_commit=commit, files_verified=len(inventory),
                source_dependencies_verified=len(sources), paired_responses_verified=count,
                originally_untracked_sources=len(recovered),
                source_snapshot_complete_at_original_commit=not recovered,
                source_recovery_commits=sorted({recovery["sources"][name]["recovery_commit"] for name in recovered}),
                runtime_rerun=False, new_hardware_acceptance=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive, args.repo), indent=2))

if __name__ == "__main__":
    main()
