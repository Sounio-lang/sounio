#!/usr/bin/env python3
"""Project archived native receipts into feedback; never decide gain in Python."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "validation/control-union-hardware-20260908"
AUDIT_SHA = "2d8230c4665825e495078286e0dad6569fc7808197c4c95b84abcea087b25559"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(source=SOURCE):
    require(sha(source / "hardware-audit.json") == AUDIT_SHA, "audit identity mismatch")
    audit = read(source / "hardware-audit.json")
    hashes = audit["file_hashes"]
    for name, expected in hashes.items():
        path = source / name
        require(path.resolve().is_relative_to(source.resolve()), "unsafe artifact path")
        require(sha(path) == expected, "artifact hash mismatch: " + name)
    members = read(source / "control-membership.json")["members"]
    require(len(members) == 64, "expected64 source occurrences")
    require({(m["source_job"], m["request_index"]) for m in members} ==
            {(job, i) for job in (11957, 11958) for i in range(32)},
            "source occurrence coverage mismatch")
    materials = []
    for i in range(18):
        rep = f"{i:03}"
        names = [rep + suffix for suffix in
                 (".proposal.json", ".receipt.json", ".ptx", ".gain.json")]
        require(all(name in hashes for name in names), "unbound material")
        proposal, receipt, gain = [read(source / (rep + suffix)) for suffix in
                                   (".proposal.json", ".receipt.json", ".gain.json")]
        require(receipt["authority"] == "Sounio" and receipt["decision"] == "ADMIT",
                "native admission missing")
        require(receipt["proposal_sha256"] == hashes[rep + ".proposal.json"],
                "admission/proposal mismatch")
        require(gain["authority"] == "Sounio" and gain["gain_gate"] == "NO_GAIN",
                "unexpected native gain decision")
        occurrences = [m for m in members if m["representative"] == rep]
        require(occurrences and all(m["plan_id"] == receipt["plan_id"] and
                m["ptx_sha256"] == hashes[rep + ".ptx"] for m in occurrences),
                "material membership mismatch")
        materials.append(dict(
            representative=rep, plan_id=receipt["plan_id"],
            proposal=proposal, native_admission=receipt, native_gain=gain,
            source_sha256={n: hashes[n] for n in names},
            occurrences=occurrences,
            parity_evidence=dict(audit_sha256=AUDIT_SHA, job=11963,
                                 scope="finite fixture;5120 exact components per node"),
            interpretation="NO_GAIN under frozen contract; not semantic rejection",
        ))
    require({m["representative"] for m in members} ==
            {m["representative"] for m in materials}, "unexpected representative")
    return dict(
        schema="pireus-control-feedback-v1",
        authority="transport of archived Sounio decisions; no new semantic authority",
        source_audit_sha256=AUDIT_SHA,
        source_scope=audit["scope"],
        source_jobs=[11957, 11958], parity_job=11963, timing_job=11964,
        material_count=18, occurrence_count=64, materials=materials,
        use="feedback for a separately frozen future experiment only",
        automatic_blacklist=False, pilot_acceptance=False,
        scientific_novelty=False, memory_root_cause_established=False,
        holdout=False, grpo_reward_qualified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.dumps(build(args.source), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
