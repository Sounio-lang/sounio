#!/usr/bin/env python3
"""Native admission/materialization of preserved controls, outside the pilot journal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, raw):
    with path.open("xb") as f:
        f.write(raw)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def run(decoded, pilot, output):
    if not os.environ.get("TMUX"):
        raise ValueError("use remote tmux")
    summary = json.loads((decoded / "summary.json").read_text())
    for name, sha in summary["file_hashes"].items():
        path = decoded / name
        if not path.resolve().is_relative_to(decoded.resolve()) or digest(path.read_bytes()) != sha:
            raise ValueError("decoded archive changed")
    receipts = [json.loads((decoded / f"rank-{r}.json").read_text()) for r in (0, 1)]
    if receipts[0]["items"] != receipts[1]["items"]:
        raise ValueError("paired decoded proposals differ")
    items = receipts[0]["items"]
    if [r["index"] for r in items] != list(range(64)):
        raise ValueError("expected exactly64 ordered control proposals")
    manifest = json.loads((pilot / "pilot-manifest.json").read_text())
    engine = manifest["engines"]["engine"]
    executable = Path(engine["path"])
    if digest(executable.read_bytes()) != engine["sha256"]:
        raise ValueError("frozen admission executable changed")
    context = (pilot / "round-0-inkling-no-ontology/context.json").read_bytes()
    context_sha = digest(context)
    for item in items:
        if json.loads(item["text"])["context"] != context_sha:
            raise ValueError("proposal context binding differs")
    output.mkdir()
    save(output / "context.json", context)
    save(output / "intent.json", encoded(dict(schema="pireus-control-native-analysis-v1",
        source_jobs=[11957, 11958], decoded_job=11962, proposals=64,
        engine=engine, context_sha256=context_sha,
        decoded_receipt_sha256=[digest((decoded / f"rank-{r}.json").read_bytes()) for r in (0, 1)],
        pilot_acceptance=False, performance_evidence=False)))
    results = []
    for index, item in enumerate(items):
        job, request = (11957 if index < 32 else 11958), index % 32
        folder = output / str(job)
        folder.mkdir(exist_ok=True)
        prefix = folder / f"{request:03d}"
        proposal = prefix.with_suffix(".proposal.json")
        raw = item["text"].encode()
        save(proposal, raw)  # Preserve decoded text; no repair or normalization.
        cmd = [str(executable), str(output / "context.json"), str(proposal)]
        admission = subprocess.run(cmd, capture_output=True, timeout=60)
        save(prefix.with_suffix(".admission.stdout"), admission.stdout)
        save(prefix.with_suffix(".admission.stderr"), admission.stderr)
        receipt = json.loads(admission.stdout)
        if receipt.get("authority") != "Sounio" or admission.returncode not in (0, 1):
            raise ValueError("invalid native admission result")
        if (admission.returncode == 0) != (receipt.get("decision") == "ADMIT"):
            raise ValueError("native decision and exit disagree")
        if receipt.get("proposal_sha256") != digest(raw) or receipt.get("context_sha256") != context_sha:
            raise ValueError("native receipt input hashes differ")
        save(prefix.with_suffix(".receipt.json"), admission.stdout)
        result = dict(source_job=job, index=request, decision=receipt["decision"],
                      proposal_sha256=digest(raw), receipt_sha256=digest(admission.stdout),
                      admission_exit=admission.returncode, plan_id=receipt.get("plan_id"), ptx_sha256=None)
        if receipt["decision"] == "ADMIT":
            material = subprocess.run(cmd + ["ptx"], capture_output=True, timeout=60)
            save(prefix.with_suffix(".materialization.stderr"), material.stderr)
            if material.returncode != 0 or not material.stdout.strip():
                save(prefix.with_suffix(".materialization.failure"), material.stdout)
                raise ValueError("native materialization refused or empty")
            save(prefix.with_suffix(".ptx"), material.stdout)
            result["ptx_sha256"] = digest(material.stdout)
        results.append(result)
    cohorts = {}
    for job in (11957, 11958):
        rows = [r for r in results if r["source_job"] == job]
        admitted = [r for r in rows if r["decision"] == "ADMIT"]
        cohorts[str(job)] = dict(proposals=len(rows), admitted=len(admitted),
            refused=len(rows)-len(admitted), materialized=sum(r["ptx_sha256"] is not None for r in rows),
            unique_native_plan_ids=len({r["plan_id"] for r in admitted}),
            unique_ptx_bytes=len({r["ptx_sha256"] for r in admitted}),
            unique_plan_and_ptx=len({(r["plan_id"], r["ptx_sha256"]) for r in admitted}))
    comparisons = []
    for i in range(32):
        a, b = results[i], results[i+32]
        comparisons.append(dict(index=i, decision_equal=a["decision"] == b["decision"],
            native_plan_equal=(a["plan_id"] == b["plan_id"]) if a["decision"] == b["decision"] == "ADMIT" else None,
            ptx_equal=(a["ptx_sha256"] == b["ptx_sha256"]) if a["ptx_sha256"] and b["ptx_sha256"] else None))
    summary = dict(schema="pireus-control-native-comparison-v1", cohorts=cohorts,
        engine_sha256=engine["sha256"], results=results, comparisons=comparisons,
        pilot_acceptance=False, gpu_parity="UNMEASURED", performance_gain="UNMEASURED",
        claim_ready=False, memory_root_cause_established=False,
        file_hashes={str(p.relative_to(output)): digest(p.read_bytes())
                    for p in output.rglob("*") if p.is_file()})
    save(output / "summary.json", encoded(summary))
    print(json.dumps(dict(cohorts=cohorts, comparisons=comparisons)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--decoded", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not all(p.is_absolute() for p in (a.decoded, a.pilot, a.output)):
        p.error("paths must be absolute")
    run(a.decoded, a.pilot, a.output)
