#!/usr/bin/env python3
"""Validate complete token receipts against the pinned diagnostic snapshot.
This is not scheduler, guardian, journal, or hardware qualification.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenized_cycle import accept_offline
from freeze_lifecycle_diagnostic import verify
from feedback_smoke import digest

def accept(frozen, arm, worker, output, job):
    if arm not in ("without-feedback", "with-feedback"):
        raise ValueError("unknown frozen arm")
    spec = verify(frozen)
    bundle = (frozen/arm/"offline-bundle.json").read_bytes()
    budget = len(json.loads(bundle)["items"])
    if budget != 8:
        raise ValueError("diagnostic requires eight frozen requests")
    for rank in (0, 1):
        r = json.loads((worker/f"rank-{rank}-complete.json").read_bytes())
        if r.get("job") != str(job) or r.get("stage") != "OFFLINE_CYCLE_COMPLETE":
            raise ValueError("diagnostic completion job/stage mismatch")
        if r.get("execution_profile", {}).get("actual_swa_tokens") != 896:
            raise ValueError("diagnostic SWA identity mismatch")
    output.mkdir(exist_ok=False)
    (output/"offline-bundle.json").write_bytes(bundle)
    accept_offline(output, dict(transport="sglang-offline-token-ids", budget=budget),
                   worker, runtime_root=frozen/"runtime")
    result = dict(schema="pireus-lifecycle-token-custody-v1", job=str(job), arm=arm,
        source_commit=spec["source_commit"],
        freeze_sha256=digest((frozen/"execution-freeze.json").read_bytes()),
        runtime_sha256=spec["runtime_sha256"]["offline_generate.py"],
        complete_token_receipts_valid=True, completed_requests=budget,
        hardware_qualified=False, pilot_acceptance=False,
        outstanding=["scheduler identity and terminal state", "guardian receipts",
                     "worker identity and raw log custody", "lifecycle journals"])
    (output/"token-custody.json").write_text(json.dumps(result, indent=2)+"\n")
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--arm", choices=["without-feedback","with-feedback"], required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job", required=True)
    a = p.parse_args()
    accept(a.frozen, a.arm, a.worker, a.output, a.job)
