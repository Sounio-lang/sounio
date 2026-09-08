#!/usr/bin/env python3
"""Wait for one diagnostic attempt and preserve receipts without pilot promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from resume_pilot import completed_accounting


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def collect(attempt, start_path, output):
    start = json.loads(start_path.read_text())
    job = str(start["job"])
    if str(attempt) != start["attempt"]:
        raise ValueError("attempt identity differs")
    manifest_raw = (attempt / "manifest.json").read_bytes()
    if digest(manifest_raw) != start["manifest_sha256"]:
        raise ValueError("diagnostic manifest changed")
    manifest = json.loads(manifest_raw)
    end = json.loads((attempt / "launcher-exit.json").read_text())
    log = (attempt / "launch.log").read_bytes()
    if digest(log) != end["log_sha256"]:
        raise ValueError("terminal log changed")
    output.mkdir()  # Never replace previous evidence.
    (output / "launch.log").write_bytes(log)
    (output / "manifest.json").write_bytes(manifest_raw)
    (output / "launcher-exit.json").write_text(json.dumps(end, indent=2) + "\n")
    all_receipts = []
    issues = []
    for rank, binding in enumerate(start["workers"]):
        worker = binding["worker"]
        prefix = ["kubectl", "-n", "slurm-pilot"]
        actual = json.loads(subprocess.check_output(prefix + ["get", "pod", worker["pod"], "-o", "json"], timeout=60))
        if actual["metadata"]["uid"] != worker["uid"]:
            raise ValueError("worker identity changed")
        # Exact bounded names; preserve any partial receipts on failed runs.
        code = ("import json;from pathlib import Path;"
                "names=" + repr([f"offline-{job}-{rank}-{i:03d}.json" for i in range(32)] +
                                [f"offline-{job}-{rank}-complete.json"]) + ";"
                "p=Path('/scratch/pireus/receipts');"
                "print(json.dumps({n:(p/n).read_text() for n in names if (p/n).exists()}))")
        receipt_texts = json.loads(subprocess.check_output(prefix + ["exec", worker["pod"], "-c", "slurmd",
                                                                   "--", "python3", "-c", code], timeout=60))
        rank_dir = output / f"rank-{rank}"
        rank_dir.mkdir()
        indexed = {}
        complete = None
        for name, text in receipt_texts.items():
            raw = text.encode()
            (rank_dir / name).write_bytes(raw)
            row = json.loads(raw)
            if str(row["job"]) != job or row["input_sha256"] != manifest["input_sha256"]:
                raise ValueError("receipt job/input mismatch")
            if name.endswith("-complete.json"):
                complete = row
                if str(row["rank"]) != str(rank) or row["helper_sha256"] != (manifest.get("runtime_sha256") or manifest["probe_sha256"]):
                    raise ValueError("completion runtime/rank mismatch")
            else:
                i = row["index"]
                if i in indexed or name != f"offline-{job}-{rank}-{i:03d}.json":
                    raise ValueError("duplicate or mismatched request index")
                indexed[i] = (row, digest(raw))
        if sorted(indexed) != list(range(32)) or complete is None:
            issues.append(f"rank{rank}: incomplete32-request receipts")
        if complete is not None:
            results = complete["results"]
            if [r["index"] for r in results] != list(range(32)):
                issues.append(f"rank{rank}: incomplete completion inventory")
            for result in results:
                row, sha = indexed[result["index"]]
                if sha != result["response_sha256"] or len(row["output_ids"]) != result["output_tokens"]:
                    raise ValueError("completion inventory differs from saved response")
        all_receipts.append(indexed)
    common = sorted(set(all_receipts[0]) & set(all_receipts[1]))
    paired = all(all_receipts[0][i] == all_receipts[1][i] for i in common)
    if not paired:
        issues.append("paired response mismatch")
    command = ["kubectl", "-n", "slurm-pilot", "exec", "slurm-pilot-controller-0", "-c", "slurmctld",
               "--", "sacct", "-j", job, "--format=JobIDRaw,State,ExitCode,NodeList,Start,End", "-n", "-P", "--noconvert"]
    accounting = subprocess.check_output(command, text=True, timeout=60)
    (output / "accounting.txt").write_text(accounting)
    try:
        completed_accounting(accounting, job)
    except ValueError as error:
        issues.append("accounting not qualified: " + str(error))
    if end["returncode"] != 0:
        issues.append("launcher nonzero exit")
    result = dict(schema="pireus-diagnostic-terminal-v1", job=int(job),
                  diagnostic_batch_complete=not issues,
                  paired_response_bytes_equal=paired, paired_responses=len(common),
                  output_tokens=sum(len(all_receipts[0][i][0]["output_ids"]) for i in common),
                  issues=issues, pilot_acceptance=False, performance_evidence=False,
                  root_cause_established=False, source_commit=manifest["source_commit"],
                  runtime_sha256=(manifest.get("runtime_sha256") or manifest["probe_sha256"]),
                  instrumentation=manifest.get("instrumentation", True),
                  input_sha256=manifest["input_sha256"],
                  file_hashes={str(p.relative_to(output)): digest(p.read_bytes())
                               for p in output.rglob("*") if p.is_file()})
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--start-evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    a = p.parse_args()
    deadline = time.monotonic() + 3 * 3600
    while not (a.attempt / "launcher-exit.json").exists():
        if not a.watch or time.monotonic() > deadline:
            raise SystemExit("diagnostic has no terminal launcher receipt")
        time.sleep(30)
    collect(a.attempt, a.start_evidence, a.output)
