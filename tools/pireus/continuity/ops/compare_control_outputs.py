#!/usr/bin/env python3
"""Compare preserved instrumented/control receipts; never promote the pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verified(root):
    s = json.loads((root / "summary.json").read_text())
    for name, expected in s["file_hashes"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("archive path escapes root")
        if sha(path.read_bytes()) != expected:
            raise ValueError("archive content differs: " + name)
    return s, json.loads((root / "manifest.json").read_text())


def compare(reference, control):
    rs, rm = verified(reference)
    cs, cm = verified(control)
    if rm["input_sha256"] != cm["input_sha256"]:
        raise ValueError("control changed frozen input")
    if cm.get("instrumentation") is not False or cm.get("parent_diagnostic_job") != rs["job"]:
        raise ValueError("control identity is not bound to diagnostic")
    if cm["parent_manifest_sha256"] != sha((reference / "manifest.json").read_bytes()):
        raise ValueError("control has a different reference manifest")
    if set(rm["runtime_files"]) != set(cm["runtime_files"]):
        raise ValueError("control changed runtime inventory")
    changed = [n for n in rm["runtime_files"] if rm["runtime_files"][n] != cm["runtime_files"][n]]
    if changed != ["offline_generate.py"] or cm["runtime_sha256"] != rm["original_runtime_sha256"]:
        raise ValueError("control changes more than removal of probes")
    result = dict(schema="pireus-instrumentation-control-comparison-v1",
                  reference_job=rs["job"], control_job=cs["job"],
                  reference_summary_sha256=sha((reference / "summary.json").read_bytes()),
                  control_summary_sha256=sha((control / "summary.json").read_bytes()),
                  input_sha256=cm["input_sha256"], runtime_delta=changed,
                  complete=False, output_ids_equal=None, execution_profiles_equal=None,
                  pilot_acceptance=False, performance_evidence=False,
                  memory_root_cause_established=False,
                  limitations=["Different run times, cache state and observation overhead remain potential confounders.",
                               "One passing control does not establish general stability or erase earlier failures."])
    if not rs["diagnostic_batch_complete"] or not cs["diagnostic_batch_complete"]:
        result["status"] = "INCOMPLETE_OR_FAILED_CONTROL"
        result["control_issues"] = cs["issues"]
        return result
    output_differences = []
    profile_differences = []
    counts = {}
    for rank in (0, 1):
        counts[str(rank)] = 0
        for i in range(32):
            a = json.loads((reference / f"rank-{rank}/offline-{rs['job']}-{rank}-{i:03d}.json").read_text())
            b = json.loads((control / f"rank-{rank}/offline-{cs['job']}-{rank}-{i:03d}.json").read_text())
            for row, job in ((a, rs["job"]), (b, cs["job"])):
                if str(row["job"]) != str(job) or row["index"] != i or row["input_sha256"] != cm["input_sha256"]:
                    raise ValueError("response identity mismatch")
                if row["completion_tokens"] != len(row["output_ids"]):
                    raise ValueError("response token count mismatch")
            if a["output_ids"] != b["output_ids"] or a["finish_reason"] != b["finish_reason"]:
                output_differences.append(dict(rank=rank, index=i))
            if a["execution_profile"] != b["execution_profile"]:
                profile_differences.append(dict(rank=rank, index=i))
            counts[str(rank)] += 1
    result.update(complete=True, status="COMPLETE_COMPARISON",
                  output_ids_equal=not output_differences,
                  execution_profiles_equal=not profile_differences,
                  output_differences=output_differences, profile_differences=profile_differences,
                  comparisons_per_rank=counts)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    a = p.parse_args()
    deadline = time.monotonic() + 3 * 3600
    while not (a.control / "summary.json").exists():
        if not a.watch or time.monotonic() > deadline:
            raise SystemExit("control archive has no terminal summary")
        time.sleep(30)
    result = compare(a.reference, a.control)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result), flush=True)
