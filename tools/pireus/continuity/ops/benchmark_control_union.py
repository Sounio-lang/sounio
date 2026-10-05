#!/usr/bin/env python3
"""Benchmark the control union without inserting any result into the pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cycle import prepare, atomic, encoded, digest, event, verify


def stage(source, pilot, output):
    summary = json.loads((source / "summary.json").read_text())
    for name, sha in summary["file_hashes"].items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()) or digest(path.read_bytes()) != sha:
            raise ValueError("native control archive changed")
    engines = json.loads((pilot / "pilot-manifest.json").read_text())["engines"]
    for name in ("engine", "fixture_engine", "parity_engine", "gain_engine"):
        if digest(Path(engines[name]["path"]).read_bytes()) != engines[name]["sha256"]:
            raise ValueError("frozen native executable changed: " + name)
    groups = {}
    members = []
    representatives = []
    for row in summary["results"]:
        if row["decision"] != "ADMIT" or row["ptx_sha256"] is None:
            raise ValueError("expected admitted/materialized control proposals")
        key = (row["plan_id"], row["ptx_sha256"])
        if key not in groups:
            groups[key] = f"{len(groups):03d}"
            representatives.append(row)
        members.append(dict(source_job=row["source_job"], request_index=row["index"],
                            representative=groups[key], plan_id=row["plan_id"],
                            ptx_sha256=row["ptx_sha256"], proposal_sha256=row["proposal_sha256"]))
    if len(groups) != 18 or len(members) != 64:
        raise ValueError("control union no longer has18 materials and64 occurrences")
    prepare(SimpleNamespace(run=output, context_engine=None, context=source / "context.json",
            evidence=[source / "summary.json", source / "qualification.json"],
            condition="control-union-analysis", budget=18, round=0,
            transport="preserved-native-control-materials",
            engine_sha256=engines["engine"]["sha256"], deduplicate_material=False))
    mapping = dict(schema="pireus-control-union-membership-v1",
                   deduplication="native plan ID and exact PTX bytes across these two control cohorts only",
                   original_occurrences=64, unique_materials=18, members=members,
                   pilot_acceptance=False)
    atomic(output / "control-membership.json", encoded(mapping))
    event(output, "control-membership", output / "control-membership.json")
    for index, row in enumerate(representatives):
        src = source / str(row["source_job"]) / f"{row['index']:03d}"
        dst = output / f"{index:03d}"
        for suffix in (".proposal.json", ".receipt.json", ".ptx"):
            raw = src.with_suffix(suffix).read_bytes()
            atomic(dst.with_suffix(suffix), raw)
            event(output, "import-preserved-control" + suffix, dst.with_suffix(suffix))
    provenance = dict(schema="pireus-control-union-benchmark-v1",
                      source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                          cwd=HERE, text=True).strip(),
                      builder_sha256=digest(Path(__file__).read_bytes()),
                      engine_identities=engines, source_jobs=[11957, 11958],
                      native_analysis_sha256=digest((source / "summary.json").read_bytes()),
                      original_pilot_progress="32/288", failed11956_preserved=True,
                      pilot_acceptance=False, automatic_retry=False)
    atomic(output / "control-analysis.json", encoded(provenance))
    event(output, "control-analysis", output / "control-analysis.json")
    verify(output)


def run(output):
    if not os.environ.get("TMUX"):
        raise ValueError("use remote tmux")
    verify(output)
    p = json.loads((output / "control-analysis.json").read_text())
    cmd = [sys.executable, str(HERE / "benchmark_pair.py"), "--run", str(output)]
    for flag, key in (("engine", "engine"), ("fixture-engine", "fixture_engine"),
                      ("parity-engine", "parity_engine"), ("gain-engine", "gain_engine")):
        engine = p["engine_identities"][key]
        if digest(Path(engine["path"]).read_bytes()) != engine["sha256"]:
            raise ValueError("native executable changed")
        cmd += ["--" + flag, engine["path"]]
    # Exclusive intent prevents a second invocation from silently reusing an attempt.
    with (output / "execution-intent.json").open("xb") as f:
        f.write(encoded(dict(command=cmd, pilot_acceptance=False, automatic_retry=False)))
    with (output / "execution.log").open("xb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    atomic(output / "execution-exit.json", encoded(dict(returncode=result.returncode,
        log_sha256=digest((output / "execution.log").read_bytes()), pilot_acceptance=False)))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["stage", "run"])
    p.add_argument("--source", type=Path)
    p.add_argument("--pilot", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.output.is_absolute():
        p.error("output must be absolute")
    if a.action == "stage":
        if a.source is None or a.pilot is None:
            p.error("stage requires source and pilot")
        stage(a.source, a.pilot, a.output)
    else:
        run(a.output)
