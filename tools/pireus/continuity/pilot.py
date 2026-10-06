#!/usr/bin/env python3
"""Frozen 3 x 3 x 32 pilot transport; native Sounio owns admission and gain."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from cycle import HERE, atomic, digest, encoded, prepare, report, verify
from benchmark_pair import remote

ROOT = HERE.parents[2]
CONDITIONS = ("deterministic", "inkling-no-ontology", "inkling-ontology")
REQUIRED_CHECKS = ("CI Decision", "transport-and-archive",
                   "Archived script custody (no runtime replay)")


def cells():
    result = []
    for round_id in range(3):
        for offset in range(3):
            condition = CONDITIONS[(round_id + offset) % 3]
            result.append(dict(id=f"round-{round_id}-{condition}",
                               round=round_id, condition=condition, budget=32))
    return result


def call(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)],
                   cwd=ROOT, check=True)


def prepare_pilot(args):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "tools/pireus/continuity"], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("commit continuity source before freezing the pilot")
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "--", "tools/pireus/continuity"], cwd=ROOT, text=True).splitlines())
    dependencies = [*HERE.glob("*.py"), *HERE.glob("*.sio")]
    for suffix in ("*.py", "*.sio", "*.sh", "*.json"):
        dependencies.extend((HERE / "runtime").glob(suffix))
    if any(p.relative_to(ROOT).as_posix() not in tracked for p in dependencies):
        raise ValueError("pilot source dependency is not committed, including ignored files")
    engines = {}
    for name in ("engine", "context_engine", "fixture_engine", "parity_engine", "gain_engine"):
        path = getattr(args, name).resolve()
        engines[name] = dict(path=str(path), sha256=digest(path.read_bytes()))
    spec = dict(schema=1, source_commit=head, cells=cells(), engines=engines,
                evidence=[dict(path=str(p.resolve()), sha256=digest(p.read_bytes())) for p in args.evidence],
                deduplication_scope="within-condition-and-round",
                proposal_budget_per_cell=32, total_proposals=288,
                native_host_floor_gib=32, early_stop_gib=33, cache_capacity_tokens=6144,
                context_length=16384, http_serving=False, general_16k_inference_accepted=False,
                context_provenance="declared research ontology, not newly observed hardware facts",
                promotion_threshold_percent=5, benchmark_blocks_per_node=30,
                formal_v13_v14="OPEN", claim_ready=False)
    args.run.mkdir(parents=True, exist_ok=True)
    atomic(args.run / "pilot-manifest.json", encoded(spec))
    for cell in spec["cells"]:
        target = args.run / cell["id"]
        if (target / "manifest.json").exists():
            old = verify(target)
            if any(old[k] != cell[k] for k in ("condition", "round", "budget")):
                raise ValueError("existing pilot cell differs")
            if old.get("deduplicate_material") is not True:
                raise ValueError("pilot requires frozen material deduplication")
            continue
        prepare(SimpleNamespace(run=target, context_engine=args.context_engine, context=None,
                                evidence=args.evidence, condition=cell["condition"],
                                budget=32, round=cell["round"], deduplicate_material=True,
                                transport=("openai-chat" if cell["condition"] == "deterministic"
                                           else "sglang-offline-token-ids"),
                                engine_sha256=engines["engine"]["sha256"]))
    return spec


def validate_checks(spec):
    head = spec["source_commit"]
    remote_head = subprocess.check_output(
        ["gh", "api", "repos/Sounio-lang/sounio/pulls/2439", "--jq", ".head.sha"],
        cwd=ROOT, text=True).strip()
    if remote_head != head:
        raise ValueError("pilot source is not the current PR head")
    raw = subprocess.check_output(
        ["gh", "api", "--paginate", f"repos/Sounio-lang/sounio/commits/{head}/check-runs",
         "--jq", '.check_runs[] | [.name,.status,.conclusion,.head_sha,.started_at,.details_url] | @json'],
        cwd=ROOT, text=True)
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    selected = {}
    for name in REQUIRED_CHECKS:
        matching = sorted((row for row in rows if row[0] == name and row[3] == head),
                          key=lambda row: row[4] or "")
        if not matching or matching[-1][1:3] != ["completed", "success"]:
            raise ValueError("required current integration check is not green: " + name)
        selected[name] = matching[-1]
    return dict(source_commit=head, required_checks=selected)


def paired_completion(log, stage):
    values = []
    launch = []
    for line in log.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        if value.get("stage") == stage:
            values.append(value)
        if "mode" in value and "command" in value and "nodes" in value:
            launch.append(value)
    if (len(values) != 2 or {str(v.get("rank")) for v in values} != {"0", "1"}
            or len({str(v.get("job")) for v in values}) != 1 or len(launch) != 1):
        raise ValueError("launch lacks a unique paired completion and worker identity")
    job = str(values[0]["job"])
    if not job.isdecimal():
        raise ValueError("invalid completed Slurm job identity")
    return job, launch[0]["nodes"]


def launch_cell_stage(root, name, mode, bundle, minutes):
    folder = root / "pilot-stages" / name
    folder.mkdir(parents=True, exist_ok=True)
    intent = folder / "intent.json"
    completed = folder / "completed.json"
    log = folder / "launch.log"
    command = [sys.executable, str(HERE / "runtime/launch_pair.py"), mode,
               "--minutes", str(minutes), "--input-bundle", str(bundle)]
    identity = dict(command=command, bundle_sha256=digest(bundle.read_bytes()))
    if completed.exists():
        if json.loads(intent.read_text()) != identity:
            raise ValueError("completed launch intent differs")
        result = json.loads(completed.read_text())
        if result["log_sha256"] != digest(log.read_bytes()):
            raise ValueError("completed launch log changed")
        return result
    if not intent.exists():
        atomic(intent, encoded(identity))
        with log.open("xb") as out:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("pilot launch failed; immutable attempt retained: " + str(folder))
    elif json.loads(intent.read_text()) != identity:
        raise ValueError("interrupted launch intent differs")
    # An interrupted coordinator may adopt a completed job, but never submits
    # the same frozen launch twice. Missing/partial logs or active jobs refuse.
    expected = "OFFLINE_CYCLE_COMPLETE" if mode == "offline-generate" else "TOKENIZER_TRANSPORT_PASS"
    job, nodes = paired_completion(log.read_text(), expected)
    raw_state = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    state = dict(x.split("=", 1) for x in raw_state.split() if "=" in x)
    if state.get("JobState") != "COMPLETED" or state.get("ExitCode") != "0:0":
        raise ValueError("paired runtime job has not completed successfully")
    result = dict(job=job, nodes=nodes, log_sha256=digest(log.read_bytes()),
                  slurm_state=raw_state, input_sha256=identity["bundle_sha256"])
    atomic(completed, encoded(result))
    return result


def collect(root, completed, offline=False):
    job = completed["job"]
    paths = []
    for rank, node in enumerate(completed["nodes"]):
        actual = json.loads(subprocess.check_output(
            ["kubectl", "-n", "slurm-pilot", "get", "pod", node["pod"], "-o", "json"]))
        if actual["metadata"]["uid"] != node["uid"]:
            raise ValueError("worker identity changed before receipt collection")
        if offline:
            target = root / "worker"
            target.mkdir(exist_ok=True)
            for suffix in ["complete"] + [f"{i:03d}" for i in range(32)]:
                raw = remote(node["pod"], "cat", f"/scratch/pireus/receipts/offline-{job}-{rank}-{suffix}.json")
                atomic(target / f"rank-{rank}-{suffix}.json", raw)
        else:
            path = root / f"tokenizer-{job}-{rank}.json"
            atomic(path, remote(node["pod"], "cat", f"/scratch/pireus/receipts/tokenizer-{job}-{rank}.json"))
            paths.append(path)
    return paths


def run_pilot(root):
    if not os.environ.get("TMUX"):
        raise ValueError("run the full pilot in remote tmux")
    spec = json.loads((root / "pilot-manifest.json").read_text())
    if spec["cells"] != cells() or spec["total_proposals"] != 288:
        raise ValueError("pilot design changed")
    for engine in spec["engines"].values():
        if digest(Path(engine["path"]).read_bytes()) != engine["sha256"]:
            raise ValueError("frozen native engine changed")
    for cell in spec["cells"]:
        verify(root / cell["id"])
    if not (root / "integration-accepted.json").exists():
        atomic(root / "integration-accepted.json", encoded(validate_checks(spec)))
    accepted = json.loads((root / "integration-accepted.json").read_text())
    if accepted["source_commit"] != spec["source_commit"] or set(accepted["required_checks"]) != set(REQUIRED_CHECKS):
        raise ValueError("integration acceptance belongs to another pilot source")
    for name, row in accepted["required_checks"].items():
        if row[0] != name or row[1:4] != ["completed", "success", spec["source_commit"]]:
            raise ValueError("recorded integration acceptance is not successful")
    engines = {key: value["path"] for key, value in spec["engines"].items()}
    for cell in spec["cells"]:
        target = root / cell["id"]
        if (target / "pilot-cell-complete.json").exists():
            previous = json.loads((target / "pilot-cell-complete.json").read_text())
            if previous != report(target, verify(target)) or previous["generated"] != 32 or previous["validated"] != 32:
                raise ValueError("completed pilot cell changed")
            continue
        print(json.dumps(dict(stage="PILOT_CELL_START", cell=cell)), flush=True)
        if cell["condition"] == "deterministic":
            call("cycle.py", "generate", "--run", target)
        else:
            call("tokenized_cycle.py", "pack-encode", "--run", target)
            encoded_job = launch_cell_stage(target, "encode", "tokenize", target / "encode-bundle.json", 15)
            receipts = collect(target, encoded_job)
            call("tokenized_cycle.py", "accept-encode", "--run", target, "--receipts", *receipts)
            call("tokenized_cycle.py", "pack-offline", "--run", target)
            generated_job = launch_cell_stage(target, "generate", "offline-generate",
                                             target / "offline-bundle.json", 120)
            collect(target, generated_job, offline=True)
            call("tokenized_cycle.py", "accept-offline", "--run", target, "--worker-dir", target / "worker")
            call("tokenized_cycle.py", "pack-decode", "--run", target)
            decoded_job = launch_cell_stage(target, "decode", "tokenize", target / "decode-bundle.json", 15)
            receipts = collect(target, decoded_job)
            call("tokenized_cycle.py", "finalize", "--run", target, "--receipts", *receipts)
        call("cycle.py", "validate", "--run", target, "--engine", engines["engine"])
        call("cycle.py", "materialize", "--run", target, "--engine", engines["engine"])
        state = report(target, verify(target))
        if state["generated"] != 32 or state["validated"] != 32:
            raise ValueError("pilot cell has incomplete proposals or admission")
        if state["admitted"]:
            call("cycle.py", "benchmark", "--run", target, "--engine", engines["engine"],
                 "--fixture-engine", engines["fixture_engine"], "--parity-engine", engines["parity_engine"],
                 "--gain-engine", engines["gain_engine"])
        state = report(target, verify(target))
        if state["hardware_benchmarked"] != state["unique_plans"]:
            raise ValueError("pilot cell lacks hardware decisions for every unique admitted plan")
        atomic(target / "pilot-cell-complete.json", encoded(state))
        print(json.dumps(dict(stage="PILOT_CELL_COMPLETE", cell=cell["id"], report=state)), flush=True)
    result = summarize(root)
    if not result["complete"]:
        raise ValueError("pilot design incomplete")
    atomic(root / "pilot-report.json", encoded(result))
    print(json.dumps(dict(stage="PILOT_COMPLETE", report=result)), flush=True)


def summarize(root):
    spec = json.loads((root / "pilot-manifest.json").read_text())
    reports = {}
    for cell in spec["cells"]:
        target = root / cell["id"]
        value = report(target, verify(target))
        marker = target / "pilot-cell-complete.json"
        if marker.exists():
            if (json.loads(marker.read_text()) != value or value["generated"] != 32
                    or value["validated"] != 32
                    or value["hardware_benchmarked"] != value["unique_plans"]):
                raise ValueError("completed pilot cell lacks matching full evidence")
        value["complete"] = marker.exists()
        reports[cell["id"]] = value
    return dict(schema=1, source_commit=spec["source_commit"], cells=reports,
                complete=len(reports) == 9 and all(v["complete"] for v in reports.values()),
                generated=sum(v["generated"] for v in reports.values()),
                admitted=sum(v["admitted"] for v in reports.values()),
                unique_plan_occurrences=sum(v["unique_plans"] for v in reports.values()),
                hardware_benchmarked=sum(v["hardware_benchmarked"] for v in reports.values()),
                gain_eligible=sum(v["gain_eligible"] for v in reports.values()),
                deduplication_scope="within-condition-and-round",
                native_host_floor_gib=32, early_stop_gib=33, cache_capacity_tokens=6144,
                formal_v13_v14="OPEN", claim_ready=False, http_serving=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "report"])
    parser.add_argument("--run", type=Path, required=True)
    for name in ("engine", "context-engine", "fixture-engine", "parity-engine", "gain-engine"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--evidence", nargs="+", type=Path)
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.command == "prepare":
        if not args.evidence or any(getattr(args, k) is None for k in
                ("engine", "context_engine", "fixture_engine", "parity_engine", "gain_engine")):
            parser.error("prepare requires five native engines and frozen evidence")
        print(json.dumps(prepare_pilot(args), indent=2))
    elif args.command == "report":
        print(json.dumps(summarize(args.run), indent=2))
    else:
        with (args.run / "pilot.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_pilot(args.run)


if __name__ == "__main__":
    main()
