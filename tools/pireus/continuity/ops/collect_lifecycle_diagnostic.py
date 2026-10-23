#!/usr/bin/env python3
"""Copy terminal diagnostic evidence; missing files remain explicit, no acceptance."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
from feedback_smoke import digest
from freeze_lifecycle_diagnostic import verify

NODES = {"gpuorangefs-multi-spark-3c59", "gpuorangefs-multi-spark-8e54"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
KUBE = ["kubectl", "-n", "slurm-pilot"]

def collect(frozen, stage, output, job, run=subprocess.check_output):
    if not str(job).isdigit():
        raise ValueError("numeric Slurm job required")
    spec = verify(frozen)
    raw = (stage/"launch.log").read_bytes()
    exit_raw = (stage/"exit-code").read_bytes()
    int(exit_raw.strip())
    logs = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict): logs.append(row)
        except (ValueError, UnicodeDecodeError):
            pass
    launches = [r for r in logs if r.get("mode") == "offline-generate"]
    if len(launches) != 1 or len(launches[0]["nodes"]) != 2:
        raise ValueError("unique two-worker launch required")
    if not any(r.get("job") == str(job) for r in logs):
        raise ValueError("job absent from runtime log")
    accounting = run(KUBE+["exec", "slurm-pilot-controller-0", "-c", "slurmctld",
        "--", "sacct", "-j", str(job), "--noheader", "--parsable2",
        "--format=JobID,JobName,State,ExitCode,NodeList,Start,End"])
    rows = [line.split("|") for line in accounting.decode().splitlines()]
    main = [r for r in rows if r[0] == str(job)]
    if (len(main) != 1 or len(main[0]) < 7
        or main[0][2].split()[0] not in TERMINAL
        or set(main[0][4].split(",")) != NODES
        or any(x in ("", "Unknown") for x in main[0][5:7])
        or main[0][1] != "pireus-inkling-offline-generate"):
        raise ValueError("unique terminal paired accounting required")
    output.mkdir(exist_ok=False)
    (output/"launch.log").write_bytes(raw)
    (output/"exit-code").write_bytes(exit_raw)
    (output/"accounting.txt").write_bytes(accounting)
    worker = output/"worker-receipts"
    worker.mkdir()
    missing = {}
    pods = launches[0]["nodes"]
    for rank, pod in enumerate(pods):
        def identity():
            live_raw = run(KUBE+["get", "pod", pod["pod"], "-o", "json"])
            live = json.loads(live_raw)
            if live["metadata"]["uid"] != pod["uid"]:
                raise ValueError("worker UID changed")
            return live_raw
        (output/f"worker-{rank}-before.json").write_bytes(identity())
        paths = {f"rank-{rank}-complete.json": f"/scratch/pireus/receipts/offline-{job}-{rank}-complete.json",
                 f"lifecycle-{job}-{rank}.jsonl": f"/scratch/pireus/receipts/lifecycle-{job}-{rank}.jsonl"}
        paths.update({f"rank-{rank}-{i:03d}.json": f"/scratch/pireus/receipts/offline-{job}-{rank}-{i:03d}.json" for i in range(8)})
        code = """import json,base64
from pathlib import Path
result={}
for name,path in PATHS.items():
 try: result[name]={"base64":base64.b64encode(Path(path).read_bytes()).decode()}
 except FileNotFoundError: result[name]={"missing":True}
print(json.dumps(result))
""".replace("PATHS", repr(paths))
        result = json.loads(run(KUBE+["exec", pod["pod"], "-c", "slurmd", "--", "python3", "-c", code]))
        if set(result) != set(paths): raise ValueError("incomplete collection inventory")
        for name, item in result.items():
            if item == {"missing": True}:
                missing[name] = paths[name]
            else:
                (worker/name).write_bytes(base64.b64decode(item["base64"], validate=True))
        (output/f"worker-{rank}-after.json").write_bytes(identity())
    if (stage/"launch.log").read_bytes() != raw or (stage/"exit-code").read_bytes() != exit_raw:
        raise ValueError("launch evidence changed during collection")
    receipt = dict(schema="pireus-lifecycle-raw-custody-v1", job=str(job),
        source_commit=spec["source_commit"],
        freeze_sha256=digest((frozen/"execution-freeze.json").read_bytes()),
        accounting_state=main[0][2], missing=missing, hardware_qualified=False,
        diagnostic_complete=False, pilot_acceptance=False,
        files_sha256={str(p.relative_to(output)):digest(p.read_bytes())
                      for p in output.rglob("*") if p.is_file()})
    (output/"collection.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "stage", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--job", required=True)
    a = p.parse_args()
    collect(a.frozen, a.stage, a.output, a.job)
