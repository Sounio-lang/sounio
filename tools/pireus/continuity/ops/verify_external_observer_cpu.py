#!/usr/bin/env python3
"""Offline reproduction of CPU-control11973; no process or GPU replay."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]/"validation/external-observer-cpu-20260909"
PIN="1ffbfed3a6855d0bb523f21b49ef700d221a5928e4e3ef9eea7ba988da06f4df"
def digest(raw):return hashlib.sha256(raw).hexdigest()

def verify(root=ROOT):
    raw=(root/"qualification.json").read_bytes()
    if digest(raw)!=PIN:raise ValueError("qualification identity mismatch")
    q=json.loads(raw)
    for name,sha in q["files_sha256"].items():
        path=root/name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path.read_bytes())!=sha:
            raise ValueError("artifact custody mismatch: "+name)
    m=json.loads((root/"manifest.json").read_bytes())
    rows=[r.split("|") for r in (root/"accounting.txt").read_text().splitlines()]
    main=[r for r in rows if r[0]=="11973"]
    if len(main)!=1 or main[0][2:4]!=["COMPLETED","0:0"]:
        raise ValueError("Slurm completion mismatch")
    if set(main[0][4].split(","))!={"gpuorangefs-multi-spark-3c59","gpuorangefs-multi-spark-8e54"}:
        raise ValueError("Slurm paired nodes mismatch")
    outcomes=[]
    for rank,worker in enumerate(m["workers"]):
        directory=root/f"rank-{rank}"
        control=json.loads((directory/"control.json").read_bytes())
        binding=json.loads((directory/"binding.json").read_bytes())
        phases=json.loads((directory/"phases.json").read_bytes())
        rows=[json.loads(r) for r in (directory/"journal.jsonl").read_bytes().splitlines()]
        if [r["phase"] for r in phases]!=["baseline","allocated","released","exiting"]:
            raise ValueError("CPU phase sequence mismatch")
        for phase in phases:
            if phase["pid"]!=binding["expected"]["pid"] or phase["starttime_ticks"]!=binding["expected"]["starttime_ticks"]:
                raise ValueError("target phase identity mismatch")
        for name in ("pod-before.json","pod-after.json"):
            if json.loads((directory/name).read_bytes())["metadata"]["uid"]!=worker["uid"]:
                raise ValueError("worker custody mismatch")
        expected=binding["expected"];observed=binding["observed"]
        if any(expected[key]!=observed[key] for key in ("job","rank","pid","starttime_ticks","worker_uid","boot_id")):
            raise ValueError("binding mismatch")
        if expected["job"]!="11973" or expected["rank"]!=str(rank) or expected["worker_uid"]!=worker["uid"] or expected["boot_id"]!=worker["boot_id"]:
            raise ValueError("job/rank/worker identity mismatch")
        if "job_11973" not in Path(observed["cgroup"]["membership"]).parts:
            raise ValueError("actual Slurm job cgroup missing")
        for name,sha in m["files_sha256"].items():
            if digest((directory/name).read_bytes())!=sha or digest((root/"source"/name).read_bytes())!=sha:
                raise ValueError("executed source mismatch")
        binding_sha=digest(json.dumps(binding,sort_keys=True).encode())
        if rows[0]["stage"]!="OBSERVER_START" or rows[-1]["stage"]!="TARGET_INVALIDATED" or rows[-1]["metrics"] is not None:
            raise ValueError("observer invalidation sequence mismatch")
        if control["observer_exit_code"]!=3 or control["observer_pid"]==control["target_pid"]:
            raise ValueError("external lifecycle mismatch")
        for row in rows:
            if (row["binding_sha256"]!=binding_sha or row["job"]!="11973" or row["rank"]!=str(rank)
                or row["target_pid"]!=expected["pid"] or row["observer_pid"]!=control["observer_pid"]):
                raise ValueError("journal attribution mismatch")
        if not phases[-1]["monotonic_ns"]<=rows[-1]["monotonic_ns"]<=phases[-1]["monotonic_ns"]+5_000_000_000:
            raise ValueError("exit detection outside bound")
        samples=[r for r in rows if r["stage"]=="SAMPLE"]
        means={}
        for first,last in zip(phases,phases[1:]):
            selected=[s for s in samples if first["monotonic_ns"]+300_000_000<s["monotonic_ns"]<last["monotonic_ns"]]
            if len(selected)<3:raise ValueError("insufficient settled samples")
            if any(v["error"] is not None for s in selected for v in s["metrics"].values()):
                raise ValueError("unexpected missing CPU-control data")
            means[first["phase"]]=dict(process_pss_bytes=statistics.median(s["metrics"]["process_smaps_rollup"]["value"]["Pss"] for s in selected),
                cgroup_current_bytes=statistics.median(s["metrics"]["cgroup_memory.current"]["value"] for s in selected))
        for key,threshold in (("process_pss_bytes",48*1024**2),("cgroup_current_bytes",32*1024**2)):
            up=means["allocated"][key]-means["baseline"][key]
            down=means["allocated"][key]-means["released"][key]
            if up<threshold or down<threshold or up!=control["deltas"][key]["increase"] or down!=control["deltas"][key]["decrease"]:
                raise ValueError("allocation/release claim not reproduced")
        outcomes.append(dict(rank=rank,samples=len(samples),phase_values=means,
            cgroup=observed["cgroup"]["membership"],
            exit_detection_ns=rows[-1]["monotonic_ns"]-phases[-1]["monotonic_ns"]))
    return dict(job=11973,paired_cpu_control_reproduced=True,ranks=outcomes,
        inference_executed=False,cuda_accounting_qualified=False,loaded_model_overhead_qualified=False)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
