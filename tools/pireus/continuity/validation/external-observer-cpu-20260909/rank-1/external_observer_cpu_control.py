#!/usr/bin/env python3
"""Slurm CPU-only 64MiB mmap control; observer and target are separate processes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from external_memory_observer import bind, sha

TARGET = """
import json,mmap,os,sys,time
from pathlib import Path
def event(phase):
 print(json.dumps(dict(phase=phase,pid=os.getpid(),monotonic_ns=time.monotonic_ns(),
  starttime_ticks=int(Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]))),flush=True)
event('baseline')
allocation=None
for line in sys.stdin:
 command=line.strip()
 if command=='allocate':
  allocation=mmap.mmap(-1,64*1024**2,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
  for offset in range(0,len(allocation),4096):allocation[offset]=1
  event('allocated')
 elif command=='release':
  allocation.close();allocation=None;event('released')
 elif command=='exit':
  event('exiting');break
 else:raise ValueError('unknown control command')
"""

def execute(output, worker_uid, boot_id):
    job=os.environ["SLURM_JOB_ID"];rank=os.environ["PIREUS_RANK"]
    if not job.isdigit() or rank not in ("0","1"):raise ValueError("invalid Slurm identity")
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent
    child=subprocess.Popen([sys.executable,"-u","-c",TARGET],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    observer=None
    try:
        def receive():
            row=json.loads(child.stdout.readline())
            if row["pid"]!=child.pid:raise ValueError("control PID mismatch")
            return row
        phases=[receive()]
        expected=dict(job=job,rank=rank,pid=child.pid,starttime_ticks=phases[0]["starttime_ticks"],
            worker_uid=worker_uid,boot_id=boot_id)
        binding=bind(expected)
        (output/"binding.json").write_text(json.dumps(binding,indent=2)+"\n")
        with (output/"observer.log").open("x") as log:
            observer=subprocess.Popen([sys.executable,str(source/"external_memory_observer.py"),"observe",
                "--binding",str(output/"binding.json"),"--output",str(output/"journal.jsonl"),
                "--interval","0.2","--seconds","30"],stdout=log,stderr=subprocess.STDOUT)
        time.sleep(2)
        child.stdin.write("allocate\n");child.stdin.flush();phases.append(receive())
        time.sleep(3)
        child.stdin.write("release\n");child.stdin.flush();phases.append(receive())
        time.sleep(2)
        child.stdin.write("exit\n");child.stdin.flush();phases.append(receive())
        if child.wait(timeout=3)!=0:raise ValueError("CPU target failed")
        rc=observer.wait(timeout=5)
        (output/"phases.json").write_text(json.dumps(phases,indent=2)+"\n")
        rows=[json.loads(x) for x in (output/"journal.jsonl").read_text().splitlines()]
        if rc!=3 or rows[-1]["stage"]!="TARGET_INVALIDATED":
            raise ValueError("observer did not detect process exit")
        samples=[r for r in rows if r["stage"]=="SAMPLE"]
        if not samples or any(r["observer_pid"]==r["target_pid"] or r["target_pid"]!=child.pid
                              or r["job"]!=job or r["rank"]!=rank for r in rows):
            raise ValueError("external process identity mismatch")
        windows={}
        for left,right in zip(phases,phases[1:]):
            selected=[r for r in samples if left["monotonic_ns"]+300_000_000<r["monotonic_ns"]<right["monotonic_ns"]]
            if len(selected)<3:raise ValueError("insufficient phase samples")
            for r in selected:
                if any(v["error"] is not None for v in r["metrics"].values()):
                    raise ValueError("missing live control metric")
            windows[left["phase"]]=dict(samples=len(selected),
                process_pss_bytes=statistics.median(r["metrics"]["process_smaps_rollup"]["value"]["Pss"] for r in selected),
                cgroup_current_bytes=statistics.median(r["metrics"]["cgroup_memory.current"]["value"] for r in selected))
        changes={}
        for key,threshold in (("process_pss_bytes",48*1024**2),("cgroup_current_bytes",32*1024**2)):
            increase=windows["allocated"][key]-windows["baseline"][key]
            decrease=windows["allocated"][key]-windows["released"][key]
            if increase<threshold or decrease<threshold:
                raise ValueError("allocation/release not visible: "+key)
            changes[key]=dict(increase=increase,decrease=decrease,threshold=threshold)
        result=dict(schema="pireus-external-observer-cpu-control-v1",job=job,rank=rank,
            worker_uid=worker_uid,boot_id=boot_id,target_pid=child.pid,observer_pid=observer.pid,
            target_starttime_ticks=expected["starttime_ticks"],target_cgroup=binding["observed"]["cgroup"],
            cpu_allocation_bytes=64*1024**2,allocation_release_visible=True,process_exit_detected=True,
            observer_exit_code=rc,samples=len(samples),phase_summary=windows,deltas=changes,
            max_sample_gap_ns=max(r["sample_gap_ns"] or 0 for r in samples),
            max_sample_duration_ns=max(r["duration_ns"] for r in samples),
            inference_executed=False,cuda_accounting_qualified=False,loaded_model_overhead_qualified=False,
            helper_sha256=sha((source/"external_memory_observer.py").read_bytes()),
            control_sha256=sha(Path(__file__).read_bytes()),
            files_sha256={p.name:sha(p.read_bytes()) for p in output.iterdir() if p.is_file()})
        (output/"control.json").write_text(json.dumps(result,indent=2)+"\n")
        return result
    finally:
        # Only these explicitly owned CPU-control children can be terminated.
        for process in (child,observer):
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker-uid",required=True);p.add_argument("--boot-id",required=True)
    a=p.parse_args()
    print(json.dumps(execute(a.output,a.worker_uid,a.boot_id)),flush=True)
