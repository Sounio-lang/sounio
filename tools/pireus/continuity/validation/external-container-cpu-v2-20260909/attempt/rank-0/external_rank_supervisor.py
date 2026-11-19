#!/usr/bin/env python3
"""Supervise observer attachment around an unchanged memory_guard command."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from external_memory_observer import bind, sha
from publish_observer_target import exclusive_json

def complete_rows(path):
    try:raw=Path(path).read_bytes()
    except FileNotFoundError:return []
    return [json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b"\n")]

def supervise(command, output, worker_uid, boot_id, entry_sha, startup_seconds=30, run_seconds=3500):
    if not 0<startup_seconds<=60 or not 0<run_seconds<=3500:
        raise ValueError("supervision bounds exceeded")
    job=os.environ["SLURM_JOB_ID"];rank=os.environ["PIREUS_RANK"]
    if not job.isdigit() or rank not in ("0","1"):raise ValueError("Slurm identity required")
    output=Path(output);output.mkdir(exist_ok=False)
    nonce=secrets.token_hex(32);handoff=output/"target.json";ack=output/"attached.json"
    env=os.environ.copy()
    env.update(PIREUS_EXTERNAL_HANDOFF=str(handoff),PIREUS_EXTERNAL_ACK=str(ack),
               PIREUS_EXTERNAL_NONCE=nonce)
    child=None;observer=None;attached=False;reason=None;previous={};binding=None
    began=time.monotonic()
    def interrupt(signum,frame):raise InterruptedError("supervisor interrupted")
    def wait_for_file():
        deadline=time.monotonic()+startup_seconds
        while time.monotonic()<deadline:
            if handoff.exists():return json.loads(handoff.read_bytes())
            if child.poll() is not None:return None
            time.sleep(0.05)
        raise TimeoutError("target handoff timeout")
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            previous[sig]=signal.signal(sig,interrupt)
        child=subprocess.Popen(command,env=env)
        target=wait_for_file()
        if target is not None:
            if (target.get("schema")!="pireus-observer-target-handoff-v1"
                or target.get("job")!=job or target.get("rank")!=rank
                or target.get("nonce")!=nonce or target.get("entry_sha256")!=entry_sha
                or target.get("boot_id")!=boot_id):
                raise ValueError("target handoff identity mismatch")
            expected={k:target[k] for k in ("job","rank","pid","starttime_ticks","boot_id")}
            expected["worker_uid"]=worker_uid
            binding=bind(expected)
            exclusive_json(output/"binding.json",binding)
            with (output/"observer.log").open("x") as log:
                observer=subprocess.Popen([sys.executable,str(Path(__file__).with_name("external_memory_observer.py")),
                    "observe","--binding",str(output/"binding.json"),"--output",str(output/"journal.jsonl"),
                    "--interval","0.2","--seconds",str(run_seconds)],stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+startup_seconds
            while time.monotonic()<deadline:
                samples=[r for r in complete_rows(output/"journal.jsonl") if r["stage"]=="SAMPLE"]
                if samples:
                    if any(r["target_pid"]!=target["pid"] or r["observer_pid"]!=observer.pid
                           or r["job"]!=job or r["rank"]!=rank or not r["identity_valid"] for r in samples):
                        raise ValueError("first observer sample mismatch")
                    acknowledgement={k:target[k] for k in ("nonce","job","rank","pid","starttime_ticks")}
                    acknowledgement.update(schema="pireus-observer-attachment-ack-v1",
                        handoff_sha256=sha(handoff.read_bytes()),observer_pid=observer.pid,
                        first_sample_valid=True,binding_sha256=sha((output/"binding.json").read_bytes()))
                    exclusive_json(ack,acknowledgement);attached=True;break
                if observer.poll() is not None:raise RuntimeError("observer failed before acknowledgement")
                if child.poll() is not None:raise RuntimeError("guardian exited before acknowledgement")
                time.sleep(0.05)
            if not attached:raise TimeoutError("first observer sample timeout")
        while child.poll() is None:
            if time.monotonic()-began>run_seconds:raise TimeoutError("external observation duration exceeded")
            if observer is not None and observer.poll() is not None:
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("observer ended while guardian remained active")
            time.sleep(0.05)
        if not attached:reason="guardian exited before target attachment"
        if observer is not None:
            try:observer.wait(timeout=3)
            except subprocess.TimeoutExpired:raise RuntimeError("observer did not terminate after guardian")
            rows=complete_rows(output/"journal.jsonl")
            if observer.returncode!=3 or not rows or rows[-1]["stage"]!="TARGET_INVALIDATED":
                raise RuntimeError("observer target termination was not recorded")
    except (OSError,ValueError,RuntimeError,TimeoutError,InterruptedError) as exc:
        reason=type(exc).__name__+": "+str(exc)
    finally:
        # Disable repeated interruption while the known guardian performs cleanup.
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                reason=(reason or "")+"; guardian cleanup exceeded20s"
                # Slurm owns final job containment; do not bypass the guardian with a target kill.
        if observer is not None and observer.poll() is None:
            observer.terminate()
            try:observer.wait(timeout=3)
            except subprocess.TimeoutExpired:observer.kill();observer.wait()
        guard_rc=child.poll() if child is not None else None
        result=dict(schema="pireus-external-rank-supervision-v1",job=job,rank=rank,
            guardian_returncode=guard_rc,observer_returncode=observer.poll() if observer else None,
            observer_attached=attached,error=reason,nonce=nonce,entry_sha256=entry_sha,
            integration_complete=attached and reason is None and guard_rc==0,
            inference_qualified=False,files_sha256={p.name:sha(p.read_bytes()) for p in output.iterdir() if p.is_file()})
        exclusive_json(output/"result.json",result)
        for sig,handler in previous.items():signal.signal(sig,handler)
    # Preserve a guardian memory-stop/nonzero status; never turn it into a pass.
    if guard_rc is not None and guard_rc!=0:return guard_rc if guard_rc>0 else 128-guard_rc
    return 0 if result["integration_complete"] else 76

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker-uid",required=True);p.add_argument("--boot-id",required=True)
    p.add_argument("--entry-sha",required=True);p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==["--"] else a.command
    if not command:p.error("guardian command required")
    raise SystemExit(supervise(command,a.output,a.worker_uid,a.boot_id,a.entry_sha))
