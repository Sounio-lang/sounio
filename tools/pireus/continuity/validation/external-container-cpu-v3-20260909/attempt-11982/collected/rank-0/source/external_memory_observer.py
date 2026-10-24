#!/usr/bin/env python3
"""External read-only observer. Identity failure stops attribution, never the target."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time

OBSERVER_PROFILE = "external-observer-deadline-v3"

class IdentityError(ValueError):
    pass

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def starttime(raw, pid):
    try:
        front, tail = raw.rsplit(") ", 1)
        if int(front.split(" (",1)[0]) != pid:
            raise IdentityError("stat PID mismatch")
        fields = tail.split()
        if fields[0] in ("Z","X","x"):
            raise IdentityError("target is no longer live")
        return int(fields[19])  # stat field22; tail starts at field3.
    except (IndexError,ValueError) as exc:
        raise IdentityError("invalid or dead target stat") from exc

def unescape(value):
    return re.sub(r"\\([0-7]{3})",lambda m:chr(int(m[1],8)),value)

def resolve_cgroup(membership, mountinfo):
    members=[x[3:] for x in membership.splitlines() if x.startswith("0::")]
    mounts=[]
    for line in mountinfo.splitlines():
        left,sep,right=line.partition(" - ")
        if sep and right.split()[0]=="cgroup2":
            fields=left.split()
            mounts.append((unescape(fields[3]),unescape(fields[4]),fields[0]))
    if len(members)!=1 or len(mounts)!=1:
        raise IdentityError("ambiguous or missing cgroup2 mapping")
    member=PurePosixPath(members[0]); root,mount,mount_id=mounts[0]
    if not member.is_absolute() or ".." in member.parts or members[0].endswith(" (deleted)"):
        raise IdentityError("invalid cgroup membership")
    try:
        relative=member.relative_to(PurePosixPath(root))
    except ValueError as exc:
        raise IdentityError("target cgroup outside visible mount") from exc
    target=Path(mount).joinpath(*relative.parts)
    if not target.resolve().is_relative_to(Path(mount).resolve()):
        raise IdentityError("cgroup path escapes mount")
    return dict(membership=members[0],mount_root=root,mountpoint=mount,
                mount_id=mount_id,path=str(target))

def capture(expected, proc=Path("/proc")):
    try:
        pid=expected["pid"]
        if type(pid) is not int or pid<=0 or not str(expected["job"]).isdigit() or expected["rank"] not in ("0","1"):
            raise IdentityError("invalid job/rank/PID identity")
        boot=(proc/"sys/kernel/random/boot_id").read_text().strip()
        if boot!=expected["boot_id"]: raise IdentityError("boot identity changed")
        own_mounts=(proc/"self/mountinfo").read_text()
        worker_root="/var/lib/kubelet/pods/"+expected["worker_uid"]+"/"
        roots=[unescape(x.split(" - ",1)[0].split()[3]) for x in own_mounts.splitlines() if " - " in x]
        if not any(x.startswith(worker_root) for x in roots):
            raise IdentityError("worker UID absent from observer mount provenance")
        target=proc/str(pid)
        before=starttime((target/"stat").read_text(),pid)
        if before!=expected["starttime_ticks"]: raise IdentityError("PID starttime changed")
        allowed=(b"SLURM_JOB_ID",b"PIREUS_RANK")
        env={}
        for item in (target/"environ").read_bytes().split(b"\0"):
            key,sep,value=item.partition(b"=")
            if sep and key in allowed:
                if key in env: raise IdentityError("duplicate target identity variable")
                env[key]=value.decode("ascii")
        if env!={b"SLURM_JOB_ID":expected["job"],b"PIREUS_RANK":expected["rank"]}:
            raise IdentityError("target job/rank mismatch")
        membership=(target/"cgroup").read_text()
        cg=resolve_cgroup(membership,own_mounts)
        if "job_"+expected["job"] not in PurePosixPath(cg["membership"]).parts:
            raise IdentityError("target is not in the declared Slurm job cgroup")
        st=Path(cg["path"]).stat()
        if not Path(cg["path"]).is_dir(): raise IdentityError("cgroup is not a directory")
        namespaces={key:os.readlink(target/"ns"/key) for key in ("pid","mnt","cgroup")}
        if starttime((target/"stat").read_text(),pid)!=before or (target/"cgroup").read_text()!=membership:
            raise IdentityError("target changed while binding")
        return dict(boot_id=boot,worker_uid=expected["worker_uid"],pid=pid,starttime_ticks=before,
            job=expected["job"],rank=expected["rank"],cgroup=cg,cgroup_device=st.st_dev,
            cgroup_inode=st.st_ino,namespaces=namespaces)
    except (OSError,UnicodeError,KeyError) as exc:
        raise IdentityError("identity unavailable: "+type(exc).__name__) from exc

def bind(expected, proc=Path("/proc")):
    first=capture(expected,proc)
    if capture(expected,proc)!=first: raise IdentityError("unstable binding")
    return dict(schema="pireus-external-observer-binding-v1",expected=expected,
        observed=first,observer_helper_sha256=sha(Path(__file__).read_bytes()))

def metric(path, kind):
    started=time.monotonic_ns()
    value=None; error=None
    try:
        raw=Path(path).read_text()
        if kind=="bytes": value=int(raw.strip())
        elif kind=="bytes-or-max":
            value="max" if raw.strip()=="max" else int(raw.strip())
            if isinstance(value,int) and value<0:raise ValueError("negative cgroup limit")
        elif kind=="kB-fields":
            value={p[0].rstrip(":"):int(p[1])*1024 for line in raw.splitlines()
                   if len(p:=line.split())==3 and p[2]=="kB"}
        else: value=raw
    except (OSError,ValueError) as exc:
        value=None; error=type(exc).__name__
    return dict(value=value,error=error,format=kind,monotonic_ns=started,
                duration_ns=time.monotonic_ns()-started)

def observer_resources(proc=Path("/proc")):
    """Observer process only; RSS is separate from target/cgroup accounting."""
    status=metric(proc/"self/status","kB-fields")
    if status["error"] is None and not {"VmRSS","VmHWM"} <= status["value"].keys():
        status.update(value=None,error="MissingObserverRSSFields")
    return dict(observer_pid=os.getpid(),process_cpu_ns=time.process_time_ns(),
                status=status,scope="observer-process-only")

def next_sample_deadline(started_ns, finished_ns, interval_ns, end_ns):
    """Anchor to actual start; overruns create no catch-up debt."""
    due=started_ns+interval_ns
    return min(end_ns,max(due,finished_ns))

def sample(binding, proc=Path("/proc")):
    started=time.monotonic_ns()
    if binding.get("schema")!="pireus-external-observer-binding-v1" or binding.get("observer_helper_sha256")!=sha(Path(__file__).read_bytes()):
        raise IdentityError("binding schema/helper mismatch")
    if capture(binding["expected"],proc)!=binding["observed"]:
        raise IdentityError("target identity or cgroup changed")
    pid=binding["expected"]["pid"]; cg=Path(binding["observed"]["cgroup"]["path"])
    sources={"host_meminfo":(proc/"meminfo","kB-fields"),
        "host_vmstat":(proc/"vmstat","raw per-field kernel units"),
        "host_pressure":(proc/"pressure/memory","raw PSI; totals in microseconds"),
        "process_status":(proc/str(pid)/"status","kB-fields"),
        "process_smaps_rollup":(proc/str(pid)/"smaps_rollup","kB-fields")}
    for name in ("memory.current","memory.peak","memory.stat","memory.events","memory.events.local","memory.pressure"):
        sources["cgroup_"+name]=(cg/name,"bytes" if name in ("memory.current","memory.peak") else "raw per-field kernel units")
    for name in ("memory.max","memory.high","memory.swap.max"):
        sources["cgroup_"+name]=(cg/name,"bytes-or-max")
    values={key:metric(path,kind) for key,(path,kind) in sources.items()}
    own=observer_resources(proc)
    if capture(binding["expected"],proc)!=binding["observed"]:
        raise IdentityError("target changed during sample; sampled values discarded")
    return dict(stage="SAMPLE",monotonic_ns=started,duration_ns=time.monotonic_ns()-started,
        identity_valid=True,metrics=values,observer_resources=own)

def observe(binding, output, interval=0.25, seconds=30, proc=Path("/proc")):
    if not 0.1<=interval<=5 or not 0<seconds<=3600:
        raise ValueError("sampling bounds exceeded")
    if capture(binding["expected"],proc)!=binding["observed"]:
        raise IdentityError("invalid identity before journal creation")
    if binding.get("observer_helper_sha256")!=sha(Path(__file__).read_bytes()):
        raise IdentityError("binding helper mismatch")
    common=dict(schema="pireus-external-memory-observation-v2",
        observer_profile=OBSERVER_PROFILE,
        binding_sha256=sha(json.dumps(binding,sort_keys=True).encode()),
        job=binding["expected"]["job"],rank=binding["expected"]["rank"],
        target_pid=binding["expected"]["pid"],observer_pid=os.getpid())
    began=time.monotonic_ns(); previous=None; deadline=began+int(seconds*1e9)
    with Path(output).open("x") as out:
        def emit(row):
            out.write(json.dumps(common|row,sort_keys=True)+"\n");out.flush()
        emit(dict(stage="OBSERVER_START",monotonic_ns=began,interval_seconds=interval,
            duration_limit_seconds=seconds,loaded_model_overhead_qualified=False,
            scheduling="actual-start-deadline-no-catchup",observer_resource_scope="observer-process-only"))
        while time.monotonic_ns()<deadline:
            try: row=sample(binding,proc)
            except IdentityError as exc:
                emit(dict(stage="TARGET_INVALIDATED",monotonic_ns=time.monotonic_ns(),
                    identity_valid=False,error=str(exc),metrics=None))
                return 3
            row["sample_gap_ns"]=None if previous is None else row["monotonic_ns"]-previous
            previous=row["monotonic_ns"]
            emit(row)
            # Read and serialization time consume the interval. An overrun starts
            # one sample immediately, then anchors its next deadline to that actual
            # start; no missed slots are replayed and actual gaps remain visible.
            wake=next_sample_deadline(row["monotonic_ns"],time.monotonic_ns(),
                                      int(interval*1e9),deadline)
            remaining=wake-time.monotonic_ns()
            while remaining>0:
                time.sleep(remaining/1e9)
                remaining=wake-time.monotonic_ns()
        emit(dict(stage="OBSERVER_END",monotonic_ns=time.monotonic_ns(),reason="duration_limit"))
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("bind"); b.add_argument("--expected",type=Path,required=True); b.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("observe");s.add_argument("--binding",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    s.add_argument("--interval",type=float,default=0.25);s.add_argument("--seconds",type=float,default=30)
    a=p.parse_args()
    if a.mode=="bind":
        result=bind(json.loads(a.expected.read_bytes()))
        with a.output.open("x") as out:out.write(json.dumps(result,indent=2)+"\n")
        return 0
    return observe(json.loads(a.binding.read_bytes()),a.output,a.interval,a.seconds)

if __name__=="__main__":
    raise SystemExit(main())
