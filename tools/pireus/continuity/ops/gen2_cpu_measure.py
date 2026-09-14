"""One-shot measurement of a pinned gen2 compilation; not a CI qualifier."""
import argparse, datetime, hashlib, json, os, pathlib, resource, signal
import socket, subprocess, tarfile, time

PROTOCOL_SHA = "3e09ae0d861c2a72b226c3e3f82f46750b9f2031b2a8b53912715c2338cd4c9c"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    try: return pathlib.Path(p).read_text()
    except OSError as e: return {"unavailable": str(e)}
def write_new(p, data):
    with p.open("x") as f: json.dump(data, f, indent=2); f.write("\n")
def snapshot():
    return {p: read(p) for p in ["/proc/meminfo", "/proc/vmstat", "/proc/loadavg",
            "/proc/pressure/memory", "/proc/pressure/cpu", "/proc/self/cgroup"]}
def measure(command, cwd, output, timeout, interval, env):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    timed_out = False
    with (output/"compile.log").open("xb") as log, (output/"samples.jsonl").open("x") as samples:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            row = {"elapsed_seconds":time.monotonic()-start, "pid":proc.pid,
                   "host":snapshot(), "process_status":read(f"/proc/{proc.pid}/status"),
                   "process_stat":read(f"/proc/{proc.pid}/stat")}
            samples.write(json.dumps(row)+"\n"); samples.flush()
            if time.monotonic()-start >= timeout:
                timed_out = True
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                break
            time.sleep(min(interval, max(0.01,timeout-(time.monotonic()-start))))
        rc = proc.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"compiler_rc":rc, "timed_out":timed_out, "wall_seconds":time.monotonic()-start,
            "user_seconds":after.ru_utime-before.ru_utime,
            "system_seconds":after.ru_stime-before.ru_stime, "maxrss_kib":after.ru_maxrss,
            "minor_faults":after.ru_minflt-before.ru_minflt,
            "major_faults":after.ru_majflt-before.ru_majflt,
            "rss_scope":"maximum child high-water mark; wrapper and its waited descendants",
            "ci_qualified":False, "inkling_qualified":False, "causal_claim":False}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--check",action="store_true")
    ap.add_argument("--boot-id")
    args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parent
    if sha(root/"protocol.json") != PROTOCOL_SHA: raise ValueError("protocol changed")
    protocol=json.loads((root/"protocol.json").read_text())
    for name,pin in protocol["files_sha256"].items():
        if sha(root/name)!=pin: raise ValueError("input changed: "+name)
    with tarfile.open(root/"source.tar") as archive:
        for m in archive.getmembers():
            if m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
                raise ValueError("unsupported source archive member")
            if m.name.startswith("/") or ".." in pathlib.PurePosixPath(m.name).parts:
                raise ValueError("unsafe source archive path")
        if args.check:
            print("PACKET_CHECK_PASS: hashes and source archive only"); return
        boot=read("/proc/sys/kernel/random/boot_id").strip()
        if not args.boot_id or boot != args.boot_id: raise ValueError("worker boot changed")
        if socket.gethostname()!=protocol["node"]: raise ValueError("wrong worker")
        if not os.environ.get("SLURM_JOB_ID"): raise ValueError("Slurm allocation required")
        if int(os.environ.get("SLURM_CPUS_PER_TASK","0"))!=protocol["cpus"]: raise ValueError("CPU allocation mismatch")
        if int(os.environ.get("SLURM_MEM_PER_NODE","0"))!=protocol["slurm_memory_mib"]: raise ValueError("memory allocation mismatch")
        write_new(root/"attempt-entered.json",{"job":os.environ["SLURM_JOB_ID"],
                  "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "hostname":socket.gethostname(),"boot_id":boot,
                  "protocol_sha256":PROTOCOL_SHA,"runner_sha256":sha(pathlib.Path(__file__)),
                  "host":snapshot(),"allocation":{k:v for k,v in os.environ.items() if k.startswith("SLURM_")}})
        tree=root/"tree"; tree.mkdir()
        archive.extractall(tree,filter="data")
    (root/"madaros").chmod(0o755)
    pins={str(p.relative_to(tree)):sha(p) for p in tree.rglob("*") if p.is_file()}
    env=os.environ.copy()
    for k in ["SOUC_BIN","SOUNIO_SOUC_BIN","MADAROS_BIN","MADAROS_RAW_BIN","_SOUNIO_SOUC_INVOKE_SOURCED"]:
        env.pop(k,None)
    env["SOUNIO_STDLIB_PATH"]=str(tree/"stdlib")
    result=measure(protocol["command"],tree,root,protocol["command_timeout_seconds"],
                   protocol["sample_seconds"],env)
    result["boot_unchanged"]=read("/proc/sys/kernel/random/boot_id").strip()==boot
    result["source_files_unchanged"]=all(sha(tree/n)==h for n,h in pins.items())
    result["compiler_unchanged"]=sha(root/"madaros")==protocol["files_sha256"]["madaros"]
    artifact=root/"madaros.gen2"
    result["artifact_exists"]=artifact.is_file()
    result["artifact_sha256"]=sha(artifact) if artifact.is_file() else None
    result["compile_complete"]=result["compiler_rc"]==0 and not result["timed_out"] and artifact.is_file() and artifact.stat().st_size>0
    result["host_final"]=snapshot()
    result["job"]=os.environ["SLURM_JOB_ID"]
    write_new(root/"result.json",result)
    print(json.dumps(result),flush=True)
if __name__=="__main__": main()
