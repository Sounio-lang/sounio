import pathlib,subprocess,os,json,hashlib,tarfile,datetime
ROOT=pathlib.Path(__file__).resolve().parent
REMOTE="/scratch/pireus-gen2-cpu-envelope-v1-20260910"
POD="slurm-pilot-worker-dl380-hvflp"
BOOT="c8e3d8c4-8c29-4caf-bbd5-95c99af8a003"
UID="aafc75ff-43b3-42db-a5ba-916baf4d3eec"
K=["kubectl","-n","slurm-pilot"]
env={**os.environ,"SLURM_CONF":"/tmp/slurm-direct.conf","TMPDIR":"/tmp"}
def save(name,data):
 with (ROOT/name).open("x") as f: json.dump(data,f,indent=2);f.write("\n")
def out(args): return subprocess.check_output(args,env=env,text=True)
save("launch-entered.json",{"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
pod=json.loads(out(K+["get","pod",POD,"-o","json"]))
assert pod["metadata"]["uid"]==UID
save("worker-before.json",pod)
assert out(["squeue","-h","-w","dl380-dl380-proxmox"]).strip()==""
save("slurm-before.json",{"node":out(["scontrol","show","node","dl380-dl380-proxmox"]),
                        "queue":out(["squeue","-h","-w","dl380-dl380-proxmox"])})
probe=json.loads(out(K+["exec",POD,"--","python3","-c",
 'import pathlib,json; print(json.dumps({"boot":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"meminfo":pathlib.Path("/proc/meminfo").read_text()}))']))
assert probe["boot"]==BOOT
available=int(next(l.split()[1] for l in probe["meminfo"].splitlines() if l.startswith("MemAvailable:")))*1024
assert available>=40*1024**3,available
save("host-before.json",probe)
with tarfile.open(ROOT/"packet.tar.gz","w:gz") as tar:
 for name in ["madaros","source.tar","protocol.json","measure.py"]:
  tar.add(ROOT/name,arcname=name)
save("transport-manifest.json",{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                              for n in ["madaros","source.tar","protocol.json","measure.py","packet.tar.gz"]})
subprocess.run(K+["exec",POD,"--","mkdir",REMOTE],check=True,env=env)
with (ROOT/"packet.tar.gz").open("rb") as data:
 subprocess.run(K+["exec","-i",POD,"--","tar","-xzf","-","-C",REMOTE],stdin=data,env=env,check=True)
subprocess.run(K+["exec",POD,"--","chown","-R","1000:1000",REMOTE],env=env,check=True)
transport_check=out(K+["exec",POD,"--","python3","-c",
 'import hashlib,pathlib,json; r=pathlib.Path("'+REMOTE+'"); print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ["madaros","source.tar","protocol.json","measure.py"]}))'])
observed=json.loads(transport_check)
for n,h in observed.items(): assert h==hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
save("transport-verified.json",observed)
subprocess.run(K+["exec",POD,"--","python3",REMOTE+"/measure.py","--check"],env=env,check=True)
assert out(["squeue","-h","-w","dl380-dl380-proxmox"]).strip()==""
current=json.loads(out(K+["get","pod",POD,"-o","json"]));assert current["metadata"]["uid"]==UID
command=["srun","--partition=all","--nodelist=dl380-dl380-proxmox","--nodes=1","--ntasks=1",
         "--cpus-per-task=4","--mem=36G","--time=01:32:00","--job-name=pireus-gen2-cpu-v1",
         "python3",REMOTE+"/measure.py","--boot-id",BOOT]
save("command.json",command)
print("SUBMIT_ONCE",flush=True)
with (ROOT/"srun.log").open("xb") as log:
 result=subprocess.run(command,env=env,cwd="/tmp",stdout=log,stderr=subprocess.STDOUT)
save("srun-terminal.json",{"rc":result.returncode,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"automatic_retry":False})
print("TERMINAL",result.returncode,flush=True)
