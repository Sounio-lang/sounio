import base64,json,os,shlex,subprocess,sys,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
if not os.environ.get("TMUX"):raise SystemExit("remote tmux required")
m=json.loads((root/"manifest.json").read_bytes())
sys.path.insert(0,"/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity/runtime")
from preflight import check_pair
(root/"preflight.json").write_text(json.dumps(check_pair(),indent=2)+"\n")
queue=subprocess.check_output(["squeue","-h","-p","gpu-orangefs"],text=True)
(root/"queue-before.txt").write_text(queue)
if queue.strip():raise ValueError("pair queue occupied")
for name,sha in m["files_sha256"].items():
 if hashlib.sha256((root/"source"/name).read_bytes()).hexdigest()!=sha:raise ValueError("source changed")

remote="/scratch/pireus/diagnostics/external-container-cpu-20260909-v2"
for w in m["workers"]:
 live=json.loads(subprocess.check_output(["kubectl","-n","slurm-pilot","get","pod",w["pod"],"-o","json"]))
 if live["metadata"]["uid"]!=w["uid"]:raise ValueError("worker changed")
 blobs={p.name:base64.b64encode(p.read_bytes()).decode() for p in (root/"source").iterdir()}
 code="import base64,os;from pathlib import Path;r=Path("+repr(remote)+");r.mkdir(parents=True,exist_ok=False);os.chown(r,1000,1000);b="+repr(blobs)+";[( (r/n).write_bytes(base64.b64decode(v)),(r/n).chmod(0o555 if n.endswith('.sh') else 0o444)) for n,v in b.items()]"
 subprocess.run(["kubectl","-n","slurm-pilot","exec",w["pod"],"-c","slurmd","--","python3","-c",code],check=True)
command=["set -euo pipefail","export TMPDIR=/tmp PIREUS_MARLIN_OVERLAY=0",
 "echo '"+m["sif_sha256"]+"  /scratch/pireus/images/inkling-spark.sif' | sha256sum -c -",
 'case "$(hostname)" in']
entry_sha=m["files_sha256"]["cpu-target.py"]
for rank,w in enumerate(m["workers"]):
 command.append("*"+w["node"].removeprefix("spark-")+") export PIREUS_RANK="+str(rank)+"; exec python3 "+remote+"/external_rank_supervisor.py --output "+remote+'/run-"$SLURM_JOB_ID-'+str(rank)+'" --worker-uid '+shlex.quote(w["uid"])+" --boot-id "+shlex.quote(w["boot_id"])+" --entry-sha "+entry_sha+" -- python3 "+remote+"/memory_guard.py --reserve-gib 33 -- "+remote+"/run_in_container.sh python3 "+remote+"/publish_observer_target.py --entry "+remote+"/cpu-target.py --entry-sha "+entry_sha+" -- ;;")
command+=["*) exit 2 ;;","esac"]
argv=["srun","-p","gpu-orangefs","-w","gpuorangefs-multi-spark-3c59,gpuorangefs-multi-spark-8e54",
 "--nodes=2","--ntasks=2","--ntasks-per-node=1","-c2","--mem=512M","--exclusive",
 "--propagate=NONE","--kill-on-bad-exit=1","--time=3","--job-name=pireus-external-container-cpu",
 "bash","-c","\n".join(command)]
(root/"command.json").write_text(json.dumps(argv,indent=2)+"\n")
env=os.environ.copy();env["TMPDIR"]="/tmp"
with (root/"launch.log").open("xb") as out:
 rc=subprocess.call(argv,stdout=out,stderr=subprocess.STDOUT,env=env,cwd="/tmp")
(root/"exit-code").write_text(str(rc)+"\n")
raise SystemExit(rc)
