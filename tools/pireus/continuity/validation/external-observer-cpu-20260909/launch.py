import base64,json,os,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
if not os.environ.get("TMUX"):raise SystemExit("remote tmux required")
manifest=json.loads((root/"manifest.json").read_bytes())
remote="/scratch/pireus/diagnostics/external-observer-cpu-20260909-v1"
for worker in manifest["workers"]:
    live=json.loads(subprocess.check_output(["kubectl","-n","slurm-pilot","get","pod",worker["pod"],"-o","json"]))
    if live["metadata"]["uid"]!=worker["uid"]:raise ValueError("worker changed")
    blobs={p.name:base64.b64encode(p.read_bytes()).decode() for p in (root/"source").iterdir()}
    script="import base64,os;from pathlib import Path;r=Path("+repr(remote)+");r.mkdir(parents=True,exist_ok=False);os.chown(r,1000,1000);b="+repr(blobs)+";[( (r/n).write_bytes(base64.b64decode(v)),(r/n).chmod(0o444)) for n,v in b.items()]"
    subprocess.run(["kubectl","-n","slurm-pilot","exec",worker["pod"],"-c","slurmd","--","python3","-c",script],check=True)
parts=["set -euo pipefail","case \"$(hostname)\" in"]
for rank,worker in enumerate(manifest["workers"]):
    parts.append("*"+worker["node"].removeprefix("spark-")+") export PIREUS_RANK="+str(rank)+"; exec python3 "+shlex.quote(remote+"/external_observer_cpu_control.py")+" --worker-uid "+shlex.quote(worker["uid"])+" --boot-id "+shlex.quote(worker["boot_id"])+" --output "+shlex.quote(remote)+"/\"$SLURM_JOB_ID-"+str(rank)+"\" ;;")
parts.extend(["*) exit 2 ;;","esac"])
argv=["srun","-p","gpu-orangefs","-w","gpuorangefs-multi-spark-3c59,gpuorangefs-multi-spark-8e54",
      "--nodes=2","--ntasks=2","--ntasks-per-node=1","-c2","--mem=512M","--exclusive",
      "--propagate=NONE","--kill-on-bad-exit=1","--time=3","--job-name=pireus-external-cpu-control",
      "bash","-c","\n".join(parts)]
(root/"command.json").write_text(json.dumps(argv,indent=2)+"\n")
with (root/"launch.log").open("xb") as log:
    rc=subprocess.call(argv,stdout=log,stderr=subprocess.STDOUT,cwd="/tmp")
(root/"exit-code").write_text(str(rc)+"\n")
raise SystemExit(rc)
