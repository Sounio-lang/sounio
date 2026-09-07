#!/usr/bin/env python3
"""Workspace launcher. Run inside remote tmux; all GPU processes belong to srun."""
import argparse, hashlib, json, os, shlex, subprocess
from pathlib import Path
from preflight import check_pair
HERE=Path(__file__).resolve().parent
NODES=["gpuorangefs-multi-spark-3c59","gpuorangefs-multi-spark-8e54"]
REVISION="b6a99534467840620d411e4cd4ad5819b2610d9c"
MODEL="/scratch/pireus/models/Inkling-Small-NVFP4/"+REVISION
SIF_SHA="3dbfccad3355b27d8a09bd4c0c5895d02a43e1b64203960905b810bbb6bccbe3"
def kube(*args):return subprocess.check_output(["kubectl",*args],text=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["qualify","qualify-model","inspect-runtime","qualify-marlin","qualify-overlay","profile-memory","profile-cuda-memory","qualify-repack","tokenize","qualify-offline-interface","offline-generate","serve-token-ids","serve"]);ap.add_argument("--minutes",type=int,default=30);ap.add_argument("--tokenizer-input","--input-bundle",dest="tokenizer_input",type=Path);args=ap.parse_args()
 if args.mode in ("tokenize","qualify-offline-interface","offline-generate") and not args.tokenizer_input:raise SystemExit("tokenize requires --tokenizer-input")
 if args.tokenizer_input and args.mode not in ("tokenize","qualify-offline-interface","offline-generate"):raise SystemExit("tokenizer input is only for tokenize")
 if not os.environ.get("TMUX"):raise SystemExit("Use remote tmux for a disconnect-safe allocation")
 if not 1<=args.minutes<=240:raise SystemExit("minutes must be 1..240")
 if kube("-n","beagle","get","lease","pireus-spark-pair","-o","jsonpath={.spec.holderIdentity}")!="slurm-owned":
  raise SystemExit("Spark pair lease is not slurm-owned")
 print(json.dumps({"host_preflight":check_pair()}),flush=True)
 pods=json.loads(kube("-n","slurm-pilot","get","pods","-o","json"))["items"]
 selected=[]
 for node in ["spark-3c59","spark-8e54"]:
  matches=[p for p in pods if p["spec"].get("nodeName")==node and p["metadata"]["name"].startswith("slurm-pilot-worker-spark-") and p["status"]["phase"]=="Running"]
  if len(matches)!=1:raise SystemExit("Expected one Spark worker for "+node)
  selected.append(matches[0])
 token_path = None
 if args.tokenizer_input:
  token_data = args.tokenizer_input.read_bytes()
  token_path = "/scratch/pireus/runtime/tokenizer-input-" + hashlib.sha256(token_data).hexdigest() + ".json"
 for pod in selected:
  base=["kubectl","-n","slurm-pilot","exec",pod["metadata"]["name"],"-c","slurmd","--"]
  for name in ["offline_generate.py","evict_checkpoint_cache.py","patch_marlin_deinterleave.py","profile_cuda_memory.py","tokenizer_transport.py","run_in_container.sh","qualify_tp2.py","inspect_runtime.py","profile_model_memory.py","qualify_marlin_placeholders.py","patch_marlin_placeholders.py","patch_marlin_repack.py","install_marlin_overlay.py","qualify_model.py","serve_rank.sh","memory_guard.py","worker_prerequisites.py","inkling-files.json"]:
   subprocess.run(base[:4]+["-i"]+base[4:]+["python3","-c","from pathlib import Path;import sys;p=Path(sys.argv[1]);p.write_bytes(sys.stdin.buffer.read());p.chmod(0o755)","/scratch/pireus/runtime/"+name],input=(HERE/name).read_bytes(),check=True)
  if token_path:
   subprocess.run(base[:4]+["-i"]+base[4:]+["python3","-c","from pathlib import Path;import sys;p=Path(sys.argv[1]);data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data);p.chmod(0o444)",token_path],input=token_data,check=True)
  subprocess.run(base+["python3","/scratch/pireus/runtime/worker_prerequisites.py"],check=True)
  subprocess.run(base+["python3","-c","from pathlib import Path;import os;paths=[Path('/scratch/pireus/receipts'),Path('/scratch/pireus/cache')];[(p.mkdir(exist_ok=True),os.chown(p,1000,1000)) for p in paths]"],check=True)
 env=os.environ.copy()
 env.update(TMPDIR="/tmp",MASTER_ADDR=selected[0]["status"]["podIP"],MASTER_PORT="29571",
            NCCL_NET="IB",NCCL_IB_DISABLE="0",NCCL_DEBUG="INFO",NCCL_SOCKET_IFNAME="eth0")
 rank='case "$(hostname)" in *3c59) export PIREUS_RANK=0;; *8e54) export PIREUS_RANK=1;; *) exit 2;; esac\n'
 check='echo "'+SIF_SHA+'  /scratch/pireus/images/inkling-spark.sif" | sha256sum -c -\n'
 if args.mode=="tokenize":
  command="exec python3 /scratch/pireus/runtime/memory_guard.py -- /scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/tokenizer_transport.py "+shlex.quote(token_path)
 elif args.mode=="qualify":
  command="exec /scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/qualify_tp2.py"
 elif args.mode=="qualify-offline-interface":
  command="PIREUS_OFFLINE_INTERFACE=1 PIREUS_OFFLINE_MODE=generate PIREUS_OFFLINE_INPUT="+shlex.quote(token_path)+" PIREUS_TOKEN_IDS=1 exec /scratch/pireus/runtime/serve_rank.sh"
 elif args.mode=="qualify-repack":
  command="PIREUS_REPACK_TEST=1 exec python3 /scratch/pireus/runtime/memory_guard.py -- /scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/qualify_marlin_placeholders.py"
 elif args.mode=="profile-cuda-memory":
  command="PIREUS_CUDA_PROBE=1 PIREUS_TOKEN_IDS=1 exec /scratch/pireus/runtime/serve_rank.sh"
 elif args.mode=="profile-memory":
  command="PIREUS_META_PROBE=1 exec /scratch/pireus/runtime/serve_rank.sh"
 elif args.mode=="qualify-overlay":
  command="PIREUS_MARLIN_OVERLAY=0 /scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/install_marlin_overlay.py\nPIREUS_MARLIN_OVERLAY=1 exec /scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/inspect_runtime.py"
 elif args.mode=="qualify-marlin":
  command="exec python3 /scratch/pireus/runtime/memory_guard.py -- /scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/qualify_marlin_placeholders.py"
 elif args.mode=="inspect-runtime":
  command="exec /scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/inspect_runtime.py"
 elif args.mode=="qualify-model":
  command="exec python3 /scratch/pireus/runtime/qualify_model.py "+MODEL+" /scratch/pireus/runtime/inkling-files.json --receipt /scratch/pireus/receipts/inkling-model.json"
 else:command="python3 /scratch/pireus/runtime/qualify_model.py "+MODEL+" /scratch/pireus/runtime/inkling-files.json --receipt /scratch/pireus/receipts/inkling-model.json\nexec /scratch/pireus/runtime/serve_rank.sh"
 if args.mode=="offline-generate":command=command.replace("exec /scratch/pireus/runtime/serve_rank.sh","PIREUS_OFFLINE_MODE=generate PIREUS_OFFLINE_INPUT="+shlex.quote(token_path)+" PIREUS_TOKEN_IDS=1 exec /scratch/pireus/runtime/serve_rank.sh")
 if args.mode=="serve-token-ids":command=command.replace("exec /scratch/pireus/runtime/serve_rank.sh","PIREUS_TOKEN_IDS=1 exec /scratch/pireus/runtime/serve_rank.sh")
 argv=["srun","-p","gpu-orangefs","-w",",".join(NODES),"--nodes=2","--ntasks=2","--ntasks-per-node=1","-c8","--gres=gpu:gb10:1","--mem=110G","--exclusive","--propagate=NONE","--kill-on-bad-exit=1","--time="+str(args.minutes),"--job-name=pireus-inkling-"+args.mode,"bash","-c","set -euo pipefail\n"+rank+check+command]
 print(json.dumps({"mode":args.mode,"nodes":[{"pod":p["metadata"]["name"],"uid":p["metadata"]["uid"],"ip":p["status"]["podIP"]} for p in selected],"command":argv}),flush=True)
 raise SystemExit(subprocess.call(argv,env=env,cwd="/tmp"))
if __name__=="__main__":main()
