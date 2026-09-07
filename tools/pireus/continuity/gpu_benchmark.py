#!/usr/bin/env python3
"""Record paired CUDA-event durations; Sounio computes all gain decisions."""
import argparse,hashlib,json,os,socket,time
from pathlib import Path
import numpy as np
import torch
from gpu_probe import Driver
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);a=ap.parse_args()
 assert os.environ.get("SLURM_JOB_ID")
 root=a.run;manifest=json.loads((root/"benchmark-manifest.json").read_text())
 assert manifest["method"]=="paired-median-ppm-bootstrap4000-v1"
 assert manifest["blocks"]==30 and manifest["vectors"]==16384 and manifest["replays"]==32
 fixtures=(root/"numeric-fixtures.jsonl").read_bytes()
 assert hashlib.sha256(fixtures).hexdigest()==manifest["fixtures_sha256"]
 rows=[json.loads(line) for line in fixtures.splitlines()][256:288]
 inputs={k:np.tile(np.array([r[k] for r in rows],dtype=np.int64),(512,1)).view(np.float64) for k in ("a_bits","b_bits")}
 driver=Driver();graphs={};retained=[]
 items=manifest["controls"]+manifest["candidates"]
 for item in items:
  p=root/item["ptx"];assert hashlib.sha256(p.read_bytes()).hexdigest()==item["ptx_sha256"]
  module,function=driver.load(p)
  tensors=[torch.from_numpy(np.ascontiguousarray(v.T if item["layout"] else v)).to("cuda") for v in inputs.values()]
  out=torch.empty_like(tensors[0])
  driver.launch(function,*tensors,out,16384);torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   for _ in range(32):driver.launch(function,*tensors,out,16384)
  for _ in range(8):graph.replay()
  torch.cuda.synchronize()
  graphs[item["id"]]=graph;retained.append((module,function,tensors,out))
 output=root/("timings-"+socket.gethostname());output.mkdir(exist_ok=True)
 for candidate in manifest["candidates"]:
  for control in manifest["controls"]:
   target=output/(candidate["id"]+"."+control["id"]+".tsv")
   if target.exists():raise ValueError("timing trial exists; use a distinct frozen run")
   cg=graphs[candidate["id"]];bg=graphs[control["id"]]
   for _ in range(8):cg.replay();bg.replay()
   torch.cuda.synchronize()
   with target.open("w") as log:
    for block in range(30):
     durations={}
     order=[("control",bg),("candidate",cg)] if block%2 else [("candidate",cg),("control",bg)]
     for label,graph in order:
      start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
      start.record();graph.replay();end.record();end.synchronize()
      durations[label]=round(start.elapsed_time(end)*1000000)
      assert 0<durations[label]<1000000000
     log.write(f"{block} {block%2} {durations['candidate']} {durations['control']}\n")
     log.flush();os.fsync(log.fileno())
   print(json.dumps(dict(stage="PAIRED_TIMING_OBSERVED",candidate=candidate["id"],
      control=control["id"],node=socket.gethostname(),job=os.environ["SLURM_JOB_ID"],
      sha256=hashlib.sha256(target.read_bytes()).hexdigest(),blocks=30)),flush=True)
 (output/"inventory.json").write_text(json.dumps(dict(job=os.environ["SLURM_JOB_ID"],
  node=socket.gethostname(),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
  torch=str(torch.__version__),timing="CUDA events around 32 captured kernel launches",
  vectors=16384,layout_conversion_included=False,observation_only=True),indent=2)+"\n")
if __name__=="__main__":main()
