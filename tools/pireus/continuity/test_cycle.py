#!/usr/bin/env python3
"""Integration regression for custody, idempotence, and a real Sounio engine."""
import argparse,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("engine",type=Path);args=ap.parse_args()
 script=Path(__file__).with_name("cycle.py")
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory);run=root/"run"
  context=root/"context.json"
  context.write_text(json.dumps(dict(schema=1,target=701202,dimension=16,precision=64,order=1,fma=0,capabilities=55,lane_width=32,facts_state=1))+"\n")
  evidence=root/"fixture-evidence.txt";evidence.write_text("Synthetic trusted context for transport regression only; not hardware evidence.\n")
  def call(command,*extra,ok=True):
   p=subprocess.run([sys.executable,str(script),command,"--run",str(run),*map(str,extra)],capture_output=True,text=True,timeout=120)
   assert (p.returncode==0)==ok,(command,p.stdout,p.stderr)
   return p
  call("prepare","--context",context,"--evidence",evidence,"--engine-sha256",hashlib.sha256(args.engine.read_bytes()).hexdigest())
  call("generate");journal=(run/"journal.jsonl").read_bytes()
  call("generate");assert (run/"journal.jsonl").read_bytes()==journal
  call("validate","--engine",args.engine)
  before={p.name:p.read_bytes() for p in run.glob("*.receipt.json")}
  call("validate","--engine",args.engine)
  assert {p.name:p.read_bytes() for p in run.glob("*.receipt.json")}==before
  call("materialize","--engine",args.engine)
  artifacts={p.name:p.read_bytes() for p in run.glob("*.ptx")}
  assert len(artifacts)==8 and len(set(artifacts.values()))==8
  call("materialize","--engine",args.engine)
  assert {p.name:p.read_bytes() for p in run.glob("*.ptx")}==artifacts
  result=json.loads(call("resume").stdout)
  assert result["admitted"]==8 and result["unique_plans"]==8 and not result["claim_ready"],result
  old=(run/"000.proposal.json").read_bytes()
  (run/"000.proposal.json").write_bytes(old+b" ")
  call("resume",ok=False);(run/"000.proposal.json").write_bytes(old)
  original=(run/"context.json").read_bytes()
  (run/"context.json").write_bytes(original+b" ")
  call("validate","--engine",args.engine,ok=False)
  (run/"context.json").write_bytes(original)
  call("validate","--engine",Path("/bin/true"),ok=False)
  print(json.dumps(result),flush=True)
 print("PIREUS_CYCLE_CUSTODY_GATE_PASS",flush=True)
if __name__=="__main__":main()
