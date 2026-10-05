#!/usr/bin/env python3
"""Exercise native Sounio material parity, ontology context, and gain decisions."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("context");ap.add_argument("admission");ap.add_argument("parity");ap.add_argument("gain");a=ap.parse_args()
 with tempfile.TemporaryDirectory() as d:
  root=Path(d)
  for control,cap,state,reason in [("",55,1,None),("1",39,1,"CAPABILITY"),("2",55,2,"UNKNOWN_OR_CONTRADICTORY_FACTS"),("3",54,0,"UNKNOWN_OR_CONTRADICTORY_FACTS")]:
   run=subprocess.run([a.context]+([control] if control else []),capture_output=True,check=True)
   c=json.loads(run.stdout);assert (c["capabilities"],c["facts_state"])==(cap,state)
   cp=root/"context.json";cp.write_bytes(run.stdout)
   p=dict(schema=1,target=701202,dimension=16,precision=64,order=1,fma=0,kind=1,lane_stride=1,lane_offset=0,load=1,layout=0,unroll=1,context=hashlib.sha256(run.stdout).hexdigest())
   pp=root/"proposal.json";pp.write_text(json.dumps(p)+"\n")
   decision=subprocess.run([a.admission,str(cp),str(pp)],capture_output=True,text=True)
   receipt=json.loads(decision.stdout)
   assert (decision.returncode==0)==(reason is None),(control,receipt)
   if reason:assert receipt["reason"]==reason
   print("CONTEXT_CONTROL",control or "production",receipt["decision"],flush=True)
  def frame(words):return "".join(format(x,"016x") for x in words)+"\n"
  ref=root/"ref.hex";obs=root/"obs.hex"
  cases=[("equal",[0]*16,[0]*16,0,0),
    ("signed-zero",[0]*16,[1<<63]+[0]*15,1,0),
    ("nan-payload",[0x7ff8000000000001]+[0]*15,[0xfff8000000000002]+[0]*15,0,1),
    ("infinity-sign",[0x7ff0000000000000]+[0]*15,[0xfff0000000000000]+[0]*15,1,0)]
  for name,x,y,rc,nans in cases:
   ref.write_text(frame(x));obs.write_text(frame(y))
   run=subprocess.run([a.parity,str(ref),str(obs)],capture_output=True,text=True)
   assert run.returncode==rc,(name,run.stdout,run.stderr)
   receipt=json.loads(run.stdout);assert receipt["nan_class_only"]==nans
   print("PARITY_CONTROL",name,receipt["decision"],flush=True)
  obs.write_text("x"*256+"\n")
  assert subprocess.run([a.parity,str(ref),str(obs)],capture_output=True).returncode==2
  paths=[root/("timing-"+str(i)+".tsv") for i in range(4)]
  def write(candidate,control=1000000):
   return "".join(f"{i} {i%2} {candidate} {control}\n" for i in range(30))
  for label,candidate,expected in [("six-percent",940000,"PASS"),("threshold",950000,"PASS"),("equal",1000000,"NO_GAIN")]:
   for p in paths:p.write_text(write(candidate))
   run=subprocess.run([a.gain,*map(str,paths)],capture_output=True,text=True,timeout=60)
   assert run.returncode==0,(label,run.stderr)
   receipt=json.loads(run.stdout);assert receipt["gain_gate"]==expected,(label,receipt)
   for c in receipt["comparisons"]:assert c["median_gain_ppm"]==1000000-candidate and c["ci95_low_ppm"]==c["ci95_high_ppm"]==c["median_gain_ppm"]
   print("GAIN_CONTROL",label,receipt["gain_gate"],flush=True)
  for p in paths:p.write_text(write(900000))
  paths[-1].write_text(write(1100000))
  receipt=json.loads(subprocess.check_output([a.gain,*map(str,paths)],text=True))
  assert receipt["gain_gate"]=="NO_GAIN"
  paths[-1].write_text(write(0))
  assert subprocess.run([a.gain,*map(str,paths)],capture_output=True).returncode==2
  print("GAIN_CONTROL one-node-failure NO_GAIN; zero-time FORMAT_REFUSE",flush=True)
 print("PIREUS_MATERIAL_CONTROLS_GATE_PASS",flush=True)
if __name__=="__main__":main()
