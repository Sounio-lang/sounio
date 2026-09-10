#!/usr/bin/env python3
"""Refusal controls for the bounded native recovery decision."""
import json,subprocess,sys,tempfile
from pathlib import Path
def main():
 engine=Path(sys.argv[1]).resolve()
 base=[1]+[1]*16+[32768,32768];base[15]=30
 with tempfile.TemporaryDirectory() as directory:
  path=Path(directory)/"facts"
  def run(values,expected,raw=None):
   path.write_text(raw if raw is not None else " ".join(map(str,values))+"\n")
   result=subprocess.run([str(engine),str(path)],capture_output=True,text=True,timeout=30)
   assert result.returncode==expected,(values,result.returncode,result.stdout)
   if expected!=2:
    r=json.loads(result.stdout)
    assert r["authority"]=="Sounio" and r["decision"]==("ALLOW" if expected==0 else "REFUSE")
    assert r["gpu_grant"] is False and r["resume"] is False and r["minimum_free_memory_mb"]==32768
  run(base,0)
  for i in range(1,17):
   if i==15:
    for bit in (2,4,8,16):
     values=base.copy();values[i]=30-bit;run(values,1)
   else:
    values=base.copy();values[i]=0;run(values,1)
    values[i]=2;run(values,2)
  for i in (17,18):
   values=base.copy();values[i]=32767;run(values,1)
  for raw in ("", " ".join(map(str,base)), " ".join(map(str,base))+"\nextra",
              " ".join(map(str,base))+"\n\n", "-1 "+"1 "*17+"32768\n"):
   run(base,2,raw)
  values=base.copy();values[15]=512;run(values,2)
 print("PIREUS_RECOVERY_DETACH_CONTROLS_PASS; no cluster mutation")
if __name__=="__main__":main()
