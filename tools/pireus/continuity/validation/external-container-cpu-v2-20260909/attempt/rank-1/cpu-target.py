import json,mmap,os,time
from pathlib import Path
path=Path(os.environ["PIREUS_EXTERNAL_HANDOFF"]).parent/"cpu-phases.jsonl"
with path.open("x") as out:
 def phase(name):
  out.write(json.dumps(dict(phase=name,pid=os.getpid(),monotonic_ns=time.monotonic_ns()))+"\n")
  out.flush()
 phase("baseline");time.sleep(2)
 value=mmap.mmap(-1,64*1024**2,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
 for offset in range(0,len(value),4096):value[offset]=1
 phase("allocated");time.sleep(3)
 value.close();phase("released");time.sleep(2)
 phase("exiting")
