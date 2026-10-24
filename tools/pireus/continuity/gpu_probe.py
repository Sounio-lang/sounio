#!/usr/bin/env python3
"""CUDA material observer: loads Sounio PTX, moves fixture bits, records output bits."""
import argparse,ctypes,hashlib,json,os,socket
from pathlib import Path
import numpy as np
import torch
class Driver:
 def __init__(self):
  self.lib=ctypes.CDLL("libcuda.so.1")
  self.lib.cuInit.argtypes=[ctypes.c_uint]
  self.lib.cuModuleLoadData.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
  self.lib.cuModuleGetFunction.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_char_p]
  self.lib.cuLaunchKernel.argtypes=[ctypes.c_void_p,*([ctypes.c_uint]*7),ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
  self.lib.cuModuleUnload.argtypes=[ctypes.c_void_p]
  self.lib.cuGetErrorString.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]
  self.check(self.lib.cuInit(0))
  torch.cuda.init()
  torch.cuda.set_device(0)
  self.context_anchor=torch.empty(1,device="cuda")
  self.lib.cuCtxGetCurrent.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
  context=ctypes.c_void_p()
  self.check(self.lib.cuCtxGetCurrent(ctypes.byref(context)))
  assert context.value,"PyTorch primary context must be current"
 def check(self,code):
  if code:
   text=ctypes.c_char_p();self.lib.cuGetErrorString(code,ctypes.byref(text))
   raise RuntimeError(str(code)+": "+str(text.value))
 def load(self,path):
  module=ctypes.c_void_p()
  data=ctypes.create_string_buffer(path.read_bytes())
  self.check(self.lib.cuModuleLoadData(ctypes.byref(module),ctypes.cast(data,ctypes.c_void_p)))
  function=ctypes.c_void_p()
  self.check(self.lib.cuModuleGetFunction(ctypes.byref(function),module,b"pireus_candidate"))
  return module,function
 def launch(self,function,a,b,c,n):
  assert 0<n<=16777216
  values=[ctypes.c_uint64(t.data_ptr()) for t in (a,b,c)]
  parameters=(ctypes.c_void_p*3)(*[ctypes.addressof(v) for v in values])
  self.check(self.lib.cuLaunchKernel(function,n,1,1,16,1,1,0,
              ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),parameters,None))
def hex_frame(values):
 return "".join("".join(format(int(x)&((1<<64)-1),"016x") for x in row)+"\n" for row in values)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);args=ap.parse_args()
 assert os.environ.get("SLURM_JOB_ID"),"Slurm allocation required"
 root=args.run
 manifest=json.loads((root/"gpu-manifest.json").read_text())
 fixtures=(root/"numeric-fixtures.jsonl").read_bytes()
 assert hashlib.sha256(fixtures).hexdigest()==manifest["fixtures_sha256"]
 rows=[json.loads(line) for line in fixtures.splitlines()]
 arrays={}
 for key in ["a_bits","b_bits","output_bits"]:
  arrays[key]=np.array([r[key] for r in rows],dtype=np.int64)
 n=len(rows);driver=Driver()
 output=root/("observed-"+socket.gethostname());output.mkdir(exist_ok=True)
 (output/"reference.hex").write_text(hex_frame(arrays["output_bits"]))
 for candidate in manifest["candidates"]:
  path=root/candidate["ptx"]
  assert hashlib.sha256(path.read_bytes()).hexdigest()==candidate["ptx_sha256"]
  module,function=driver.load(path)
  tensors=[]
  for key in ["a_bits","b_bits"]:
   value=arrays[key].view(np.float64)
   if candidate["layout"]==1:value=value.T
   tensors.append(torch.from_numpy(np.ascontiguousarray(value)).to("cuda"))
  out=torch.empty_like(tensors[0])
  driver.launch(function,*tensors,out,n)
  torch.cuda.synchronize()
  raw=out.cpu().numpy()
  if candidate["layout"]==1:raw=raw.T
  bits=np.ascontiguousarray(raw).view(np.int64)
  destination=output/(candidate["id"]+".hex")
  destination.write_text(hex_frame(bits))
  print(json.dumps(dict(stage="MATERIAL_OBSERVATION",candidate=candidate["id"],
    sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),components=int(bits.size),
    node=socket.gethostname(),job=os.environ["SLURM_JOB_ID"])),flush=True)
  driver.check(driver.lib.cuModuleUnload(module))
 (output/"inventory.json").write_text(json.dumps(dict(job=os.environ["SLURM_JOB_ID"],
    node=socket.gethostname(),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
    observation_only=True),indent=2)+"\n")
if __name__=="__main__":main()
