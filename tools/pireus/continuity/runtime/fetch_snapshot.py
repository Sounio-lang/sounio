#!/usr/bin/env python3
"""Resume direct internal transport; SHA qualification remains a separate gate."""
import argparse,json,os,time,urllib.request,urllib.error,http.client
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--endpoint",required=True);a=ap.parse_args()
 a.root.mkdir(parents=True,exist_ok=True)
 pending={e["rfilename"]:e for e in json.loads(a.manifest.read_text())}
 while pending:
  progress=False
  for name,e in list(pending.items()):
   p=a.root/name;part=p.with_name(p.name+".partial")
   p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():
    if p.stat().st_size!=e["size"]:raise ValueError("existing size mismatch")
    del pending[name];continue
   offset=part.stat().st_size if part.exists() else 0
   if offset>e["size"]:raise ValueError("partial too large")
   if offset==e["size"]:
    os.replace(part,p);p.chmod(0o444);del pending[name];continue
   request=urllib.request.Request(a.endpoint.rstrip("/")+"/"+name,headers={"Range":f"bytes={offset}-"})
   try:
    with urllib.request.urlopen(request,timeout=120) as response:
     if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {offset}-{e['size']-1}/{e['size']}":
      raise ValueError("range response mismatch")
     print("FETCH",name,offset,flush=True)
     with part.open("ab" if offset else "wb") as out:
      for block in iter(lambda:response.read(8*1024**2),b""):
       out.write(block);offset+=len(block)
       if offset>e["size"]:raise ValueError("oversized response")
      out.flush();os.fsync(out.fileno())
     if offset!=e["size"]:raise OSError("early EOF; retained")
    os.replace(part,p);p.chmod(0o444)
    print("COPIED_UNQUALIFIED",name,offset,flush=True)
    del pending[name];progress=True
   except urllib.error.HTTPError as error:
    if error.code!=404:raise
   except (OSError,TimeoutError,http.client.HTTPException) as error:print("RETRY",name,type(error).__name__,flush=True)
  if pending and not progress:time.sleep(15)
 print("SNAPSHOT_TRANSPORT_COMPLETE",flush=True)
if __name__=="__main__":main()
