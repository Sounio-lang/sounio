#!/usr/bin/env python3
"""Verify frozen Hub bytes on a Spark before authorizing this snapshot to load."""
import argparse, hashlib, json, os
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("snapshot");ap.add_argument("manifest");ap.add_argument("--receipt",type=Path,required=True);args=ap.parse_args()
 root=Path(args.snapshot);manifest=Path(args.manifest);doc=json.loads(manifest.read_text())
 entries=doc if isinstance(doc,list) else doc["siblings"] if "siblings" in doc else doc["files"]
 results=[]
 for entry in entries:
  name=entry["rfilename"];p=root/name
  if p.stat().st_size!=entry["size"]:raise ValueError("size: "+name)
  sha=hashlib.sha256()
  git=hashlib.sha1(b"blob "+str(entry["size"]).encode()+b"\0")
  with p.open("rb") as f:
   for chunk in iter(lambda:f.read(8*1024**2),b""):sha.update(chunk);git.update(chunk)
  expected=entry.get("lfs",{}).get("sha256")
  if expected:
   if sha.hexdigest()!=expected:raise ValueError("sha256: "+name)
  elif git.hexdigest()!=entry["blobId"]:raise ValueError("git blob: "+name)
  results.append(dict(file=name,size=entry["size"],sha256=sha.hexdigest()))
  print("VERIFIED",name,sha.hexdigest(),flush=True)
 receipt=dict(schema=1,revision=root.name,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),files=results)
 temporary=args.receipt.with_name(args.receipt.name+".partial")
 temporary.write_text(json.dumps(receipt,indent=2)+"\n")
 with temporary.open("rb") as f:os.fsync(f.fileno())
 os.replace(temporary,args.receipt)
 print("MODEL_QUALIFICATION_PASS",flush=True)
if __name__=="__main__":main()
