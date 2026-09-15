#!/usr/bin/env python3
"""Temporary internal HTTP transport exposing only completed pinned snapshot files."""
import argparse, http.server, json, re, shutil, urllib.parse
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--bind",required=True);ap.add_argument("--port",type=int,default=18091);a=ap.parse_args()
 files={e["rfilename"]:e["size"] for e in json.loads(a.manifest.read_text())}
 class Handler(http.server.BaseHTTPRequestHandler):
  def do_GET(self):
   name=urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).lstrip("/")
   p=a.root/name
   if name not in files or not p.is_file() or p.stat().st_size!=files[name]:
    self.send_error(404);return
   start=0;end=files[name]-1
   value=self.headers.get("Range")
   if value:
    match=re.fullmatch(r"bytes=(\d+)-(\d*)",value)
    if not match:self.send_error(416);return
    start=int(match[1])
    if match[2]:end=int(match[2])
    if start>end or end>=files[name]:self.send_error(416);return
   self.send_response(206 if value else 200)
   self.send_header("Content-Length",str(end-start+1))
   self.send_header("Accept-Ranges","bytes")
   if value:self.send_header("Content-Range",f"bytes {start}-{end}/{files[name]}")
   self.end_headers()
   try:
    with p.open("rb") as f:
     f.seek(start);remaining=end-start+1
     while remaining:
      block=f.read(min(8*1024**2,remaining))
      if not block:break
      self.wfile.write(block);remaining-=len(block)
   except (BrokenPipeError,ConnectionResetError):pass
 http.server.ThreadingHTTPServer((a.bind,a.port),Handler).serve_forever()
if __name__=="__main__":main()
