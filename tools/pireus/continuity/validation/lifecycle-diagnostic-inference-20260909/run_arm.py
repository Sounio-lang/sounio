#!/usr/bin/env python3
import subprocess,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
arm=sys.argv[1]
if arm not in ("without-feedback","with-feedback"): raise SystemExit("unknown arm")
stage=root/arm/"generation"
stage.mkdir(exist_ok=False)
with (stage/"launch.log").open("xb") as out:
 rc=subprocess.call([sys.executable,str(root/"runtime/launch_pair.py"),"offline-generate","--minutes","90","--input-bundle",str(root/arm/"offline-bundle.json")],stdout=out,stderr=subprocess.STDOUT)
(stage/"exit-code").write_text(str(rc)+"\n")
raise SystemExit(rc)
