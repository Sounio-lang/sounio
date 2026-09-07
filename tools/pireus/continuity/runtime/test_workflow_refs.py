#!/usr/bin/env python3
"""Workflow reference checker controls, including frozen path-filter globs."""
import subprocess,tempfile,sys
from pathlib import Path
def main():
 source=Path(sys.argv[1]).read_bytes()
 cases=[("scripts/ci/gate*",True,True,0),("scripts/ci/missing*",True,True,1),
        ("scripts/ci/gate.sh",True,True,0),("scripts/ci/missing.sh",True,True,1),
        ("scripts/ci/gate*",True,False,1),("scripts/**/*.sh",True,True,0)]
 for reference,exists,executable,expected in cases:
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);checker=root/"scripts/dev/check_workflow_script_refs.sh"
   checker.parent.mkdir(parents=True);checker.write_bytes(source);checker.chmod(0o755)
   target=root/"scripts/ci/gate.sh";target.parent.mkdir(parents=True)
   if exists:target.write_text("#!/bin/sh\nexit 0\n");target.chmod(0o755 if executable else 0o644)
   workflows=root/".github/workflows";workflows.mkdir(parents=True)
   (workflows/"test.yml").write_text("on:\n  push:\n    paths:\n      - '"+reference+"'\n")
   run=subprocess.run(["bash",str(checker)],capture_output=True,text=True,timeout=10)
   assert run.returncode==expected,(reference,run.returncode,run.stdout,run.stderr)
 print("PIREUS_WORKFLOW_REF_GLOBS_PASS existing_glob=1 missing_glob_refused=1 literals_preserved=1 nonexecutable_refused=1 recursive_glob=1")
if __name__=="__main__":main()
