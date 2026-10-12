#!/usr/bin/env python3
"""Run the installed CLI outside any checkout, with compiler overrides removed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
witnesses = Path(__file__).parent / 'witnesses'
env = {k:v for k,v in os.environ.items() if not k.startswith(('SOUNIO_', 'MADAROS_', 'SOUC'))}
results = []
with tempfile.TemporaryDirectory(prefix='sounio-external-consumer-') as tmp:
    cwd = Path(tmp)
    env['TMPDIR'] = tmp
    def call(args, expected=0, stdout=None):
        result = subprocess.run([str(a) for a in args], cwd=cwd, env=env,
                                capture_output=True, text=True, timeout=120)
        if result.returncode != expected or (stdout is not None and result.stdout.split() != stdout):
            raise AssertionError(dict(args=list(map(str,args)), rc=result.returncode,
                                      stdout=result.stdout, stderr=result.stderr[-4000:]))
        results.append(dict(command=list(map(str,args)), rc=result.returncode,
                            stdout=result.stdout[-2000:]))
        return result
    call([sys.executable, root/'verify-distribution.py', '--verify', root])
    version = call([root/'bin/souc', '--version'])
    assert 'Madaros' in version.stdout
    for name, rc, output in [('same_name',0,['14','3.000000']),
                             ('multi_call',0,['11','5','6','multi_call','PASS']),
                             ('w1_rmix',23,[]),('w2_g',60,[])]:
        src = cwd/(name+'.sio')
        src.write_text((witnesses/src.name).read_text())
        elf = cwd/(name+'.elf')
        call([root/'bin/souc','check',src])
        call([root/'bin/souc','compile',src,'-o',elf])
        call([elf],rc,output)
    source = cwd/'stdlib.sio'
    source.write_text('use cmp::lib::max_i64\nfn main() -> i32 {\n if max_i64(7, 3) == 7 { 0 } else { 1 }\n}\n')
    call([root/'bin/souc','run',source])
    bad = cwd/'invalid.sio'
    bad.write_text('fn main() -> i32 { missing_distribution_symbol() }\n')
    failed = subprocess.run([str(root/'bin/souc'),'compile',str(bad),'-o',str(cwd/'bad.elf')],
                            cwd=cwd,env=env,capture_output=True,text=True,timeout=120)
    assert failed.returncode != 0 and not (cwd/'bad.elf').exists(), failed.stdout
    results.append(dict(case='invalid_source_refused',rc=failed.returncode))
print(json.dumps(dict(status='PASS',checkout_required=False,results=results),indent=2))
