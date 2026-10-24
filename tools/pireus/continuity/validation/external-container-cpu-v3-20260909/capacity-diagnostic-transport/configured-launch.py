import json,os,subprocess
from pathlib import Path
p=Path('/workspace/.cache/pireus-continuity/madaros-capacity-diagnostic-20260910')
env=os.environ.copy();env['TMPDIR']='/tmp';env['SLURM_CONF']='/tmp/slurm-direct.conf'
with (p/'configured-launch.log').open('xb') as log:
 r=subprocess.run(json.loads((p/'configured-launch-command.json').read_bytes()),cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT)
(p/'configured-exit-code').write_text(str(r.returncode)+'\n')
