import json,os,subprocess
from pathlib import Path
p=Path('/workspace/.cache/pireus-continuity/madaros-capacity-diagnostic-20260910')
env=os.environ.copy();env['TMPDIR']='/tmp'
with (p/'direct-launch.log').open('xb') as log:
 r=subprocess.run(json.loads((p/'direct-launch-command.json').read_bytes()),cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT)
(p/'direct-exit-code').write_text(str(r.returncode)+'\n')
