import json,subprocess
from pathlib import Path
p=Path('/workspace/.cache/pireus-continuity/madaros-capacity-diagnostic-20260910')
with (p/'launch.log').open('xb') as log:
 r=subprocess.run(json.loads((p/'launch-command.json').read_bytes()),stdout=log,stderr=subprocess.STDOUT)
(p/'exit-code').write_text(str(r.returncode)+'\n')
