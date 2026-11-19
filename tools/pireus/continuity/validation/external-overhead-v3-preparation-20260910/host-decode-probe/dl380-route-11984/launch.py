import pathlib,subprocess,os,json
p=pathlib.Path('/workspace/.cache/pireus-continuity/dl380-transport-check-20260910')
env=os.environ.copy();env.update(SLURM_CONF='/tmp/slurm-direct.conf',TMPDIR='/tmp')
with (p/'launch.log').open('xb') as f:
 r=subprocess.run(json.loads((p/'command.json').read_text()),env=env,cwd='/tmp',stdout=f,stderr=subprocess.STDOUT)
(p/'exit-code').write_text(str(r.returncode)+'\n')
