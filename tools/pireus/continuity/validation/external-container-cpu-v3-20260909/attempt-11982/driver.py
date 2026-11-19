import pathlib,json,hashlib,os,subprocess,sys
sys.dont_write_bytecode=True
base=pathlib.Path('/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity')
root=pathlib.Path('/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity/validation/external-container-cpu-v3-20260909/freeze-source-7617d6ff')
d=pathlib.Path('/workspace/.cache/pireus-continuity/cpu-v3-source-7617d6ff-20260910')
c=json.loads((d/'contract.json').read_bytes())
try:
 assert os.environ.get('TMUX')
 assert hashlib.sha256(pathlib.Path(c['slurm_conf']).read_bytes()).hexdigest()==c['slurm_conf_sha256']
 assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==c['protocol_sha256']
 for n,h in c['orchestration_sha256'].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h
 os.environ['SLURM_CONF']=c['slurm_conf']
 sys.path.insert(0,str(base/'ops'))
 import cpu_v3_attempt as a
 rc=a.launch(root,d/'attempt')
except Exception as e:
 print(type(e).__name__+': '+str(e),flush=True)
 rc=1
(d/'driver-exit-code').write_text(str(rc)+'\n')
raise SystemExit(rc)
