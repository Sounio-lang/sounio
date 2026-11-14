import pathlib,json,hashlib,os,sys
sys.dont_write_bytecode=True
base=pathlib.Path('/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity')
frozen=pathlib.Path('/workspace/.cache/pireus-continuity/external-overhead-v3-freeze-20260910')
root=pathlib.Path('/workspace/.cache/pireus-continuity/external-overhead-v3-run-20260910')
contract=json.loads((root/'contract.json').read_bytes())
try:
 if not __debug__ or not os.environ.get('TMUX'):raise ValueError('tmux and checked Python required')
 if hashlib.sha256(pathlib.Path(contract['slurm_conf']).read_bytes()).hexdigest()!=contract['slurm_conf_sha256']:raise ValueError('Slurm configuration changed')
 if hashlib.sha256((frozen/'execution-freeze.json').read_bytes()).hexdigest()!=contract['freeze_sha256']:raise ValueError('freeze changed')
 for name,pin in contract['orchestration_sha256'].items():
  if hashlib.sha256((base/name).read_bytes()).hexdigest()!=pin:raise ValueError('orchestration changed: '+name)
 os.environ['SLURM_CONF']=contract['slurm_conf']
 os.environ['PYTHONOPTIMIZE']='0'
 os.environ['PYTHONDONTWRITEBYTECODE']='1'
 sys.path.insert(0,str(base/'ops'))
 import launch_external_overhead_v3 as launcher
 rc=launcher.launch(frozen,'baseline',root/'baseline-stage')
except Exception as e:
 print(type(e).__name__+': '+str(e),flush=True)
 rc=1
(root/'baseline-driver-exit-code').write_text(str(rc)+'\n')
raise SystemExit(rc)
