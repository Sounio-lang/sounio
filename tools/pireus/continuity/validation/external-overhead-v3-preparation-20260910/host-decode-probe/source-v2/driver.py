#!/usr/bin/env python3
"""Pinned host diagnostic entrypoint. No waiting loop and no automatic retry."""
import hashlib,json,os,pathlib,sys
sys.dont_write_bytecode=True
BASE=pathlib.Path('/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity')
ROOT=pathlib.Path('/workspace/.cache/pireus-continuity/host-decode-diagnostic-source-v2-run-20260910')
FROZEN=pathlib.Path('/workspace/.cache/pireus-continuity/host-decode-diagnostic-source-v2-freeze-20260910')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def main():
    require(__debug__,'checked Python required')
    require(sys.argv[1:] in (['--check'],['--execute']),'choose --check or --execute')
    contract=json.loads((ROOT/'contract.json').read_bytes())
    require(digest(pathlib.Path(__file__))==contract['driver_sha256'],'driver changed')
    require(digest(pathlib.Path(contract['slurm_conf']))==contract['slurm_conf_sha256'],'Slurm configuration changed')
    require(digest(FROZEN/'execution-freeze.json')==contract['freeze_sha256'],'freeze changed')
    for name,pin in contract['orchestration_sha256'].items():
        require(digest(BASE/name)==pin,'orchestration changed: '+name)
    os.environ.update(SLURM_CONF=contract['slurm_conf'],PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,str(BASE/'ops'))
    from host_decode_attempt_v2 import context
    attempt=context()
    spec=attempt.verify(FROZEN)
    require(spec['source_commit']==contract['source_commit'],'source identity changed')
    if sys.argv[1:] == ['--check']:
        print(json.dumps({'packet_verified':True,'source_commit':spec['source_commit'],
                          'source_ci_checked':False,'hardware_submitted':False}))
        return 0
    require(os.environ.get('TMUX'),'remote tmux required')
    require(not (ROOT/'attempt-entered.json').exists(),'attempt already entered; no retry')
    ready=attempt.readiness(FROZEN)
    with (ROOT/'attempt-entered.json').open('x') as f:
        json.dump({'readiness':ready,'automatic_retry':False},f,indent=2)
        f.write('\n')
    rc=1
    try:
        rc=attempt.launch(FROZEN,'baseline',ROOT/'stage')
    except Exception as e:
        print(type(e).__name__+': '+str(e),flush=True)
    finally:
        with (ROOT/'driver-exit-code').open('x') as f:f.write(str(rc)+'\n')
    return rc
if __name__=='__main__':
    raise SystemExit(main())
