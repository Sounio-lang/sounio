#!/usr/bin/env python3
import hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path('/workspace/.cache/pireus-continuity/host-decode-journal-v1-run-20260910')
REPO=pathlib.Path('/workspace/.wt/pireus-integration-20260906')
SOURCE='342b3f4e36c78c48d70b3ca27b7ef69eb659001e'
RUN=34443820184
PINS={'driver.py': '094452e106c4cc46a7c854ad77c1bdd489fc4c18cc53771ffd2926f24a3c37c3', 'contract.json': '6f5eef0a56b8840a416f11ab7c30698bbec6664efc250c8acb6e6059dde19302'}
def decision(run):
    if run['id']!=RUN or run['head_sha']!=SOURCE or run['run_attempt']!=1:
        raise ValueError('CI identity/attempt changed')
    if run['status']!='completed':return 'WAIT_SOURCE_CI'
    return 'READY_TO_CHECK_DRIVER' if run['conclusion']=='success' else 'STOP_SOURCE_CI'
def record(value):
    value=dict(value,observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    print(json.dumps(value),flush=True)
    (ROOT/'waiter-status.json').write_text(json.dumps(value,indent=2)+'\n')
def main():
    if not __debug__ or not os.environ.get('TMUX'):raise ValueError('checked Python and tmux required')
    with (ROOT/'waiter-entered.json').open('x') as f:
        json.dump({'run':RUN,'source':SOURCE,'maximum_wait_seconds':7200,'automatic_model_retry':False},f)
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        try:
            raw=subprocess.check_output(['gh','api',f'repos/Sounio-lang/sounio/actions/runs/{RUN}'],cwd=REPO,timeout=45)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
            record({'state':'OBSERVATION_ERROR','error':type(e).__name__})
            time.sleep(45)
            continue
        run=json.loads(raw)
        state=decision(run)
        record({'state':state,'run':RUN,'source':SOURCE})
        if state=='STOP_SOURCE_CI':
            (ROOT/'source-ci-terminal.json').write_bytes(raw)
            return 1
        if state=='READY_TO_CHECK_DRIVER':
            (ROOT/'source-ci-terminal.json').write_bytes(raw)
            for name,pin in PINS.items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=pin:raise ValueError('driver identity changed')
            with (ROOT/'waiter-driver-invocation.json').open('x') as f:
                json.dump({'source':SOURCE,'run':RUN,'driver_pins':PINS,'maximum_invocations':1},f)
            with (ROOT/'driver.log').open('xb') as log:
                rc=subprocess.call([sys.executable,'-B',str(ROOT/'driver.py'),'--execute'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            record({'state':'DRIVER_TERMINAL','driver_exit_code':rc,'automatic_retry':False})
            return rc
        time.sleep(45)
    record({'state':'STOP_WAIT_DEADLINE','hardware_submitted':False})
    return 1
if __name__=='__main__':
    try:rc=main()
    except Exception as e:
        record({'state':'STOP_WAITER_ERROR','error':type(e).__name__+': '+str(e)})
        rc=1
    raise SystemExit(rc)
