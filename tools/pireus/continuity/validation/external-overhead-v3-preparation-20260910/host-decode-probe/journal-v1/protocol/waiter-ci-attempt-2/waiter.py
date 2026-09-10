#!/usr/bin/env python3
import hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path('/workspace/.cache/pireus-continuity/host-decode-journal-v1-ci-recovery-20260910')
DRIVER_ROOT=pathlib.Path('/workspace/.cache/pireus-continuity/host-decode-journal-v1-run-20260910')
REPO=pathlib.Path('/workspace/.wt/pireus-integration-20260906')
SOURCE='342b3f4e36c78c48d70b3ca27b7ef69eb659001e'
RUN=34443820184
PINS={'driver.py': '094452e106c4cc46a7c854ad77c1bdd489fc4c18cc53771ffd2926f24a3c37c3', 'contract.json': '6f5eef0a56b8840a416f11ab7c30698bbec6664efc250c8acb6e6059dde19302'}
def decision(run):
    if run['id']!=RUN or run['head_sha']!=SOURCE or run['run_attempt']!=2:
        raise ValueError('CI identity/attempt changed')
    if run['status']!='completed':return 'WAIT_SOURCE_CI'
    return 'READY_TO_CHECK_DRIVER' if run['conclusion']=='success' else 'STOP_SOURCE_CI'
PRIOR_TERMINAL_SHA='a2c6ac1a1ce976718f2450e51452478476e1afd7f1559c62771639afe8e77b83'
def verify_prior():
    raw=(DRIVER_ROOT/'source-ci-terminal.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PRIOR_TERMINAL_SHA:raise ValueError('prior terminal changed')
    prior=json.loads(raw)
    if prior['id']!=RUN or prior['head_sha']!=SOURCE or prior['run_attempt']!=1 or prior['status']!='completed' or prior['conclusion']!='cancelled':
        raise ValueError('prior terminal identity')
    if (DRIVER_ROOT/'waiter-driver-invocation.json').exists() or (DRIVER_ROOT/'attempt-entered.json').exists():
        raise ValueError('prior driver already invoked')

def record(value):
    value=dict(value,observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    print(json.dumps(value),flush=True)
    (ROOT/'waiter-status.json').write_text(json.dumps(value,indent=2)+'\n')
def main():
    if not __debug__ or not os.environ.get('TMUX'):raise ValueError('checked Python and tmux required')
    with (ROOT/'waiter-entered.json').open('x') as f:
        json.dump({'run':RUN,'run_attempt':2,'source':SOURCE,'maximum_wait_seconds':7200,'automatic_model_retry':False},f)
    verify_prior()
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
        record({'state':state,'run':RUN,'run_attempt':2,'source':SOURCE})
        if state=='STOP_SOURCE_CI':
            (ROOT/'source-ci-terminal.json').write_bytes(raw)
            return 1
        if state=='READY_TO_CHECK_DRIVER':
            verify_prior()
            (ROOT/'source-ci-terminal.json').write_bytes(raw)
            for name,pin in PINS.items():
                if hashlib.sha256((DRIVER_ROOT/name).read_bytes()).hexdigest()!=pin:raise ValueError('driver identity changed')
            with (ROOT/'waiter-driver-invocation.json').open('x') as f:
                json.dump({'source':SOURCE,'run':RUN,'run_attempt':2,'driver_pins':PINS,'maximum_invocations':1},f)
            with (ROOT/'driver.log').open('xb') as log:
                rc=subprocess.call([sys.executable,'-B',str(DRIVER_ROOT/'driver.py'),'--execute'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
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
