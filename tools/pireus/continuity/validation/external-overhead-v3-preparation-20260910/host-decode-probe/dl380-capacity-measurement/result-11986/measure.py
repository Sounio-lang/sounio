import hashlib,json,os,pathlib,resource,signal,socket,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_bytes())
for name,pin in manifest['files_sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=pin:raise ValueError('file changed: '+name)
def read(path):
    try:return pathlib.Path(path).read_text()
    except OSError as e:return {'error':str(e)}
def limits():
    resource.setrlimit(resource.RLIMIT_AS,(32*1024**3,32*1024**3))
    resource.setrlimit(resource.RLIMIT_STACK,(512*1024**2,512*1024**2))
start=dict(schema='pireus-raw-capacity-measurement-v1',job=os.environ['SLURM_JOB_ID'],
           hostname=socket.gethostname(),boot_id=read('/proc/sys/kernel/random/boot_id'),
           timeout_seconds=600,vmem_bytes=32*1024**3,stack_bytes=512*1024**2,
           manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),
           host_meminfo=read('/proc/meminfo'),host_loadavg=read('/proc/loadavg'),
           ci_equivalence_claimed=False,pireus_qualified=False)
with (ROOT/'start.json').open('x') as f:json.dump(start,f,indent=2)
print(json.dumps(start),flush=True)
began=time.monotonic();timed_out=False
with (ROOT/'compile.log').open('xb') as log, (ROOT/'samples.jsonl').open('x') as samples:
    p=subprocess.Popen([str(ROOT/'madaros'),str(ROOT/'boundary_main.sio'),'-o',str(ROOT/'boundary.elf')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
    while p.poll() is None:
        now=time.monotonic()
        row=dict(elapsed_seconds=now-began,pid=p.pid,status=read(f'/proc/{p.pid}/status'),
                 process_stat=read(f'/proc/{p.pid}/stat'),host_loadavg=read('/proc/loadavg'),
                 memory_pressure=read('/proc/pressure/memory'),cpu_pressure=read('/proc/pressure/cpu'),
                 compile_log_bytes=(ROOT/'compile.log').stat().st_size)
        samples.write(json.dumps(row)+'\n');samples.flush()
        if now-began>=600:
            timed_out=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            break
        time.sleep(1)
    rc=p.wait()
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(job=start['job'],compiler_rc=rc,timed_out=timed_out,compile_seconds=time.monotonic()-began,
            user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,maxrss_kib=usage.ru_maxrss,
            minor_faults=usage.ru_minflt,major_faults=usage.ru_majflt,voluntary_switches=usage.ru_nvcsw,
            involuntary_switches=usage.ru_nivcsw,artifact_exists=(ROOT/'boundary.elf').exists(),
            runtime_rc=None,ci_qualified=False,pireus_qualified=False)
if rc==0 and result['artifact_exists']:
    (ROOT/'boundary.elf').chmod(0o755)
    with (ROOT/'run.log').open('xb') as log:
        run=subprocess.run(['timeout','10',str(ROOT/'boundary.elf')],stdout=log,stderr=subprocess.STDOUT)
    result['runtime_rc']=run.returncode
    result['artifact_sha256']=hashlib.sha256((ROOT/'boundary.elf').read_bytes()).hexdigest()
result['raw_witness_pass']=rc==0 and result['runtime_rc']==7
with (ROOT/'result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
