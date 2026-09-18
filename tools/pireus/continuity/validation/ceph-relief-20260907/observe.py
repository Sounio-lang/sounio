import json,pathlib,subprocess,time,datetime
out=pathlib.Path('/var/tmp/pireus-ceph-relief-20260907')
for tick in range(35):
 report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'devices':[]}
 for i in [0,1]:
  lines=pathlib.Path('/var/log/ceph/ceph-osd.'+str(i)+'.log').read_text(errors='replace').splitlines()
  slow=[x for x in lines if 'slow operation observed' in x]
  stalled=[x for x in lines if 'stalled read' in x]
  report['devices'].append({'osd':i,'slow_count':len(slow),'stalled_count':len(stalled),'last_slow_at':slow[-1].split()[0] if slow else None,'last_stalled_at':stalled[-1].split()[0] if stalled else None})
 report['perf']=json.loads(subprocess.check_output(['ceph','osd','perf','--format','json'],timeout=30))
 with (out/'io-monitor.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
 print(report['at'],flush=True)
 if (out/'after-df.json').exists(): break
 time.sleep(60)
