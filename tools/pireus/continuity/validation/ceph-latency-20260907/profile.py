import json,pathlib,subprocess,time,gzip
out=pathlib.Path('/var/tmp/pireus-ceph-latency-20260907')
out.mkdir(mode=0o700,exist_ok=True)
def disk(i):return list(map(int,pathlib.Path('/sys/block/nvme'+str(i)+'n1/stat').read_text().split()))
def events(i):
 p=pathlib.Path('/var/log/ceph/ceph-osd.'+str(i)+'.log')
 lines=p.read_text(errors='replace').splitlines()
 if pathlib.Path(str(p)+'.1.gz').exists():lines+=gzip.open(str(p)+'.1.gz','rt',errors='replace').read().splitlines()
 slow=sorted(l for l in lines if 'slow operation observed' in l)
 stall=sorted(l for l in lines if 'stalled read' in l)
 return {'slow_count':len(slow),'stalled_count':len(stall),'latest_slow':slow[-1].split()[0] if slow else None,'latest_stalled':stall[-1].split()[0] if stall else None}
for n in range(10):
 a={i:disk(i) for i in [0,1]};start=time.monotonic();time.sleep(10);dt=time.monotonic()-start
 record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'devices':[]}
 for i in [0,1]:
  b=disk(i);d=[y-x for x,y in zip(a[i],b)]
  record['devices'].append({'osd':i,'read_MBps':d[2]*512/dt/1e6,'write_MBps':d[6]*512/dt/1e6,'write_iops':d[4]/dt,'write_await_ms':d[7]/max(d[4],1),'inflight':b[8],'mean_queue':d[10]/dt/1000,'discard_completed_total':b[11],'discard_sectors_total':b[13],**events(i)})
 with (out/'profile.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 print(json.dumps(record),flush=True)
 time.sleep(20)
