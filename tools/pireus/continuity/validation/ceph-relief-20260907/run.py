import json,subprocess,pathlib,time,datetime
out=pathlib.Path('/var/tmp/pireus-ceph-relief-20260907')
def ceph(*args):
 return json.loads(subprocess.check_output(['ceph',*args,'--format','json'],timeout=45))
def save(name,data):
 (out/name).write_text(json.dumps(data,indent=2))
m=ceph('osd','dump')
pg=ceph('pg','ls-by-osd','11')['pg_stats']
p=next(p for p in pg if p['pgid']=='5.b')
assert p['state']=='active+clean' and p['up']==[0,11,20] and p['acting']==p['up']
assert not any(x['pgid']=='5.b' for x in m.get('pg_upmap_items',[])+m.get('pg_upmap',[]))
df=ceph('osd','df')
dest=next(x for x in df['nodes'] if x['id']==23)
assert dest['kb_avail']*1024 > 3*p['stat_sum']['num_bytes']
assert dest['utilization'] < 60
bal=ceph('balancer','status')
save('before-map.json',m);save('before-df.json',df);save('before-pg.json',p);save('before-balancer.json',bal)
save('before-health.json',ceph('health','detail'))
print('BEGIN',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
try:
 if bal['active']: subprocess.run(['ceph','balancer','off'],check=True,timeout=45)
 subprocess.run(['ceph','osd','pg-upmap-items','5.b','11','23'],check=True,timeout=45)
 for i in range(120):
  d=ceph('pg','ls-by-pool','rbd_ssd')
  current=next(x for x in d['pg_stats'] if x['pgid']=='5.b')
  summary={k:current[k] for k in ['pgid','state','up','acting']}
  summary['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(summary)+'\n')
  print(json.dumps(summary),flush=True)
  if current['state']=='active+clean' and current['up']==[0,23,20] and current['acting']==current['up']:
   save('after-df.json',ceph('osd','df'));save('after-health.json',ceph('health','detail'))
   print('REMAP_COMPLETE',flush=True)
   break
  if any(x in current['state'] for x in ['incomplete','inconsistent','down']):
   raise RuntimeError('Unhealthy PG; retain evidence and investigate')
  time.sleep(15)
 else: raise RuntimeError('Observation timeout; remap may still be progressing')
finally:
 if bal['active']: subprocess.run(['ceph','balancer','on'],check=True,timeout=45)
