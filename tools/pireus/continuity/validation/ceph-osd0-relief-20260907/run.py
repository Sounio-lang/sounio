import json,pathlib,subprocess,time,os
os.umask(0o077)
out=pathlib.Path('/var/tmp/pireus-ceph-osd0-relief-20260907')
out.mkdir(mode=0o700,exist_ok=True)
def ceph(*args):
 return json.loads(subprocess.check_output(['ceph',*args,'--format','json'],timeout=30))
def command(*args):
 return subprocess.check_output(['ceph',*args],text=True,timeout=30)
def save(name,data):(out/name).write_text(json.dumps(data,indent=2))
pg=ceph('pg','5.b','query')
assert pg['state']=='active+clean' and pg['up']==pg['acting']==[0,23,20]
before_map=next(x for x in ceph('osd','dump')['pg_upmap_items'] if x['pgid']=='5.b')
assert before_map['mappings']==[{'from':11,'to':23}]
health=ceph('health','detail')
assert set(health.get('checks',{})) <= {'BLUESTORE_SLOW_OP_ALERT','DB_DEVICE_STALLED_READ_ALERT'}
df=ceph('osd','df')
byid={x['id']:x for x in df['nodes']}
assert byid[1]['utilization']<60 and byid[1]['kb_avail']*1024>2*pg['info']['stats']['stat_sum']['num_bytes']
assert byid[0]['device_class']==byid[1]['device_class']=='ssd'
locations=[ceph('osd','find',str(i))['crush_location']['host'] for i in [0,1,23,20]]
assert locations[0]==locations[1]=='t560-proxmox' and len(set(locations[1:]))==3
assert command('config','show','osd.0','bdev_enable_discard').strip()=='true'
assert command('config','show','osd.0','bdev_async_discard_threads').strip()=='1'
balancer=ceph('balancer','status')
save('before.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pg':pg,'df':df,'health':health,'mapping':before_map,'balancer':balancer,'locations':locations})
try:
 if balancer['active']:command('balancer','off')
 command('osd','pg-upmap-items','5.b','11','23','0','1')
 started=time.monotonic()
 while time.monotonic()-started<3600:
  current=ceph('pg','5.b','query')
  stat=current['info']['stats']['stat_sum']
  record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'elapsed_seconds':time.monotonic()-started,
          'state':current['state'],'up':current['up'],'acting':current['acting'],
          'num_objects_degraded':stat.get('num_objects_degraded'),'num_objects_unfound':stat.get('num_objects_unfound'),
          'num_objects_misplaced':stat.get('num_objects_misplaced')}
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
  if current['state']=='active+clean' and current['up']==current['acting']==[1,23,20]:
   save('after.json',{'utc':record['utc'],'pg':current,'df':ceph('osd','df'),'health':ceph('health','detail'),'latency_acceptance':False})
   print('REPLICA_RELOCATION_COMPLETE_LATENCY_GATE_OPEN',flush=True)
   break
  if stat.get('num_objects_unfound',0):
   raise RuntimeError('Unfound objects observed; retain active recovery mapping and investigate')
  time.sleep(15)
 else:raise RuntimeError('Observation deadline; recovery mapping retained, not reversed')
finally:
 if balancer['active']:command('balancer','on')
 save('balancer-final.json',ceph('balancer','status'))
