import json,pathlib,subprocess,time,datetime
root=pathlib.Path('/workspace/.wt/pireus-integration-20260906')
out=root/'tools/pireus/continuity/validation/relocation-storage-canary-20260907'
out.mkdir(parents=True,exist_ok=True)
def k(*args):
 return subprocess.check_output(['kubectl',*args],text=True,timeout=40)
host=['-n','beagle','exec','node-ephemeral-governance-kp96t','-c','governor','--','nsenter','-t','1','-m','-p','--']
def ceph(*args): return json.loads(k(*host,'ceph',*args,'--format','json'))
health=ceph('health','detail')
allowed={'BLUESTORE_SLOW_OP_ALERT','DB_DEVICE_STALLED_READ_ALERT'}
assert set(health.get('checks',{})) <= allowed, 'Unresolved capacity or other health condition'
pg=ceph('pg','5.b','query')
assert pg['state']=='active+clean' and pg['up']==[0,23,20] and pg['acting']==pg['up']
df=ceph('osd','df')
for osd,limit in [(11,80),(23,75)]:
 assert next(x for x in df['nodes'] if x['id']==osd)['utilization'] < limit
io_script="""import pathlib,json,subprocess,gzip
base=json.loads(pathlib.Path('/var/tmp/pireus-ceph-relief-20260907/io-observation.json').read_text())
for d in base['devices']:
 i=d['osd']
 log=pathlib.Path('/var/log/ceph/ceph-osd.'+str(i)+'.log')
 lines=log.read_text(errors='replace').splitlines()+gzip.open(str(log)+'.1.gz','rt',errors='replace').read().splitlines()
 slow=[l for l in lines if 'slow operation observed' in l]
 stall=[l for l in lines if 'stalled read' in l]
 assert len(slow)==d['slow_count_current_log'] and len(stall)==d['stalled_count_current_log'], 'New IO event or log rotation requires investigation'
 s=json.loads(subprocess.check_output(['smartctl','-a','-j','/dev/nvme'+str(i)+'n1']))
 assert s['smart_status']['passed'] and s['nvme_smart_health_information_log']['critical_warning']==0 and s['nvme_smart_health_information_log']['media_errors']==0
print(json.dumps({'unchanged_io_counters':True,'log_rotation_accounted_for':True,'smart_passed':True,'baseline':base}))
"""
io=json.loads(k(*host,'python3','-c',io_script))
sc=json.loads(k('get','storageclass','ceph-rbd-ssd-checkpoints','-o','json'))
assert sc['reclaimPolicy']=='Retain' and sc['parameters']['pool']=='rbd_ssd'
pre={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'health':health,'io':io,'scope':'New isolated volume qualification only; no production cutover acceptance'}
(out/'precondition.json').write_text(json.dumps(pre,indent=2))
pvc={'apiVersion':'v1','kind':'PersistentVolumeClaim','metadata':{'name':'pireus-pg-relocation-data','namespace':'beagle'},'spec':{'accessModes':['ReadWriteOnce'],'storageClassName':'ceph-rbd-ssd-checkpoints','resources':{'requests':{'storage':'64Gi'}}}}
subprocess.run(['kubectl','create','-f','-'],input=json.dumps(pvc),text=True,check=True,timeout=40)
for i in range(120):
 p=json.loads(k('-n','beagle','get','pvc','pireus-pg-relocation-data','-o','json'))
 if p['status'].get('phase')=='Bound': break
 time.sleep(2)
assert p['status']['phase']=='Bound'
pv=json.loads(k('get','pv',p['spec']['volumeName'],'-o','json'))
assert pv['spec']['persistentVolumeReclaimPolicy']=='Retain'
(out/'volume.json').write_text(json.dumps({'pvc_uid':p['metadata']['uid'],'pv_uid':pv['metadata']['uid'],'pv_name':pv['metadata']['name'],'class':p['spec']['storageClassName'],'capacity':p['status']['capacity'],'reclaim':pv['spec']['persistentVolumeReclaimPolicy']},indent=2))
for stem,file in [('write','relocation_storage_job.json'),('read','relocation_storage_reader_job.json')]:
 d=json.loads((root/'tools/pireus/continuity/runtime'/file).read_text());name=d['metadata']['name']
 subprocess.run(['kubectl','create','-f','-'],input=json.dumps(d),text=True,check=True,timeout=40)
 pod=None
 for tick in range(180):
  pods=json.loads(k('-n','beagle','get','pods','-l','job-name='+name,'-o','json'))['items']
  if pods:
   pod=pods[0];statuses=pod['status'].get('containerStatuses',[])
   if statuses and ('running' in statuses[0]['state'] or 'terminated' in statuses[0]['state']): break
  time.sleep(2)
 assert pod is not None
 pname=pod['metadata']['name']
 with (out/(stem+'.log')).open('w') as f:
  subprocess.run(['kubectl','-n','beagle','logs','-f',pname],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
 for tick in range(30):
  pod=json.loads(k('-n','beagle','get','pod',pname,'-o','json'))
  if pod['status']['phase'] in ['Succeeded','Failed']: break
  time.sleep(1)
 report={'name':name,'pod_uid':pod['metadata']['uid'],'node':pod['spec']['nodeName'],'phase':pod['status']['phase'],'containers':pod['status']['containerStatuses']}
 (out/(stem+'.json')).write_text(json.dumps(report,indent=2))
 assert report['phase']=='Succeeded', 'Storage canary failed'
 print(name,report['phase'],flush=True)
end_io=json.loads(k(*host,'python3','-c',io_script))
(out/'postcondition.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'health':ceph('health','detail'),'io':end_io,'result':'ISOLATED_VOLUME_IO_PROBE_COMPLETE','production_cutover_accepted':False},indent=2))
print('ISOLATED_VOLUME_IO_PROBE_COMPLETE',flush=True)
