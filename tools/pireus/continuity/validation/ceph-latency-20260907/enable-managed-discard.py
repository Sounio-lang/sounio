import json,pathlib,subprocess,time,os
os.umask(0o077)
out=pathlib.Path('/var/tmp/pireus-ceph-latency-20260907')
def ceph(*args):
 return subprocess.check_output(['ceph',*args],text=True,timeout=30)
settings={'bdev_async_discard_threads':'1','bdev_enable_discard':'true'}
existing=json.loads(ceph('config','dump','--format','json'))
prior={k:[x for x in existing if x.get('section')=='osd.0' and x.get('name')==k] for k in settings}
for k in settings:
 meta=json.loads(ceph('config','help',k,'--format','json'))
 assert meta['can_update_at_runtime']
 assert not prior[k], 'Existing targeted override requires review'
assert int(pathlib.Path('/sys/block/nvme0n1/queue/discard_max_bytes').read_text())>0
before={k:ceph('config','show','osd.0',k).strip() for k in settings}
assert before=={'bdev_async_discard_threads':'0','bdev_enable_discard':'false'}
receipt={'started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before':before,'prior_explicit_overrides':prior,'changes':settings,'scope':'OSD0 managed async discard of allocator-released extents only','raw_device_discard':False,'daemon_restart':False,'performance_acceptance':False}
(out/'discard-change.json').write_text(json.dumps(receipt,indent=2))
changed=[]
try:
 for k,v in settings.items():
  ceph('config','set','osd.0',k,v);changed.append(k)
 receipt['observed']={k:ceph('config','show','osd.0',k).strip() for k in settings}
 assert receipt['observed']==settings
 receipt['runtime_configuration_applied']=True
except BaseException:
 for k in reversed(changed):ceph('config','rm','osd.0',k)
 receipt['reverted_on_failure']=True
 (out/'discard-change.json').write_text(json.dumps(receipt,indent=2))
 raise
(out/'discard-change.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
