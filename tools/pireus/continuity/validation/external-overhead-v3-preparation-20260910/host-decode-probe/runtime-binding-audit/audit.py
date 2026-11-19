#!/usr/bin/env python3
"""Supplement frozen inspection with in-allocation runtime receipt binding."""
import argparse,hashlib,json
from pathlib import Path
def require(ok,message):
    if not ok:raise ValueError(message)
def validate(receipt,start,job,rank,lifecycle):
    worker=start['workers'][rank]
    require(receipt['stage']=='OVERHEAD_RUNTIME_VERIFIED','runtime stage')
    require(receipt['job']==job and receipt['rank']==str(rank),'runtime job/rank')
    require(receipt['worker_uid']==worker['uid'] and receipt['boot_id']==worker['boot_id'],'runtime worker/boot')
    require(receipt['runtime_sha256']==start['runtime_before_sha256'][rank],'in-allocation runtime hashes')
    require(receipt['input_sha256']==start['input_sha256'],'in-allocation input hash')
    stamp=receipt['monotonic_ns']
    require(type(stamp) is int and stamp>=0,'runtime timestamp')
    for row in lifecycle:
        require(row['job']==job and row['rank']==str(rank),'lifecycle job/rank')
        require(type(row['monotonic_ns']) is int and row['monotonic_ns']>=stamp,'runtime/lifecycle chronology')
    return dict(rank=rank,runtime_binding_verified=True,lifecycle_records=len(lifecycle),
                lifecycle_present=bool(lifecycle))
def audit(root,pin):
    raw=(root/'collection.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==pin,'collection pin')
    collection=json.loads(raw)
    def read(name):
        require(name in collection['files_sha256'],'artifact absent from manifest')
        path=root/name
        require(path.resolve().is_relative_to(root.resolve()),'artifact escapes collection')
        raw=path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==collection['files_sha256'][name],'artifact hash')
        return raw
    start=json.loads(read('start.json'))
    require(len(start['workers'])==2,'worker count')
    ranks=[]
    for rank in range(2):
        receipt=json.loads(read(f'rank-{rank}/runtime-before.json'))
        name=f"worker-receipts/lifecycle-{collection['job']}-{rank}.jsonl"
        if name in collection['files_sha256']:
            lifecycle=[json.loads(line) for line in read(name).splitlines() if line.strip()]
        else:
            require(name in collection['missing'],'unclassified missing lifecycle')
            lifecycle=[]
        ranks.append(validate(receipt,start,collection['job'],rank,lifecycle))
    return dict(schema='pireus-runtime-binding-supplement-v1',job=collection['job'],
                collection_sha256=pin,ranks=ranks,full_custody_qualified=False,
                loaded_model_qualified=False,timing_eligible=False,pilot_acceptance=False)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--pin',required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.collection,a.pin),indent=2))
