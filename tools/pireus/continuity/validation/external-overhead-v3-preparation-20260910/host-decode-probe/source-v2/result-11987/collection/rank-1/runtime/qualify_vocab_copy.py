#!/usr/bin/env python3
"""GB10 stock/staged BF16 vocabulary sharding and padding controls."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import sglang.srt.layers.vocab_parallel_embedding as vocab
from patch_vocab_copy import patched_source
assert os.environ.get("SLURM_JOB_ID")
assert torch.cuda.get_device_capability() == (12,1)
os.environ["PIREUS_OFFLINE_MODE"]="generate"
raw=Path(vocab.__file__).read_bytes();changed=patched_source(raw)
tree=ast.parse(changed)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="VocabParallelEmbedding")
nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="weight_loader"]
nodes += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_pireus_vocab_copy"]
ns=dict(vars(vocab))
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),"<staged-vocab-copy>","exec"),ns)
reports=[]
for total_rows,hidden,padding in [(12,32,2),(201024,4096,0)]:
    torch.manual_seed(20260907)
    source=torch.randint(0,127,(total_rows,hidden),dtype=torch.int8).to(torch.bfloat16)
    for rank in (0,1):
        rows=total_rows//2
        layer=SimpleNamespace(tp_size=2,org_vocab_size=total_rows,use_presharded_weights=False,
            shard_indices=SimpleNamespace(org_vocab_start_index=rank*rows,org_vocab_end_index=(rank+1)*rows))
        def make():
            p=torch.nn.Parameter(torch.full((rows+padding,hidden),-3.,dtype=torch.bfloat16,device="cuda"),requires_grad=False)
            p.output_dim=0
            return p
        reference=make();candidate=make();pointer=candidate.data_ptr()
        vocab.VocabParallelEmbedding.weight_loader(layer,reference,source)
        ns["weight_loader"](layer,candidate,source)
        torch.cuda.synchronize()
        assert candidate.data_ptr()==pointer
        assert torch.equal(candidate.view(torch.uint8),reference.view(torch.uint8))
        assert torch.equal(candidate[:rows].cpu(),source[rank*rows:(rank+1)*rows])
        if padding:assert torch.count_nonzero(candidate[rows:])==0
        candidate.data.view(torch.uint8)[0,0]^=1
        assert not torch.equal(candidate.view(torch.uint8),reference.view(torch.uint8))
        reports.append(dict(vocab_rows=total_rows,hidden=hidden,tp_rank=rank,padded_rows=padding,
            exact=True,original_storage_retained=True,negative_control=True,staging_limit_bytes=4*1024**2))
        del candidate,reference;torch.cuda.empty_cache()
    del source
print(json.dumps(dict(stage="VOCAB_COPY_GPU_PASS",job=os.environ["SLURM_JOB_ID"],
    rank=os.environ["PIREUS_RANK"],source_sha256=hashlib.sha256(raw).hexdigest(),
    patched_sha256=hashlib.sha256(changed).hexdigest(),controls=reports,full_model_loaded=False)),flush=True)
