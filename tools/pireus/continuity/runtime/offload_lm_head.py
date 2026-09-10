#!/usr/bin/env python3
"""Scoped BF16 file-backed LM-head with4096-row GPU projections; offline M1 only."""
import gc
import hashlib
import mmap
import os
from pathlib import Path
import shutil
import types
import torch
import torch.nn.functional as F

BLOCK_BYTES = 4 * 1024**2

def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(BLOCK_BYTES):
            h.update(block)
            os.posix_fadvise(f.fileno(), f.tell()-len(block), len(block), os.POSIX_FADV_DONTNEED)
    return h.hexdigest()

from offload_embedding import FileEmbedding

class FileProjection(FileEmbedding):
    def project(self, hidden):
        if hidden.ndim != 2 or tuple(hidden.shape) != (1, self.weight.shape[1]) or not hidden.is_cuda:
            raise ValueError("LM-head profile supports one GPU hidden row only")
        values = hidden.to(torch.bfloat16)
        output = torch.empty((1,self.weight.shape[0]), dtype=torch.bfloat16, device=hidden.device)
        for start in range(0,self.weight.shape[0],4096):
            cpu = self.weight[start:start+4096].clone()
            self.mapping.madvise(mmap.MADV_DONTNEED)
            os.posix_fadvise(self.file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            tile = cpu.to(hidden.device, non_blocking=False)
            output[:,start:start+len(cpu)] = torch.matmul(values, tile.T)
            del tile, cpu
        return output

@torch.no_grad()
def offload(module, processor, directory):
    if (type(module).__name__ != "ParallelLMHead"
        or processor.use_fp32_lm_head or processor.rl_on_policy_target is not None
        or module.bias is not None or hasattr(module, "set_lora") or hasattr(module, "apply_lora")
        or type(module.quant_method).__name__ != "UnquantizedEmbeddingMethod"
        or not module.weight.is_cuda or module.weight.dtype != torch.bfloat16
        or module.weight.ndim != 2 or not module.weight.is_contiguous()
        or module.weight.shape[1] * module.weight.element_size() > BLOCK_BYTES):
        raise ValueError("unsupported LM-head offload")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shape = tuple(module.weight.shape)
    size = module.weight.numel() * module.weight.element_size()
    if shutil.disk_usage(directory).free < 2 * size:
        raise ValueError("insufficient local storage for owned LM-head copy")
    path = directory / ("lm-head-" + os.environ["SLURM_JOB_ID"] + "-" + os.environ["PIREUS_RANK"] + ".bf16")
    before = torch.cuda.memory_allocated()
    rows = BLOCK_BYTES // (shape[1] * module.weight.element_size())
    source_hash = hashlib.sha256()
    # Flush/drop each bounded block before continuing: do not accumulate824MB
    # dirty file pages while the original GPU allocation remains resident.
    with path.open("xb") as f:
        for start in range(0, shape[0], rows):
            cpu = module.weight[start:start+rows].cpu()
            raw = memoryview(cpu.view(torch.uint8).numpy())
            source_hash.update(raw)
            written = f.write(raw)
            if written != raw.nbytes:
                raise OSError("short LM-head write")
            f.flush()
            os.fdatasync(f.fileno())
            os.posix_fadvise(f.fileno(), f.tell()-written, written, os.POSIX_FADV_DONTNEED)
            del raw, cpu
    expected = source_hash.hexdigest()
    if path.stat().st_size != size or file_digest(path) != expected:
        raise ValueError("LM-head storage bytes differ from GPU source")
    path.chmod(0o444)
    backing = FileProjection(path, shape)
    old = module.weight
    replacement = torch.nn.Parameter(backing.weight, requires_grad=False)
    replacement.__dict__.update(old.__dict__)
    module.weight = replacement

    def compute(owner, hidden_states, lm_head, embedding_bias=None):
        if (owner is not processor or lm_head is not module or embedding_bias is not None
            or owner.use_fp32_lm_head or owner.rl_on_policy_target is not None
            or hasattr(lm_head, "set_lora") or hasattr(lm_head, "apply_lora")
            or lm_head.weight.data_ptr() != backing.weight.data_ptr()
            or type(lm_head.quant_method).__name__ != "UnquantizedEmbeddingMethod"):
            raise ValueError("unsupported LM-head projection profile")
        return backing.project(hidden_states)
    processor._compute_lm_head = types.MethodType(compute, processor)
    module._pireus_file_projection = backing
    del old
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    if before - after < size:
        raise ValueError("GPU LM-head storage was retained or aliased")
    return dict(schema=1, job=os.environ["SLURM_JOB_ID"], rank=os.environ["PIREUS_RANK"],
                placement="file-backed-gpu-tiles", tile_rows=4096, hidden_rows=1, shape=list(shape), dtype="torch.bfloat16",
                bytes=size, source_gpu_sha256=expected, file_sha256=file_digest(path),
                cuda_allocated_before=before, cuda_allocated_after=after,
                copy_block_bytes=BLOCK_BYTES, checkpoint_precision_changed=False)
