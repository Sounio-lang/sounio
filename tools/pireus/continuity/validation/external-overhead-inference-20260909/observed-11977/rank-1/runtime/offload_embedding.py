#!/usr/bin/env python3
"""Lossless, offline-only file-backed CPU input embedding; stock TP mask/reduce."""
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

class FileEmbedding:
    def __init__(self, path, shape):
        self.file = path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)
        self.weight = torch.frombuffer(self.mapping, dtype=torch.bfloat16).reshape(shape)

    def gather(self, ids):
        # index_select produces owned CPU output: CUDA never reads the mmap itself.
        cpu = F.embedding(ids.cpu(), self.weight)
        self.mapping.madvise(mmap.MADV_DONTNEED)
        os.posix_fadvise(self.file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return cpu.to(ids.device, non_blocking=False)

@torch.no_grad()
def offload(module, directory):
    if (type(module).__name__ != "VocabParallelEmbedding"
        or type(module.quant_method).__name__ != "UnquantizedEmbeddingMethod"
        or not module.weight.is_cuda or module.weight.dtype != torch.bfloat16
        or module.weight.ndim != 2 or not module.weight.is_contiguous()
        or module.weight.shape[1] * module.weight.element_size() > BLOCK_BYTES):
        raise ValueError("unsupported input embedding offload")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shape = tuple(module.weight.shape)
    size = module.weight.numel() * module.weight.element_size()
    if shutil.disk_usage(directory).free < 2 * size:
        raise ValueError("insufficient local storage for owned embedding copy")
    path = directory / ("embedding-" + os.environ["SLURM_JOB_ID"] + "-" + os.environ["PIREUS_RANK"] + ".bf16")
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
                raise OSError("short embedding write")
            f.flush()
            os.fdatasync(f.fileno())
            os.posix_fadvise(f.fileno(), f.tell()-written, written, os.POSIX_FADV_DONTNEED)
            del raw, cpu
    expected = source_hash.hexdigest()
    if path.stat().st_size != size or file_digest(path) != expected:
        raise ValueError("embedding storage bytes differ from GPU source")
    path.chmod(0o444)
    backing = FileEmbedding(path, shape)
    old = module.weight
    replacement = torch.nn.Parameter(backing.weight, requires_grad=False)
    replacement.__dict__.update(old.__dict__)
    module.weight = replacement
    def embedding(method, layer, ids):
        if layer is not module or layer.weight.data_ptr() != backing.weight.data_ptr():
            raise ValueError("embedding backing identity changed")
        return backing.gather(ids)
    module.quant_method.embedding = types.MethodType(embedding, module.quant_method)
    module._pireus_file_embedding = backing
    del old
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    if before - after < size:
        raise ValueError("GPU embedding storage was retained or aliased")
    return dict(schema=1, job=os.environ["SLURM_JOB_ID"], rank=os.environ["PIREUS_RANK"],
                placement="file-backed-cpu", shape=list(shape), dtype="torch.bfloat16",
                bytes=size, source_gpu_sha256=expected, file_sha256=file_digest(path),
                cuda_allocated_before=before, cuda_allocated_after=after,
                copy_block_bytes=BLOCK_BYTES, checkpoint_precision_changed=False)
