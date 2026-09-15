#!/usr/bin/env python3
"""Stage one verified source overlay without rewriting the pinned SIF."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from patch_marlin_placeholders import patched_source
from patch_marlin_deinterleave import patched_source as patched_deinterleave_source
from patch_marlin_repack import patched_source as patched_repack_source

PATCHED_SHA256 = "2aa3b391d05cbead8c23c2dfc6fac88c425eae74938ce7e1cbae871b4b47a36e"
assert os.environ.get("SLURM_JOB_ID")
root = Path(importlib.util.find_spec("sglang").origin).parent
source = root / "srt/layers/quantization/modelopt_quant.py"
changed = patched_deinterleave_source(patched_source(source.read_bytes()))
assert hashlib.sha256(changed).hexdigest() == PATCHED_SHA256
out = Path("/scratch/pireus/cache") / ("modelopt-" + PATCHED_SHA256 + ".py")
if out.exists():
    assert not out.is_symlink() and out.read_bytes() == changed
else:
    with out.open("xb") as stream:
        stream.write(changed)
        stream.flush()
        os.fsync(stream.fileno())
    out.chmod(0o444)
print(json.dumps(dict(stage="MARLIN_OVERLAY_STAGED",job=os.environ["SLURM_JOB_ID"],
                     rank=os.environ["PIREUS_RANK"],patched_sha256=PATCHED_SHA256,
                     original_sif_modified=False)),flush=True)

# A second content-addressed overlay bounds repack temporaries per expert.
repack_source = root / "srt/layers/quantization/marlin_utils_fp4.py"
repack_changed = patched_repack_source(repack_source.read_bytes())
repack_sha = "2d3207f11b4c7a0023fa508666b447170ff8325e06248b4928f21df77b03fc0f"
assert hashlib.sha256(repack_changed).hexdigest() == repack_sha
repack_out = Path("/scratch/pireus/cache") / ("marlin-utils-" + repack_sha + ".py")
if repack_out.exists():
    assert not repack_out.is_symlink() and repack_out.read_bytes() == repack_changed
else:
    with repack_out.open("xb") as stream:
        stream.write(repack_changed)
        stream.flush()
        os.fsync(stream.fileno())
    repack_out.chmod(0o444)
print(json.dumps(dict(stage="MARLIN_REPACK_OVERLAY_STAGED",job=os.environ["SLURM_JOB_ID"],
                     rank=os.environ["PIREUS_RANK"],patched_sha256=repack_sha,
                     original_sif_modified=False)),flush=True)

# Third overlay bounds offline packed checkpoint copies; original SIF retained.
from patch_checkpoint_copy import patched_source as patched_checkpoint_source
copy_changed = patched_checkpoint_source((root / "srt/layers/moe/fused_moe_triton/layer.py").read_bytes())
copy_sha = "8f97a3bcb419e4a48d6d94245e5747fd918b4ab7e0c785f036099e7ff96e6112"
assert hashlib.sha256(copy_changed).hexdigest() == copy_sha
copy_out = Path("/scratch/pireus/cache") / ("checkpoint-copy-" + copy_sha + ".py")
if copy_out.exists():
    assert not copy_out.is_symlink() and copy_out.read_bytes() == copy_changed
else:
    with copy_out.open("xb") as stream:
        stream.write(copy_changed)
        stream.flush()
        os.fsync(stream.fileno())
    copy_out.chmod(0o444)
print(json.dumps(dict(stage="CHECKPOINT_COPY_OVERLAY_STAGED",job=os.environ["SLURM_JOB_ID"],
                     rank=os.environ["PIREUS_RANK"],patched_sha256=copy_sha,
                     original_sif_modified=False)),flush=True)

# Vocabulary copies retain stock shard selection and padding.
from patch_vocab_copy import patched_source as patched_vocab_source
vocab_changed = patched_vocab_source((root / "srt/layers/vocab_parallel_embedding.py").read_bytes())
vocab_sha = "fda7e7fdc854100a6250e2e24fd3242cdbee919546b66185f6920af72ee1f194"
assert hashlib.sha256(vocab_changed).hexdigest() == vocab_sha
vocab_out = Path("/scratch/pireus/cache") / ("vocab-copy-" + vocab_sha + ".py")
if vocab_out.exists():
    assert not vocab_out.is_symlink() and vocab_out.read_bytes() == vocab_changed
else:
    with vocab_out.open("xb") as stream:
        stream.write(vocab_changed)
        stream.flush()
        os.fsync(stream.fileno())
    vocab_out.chmod(0o444)
print(json.dumps(dict(stage="VOCAB_COPY_OVERLAY_STAGED",job=os.environ["SLURM_JOB_ID"],
                     rank=os.environ["PIREUS_RANK"],patched_sha256=vocab_sha,
                     original_sif_modified=False)),flush=True)
