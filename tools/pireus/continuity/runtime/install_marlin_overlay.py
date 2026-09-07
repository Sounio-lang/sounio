#!/usr/bin/env python3
"""Stage one verified source overlay without rewriting the pinned SIF."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from patch_marlin_placeholders import patched_source
from patch_marlin_repack import patched_source as patched_repack_source

PATCHED_SHA256 = "26b999da4d72fd238c32331782f72b6aa110165adec96fb8885f4252a5a7099c"
assert os.environ.get("SLURM_JOB_ID")
root = Path(importlib.util.find_spec("sglang").origin).parent
source = root / "srt/layers/quantization/modelopt_quant.py"
changed = patched_source(source.read_bytes())
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
