#!/usr/bin/env python3
"""Stage one verified source overlay without rewriting the pinned SIF."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from patch_marlin_placeholders import patched_source

PATCHED_SHA256 = "26b999da4d72fd238c32331782f72b6aa110165adec96fb8885f4252a5a7099c"
assert os.environ.get("SLURM_JOB_ID")
root = Path(importlib.util.find_spec("sglang").origin).parent
source = root / "srt/layers/quantization/modelopt_quant.py"
changed = patched_source(source.read_bytes())
assert hashlib.sha256(changed).hexdigest() == PATCHED_SHA256
out = Path("/scratch/pireus/runtime") / ("modelopt-" + PATCHED_SHA256 + ".py")
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
