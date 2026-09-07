#!/usr/bin/env python3
"""Read the installed pinned runtime; do not allocate model/GPU tensors."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
assert os.environ.get("SLURM_JOB_ID")
root = Path(importlib.util.find_spec("sglang").origin).parent
result = dict(stage="RUNTIME_SOURCE_INVENTORY", job=os.environ["SLURM_JOB_ID"],
              rank=os.environ["PIREUS_RANK"], root=str(root), files={})
for name in ["srt/layers/moe/fused_moe_triton/layer.py","srt/layers/quantization/modelopt_quant.py",
             "srt/layers/quantization/marlin_utils_fp4.py",
             "srt/models/inkling.py"]:
    p = root / name
    result["files"][name] = dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(result),flush=True)
