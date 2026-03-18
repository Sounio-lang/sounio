#!/usr/bin/env python3
"""Valida epistemic matmul GUM vs manual (erro < 1e-10)"""
import json
import numpy as np

with open(&#x27;sounio.json&#x27;) as f:
    sounio = json.load(f)
with open(&#x27;ref.json&#x27;) as f:
    ref = json.load(f)

keys = [&#x27;c00_val&#x27;, &#x27;c00_unc&#x27;, &#x27;c00_conf&#x27;]

max_err = 0.0
for k in keys:
    err = np.abs(sounio[k] - ref[k]) / np.max(np.abs(ref[k]), 1e-12)
    max_err = np.max([max_err, err])

print(f"Epistemic Matmul: max rel err = {max_err:.2e}")
print("PASS" if max_err < 1e-10 else "FAIL")
