#!/usr/bin/env python3
"""Valida linalg matmul/SVD vs NumPy (erro < 1e-12)"""
import json
import numpy as np

# Assume sounio run > sounio.json ; python python_bench.py > ref.json
with open(&#x27;sounio.json&#x27;) as f:
    sounio = json.load(f)
with open(&#x27;ref.json&#x27;) as f:
    ref = json.load(f)

keys = [&#x27;matmul_00&#x27;, &#x27;matmul_11&#x27;, &#x27;matmul_22&#x27;, &#x27;matmul_33&#x27;, &#x27;svd_sigma0&#x27;, &#x27;svd_sigma1&#x27;, &#x27;svd_sigma2&#x27;, &#x27;svd_sigma3&#x27;]

max_err = 0.0
for k in keys:
    err = np.abs(sounio[k] - ref[k]) / np.max(np.abs(ref[k]), 1e-12)
    max_err = np.max([max_err, err])

print(f"Linalg: max rel err = {max_err:.2e}")
print("PASS" if max_err < 1e-12 else "FAIL")
