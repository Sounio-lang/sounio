#!/usr/bin/env python3
"""Valida erro relativo Sounio vs SciPy OLS."""
import json
import sys
import subprocess

# Run benches
subprocess.run(["sounio", "run", "sounio_bench.sio"], stdout=open(&#x27;sounio.json&#x27;, &#x27;w&#x27;))
subprocess.run(["python3", "python_bench.py"], stdout=open(&#x27;ref.json&#x27;, &#x27;w&#x27;))

with open(&#x27;sounio.json&#x27;) as f:
    sounio = json.load(f)
with open(&#x27;ref.json&#x27;) as f:
    ref = json.load(f)

def rel_err(s, r):
    return abs(s - r) / max(abs(r), 1e-12)

err_int = rel_err(sounio[&#x27;intercept&#x27;], ref[&#x27;intercept&#x27;])
err_slope = rel_err(sounio[&#x27;slope&#x27;], ref[&#x27;slope&#x27;])
err_r2 = rel_err(sounio[&#x27;r_squared&#x27;], ref[&#x27;r_squared&#x27;])

max_err = max(err_int, err_slope, err_r2)

print(f"OLS Stats: max rel err = {max_err:.2e}")
if max_err < 1e-10:
    print("PASS")
else:
    print("FAIL")
