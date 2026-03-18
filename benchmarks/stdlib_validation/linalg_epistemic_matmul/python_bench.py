#!/usr/bin/env python3
"""Ref epistemic matmul GUM manual computation."""
import numpy as np
import json
import math

# Same data as sounio
a_vals = np.array([[1.0, 0.5], [0.5, 2.0]])
a_unc = np.array([[1e-6, 1e-6], [1e-6, 1e-6]])
a_conf = np.array([[0.95, 0.95], [0.95, 0.95]])

b_vals = np.array([[3.0, 1.0], [1.0, 4.0]])
b_unc = np.array([[1e-6, 1e-6], [1e-6, 1e-6]])
b_conf = np.array([[0.95, 0.95], [0.95, 0.95]])

# c00 = a00*b00 + a01*b10
c00_val = a_vals[0,0]*b_vals[0,0] + a_vals[0,1]*b_vals[1,0]
c00_unc_sq = (a_unc[0,0]**2 * b_vals[0,0]**2 + 
              a_vals[0,0]**2 * b_unc[0,0]**2 + 
              a_unc[0,1]**2 * b_vals[1,0]**2 + 
              a_vals[0,1]**2 * b_unc[1,0]**2)
c00_unc = math.sqrt(c00_unc_sq)
c00_conf = min(np.min(a_conf), np.min(b_conf))

print(json.dumps({
    "c00_val": float(c00_val),
    "c00_unc": float(c00_unc),
    "c00_conf": float(c00_conf)
}))
