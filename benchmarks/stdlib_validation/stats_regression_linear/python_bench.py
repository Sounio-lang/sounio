#!/usr/bin/env python3
"""Ref SciPy linregress para OLS benchmark."""
import numpy as np
from scipy.stats import linregress
import json

x = np.arange(20.0)
y = 2.0 + 3.0 * x + 0.05 * (x % 2 - 0.5)

result = linregress(x, y)

print(json.dumps({
    "intercept": float(result.intercept),
    "slope": float(result.slope),
    "r_squared": float(result.rvalue**2)
}, indent=None))
