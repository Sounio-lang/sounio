#!/usr/bin/env python3
"""Ref GUM propagation R=V/I NIST-like."""
import math
import json

# Type B uniform
def type_b_uniform(half_width):
    return half_width / math.sqrt(12)

u_v = type_b_uniform(0.02)
v = 10.0
u_i = type_b_uniform(0.003)
i = 1.005

r = v / i

# GUM div u_r^2 = (u_v/i)^2 + (v*u_i/i^2)^2
u_r_sq = (u_v / i)**2 + (v * u_i / i**2)**2
u_r = math.sqrt(u_r_sq)

# k95 infinite dof = 1.96
k95 = 1.96
u95 = k95 * u_r

print(json.dumps({
    "r_value": r,
    "r_std_unc": u_r,
    "r_exp_unc95": u95
}))
