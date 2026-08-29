#!/usr/bin/env python3
"""Reference arithmetic for the Sounio PPCR dosing demo.

This script validates the integer-scaled calculations in fregni_demo.sio
using ordinary Python floats. The Sounio demo uses x100 scaling for all
clinical quantities to stay in fixed-point integer arithmetic.
"""

# Scenario 1: measured clearance 80.00 L/h, target AUC 400.00 mg*h/L, F=1.00
target_auc = 400.00
cl = 80.00
f = 1.00
dose = target_auc * cl / f
print(f"Scenario 1 reference dose = {dose:.2f} mg")
assert abs(dose - 32000.00) < 0.01

# Scenario 2: confidence gating
sim_conf = 700   # permille
min_conf = 950   # permille
print(f"Scenario 2 confidence {sim_conf} permille < threshold {min_conf} permille -> REJECT")
assert sim_conf < min_conf

# Scenario 3: CV gating
imp_value = 78.00
imp_std = 23.40
cv_percent = 100.0 * imp_std / imp_value
max_cv = 25.00
print(f"Scenario 3 CV = {cv_percent:.2f}% > threshold {max_cv:.2f}% -> REJECT")
assert cv_percent > max_cv

print("REFERENCE: all checks match demo expectations")
