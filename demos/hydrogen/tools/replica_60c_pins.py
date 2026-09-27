#!/usr/bin/env python3
# demos/hydrogen/tools/replica_60c_pins.py
#
# Independent Python replica of the value trajectory of
# stdlib chemistry::kinetics::simulate_general_epistemic for the
# UHS H2-brine-calcite network (demos/hydrogen/uhs_brine_calcite.sio,
# site_screening.sio) at T = 333.15 K (60 C — Pentalofos bracket
# interior). Mirrors the engine exactly: mass-action rates
# rate_j = k_j * prod_reactants c^order (integer orders from nu),
# dc/dt = nu . rates, classical RK4, dt = 0.05 yr x 600 steps.
# The rate constants use the same public PWP/phreeqc.dat formulas the
# demo uses (its ss_exp/ss_pow10 series agree with libm to ~1e-12,
# far inside the 1e-7 selftest tolerance).
#
# Output: pin literals for tests/run-pass/site_screening_selftest.sio.

import math

T = 333.15
SPY = 31557600.0
A = 1.0e-5
KM = 0.0135  # below the 343.15 K A2 cutoff, so k_m stays active

k1 = 10 ** (0.198 - 444.0 / T)
k2 = 10 ** (2.84 - 2177.0 / T)
k3 = 10 ** (-1.1 - 1737.0 / T)  # T > 298.15 branch
logk = -8.45 - ((-3.15 * 4184.0) / 19.14476) * (1.0 / T - 1.0 / 298.15)
k4 = 10 ** (logk + 10.33) * 1.0e-7

ks1 = A * k1 * 1.0e-7 * SPY
ks2 = A * k2 * SPY
ks3 = A * k3 * SPY
kb = ks3 / k4
ks = [KM, ks1, ks2, ks3, kb]

# nu (species x reactions): 0 H2, 1 CH4, 2 Calcite, 3 Ca, 4 HCO3, 5 CO2
nu = [
    [-1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, -1.0],
    [0.0, 1.0, 2.0, 1.0, -1.0],
    [-1.0, 0.0, -1.0, 0.0, 0.0],
]

H2INIT = 7.8e-4 * 15.0 * 0.85
y = [H2INIT, 0.0, 1.0, 1.0e-3, 5.0e-4, 5.0e-2]

def rates(c):
    r = []
    for j in range(5):
        rate = ks[j]
        for s in range(6):
            coeff = nu[s][j]
            if coeff < 0.0:
                rate *= c[s] ** int(-coeff)
        r.append(rate)
    return r

def dc(c):
    r = rates(c)
    return [sum(nu[s][j] * r[j] for j in range(5)) for s in range(6)]

DT = 0.05
for _ in range(600):
    k1v = dc(y)
    y2 = [y[i] + 0.5 * DT * k1v[i] for i in range(6)]
    k2v = dc(y2)
    y3 = [y[i] + 0.5 * DT * k2v[i] for i in range(6)]
    k3v = dc(y3)
    y4 = [y[i] + DT * k3v[i] for i in range(6)]
    k4v = dc(y4)
    y = [y[i] + (DT / 6.0) * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) for i in range(6)]

print(f"ks literals: km={KM}, ks1={ks1:.12e}, ks2={ks2:.12e}, ks3={ks3:.12e}, kb={kb:.12e}")
print(f"K4 = {k4:.12e}")
print(f"H2      = {y[0]:.13f}")
print(f"Calcite = {y[2]:.13f}")
print(f"CO2     = {y[5]:.13f}")
print(f"loss %  = {100.0 * (1.0 - y[0] / H2INIT):.6f}")
