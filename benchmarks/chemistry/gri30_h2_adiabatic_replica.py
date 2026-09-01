#!/usr/bin/env python3
"""Independent Python replica of the ADIABATIC (constant-U,V) reactor path in
stdlib/chemistry/gri30_h2.sio (g30_uv_rhs / g30_adiabatic_delay).

Same mechanism data and rate code as gri30_h2_python_replica.py (extracted
from Cantera's gri30.yaml), same fixed-step explicit RK4 integrator and the
same delay definition (mid-step finite-difference maxima of d[H2O]/dt and
dT/dt), now with the temperature coupled through sensible energy
conservation:

  u_mix = sum_i c_i (h_i(T) - R T) = const     (ideal-gas species, mol/cm^3)
  dT/dt = -T * sum_i (h_rt_i - 1) w_i / sum_i c_i (cp_r_i - 1)

Protocol (shock-tube-like, identical to the Sounio module):
  2% H2 / 1% O2 / 97% N2, 1 atm initial, H-atom seed 1e-11 mol/cm^3
  (chain initiation surrogate — same protocol as the isothermal work).

Outputs:
  - NASA-7 cp/R and h/(RT) spot values (pins for the Sounio test)
  - adiabatic delays for a 1000..2000 K sweep (dt = 1e-8)
  - dt-convergence at two temperatures (1e-8 vs 5e-9)

Run:  python3 benchmarks/chemistry/gri30_h2_adiabatic_replica.py
"""
import json, math, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
SP = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2", "AR"]
NSP = 10
D = json.load(open(os.path.join(HERE, "gri30_h2_mechanism.json")))
RX = D["reactions"]
NR = len(RX)
R_SI = 8.31446261815324
P0 = 101325.0  # CHEMKIN/GRI standard state: 1 atm

nasa = {s: D["species"][s]["coeffs"] for s in SP}
A = [r["fwd"][0] for r in RX]
B = [r["fwd"][1] for r in RX]
EA = [r["fwd"][2] for r in RX]
TYPE = [{"arrhenius": 0, "three-body": 1, "falloff": 2}[r["type"]] for r in RX]
reac = [[0.0] * NSP for _ in range(NR)]
prod = [[0.0] * NSP for _ in range(NR)]
nu = [[0.0] * NSP for _ in range(NR)]
for i, r in enumerate(RX):
    for s, v in r["react"].items():
        reac[i][SP.index(s)] = v
        nu[i][SP.index(s)] -= v
    for s, v in r["prod"].items():
        prod[i][SP.index(s)] = v
        nu[i][SP.index(s)] += v
dn = [sum(nu[i]) for i in range(NR)]
eff = [[1.0] * NSP for _ in range(NR)]
for i, r in enumerate(RX):
    for s, v in r["eff"].items():
        eff[i][SP.index(s)] = v
LOW = TROE = None
for i, r in enumerate(RX):
    if r["type"] == "falloff":
        LOW = (i, r["low"])
        TROE = (i, r["troe"])


def nasa7(si, t):
    return nasa[SP[si]][0] if t <= 1000.0 else nasa[SP[si]][1]


def cp_r(si, t):
    a = nasa7(si, t)
    return a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4


def h_rt(si, t):
    a = nasa7(si, t)
    return a[0] + a[1] * t / 2 + a[2] * t**2 / 3 + a[3] * t**3 / 4 + a[4] * t**4 / 5 + a[5] / t


def g_rt(si, t):
    a = nasa7(si, t)
    s_r = a[0] * math.log(t) + a[1] * t + a[2] * t**2 / 2 + a[3] * t**3 / 3 + a[4] * t**4 / 4 + a[6]
    return h_rt(si, t) - s_r


def kfwd_eff(r, t, m_eff):
    kf = A[r] * t**B[r] * math.exp(-EA[r] / (1.9872041 * t))
    if TYPE[r] == 1:
        return kf * m_eff
    if TYPE[r] == 2:
        _, low = LOW
        k0 = low[0] * t**low[1] * math.exp(-low[2] / (1.9872041 * t))
        pr = k0 * m_eff / kf
        at, t3, t1, t2 = TROE[1]
        fcent = (1 - at) * math.exp(-t / t3) + at * math.exp(-t / t1) + math.exp(-t2 / t)
        lfc = math.log10(fcent)
        c = -0.4 - 0.67 * lfc
        n = 0.75 - 1.27 * lfc
        lpr = math.log10(pr)
        x = (lpr + c) / (n - 0.14 * (lpr + c))
        return kf * (pr / (1 + pr)) * 10 ** (lfc / (1 + x * x))
    return kf


def uv_rhs(t, c):
    """One adiabatic RHS evaluation at (t, c): returns (dT/dt, dc/dt)."""
    grt = [g_rt(s, t) for s in range(NSP)]
    c0 = P0 / (R_SI * t) * 1e-6
    kc = [math.exp(-sum(nu[r][s] * grt[s] for s in range(NSP))) * c0 ** dn[r]
          for r in range(NR)]
    rn = []
    for r in range(NR):
        m_eff = sum(eff[r][s] * c[s] for s in range(NSP))
        kf = kfwd_eff(r, t, m_eff)
        fwd = kf
        for s in range(NSP):
            if reac[r][s] > 0:
                fwd *= c[s] ** reac[r][s]
        rev = kf / kc[r]
        for s in range(NSP):
            if prod[r][s] > 0:
                rev *= c[s] ** prod[r][s]
        rn.append(fwd - rev)
    dc = [sum(nu[r][s] * rn[r] for r in range(NR)) for s in range(NSP)]
    num = sum((h_rt(s, t) - 1.0) * dc[s] for s in range(NSP))
    den = sum(c[s] * (cp_r(s, t) - 1.0) for s in range(NSP))
    return -t * num / den, dc


def demo_init(t):
    m = P0 / (R_SI * t) * 1e-6
    c = [0.0] * NSP
    c[0], c[3], c[8] = m * 0.02, m * 0.01, m * 0.97
    c[1] = 1.0e-11  # H-atom seed, mol/cm^3 absolute (chain initiation)
    return c


def adiabatic_delay(t0, dt, max_steps):
    """RK4 on the coupled (T, c) system; returns (t_h2o, t_dT, steps_used).

    Delays are mid-step times of max d[H2O]/dt and max dT/dt; (-1, -1) when
    no completed front occurs within max_steps*dt.
    """
    t = t0
    y = demo_init(t0)
    prev_h2o, prev_t = y[5], t
    best_h2o, best_h2o_t = 0.0, -1.0
    best_dt, best_dt_t = 0.0, -1.0
    done = False
    s = 0
    while s < max_steps and not done:
        d1t, d1c = uv_rhs(t, y)
        yt = [y[i] + 0.5 * dt * d1c[i] for i in range(NSP)]
        d2t, d2c = uv_rhs(t + 0.5 * dt * d1t, yt)
        yt = [y[i] + 0.5 * dt * d2c[i] for i in range(NSP)]
        d3t, d3c = uv_rhs(t + 0.5 * dt * d2t, yt)
        yt = [y[i] + dt * d3c[i] for i in range(NSP)]
        d4t, d4c = uv_rhs(t + dt * d3t, yt)
        y = [y[i] + dt / 6 * (d1c[i] + 2 * d2c[i] + 2 * d3c[i] + d4c[i]) for i in range(NSP)]
        t = t + dt / 6 * (d1t + 2 * d2t + 2 * d3t + d4t)
        s += 1
        tnow = s * dt
        rate_h2o = (y[5] - prev_h2o) / dt
        rate_t = (t - prev_t) / dt
        prev_h2o, prev_t = y[5], t
        if rate_h2o > best_h2o:
            best_h2o, best_h2o_t = rate_h2o, tnow - 0.5 * dt
        if rate_t > best_dt:
            best_dt, best_dt_t = rate_t, tnow - 0.5 * dt
        if best_h2o_t > 0.0 and rate_h2o < 0.05 * best_h2o and tnow > 2.0 * best_h2o_t:
            done = True
    if done:
        return best_h2o_t, best_dt_t, s
    return -1.0, -1.0, s


def main():
    print(f"NR = {NR}, NSP = {NSP} (gri30_h2 sub-mechanism, adiabatic UV)")
    print("NASA-7 spot values (Sounio test pins):")
    for sp, t in (("H2", 1500.0), ("O2", 1500.0), ("H2O", 1500.0), ("N2", 1500.0),
                  ("H", 1500.0), ("H2", 900.0), ("H2O", 900.0)):
        si = SP.index(sp)
        print(f"  cp_r({sp:4s},{t:6.0f}K) = {cp_r(si, t):.6f}   h_rt({sp},{t:.0f}K) = {h_rt(si, t):.6f}")

    print("ADIABATIC DELAYS (2%H2/1%O2/97%N2, 1 atm, seed H=1e-11, dt=1e-8)")
    print("  T0(K)  1000/T    t_h2o(us)  t_dT(us)   steps")
    for t0 in range(1000, 2001, 100):
        # uniform 3 ms horizon: the early-exit certificate needs t > 2*peak,
        # so a shorter horizon can false-negative slow-but-real ignitions
        max_steps = 300000
        th, tt, s = adiabatic_delay(float(t0), 1e-8, max_steps)
        if th > 0:
            print(f"  {t0:5d}  {1000.0/t0:7.4f}  {th*1e6:9.2f}  {tt*1e6:9.2f}  {s:7d}")
        else:
            print(f"  {t0:5d}  {1000.0/t0:7.4f}  no-ign<{max_steps*1e-8*1e6:.0f}us  no-ign     {s:7d}")

    print("DT-CONVERGENCE (t_h2o, us):")
    for t0 in (1500.0, 1700.0):
        a, _, _ = adiabatic_delay(t0, 1e-8, 60000)
        b, _, _ = adiabatic_delay(t0, 5e-9, 120000)
        print(f"  T0={t0:.0f}K  dt=1e-8: {a*1e6:.3f}   dt=5e-9: {b*1e6:.3f}")


if __name__ == "__main__":
    t_start = time.perf_counter()
    main()
    print(f"wall={time.perf_counter()-t_start:.1f}s")
