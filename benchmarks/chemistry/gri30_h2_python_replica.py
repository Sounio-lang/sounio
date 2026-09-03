#!/usr/bin/env python3
"""Independent Python replica of stdlib/chemistry/gri30_h2.sio.

Recomputes, from benchmarks/chemistry/gri30_h2_mechanism.json (extracted from
Cantera's gri30.yaml by extract_gri30_h2.py), every number the Sounio module's
run-pass tests assert:

  - forward effective rate constants and Kc at 1200 K (air-like bath)
  - isothermal ignition trajectory checkpoint at t = 1e-4 s
    (2% H2 / 1% O2 / 97% N2, 1 atm, H-atom seed 1e-11, RK4 dt = 1e-8)
  - native 1-sigma uncertainty band at t = 1e-4 s from the first-order
    diagonal GUM delta method on the rate constants (same formula as the
    Sounio module, including the dt^2 scaling)

The ignition front itself is exponentially phase-sensitive; only the pre-front
checkpoint is a cross-language parity target (verified by dt-convergence:
dt = 1e-8 vs 5e-9 agree to 4 significant figures).

Run:  python3 benchmarks/chemistry/gri30_h2_python_replica.py
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
SP = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2", "AR"]
NSP = 10
D = json.load(open(os.path.join(HERE, "gri30_h2_mechanism.json")))
RX = D["reactions"]
NR = len(RX)
R_SI = 8.31446261815324
# Arrhenius activation-energy gas constant, cal/(mol*K).
# Cantera converts the cal/mol Ea values of gri30.yaml with 4.184 J/cal exactly
# and its own R, giving 8.31446261815324/4.184 = 1.9872042586408316.  The older
# 1.9872041 is 7.98e-08 away, which sits inside exp(-Ea/(R*T)) and therefore made
# this replica run DIFFERENT rate constants from the mechanism Cantera reads.
R_CAL = 1.9872042586408316
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

# representative 1-sigma relative uncertainties on k (Baulch 2005 / Konnov 2008 /
# Hong 2011 order-of-magnitude fidelity; NOT a refit)
U = [0.30] * NR
NAMED = {
    "H + O2 <=> O + OH": 0.10,
    "OH + H2 <=> H + H2O": 0.10,
    "O + H2 <=> H + OH": 0.15,
    "2 OH <=> O + H2O": 0.20,
    "H + O2 + M <=> HO2 + M": 0.25,
    "H + OH + M <=> H2O + M": 0.30,
    "2 O + M <=> O2 + M": 0.25,
    "O + H + M <=> OH + M": 0.25,
}
for i, r in enumerate(RX):
    if r["eq"] in NAMED:
        U[i] = NAMED[r["eq"]]


def nasa_g_rt(s, t):
    a = nasa[s][0] if t <= 1000.0 else nasa[s][1]
    h_rt = a[0] + a[1] * t / 2 + a[2] * t**2 / 3 + a[3] * t**3 / 4 + a[4] * t**4 / 5 + a[5] / t
    s_r = a[0] * math.log(t) + a[1] * t + a[2] * t**2 / 2 + a[3] * t**3 / 3 + a[4] * t**4 / 4 + a[6]
    return h_rt - s_r


def kp(r, t):
    return math.exp(-sum(nu[r][s] * nasa_g_rt(SP[s], t) for s in range(NSP)))


def kc_cm(r, t):
    c0 = P0 / (R_SI * t)
    return kp(r, t) * (c0 * 1e-6) ** dn[r]


def kfwd_eff(r, t, m_eff):
    kf = A[r] * t**B[r] * math.exp(-EA[r] / (R_CAL * t))
    if TYPE[r] == 1:
        return kf * m_eff
    if TYPE[r] == 2:
        _, low = LOW
        k0 = low[0] * t**low[1] * math.exp(-low[2] / (R_CAL * t))
        pr = k0 * m_eff / kf
        at, t3, t1, t2 = TROE[1]
        fcent = (1 - at) * math.exp(-t / t3) + at * math.exp(-t / t1) + math.exp(-t2 / t)
        lfc = math.log10(fcent)
        c = -0.4 - 0.67 * lfc
        n = 0.75 - 1.27 * lfc
        lpr = math.log10(pr)
        x = (lpr + c) / (0.75 - 1.27 * lfc - 0.14 * (lpr + c))
        return kf * (pr / (1 + pr)) * 10 ** (lfc / (1 + x * x))
    return kf


def rates_net(t, c, kc):
    out = []
    for r in range(NR):
        m_eff = sum(eff[r][s] * c[s] for s in range(NSP))
        kf = kfwd_eff(r, t, m_eff)
        fwd = kf
        for s in range(NSP):
            if reac[r][s] > 0:
                fwd *= c[s] ** reac[r][s]
        rev = kf / kc[r]
        for s in range(NSP):
            p = prod[r][s]
            if p > 0:
                rev *= c[s] ** p
        out.append(fwd - rev)
    return out


def dc_dt(t, c, kc):
    rn = rates_net(t, c, kc)
    return [sum(nu[r][s] * rn[r] for r in range(NR)) for s in range(NSP)]


def rk4_step(c, t, dt, kc):
    k1 = dc_dt(t, c, kc)
    k2 = dc_dt(t, [c[i] + 0.5 * dt * k1[i] for i in range(NSP)], kc)
    k3 = dc_dt(t, [c[i] + 0.5 * dt * k2[i] for i in range(NSP)], kc)
    k4 = dc_dt(t, [c[i] + dt * k3[i] for i in range(NSP)], kc)
    return [c[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(NSP)]


def propagate_unc(t, c, kc, varc, dt):
    """First-order diagonal GUM delta method, per step (dt^2 scaling)."""
    eps = 1e-12
    f0 = dc_dt(t, c, kc)
    rn = rates_net(t, c, kc)
    out = []
    for i in range(NSP):
        cp = list(c); cp[i] += eps
        cm = list(c); cm[i] -= eps
        jii = (dc_dt(t, cp, kc)[i] - dc_dt(t, cm, kc)[i]) / (2 * eps)
        acc = varc[i] + 2 * jii * varc[i] * dt
        for k in range(NSP):
            if k == i:
                continue
            cp = list(c); cp[k] += eps
            cm = list(c); cm[k] -= eps
            jik = (dc_dt(t, cp, kc)[i] - dc_dt(t, cm, kc)[i]) / (2 * eps)
            acc += (jik * dt) ** 2 * varc[k]
        for r in range(NR):
            acc += (nu[r][i] * rn[r] * dt * U[r]) ** 2
        out.append(max(acc, 0.0))
    return out


def main():
    t = 1200.0
    mtot = P0 / (R_SI * t) * 1e-6
    kc = [kc_cm(r, t) for r in range(NR)]

    print(f"NR = {NR}")
    print(f"[M]@1200K 1atm = {mtot:.6e}")
    c_air = [0.0] * NSP
    c_air[8], c_air[3] = mtot * 0.79, mtot * 0.21
    for r in range(NR):
        m_eff = sum(eff[r][s] * c_air[s] for s in range(NSP))
        print("k%02d %-28s %.4e  (Kc=%.3e, u=%.2f)"
              % (r, RX[r]["eq"][:28], kfwd_eff(r, t, m_eff), kc[r], U[r]))

    # --- deterministic trajectory checkpoint, H-seeded (Sounio test protocol:
    # --- T=1500 K, seed H = 1e-11 mol/cm^3 absolute, pre-front t = 1e-4 s)
    ts = 1500.0
    ms = P0 / (R_SI * ts) * 1e-6
    kcs = [kc_cm(r, ts) for r in range(NR)]
    c = [0.0] * NSP
    c[0], c[3], c[8] = ms * 0.02, ms * 0.01, ms * 0.97
    c[1] = 1.0e-11  # H-atom seed, mol/cm^3 absolute (chain initiation)
    dt = 1e-8
    for _ in range(10000):  # t = 1e-4 s
        c = rk4_step(c, ts, dt, kcs)
    print("DET T=1500 t=1e-4: " + " ".join("%s=%.6e" % (SP[i], c[i]) for i in range(NSP)))

    # --- epistemic band at the same checkpoint; 1% standard uncertainty on
    # --- initial H2 and O2 concentrations (matches the Sounio test protocol)
    ce = [0.0] * NSP
    ce[0], ce[3], ce[8], ce[1] = ms * 0.02, ms * 0.01, ms * 0.97, 1.0e-11
    v = [0.0] * NSP
    v[0] = (0.01 * ce[0]) ** 2
    v[3] = (0.01 * ce[3]) ** 2
    for _ in range(10000):
        ce = rk4_step(ce, ts, dt, kcs)
        v = propagate_unc(ts, ce, kcs, v, dt)
    u = [math.sqrt(x) for x in v]
    print("EPI T=1500 t=1e-4: " + " ".join("u(%s)=%.4e" % (SP[i], u[i]) for i in range(NSP)))


if __name__ == "__main__":
    main()
