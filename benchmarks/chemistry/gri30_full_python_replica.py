#!/usr/bin/env python3
"""Independent Python replica of stdlib/chemistry/gri30_full.sio.

Recomputes, from benchmarks/chemistry/gri30_full_mechanism.json (the FULL
GRI-Mech 3.0 mechanism: 53 species, 325 reactions, extracted from Cantera's
gri30.yaml), every number the Sounio module's run-pass tests assert:

  - forward effective rate constants and Kc at 1200 K (air-like bath),
    including a Troe falloff (2 CH3 (+M) <=> C2H6 (+M)) and a Lindemann
    falloff (O + CO (+M) <=> CO2 (+M), broadening factor F = 1)
  - isothermal ignition trajectory checkpoint at t = 4e-6 s
    (2% H2 / 1% O2 / 97% N2, 1 atm, H-atom seed 1e-11, RK4 dt = 2e-9,
    2000 steps -- shorter than the H/O module's 10000 for runtime; the
    ignition front itself is exponentially phase-sensitive, so only this
    pre-front checkpoint is a cross-language parity target).
    NOTE on dt: the full mechanism contains NNH <=> N2 + H with
    k = 3.3e8 /s at 1500 K (plus NNH + M at ~9e7 /s), so the H/O module's
    dt = 1e-8 is explicitly unstable (k*dt > 4 vs the RK4 limit ~2.785)
    and the trajectory blows up within ~20 steps. dt = 2e-9 gives
    k*dt ~ 0.84 and matches dt = 1e-9 / 5e-10 to every printed digit.
  - native 1-sigma uncertainty band at t = 4e-7 s (200 steps) from the
    first-order diagonal GUM delta method on the rate constants (same
    formula as the Sounio module, including the dt^2 scaling)

16 reactions are irreversible in GRI-Mech 3.0 ('reversible: false'): for
those the reverse rate is exactly 0 (the Kc division is skipped), matching
the Sounio module.

Sparse-index note: the loops below iterate only over precomputed nonzero
stoichiometric entries, in ascending species/reaction order. Skipping exact
0.0 terms is an IEEE identity, so the arithmetic is bit-for-bit the same
summation order as the dense Sounio loops (only the transcendental-function
implementations differ, at the ~1e-15 level).

Run:  python3 benchmarks/chemistry/gri30_full_python_replica.py
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "gri30_full_mechanism.json")))
SP = list(D["species"].keys())  # json order: H2 H O O2 OH H2O HO2 H2O2 C ... N2 AR ...
NSP = len(SP)
RX = D["reactions"]
NR = len(RX)
R_SI = 8.31446261815324
P0 = 101325.0  # CHEMKIN/GRI standard state: 1 atm

nasa = {s: D["species"][s]["coeffs"] for s in SP}
A = [r["fwd"][0] for r in RX]
B = [r["fwd"][1] for r in RX]
EA = [r["fwd"][2] for r in RX]
TYPE = [{"arrhenius": 0, "three-body": 1, "falloff": 2}[r["type"]] for r in RX]
REVERSIBLE = [1 if r["reversible"] else 0 for r in RX]
LOW = [r["low"] if r["low"] else [0.0, 0.0, 0.0] for r in RX]
TROE = [r["troe"] if r["troe"] else [0.0, 0.0, 0.0, 0.0] for r in RX]
HAS_TROE = [1 if (r["type"] == "falloff" and r["troe"]) else 0 for r in RX]

reac = [[0.0] * NSP for _ in range(NR)]
prod = [[0.0] * NSP for _ in range(NR)]
nu = [[0.0] * NSP for _ in range(NR)]
eff = [[1.0] * NSP for _ in range(NR)]
for i, r in enumerate(RX):
    for s, v in r["react"].items():
        reac[i][SP.index(s)] = v
        nu[i][SP.index(s)] -= v
    for s, v in r["prod"].items():
        prod[i][SP.index(s)] = v
        nu[i][SP.index(s)] += v
    for s, v in r["eff"].items():
        eff[i][SP.index(s)] = v
dn = [sum(nu[i]) for i in range(NR)]

# sparse nonzero lists (ascending order == dense Sounio summation order)
reac_nz = [[s for s in range(NSP) if reac[r][s] != 0.0] for r in range(NR)]
prod_nz = [[s for s in range(NSP) if prod[r][s] != 0.0] for r in range(NR)]
nu_nz = [[s for s in range(NSP) if nu[r][s] != 0.0] for r in range(NR)]
eff_nz = [[s for s in range(NSP) if eff[r][s] != 0.0] for r in range(NR)]
rxn_of_sp = [[r for r in range(NR) if nu[r][s] != 0.0] for s in range(NSP)]

# representative 1-sigma relative uncertainties on k (Baulch 2005 / Konnov 2008 /
# Hong 2011 order-of-magnitude fidelity; NOT a refit)
U = [0.30] * NR
NAMED = {
    "H + O2 <=> O + OH": 0.10,
    "OH + H2 <=> H + H2O": 0.10,
    "O + H2 <=> H + OH": 0.15,
    "2 OH <=> O + H2O": 0.20,
    "H + O2 + M <=> HO2 + M": 0.25,
    "2 O + M <=> O2 + M": 0.25,
    "O + H + M <=> OH + M": 0.25,
    "H + OH + M <=> H2O + M": 0.30,
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
    return math.exp(-sum(nu[r][s] * nasa_g_rt(SP[s], t) for s in nu_nz[r]))


def kc_cm(r, t):
    c0 = P0 / (R_SI * t)
    return kp(r, t) * (c0 * 1e-6) ** dn[r]


def m_eff(r, c):
    return sum(eff[r][s] * c[s] for s in eff_nz[r])


def kfwd_eff(r, t, me):
    kf = A[r] * t**B[r] * math.exp(-EA[r] / (1.9872041 * t))
    if TYPE[r] == 1:
        return kf * me
    if TYPE[r] == 2:
        low = LOW[r]
        k0 = low[0] * t**low[1] * math.exp(-low[2] / (1.9872041 * t))
        pr = k0 * me / kf
        if pr <= 0.0:
            return 0.0
        if HAS_TROE[r]:
            at, t3, t1, t2 = TROE[r]
            fcent = (1 - at) * math.exp(-t / t3) + at * math.exp(-t / t1) + math.exp(-t2 / t)
            lfc = math.log10(fcent)
            cc = -0.4 - 0.67 * lfc
            nn = 0.75 - 1.27 * lfc
            lpr = math.log10(pr)
            x = (lpr + cc) / (nn - 0.14 * (lpr + cc))
            return kf * (pr / (1 + pr)) * 10 ** (lfc / (1 + x * x))
        return kf * pr / (1 + pr)  # Lindemann: broadening factor F = 1
    return kf


def rates_fb(t, c, kc):
    """Per-reaction forward and reverse rates (mirrors the Sounio loop)."""
    fwd = [0.0] * NR
    rev = [0.0] * NR
    for r in range(NR):
        me = m_eff(r, c) if TYPE[r] != 0 else 0.0
        kf = kfwd_eff(r, t, me)
        f = kf
        for s in reac_nz[r]:
            o = reac[r][s]
            if o == 2.0:
                f = f * c[s] * c[s]
            if o == 1.0:
                f = f * c[s]
        rv = 0.0
        if REVERSIBLE[r] and kc[r] > 0.0:
            rv = kf / kc[r]
            for s in prod_nz[r]:
                o = prod[r][s]
                if o == 2.0:
                    rv = rv * c[s] * c[s]
                if o == 1.0:
                    rv = rv * c[s]
        fwd[r] = f
        rev[r] = rv
    return fwd, rev


def dc_dt(t, c, kc):
    fwd, rev = rates_fb(t, c, kc)
    return [sum(nu[r][s] * (fwd[r] - rev[r]) for r in rxn_of_sp[s]) for s in range(NSP)]


def rk4_step(c, t, dt, kc):
    k1 = dc_dt(t, c, kc)
    k2 = dc_dt(t, [c[i] + 0.5 * dt * k1[i] for i in range(NSP)], kc)
    k3 = dc_dt(t, [c[i] + 0.5 * dt * k2[i] for i in range(NSP)], kc)
    k4 = dc_dt(t, [c[i] + dt * k3[i] for i in range(NSP)], kc)
    return [c[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(NSP)]


def jacobian(t, c, kc, eps=1e-12):
    jac = [[0.0] * NSP for _ in range(NSP)]
    for s in range(NSP):
        cp = list(c); cp[s] += eps
        cm = list(c); cm[s] -= eps
        dcp = dc_dt(t, cp, kc)
        dcm = dc_dt(t, cm, kc)
        for i in range(NSP):
            jac[i][s] = (dcp[i] - dcm[i]) / (2 * eps)
    return jac


def propagate_unc(t, c, kc, varc, dt):
    """First-order diagonal GUM delta method, per step (dt^2 scaling).

    Identical formula to the Sounio module:
      v+_i = v_i + 2 J_ii v_i dt + sum_k (J_ik dt)^2 v_k
           + sum_r (nu_ir * net_r * dt * u_rel_r)^2
    """
    jac = jacobian(t, c, kc)
    fwd, rev = rates_fb(t, c, kc)
    out = []
    for i in range(NSP):
        acc = varc[i] + 2.0 * jac[i][i] * varc[i] * dt
        for k in range(NSP):
            jdt = jac[i][k] * dt
            acc += jdt * jdt * varc[k]
        for r in rxn_of_sp[i]:
            sens = nu[r][i] * (fwd[r] - rev[r]) * dt * U[r]
            acc += sens * sens
        out.append(acc)
    return out


def demo_init(t):
    """2% H2 / 1% O2 / 97% N2 at 1 atm, H-atom seed 1e-11 (chain initiation)."""
    m = P0 / (R_SI * t) * 1e-6
    c = [0.0] * NSP
    c[SP.index("H2")] = m * 0.02
    c[SP.index("O2")] = m * 0.01
    c[SP.index("N2")] = m * 0.97
    c[SP.index("H")] = 1.0e-11
    return c


def main():
    print(f"NSP = {NSP}  NR = {NR}  (falloff={sum(1 for t in TYPE if t == 2)}, "
          f"three-body={sum(1 for t in TYPE if t == 1)}, "
          f"irreversible={sum(1 for f in REVERSIBLE if f == 0)})")

    # --- forward effective rates + Kc at 1200 K, air-like bath ---
    t = 1200.0
    mtot = P0 / (R_SI * t) * 1e-6
    kc = [kc_cm(r, t) for r in range(NR)]
    c_air = [0.0] * NSP
    c_air[SP.index("N2")] = mtot * 0.79
    c_air[SP.index("O2")] = mtot * 0.21
    spot = ["2 O + M <=> O2 + M", "O + H2 <=> H + OH", "H + O2 <=> O + OH",
            "OH + H2 <=> H + H2O", "OH + CH4 <=> CH3 + H2O",
            "H + O2 + M <=> HO2 + M", "2 CH3 (+M) <=> C2H6 (+M)",
            "O + CO (+M) <=> CO2 (+M)", "H + OH + M <=> H2O + M"]
    for eq in spot:
        r = next(i for i, x in enumerate(RX) if x["eq"] == eq)
        me = m_eff(r, c_air) if TYPE[r] != 0 else mtot
        print("k%03d %-26s k=%.6e  Kc=%.6e  (type=%d, u=%.2f)"
              % (r, eq[:26], kfwd_eff(r, t, me), kc[r], TYPE[r], U[r]))

    # --- deterministic checkpoint, T=1500 K, dt=2e-9, 2000 steps (t=4e-6 s) ---
    ts = 1500.0
    kcs = [kc_cm(r, ts) for r in range(NR)]
    c = demo_init(ts)
    dt = 2e-9
    for _ in range(2000):
        c = rk4_step(c, ts, dt, kcs)
    keep = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2"]
    print("DET T=1500 t=4e-6 (2000 steps): "
          + " ".join("%s=%.6e" % (s, c[SP.index(s)]) for s in keep))

    # --- epistemic band at t=4e-7 s (200 steps); 1% std uncertainty on H2/O2 ---
    ce = demo_init(ts)
    v = [0.0] * NSP
    v[SP.index("H2")] = (0.01 * ce[SP.index("H2")]) ** 2
    v[SP.index("O2")] = (0.01 * ce[SP.index("O2")]) ** 2
    for _ in range(200):
        ce = rk4_step(ce, ts, dt, kcs)
        v = propagate_unc(ts, ce, kcs, v, dt)
    u = [math.sqrt(x) for x in v]
    print("EPI T=1500 t=4e-7 (200 steps): "
          + " ".join("%s=%.6e" % (s, ce[SP.index(s)]) for s in ["H2", "H2O", "OH", "H"]))
    print("EPI uncertainties: "
          + " ".join("u(%s)=%.6e" % (s, u[SP.index(s)]) for s in ["H2", "H2O", "OH", "H"]))


if __name__ == "__main__":
    main()
