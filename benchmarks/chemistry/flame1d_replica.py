#!/usr/bin/env python3
"""1-D premixed freely-propagating H2/air flame — independent pure-Python
replica of the Sounio flame1d solver (house pattern: no dependencies).

Model (identical equations as stdlib/chemistry/flame1d.sio):
  low-Mach 1-D premixed flame at p = 1 atm, unburned phi=1 H2/air
  (H2:O2:N2 = 2:1:3.76) at T_u = 300 K. Unity Lewis number: one shared
  diffusivity D(T) for all species and the heat flux, with D(T) = D0 (T/T0)^n
  fit to Cantera's operative lambda/(rho cp) along its own unity-Lewis
  free-flame solution (samples below). Kinetics/thermo: the merged
  benchmarks/chemistry/gri30_h2_adiabatic_replica.py (10 species, 29
  reactions, NASA-7, detailed balance) — single source of truth.

  Godunov operator split per macro-step dt_m:
    (1) chemistry: isobaric batch per ACTIVE cell (T > T_CHEM, excluding the
        near-equilibrium burned plateau T > T_ad-50 with Y_H2O >= 0.99*Y_H2O,eq),
        stage-clamped explicit RK4 substeps at dt_c ~ 1e-8 s. Clamping
        (Y <- max(Y,0) on every RK stage state and the final update) is
        REQUIRED: unclamped RK4 and Euler/RK2 at any dt_c blow up above
        ~2400 K through HO2/H2O2 stage negativity (0-D probed). dT/dt =
        -T sum(h_rt omega) / sum(c cp_r)  (isobaric; NOT the shock-tube
        constant-volume form), dc/dt from the merged kinetics.
    (2) transport: explicit FTCS Euler over all cells — species/heat
        diffusion with the enthalpy-flux correction -sum(j_k cp_k) dT/dx,
        continuity-induced velocity rho*u from -integral(d rho/dt) dx, and
        upwind advection of Y and T. dt_m limited by the FTCS bound
        dx^2/(2 D_max).

  Initial condition: the converged Cantera unity-Lewis phi=1 free-flame
  profile (flame1d_profile_phi1.json), anchored with its T = 1500 K point at
  x = x0. The solver's own slab-ignition transient takes ~1 ms to reach the
  steady flame (probed: a hot-products slab collapses diffusively and the
  front re-establishes through a ~3.5 m/s transient into slab-preheated
  gas), so the benchmark is a pure propagation-speed (eigenvalue)
  measurement: operator-splitting inconsistencies show up as profile drift.

  Frame and velocity: FLAME FRAME with prescribed mass flux. rho*u is
  uniform (= rho_u S_L_G with S_L_G the Cantera unity-Lewis guess); fresh
  mixture flows in through the right boundary (upwind inflow BC), burned
  gas flows out the left. This is exactly the frame of the Cantera
  FreeFlame solution the IC is taken from, so there is no startup
  transient. (The alternative — lab frame with u from the integrated
  continuity equation — was probed and rejected: the velocity field builds
  through a ~ms flame-acceleration transient that dwarfs the runtime
  budget.) For the steady equations d(rho u)/dx = 0 is exact, so a
  prescribed uniform mass flux is the consistent closure, not an
  approximation.

  Speed extraction: with the inflow at S_L_G, the flame drifts at
  dx_f/dt = S_L - S_L_G, so S_L = S_L_G + least-squares slope of x_f(t)
  over the measurement window (front position = centroid of
  (max(-dT/dx,0))^2). Cross-check: mass-flux integral
  S_L = integral(-omega_H2 W_H2) dx / (rho_u Y_H2,u), exact for a steady
  flame in any frame.

Run: python3 flame1d_replica.py            (three grids, ~2 h)
"""

import importlib.util
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_kinetics():
    spec = importlib.util.spec_from_file_location(
        "g30k", os.path.join(HERE, "gri30_h2_adiabatic_replica.py"))
    k = importlib.util.module_from_spec(spec)
    src = open(os.path.join(HERE, "gri30_h2_adiabatic_replica.py")).read()
    exec(compile(src.split('def main')[0], "g30k", "exec"), k.__dict__)
    return k


K = _load_kinetics()
SP = K.SP
NSP = 10

P0 = 101325.0          # Pa
R = 8.314462618        # J/mol/K
# molecular weights (g/mol) matching Cantera's gri30.yaml
MW = [2.016, 1.008, 15.999, 31.998, 17.007, 18.015, 33.006, 34.014,
      28.014, 39.95]

# ---- unity-Lewis diffusivity D(T) = lambda/(rho cp) from Cantera's own
# 29-reaction unity-Lewis free-flame solution at phi=1 (flame1d_cantera.py) ----
D_SAMPLES = [
    (300.0, 4.637e-05), (500.0, 1.042e-04), (800.0, 2.168e-04),
    (1200.0, 3.936e-04), (1600.0, 5.831e-04), (2000.0, 7.375e-04),
    (2300.0, 8.600e-04),
]
_lx = [math.log(t / 300.0) for t, _ in D_SAMPLES]
_ly = [math.log(d) for _, d in D_SAMPLES]
_mx = sum(_lx) / len(_lx)
_my = sum(_ly) / len(_ly)
D_N = sum((a - _mx) * (b - _my) for a, b in zip(_lx, _ly)) / \
    sum((a - _mx) ** 2 for a in _lx)
D0 = math.exp(_my - D_N * _mx)   # D = D0 * (T/300)^D_N


def D_of(T):
    return D0 * (T / 300.0) ** D_N


# ---- unburned mixture, phi=1 H2/air ----
X_U = {"H2": 2.0, "O2": 1.0, "N2": 3.76}
T_U = 300.0


def mixture_Y(xmap):
    tot = sum(xmap.values())
    num = [0.0] * NSP
    den = 0.0
    for i, s in enumerate(SP):
        x = xmap.get(s, 0.0) / tot
        num[i] = x * MW[i]
        den += x
    mtot = sum(num)
    return [n / mtot for n in num]


Y_U = mixture_Y(X_U)
W_U = sum(X_U[s] * MW[SP.index(s)] for s in X_U) / sum(X_U.values())  # g/mol
RHO_U = P0 / (R / (W_U * 1e-3) * T_U)

# Initial condition: the CONVERGED Cantera unity-Lewis phi=1 free-flame
# profile (flame1d_profile_phi1.json, dumped by flame1d_cantera.py) on a
# uniform 20 um grid, x = 0 at the T = 1500 K point, covering [-1, +5] mm.
# Rationale: the solver's own slab-ignition transient takes ~1 ms to reach
# the steady flame (probed: a hot-products slab collapses diffusively and
# the front re-establishes through a ~3.5 m/s transient into slab-preheated
# gas), which would blow the lean_single runtime budget. Starting at the
# converged profile turns the benchmark into a pure propagation-speed
# (eigenvalue) measurement: any inconsistency in our discrete operators
# shows up as profile drift, and S_L is measured on the relaxed flame.
_T_AD = 2387.64  # HP-equilibrium value, documentation only
# HP-equilibrium products at phi=1 (Cantera): Y_H2O at equilibrium,
# used by the activity mask (gas this close to equilibrium is skipped)
Y_H2O_EQ = 2.406933e-01


def load_profile(path=None):
    if path is None:
        path = os.path.join(HERE, "flame1d_profile_phi1.json")
    with open(path) as f:
        return json.load(f)


def rho_of(T, Y):
    # rho = p / (R_s T), R_s = R sum Y_i/W_i
    rs = R * sum(Y[i] / (MW[i] * 1e-3) for i in range(NSP))
    return P0 / (rs * T)


def cp_of(T, Y):
    # mixture isobaric heat capacity, J/kg/K
    return sum(Y[i] * (K.cp_r(i, T) * R / (MW[i] * 1e-3)) for i in range(NSP))


def cell_c(rho, Y):
    # mol/cm^3 from rho [kg/m3] and mass fractions
    return [rho * Y[i] / (MW[i] * 1e-3) / 1e6 for i in range(NSP)]


def cp_rhs(T, Y):
    """Isobaric (T, Y) right-hand side: dT/dt and dY_i/dt."""
    rho = rho_of(T, Y)
    c = cell_c(rho, Y)
    _, dc = K.uv_rhs(T, c)
    num = sum(K.h_rt(i, T) * dc[i] for i in range(NSP))
    den = sum(c[i] * K.cp_r(i, T) for i in range(NSP))
    dTdt = -T * num / den
    dY = [dc[i] * MW[i] * 1e3 / rho for i in range(NSP)]
    return dTdt, dY


def chem_substep(T, Y, dt_sub, n_sub):
    """Isobaric batch chemistry in one cell, stage-clamped RK4 substeps.

    Every RK stage state and the final update clamps Y to >= 0; without the
    clamp the HO2/H2O2 stages go negative above ~2400 K and the detailed-
    balance exp() overflows."""
    Y = [max(y, 0.0) for y in Y]
    for _ in range(n_sub):
        k1T, k1Y = cp_rhs(T, Y)
        T2 = T + 0.5 * dt_sub * k1T
        Y2 = [max(Y[i] + 0.5 * dt_sub * k1Y[i], 0.0) for i in range(NSP)]
        k2T, k2Y = cp_rhs(T2, Y2)
        T3 = T + 0.5 * dt_sub * k2T
        Y3 = [max(Y[i] + 0.5 * dt_sub * k2Y[i], 0.0) for i in range(NSP)]
        k3T, k3Y = cp_rhs(T3, Y3)
        T4 = T + dt_sub * k3T
        Y4 = [max(Y[i] + dt_sub * k3Y[i], 0.0) for i in range(NSP)]
        k4T, k4Y = cp_rhs(T4, Y4)
        T += dt_sub / 6.0 * (k1T + 2.0 * k2T + 2.0 * k3T + k4T)
        Y = [max(Y[i] + dt_sub / 6.0 *
                 (k1Y[i] + 2.0 * k2Y[i] + 2.0 * k3Y[i] + k4Y[i]), 0.0)
             for i in range(NSP)]
    return T, Y


def run(dx=4e-5, L=8e-3, x0=1.0e-3, dt_m=5e-7, t_end=4e-4,
        dt_c=1e-8, t_chem=1200.0, t_fast=2200.0, s_l_g=1.6543, label="base"):
    nc = int(round(L / dx))
    xs = [(j + 0.5) * dx for j in range(nc)]
    prof = load_profile()
    pdx = prof["dx"]
    px0 = prof["x0"]         # profile x=0 sits at domain x = x0
    pn = prof["n"]
    pT = prof["T"]
    pY = prof["Y"]
    T = [T_U] * nc
    Y = [list(Y_U) for _ in range(nc)]
    for j in range(nc):
        s = xs[j] - x0       # position relative to the flame anchor
        if s <= px0:
            T[j] = pT[0]
            Y[j] = list(pY[0])
        elif s >= px0 + (pn - 1) * pdx:
            T[j] = T_U
            Y[j] = list(Y_U)
        else:
            u = (s - px0) / pdx
            k = int(u)
            f = u - k
            T[j] = pT[k] * (1.0 - f) + pT[k + 1] * f
            Y[j] = [pY[k][i] * (1.0 - f) + pY[k + 1][i] * f
                    for i in range(NSP)]
    # pinned hot-bath zone: cells with s <= -6e-4 (0.6 mm behind the flame
    # anchor) are frozen at the profile state and act as a Dirichlet burned
    # reservoir. Without it the ~1 mm of hot gas behind the anchor drains
    # out the left boundary at |u| ~ 11 m/s in ~90 us, the plateau cools,
    # the front loses its thermal anchor and is flushed downstream
    # (smoke v6: drift ~ -1.4 m/s). Both chemistry and transport updates
    # skip pinned cells; fluxes from pinned cells into live neighbors remain.
    pinned = [(xs[j] - x0) <= -6e-4 for j in range(nc)]
    # burned-plateau state for the activity mask: skip chemistry ONLY for
    # gas essentially at HP equilibrium (T within 35 K of T_ad = 2387.64 K
    # AND Y_H2O >= 0.99*Y_H2O,eq). A looser plateau test was probed and
    # rejected: masking the recombination tail (2000-2350 K) freezes its
    # exothermic radical recombination, the plateau becomes a heat sink,
    # and the front is flushed downstream (drift ~ -1.4 m/s, smoke v5).
    y_h2o_mask = 0.99 * Y_H2O_EQ
    t_plateau = 2350.0
    # chemistry substepping: n_sub equal RK4 steps of <= dt_c per macro-step
    n_sub = max(1, int(dt_m / dt_c + 0.5))
    # adaptive: cells below t_fast react slowly and take 4x larger substeps
    n_sub_slow = max(1, n_sub // 4)
    hist = []
    t = 0.0
    step = 0
    # flame frame: prescribed uniform mass flux (negative: gas flows
    # right-to-left, fresh inflow through the right boundary)
    rhou = -RHO_U * s_l_g
    while t < t_end:
        # (1) chemistry on reactive cells. Reactivity test: hot gas
        # (T > t_chem), OR cold gas carrying a radical pool (H+O+OH+HO2 >
        # 1e-8). A pure T cutoff starves the flame: the Cantera profile's
        # 700-1200 K preheat foot carries diffused radicals whose
        # radical-seeded heat release holds the flame foot; masking it
        # blows the flame off (smoke v6/v7: foot dies, front collapses,
        # drift ~ -1.5 m/s). Radical-free fresh cells still skip cheaply.
        # Pinned reservoir cells and near-equilibrium burned gas skip too.
        n_active = 0
        for j in range(nc):
            if pinned[j]:
                continue
            if T[j] > t_plateau and Y[j][5] >= y_h2o_mask:
                continue
            if T[j] <= t_chem and (
                    Y[j][1] + Y[j][2] + Y[j][4] + Y[j][6]) <= 1e-8:
                continue
            ns = n_sub if T[j] > t_fast else n_sub_slow
            T[j], Y[j] = chem_substep(T[j], Y[j], dt_m / ns, ns)
            n_active += 1
        # (2) transport: diffusion, prescribed-flux advection, inflow/outflow
        rho = [rho_of(T[j], Y[j]) for j in range(nc)]
        cp = [cp_of(T[j], Y[j]) for j in range(nc)]
        Dd = [D_of(T[j]) for j in range(nc)]
        lam = [rho[j] * cp[j] * Dd[j] for j in range(nc)]
        # face quantities (nc-1 interior faces)
        rD_f = [0.5 * (rho[j] * Dd[j] + rho[j + 1] * Dd[j + 1])
                for j in range(nc - 1)]
        lam_f = [0.5 * (lam[j] + lam[j + 1]) for j in range(nc - 1)]
        rY = [[0.0] * NSP for _ in range(nc)]
        rT = [0.0] * nc
        # diffusion (zero-flux boundaries): rho cp dT/dt = -dq/dx,
        # rho dY/dt = -dj/dx with q = -lam dT/dx, j = -rho D dY/dx
        for f in range(nc - 1):
            dtx = (T[f + 1] - T[f]) / dx
            q = -lam_f[f] * dtx
            rT[f] -= q / (rho[f] * cp[f] * dx)
            rT[f + 1] += q / (rho[f + 1] * cp[f + 1] * dx)
            for i in range(NSP):
                jy = -rD_f[f] * (Y[f + 1][i] - Y[f][i]) / dx
                rY[f][i] -= jy / (rho[f] * dx)
                rY[f + 1][i] += jy / (rho[f + 1] * dx)
        # enthalpy-flux correction -sum(j_k cp_k) dT/dx at cell centers
        for j in range(nc):
            jm = max(j - 1, 0)
            jp = min(j + 1, nc - 1)
            dtx = (T[jp] - T[jm]) / ((jp - jm) * dx)
            s = 0.0
            for i in range(NSP):
                jyc = -rho[j] * Dd[j] * (Y[jp][i] - Y[jm][i]) / ((jp - jm) * dx)
                s += jyc * (K.cp_r(i, T[j]) * R / (MW[i] * 1e-3))
            rT[j] -= s * dtx / (rho[j] * cp[j])
        # advection (upwind) with the prescribed uniform mass flux:
        # interior species face fluxes
        for f in range(nc - 1):
            up = f if rhou >= 0.0 else f + 1
            for i in range(NSP):
                fl = rhou * Y[up][i]
                rY[f][i] -= fl / (rho[f] * dx)
                rY[f + 1][i] += fl / (rho[f + 1] * dx)
        # boundary faces: fresh inflow on the right, burned outflow on the left
        for i in range(NSP):
            rY[nc - 1][i] -= rhou * Y_U[i] / (rho[nc - 1] * dx)
            rY[0][i] += rhou * Y[0][i] / (rho[0] * dx)
        # T advection, cell-centered upwind; fresh ghost cell T_u on the right
        for j in range(nc):
            u = rhou / rho[j]
            if u >= 0.0:
                rT[j] -= u * (T[j] - T[max(j - 1, 0)]) / dx
            elif j < nc - 1:
                rT[j] -= u * (T[j + 1] - T[j]) / dx
            else:
                rT[j] -= u * (T_U - T[j]) / dx
        for j in range(nc):
            if pinned[j]:
                continue
            T[j] += dt_m * rT[j]
            ys = 0.0
            for i in range(NSP):
                Y[j][i] = min(1.0, max(0.0, Y[j][i] + dt_m * rY[j][i]))
                ys += Y[j][i]
            # renormalize: clipped mass fractions must still sum to 1
            if ys > 0.0:
                for i in range(NSP):
                    Y[j][i] /= ys
            else:
                Y[j] = list(Y_U)
        t += dt_m
        step += 1
        # tracked front: centroid of (max(-dT/dx,0))^2
        wsum = 0.0
        xsum = 0.0
        for f in range(nc - 1):
            g = -(T[f + 1] - T[f]) / dx
            if g > 0.0:
                w = g * g
                wsum += w
                xsum += (f + 1) * dx * w
        xf = xsum / wsum if wsum > 0.0 else float("nan")
        hist.append((t, xf, max(T)))
        if step % 50 == 0:
            print(f"[{label}] t={t*1e6:.0f}us T_max={max(T):.0f}K "
                  f"x_f={xf*1e3:.2f}mm active={n_active}", flush=True)
        if max(T) > 3500.0 or min(T) < 150.0:
            raise RuntimeError("diverged at t=%g" % t)
    # S_L = S_L_G + drift of the front: least-squares slope of x_f(t)
    # over the last 60% of the window
    t1 = 0.4 * t_end
    pts = [(tt, xx) for tt, xx, _ in hist if tt >= t1 and xx == xx]
    n = len(pts)
    mt = sum(p[0] for p in pts) / n
    mx = sum(p[1] for p in pts) / n
    drift = sum((p[0] - mt) * (p[1] - mx) for p in pts) / \
        sum((p[0] - mt) ** 2 for p in pts)
    sl = s_l_g + drift
    # mass-flux integral cross-check on the final state
    acc = 0.0
    for j in range(nc):
        if T[j] > 600.0:
            rho = rho_of(T[j], Y[j])
            c = cell_c(rho, Y[j])
            _, dc = K.uv_rhs(T[j], c)
            acc += -dc[0] * 1e6 * MW[0] * 1e-3 * dx   # kg/m2/s of H2
    sl_int = acc / (RHO_U * Y_U[0])
    print(f"[{label}] nc={nc} dx={dx*1e6:.0f}um dt_m={dt_m:.1e} "
          f"n_sub={n_sub} steps={step} active_cells(last)={n_active}")
    print(f"[{label}] S_L(guess)={s_l_g:.4f} m/s   drift={drift:+.4f} m/s   "
          f"S_L(slope)={sl:.4f} m/s   S_L(integral)={sl_int:.4f} m/s   "
          f"T_max={max(T):.1f} K   x_front={hist[-1][1]*1e3:.2f} mm")
    return sl, sl_int


def main():
    print(f"D(T) fit: D = {D0:.4e} * (T/300)^{D_N:.4f}  m^2/s")
    for tv, dv in D_SAMPLES:
        print(f"   T={tv:5.0f}  D_sample={dv:.3e}  D_fit={D_of(tv):.3e}  "
              f"err={100*(D_of(tv)/dv-1):+.2f}%")
    print(f"Y_U: H2={Y_U[0]:.5f} O2={Y_U[3]:.5f} N2={Y_U[8]:.5f}  "
          f"rho_u={RHO_U:.4f} kg/m3")
    run(label="base")
    run(dx=2e-5, L=8e-3, dt_m=1.25e-7, t_end=2e-4, label="fine")
    run(dx=8e-5, L=8e-3, dt_m=2e-6, t_end=4e-4, label="coarse")


if __name__ == "__main__":
    main()
