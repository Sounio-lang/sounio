#!/usr/bin/env python3
# Dev predictor for the sourced k_m(T) law path (Rosso 1993 CTMI).
# Mirrors demos/hydrogen/site_screening.sio exactly (RK4 replica idiom of
# replica_60c_pins.py) so we can predict the law-path numbers before
# running the 10-min demo, and check whether the 8-corner cardinal scan
# brackets the true min/max of ctmi over the 3-D cardinal box.

import math

SPY = 31557600.0

def ctmi(tc, tmin, topt, tmax):
    if tc <= tmin or tc >= tmax:
        return 0.0
    num = (tc - tmax) * (tc - tmin) ** 2
    den = (topt - tmin) * ((topt - tmin) * (tc - topt)
                           - (topt - tmax) * (topt + tmin - 2.0 * tc))
    return num / den

TMIN = (25.0, 40.0)   # 40 = Zeikus & Wolfe 1972; 25 = LABELED community floor (Tyne 2021 active at 29.2 C)
TOPT = (65.0, 70.0)   # Zeikus & Wolfe 1972 optimum
TMAX = (75.0, 90.0)   # 75 = Zeikus & Wolfe 1972 max; 90 = Head 2014 / Wilhelms 2001 field cutoff

def f_scan_corners(tc):
    vals = [ctmi(tc, a, b, c) for a in TMIN for b in TOPT for c in TMAX]
    return min(vals), max(vals)

def f_scan_dense(tc, n=21):
    lo, hi = 1e9, -1e9
    for i in range(n):
        tmin = TMIN[0] + (TMIN[1] - TMIN[0]) * i / (n - 1)
        for j in range(n):
            topt = TOPT[0] + (TOPT[1] - TOPT[0]) * j / (n - 1)
            for k in range(n):
                tmax = TMAX[0] + (TMAX[1] - TMAX[0]) * k / (n - 1)
                v = ctmi(tc, tmin, topt, tmax)
                lo, hi = min(lo, v), max(hi, v)
    return lo, hi

def corner_loss(t_k, a, km, salt, steps=600):
    t = t_k
    k1 = 10 ** (0.198 - 444.0 / t)
    k2 = 10 ** (2.84 - 2177.0 / t)
    k3 = 10 ** (-1.1 - 1737.0 / t)
    logk = -8.45 - ((-3.15 * 4184.0) / 19.14476) * (1.0 / t - 1.0 / 298.15)
    k4 = 10 ** (logk + 10.33) * 1.0e-7
    ks = [km, a * k1 * 1.0e-7 * SPY, a * k2 * SPY, a * k3 * SPY, a * k3 * SPY / k4]
    nu = [[-1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, -1.0, -1.0, -1.0, 1.0], [0.0, 1.0, 1.0, 1.0, -1.0],
          [0.0, 1.0, 2.0, 1.0, -1.0], [-1.0, 0.0, -1.0, 0.0, 0.0]]
    h2init = 7.8e-4 * 15.0 * salt
    y = [h2init, 0.0, 1.0, 1.0e-3, 5.0e-4, 5.0e-2]
    def dc(c):
        r = []
        for j in range(5):
            rate = ks[j]
            for s in range(6):
                cf = nu[s][j]
                if cf < 0.0:
                    rate *= c[s] ** int(-cf)
            r.append(rate)
        return [sum(nu[s][j] * r[j] for j in range(5)) for s in range(6)]
    DT = 0.05
    for _ in range(steps):
        k1v = dc(y)
        y2 = [y[i] + 0.5 * DT * k1v[i] for i in range(6)]
        k2v = dc(y2)
        y3 = [y[i] + 0.5 * DT * k2v[i] for i in range(6)]
        k3v = dc(y3)
        y4 = [y[i] + DT * k3v[i] for i in range(6)]
        k4v = dc(y4)
        y = [y[i] + (DT / 6.0) * (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i]) for i in range(6)]
    return 100.0 * (1.0 - y[0] / h2init)

AA = (3.0e-6, 3.0e-5)
SS = (0.70, 1.00)
KM = (0.0048, 0.0187)

def law_box(t_k):
    tc = t_k - 273.15
    flo, fhi = f_scan_dense(tc, 11)
    kks = (KM[0] * flo, KM[1] * fhi)
    if fhi == 0.0:
        kks = (0.0, 0.0)
        nk = 1
    else:
        nk = 2
    lo, hi = 1e9, -1e9
    for a in AA:
        for kk in kks[:nk]:
            for s in SS:
                l = corner_loss(t_k, a, kk, s)
                lo, hi = min(lo, l), max(hi, l)
    return lo, hi, flo, fhi

print("== corner-scan vs dense-grid (21^3) at site T points ==")
for tc in [52.5, 60.75, 65.0, 69.0, 70.0, 75.0, 76.0, 80.0, 85.0, 87.0, 95.0]:
    cl, ch = f_scan_corners(tc)
    dl, dh = f_scan_dense(tc)
    ok = "OK " if (cl <= dl + 1e-12 and ch >= dh - 1e-12) else "FAIL"
    print(f"  tc={tc:6.2f}  corners [{cl:.6f},{ch:.6f}]  dense [{dl:.6f},{dh:.6f}]  {ok}")

print("== predicted LAW-path boxes ==")
sites = [("S1", 368.15, 368.15), ("S2", 325.65, 342.15), ("S3", 338.15, 360.15)]
for name, tlo, thi in sites:
    tmid = 0.5 * (tlo + thi)
    res = {}
    for tag, tk in [("lo", tlo), ("mid", tmid), ("hi", thi)]:
        lo, hi, flo, fhi = law_box(tk)
        res[tag] = (lo, hi)
        print(f"  {name} T={tk-273.15:6.2f} C  f=[{flo:.4f},{fhi:.4f}]  law box [{lo:.6f},{hi:.6f}]")
    p_lo = min(res["lo"][0], res["hi"][0])
    p_hi = max(res["lo"][1], res["hi"][1])
    print(f"  {name} LAW p-box [{p_lo:.6f}, {p_hi:.6f}]   (slot was: "
          + {"S1": "[0,0]", "S2": "[0.084517, 2.275996]", "S3": "[0, 1.980550]"}[name] + ")")
    if name == "S3":
        for tc in [70.0, 75.0, 80.0, 85.0]:
            lo, hi, flo, fhi = law_box(tc + 273.15)
            print(f"    S3 fan extra T={tc:5.1f} C  f=[{flo:.4f},{fhi:.4f}]  law box [{lo:.6f},{hi:.6f}]")

# ─── FIELD VALIDATION predictor (mirrors the additive .sio section) ───
# field_pred_box: envelope of (k_eff x A x salt) corners at field tc over
# `steps` 0.05-yr steps; k_eff = k_m x f from the same cardinal p-box.
def field_pred_box(tc, steps):
    flo, fhi = f_scan_dense(tc, 11)
    if fhi == 0.0:
        kks, nk = (0.0, 0.0), 1
    else:
        kks, nk = (KM[0] * flo, KM[1] * fhi), 2
    lo, hi = 1e9, -1e9
    for a in AA:
        for kk in kks[:nk]:
            for s in SS:
                l = corner_loss(tc + 273.15, a, kk, s, steps)
                lo, hi = min(lo, l), max(hi, l)
    return lo, hi, flo, fhi

print("== FIELD VALIDATION predictions ==")
# LEHEN (Hellerschmied 2024, DOI 10.1038/s41560-024-01458-1): T=40 C
# MEASURED, tau 285 d -> steps 15 (273.9 d) / 16 (292.2 d) bracket it.
for st in (15, 16):
    lo, hi, flo, fhi = field_pred_box(40.0, st)
    print(f"  LEHEN tc=40 steps={st} ({st*0.05*365.25:.1f} d)  f=[{flo:.6f},{fhi:.6f}]  pred [{lo:.9f},{hi:.9f}]")
llo = min(field_pred_box(40.0, 15)[0], field_pred_box(40.0, 16)[0])
lhi = max(field_pred_box(40.0, 15)[1], field_pred_box(40.0, 16)[1])
print(f"  LEHEN envelope pred [{llo:.9f},{lhi:.9f}]  obs [3.0,3.2]  ratio obs_lo/pred_hi = {3.0/lhi:.2f}")
# LOBODICE (Smigan 1990 / Buzek 1994 via Tremosa 2023): T 25-45 C,
# tau 7 months ~ 210 d -> steps 11 (200.9 d) / 12 (219.2 d).
blo, bhi = 1e9, -1e9
for tc in (25.0, 30.0, 35.0, 40.0, 45.0):
    for st in (11, 12):
        lo, hi, flo, fhi = field_pred_box(tc, st)
        blo, bhi = min(blo, lo), max(bhi, hi)
        print(f"  LOBODICE tc={tc} steps={st} ({st*0.05*365.25:.1f} d)  f=[{flo:.6f},{fhi:.6f}]  pred [{lo:.9f},{hi:.9f}]")
print(f"  LOBODICE envelope pred [{blo:.9f},{bhi:.9f}]  obs [17.0,31.5]  ratio obs_lo/pred_hi = {17.0/bhi:.2f}")
print("== f-scan pins ==")
for tc in (40.0, 45.0):
    flo, fhi = f_scan_dense(tc, 11)
    print(f"  ctmi_scan({tc}) = [{flo:.12f}, {fhi:.12f}]")

# ─── FIELD CALIBRATION predictor (mirrors the additive .sio [A4] section) ──
# Inverse calibration: bisect k_m_eff (the ks[0] second-order constant at
# field T, units 1/(mol/L)/yr in the model's concentration-time units) so
# the RK4 network loss over the field horizon equals the observed extent.
# The H2 channel decouples: rate R0 = ks[0]*[H2]*[CO2] is the ONLY H2 sink,
# so the fractional loss is independent of the A and salt corners (verified
# below) — the inverse is taken at the (a_lo, salt=1.0) representative and
# the decoupling is printed as evidence.

def invert_k(tc, target_pct, steps, a, salt, iters=80):
    # bracket hi = 100: above ~1e3 the 0.05-yr RK4 step overshoots the
    # H2 charge and the loss goes non-monotone; 100 is stable and
    # already gives > 90 % loss at every field horizon.
    lo, hi = 1.0e-12, 100.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        l = corner_loss(tc + 273.15, a, mid, salt, steps)
        if l < target_pct:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def invert_box(tc, e_lo, e_hi, steps_lo, steps_hi):
    # envelope over the (extent edge x horizon edge x A x salt) corners:
    # k_lo = least k consistent (low extent, LONG horizon), k_hi = most
    # (high extent, SHORT horizon); A/salt corners shift the network's
    # CO2 supply weakly (decoupling check above) and are enveloped too.
    klo, khi = 1e9, -1e9
    for a in AA:
        for s in SS:
            klo = min(klo, invert_k(tc, e_lo, steps_hi, a, s))
            khi = max(khi, invert_k(tc, e_hi, steps_lo, a, s))
    return klo, khi

print("== FIELD CALIBRATION: decoupling check ==")
# fractional H2 loss must be (near-)independent of A and salt corners
for a in AA:
    for s in SS:
        l = corner_loss(313.15, a, 0.5, s, 15)
        print(f"  a={a:.1e} salt={s:.2f}  loss(15 steps, k=0.5) = {l:.9f}")

print("== FIELD CALIBRATION: inverse k_m_eff (field T, model units) ==")
# LEHEN: extent [3.0, 3.2] % over 285 d -> steps 15 (273.9 d) / 16 (292.2 d)
leh_k_lo, leh_k_hi = invert_box(40.0, 3.0, 3.2, 15, 16)
print(f"  LEHEN k_eff box [{leh_k_lo:.6f}, {leh_k_hi:.6f}]  (40 C, steps 15/16, A x salt enveloped)")
# cross-check the ends reproduce the observations
print(f"    check: loss(k_lo,16)={corner_loss(313.15, AA[0], leh_k_lo, 1.0, 16):.6f}  "
      f"loss(k_hi,15)={corner_loss(313.15, AA[1], leh_k_hi, 1.0, 15):.6f}")
# LOBODICE: extent [17.0, 31.5] % over ~210 d -> steps 11/12; T in 25..45 C
lob_k_lo, lob_k_hi = 1e9, -1e9
for tc in (25.0, 30.0, 35.0, 40.0, 45.0):
    k_lo, k_hi = invert_box(tc, 17.0, 31.5, 11, 12)
    lob_k_lo, lob_k_hi = min(lob_k_lo, k_lo), max(lob_k_hi, k_hi)
    print(f"  LOBODICE tc={tc}: k_eff [{k_lo:.6f}, {k_hi:.6f}]")
print(f"  LOBODICE k_eff envelope [{lob_k_lo:.6f}, {lob_k_hi:.6f}]  (BUZEK CAVEAT: leakage-inflated)")

# TYNE bridge: in-situ 73-109 mmol CH4 m^-3 yr^-1 (Olla, 29.2-50.7 C,
# DOI 10.1038/s41586-021-04153-3). Bridge to the model's ks[0]:
#   R0 volumetric rate = ks[0]*[H2]*[CO2] mol/L/yr -> x1e6 mmol/m^3/yr
#   [H2] = h2 charge 7.8e-4*15*salt mol/L (salt in [0.70,1.00]), [CO2]=0.05
#   stoichiometry 4 H2 : 1 CH4  ->  r_H2 = 4*r_CH4 in [292, 436] mmol/m^3/yr
# ASSUMPTION (labeled): Olla reservoir water carries the demo's screening
# H2 charge; the paper's "per cubic metre" reads as per m^3 of reservoir
# water. k scales inversely with the actual dissolved H2.
TYNE_R = (4.0 * 73.0, 4.0 * 109.0)   # mmol H2 m^-3 yr^-1
def tyne_k(r, salt):
    c0 = 7.8e-4 * 15.0 * salt
    return r / (c0 * 0.05 * 1.0e6)
tyne_k_lo = tyne_k(TYNE_R[0], SS[1])   # least k: lowest rate, highest [H2]
tyne_k_hi = tyne_k(TYNE_R[1], SS[0])   # most k: highest rate, lowest [H2]
print(f"== TYNE bridge (labeled assumptions) ==")
print(f"  r_H2 = 4 x [73,109] = [{TYNE_R[0]:.0f},{TYNE_R[1]:.0f}] mmol/m^3/yr; "
      f"[H2] charge [{7.8e-4*15*SS[0]:.5f},{7.8e-4*15*SS[1]:.5f}] mol/L")
print(f"  TYNE k_eff box [{tyne_k_lo:.6f}, {tyne_k_hi:.6f}]  (29.2-50.7 C)")

print("== OVERLAP ANALYSIS (k_eff at field T) ==")
ov_lo = max(leh_k_lo, tyne_k_lo)
ov_hi = min(leh_k_hi, tyne_k_hi)
print(f"  LEHEN [{leh_k_lo:.6f},{leh_k_hi:.6f}] n TYNE [{tyne_k_lo:.6f},{tyne_k_hi:.6f}]"
      f" = [{ov_lo:.6f},{ov_hi:.6f}]  {'NONEMPTY' if ov_lo <= ov_hi else 'EMPTY'}")
print(f"  LOBODICE [{lob_k_lo:.6f},{lob_k_hi:.6f}] disjoint above by "
      f"{lob_k_lo/tyne_k_hi:.2f}x (caveated)")

# FIELD-CALIBRATED k_m magnitude p-box at Topt (f = 1), model units:
#   LO = LEHEN k_lo / f_hi(40 C)   — caveat-free site, biology at p-box-best
#   HI = LOBODICE k_hi / f_lo(45 C) — caveated site, biology at p-box-worst
#     (labeled edge, NOT a strict bound: f at its p-box min at the warmest
#     reported T; smaller f at cooler T would push k_m higher)
#   TYNE cross-check minimal k_m = tyne_k_lo / f_hi(50.7 C) must lie <= LO.
f40_lo, f40_hi = f_scan_dense(40.0, 11)
f45_lo, f45_hi = f_scan_dense(45.0, 11)
f507_lo, f507_hi = f_scan_dense(50.7, 11)
KMF_LO = leh_k_lo / f40_hi
KMF_HI = lob_k_hi / f45_lo
tyne_min = tyne_k_lo / f507_hi
print("== FIELD-CALIBRATED k_m p-box at Topt (model units) ==")
print(f"  f_hi(40) = {f40_hi:.12f}  f_lo(45) = {f45_lo:.12f}  f_hi(50.7) = {f507_hi:.12f}")
print(f"  KMF = [{KMF_LO:.6f}, {KMF_HI:.6f}]   vs Bo-2021 [0.0048, 0.0187] LAB-FALSIFIED")
print(f"  ratio lo {KMF_LO/0.0187:.2f}x over Bo hi; hi {KMF_HI/0.0187:.2f}x")
print(f"  TYNE minimal k_m = {tyne_min:.6f}  ({'<= KMF_LO: consistent' if tyne_min <= KMF_LO else '> KMF_LO: TENSION'})")

# ─── field-calibrated LAW boxes (same machinery, KM -> KMF) ───
def law_box_field(t_k):
    tc = t_k - 273.15
    flo, fhi = f_scan_dense(tc, 11)
    kks = (KMF_LO * flo, KMF_HI * fhi)
    if fhi == 0.0:
        kks = (0.0, 0.0)
        nk = 1
    else:
        nk = 2
    lo, hi = 1e9, -1e9
    for a in AA:
        for kk in kks[:nk]:
            for s in SS:
                l = corner_loss(t_k, a, kk, s)
                lo, hi = min(lo, l), max(hi, l)
    return lo, hi, flo, fhi

print("== FIELD-CALIBRATED LAW p-boxes (dense T grid 2.5 C) ==")
FIELD_SITES = [("S1", 95.0, 95.0), ("S2", 52.5, 69.0), ("S3", 65.0, 87.0)]
site_fbox = {}
for name, tlo_c, thi_c in FIELD_SITES:
    plo, phi = 1e9, -1e9
    # .sio grid: n_g = ceil((thi-tlo)/2.5); ig = 0..n_g; tcg capped at thi
    n_g = max(1, math.ceil((thi_c - tlo_c) / 2.5))
    for ig in range(n_g + 1):
        t = min(tlo_c + 2.5 * ig, thi_c)
        lo, hi, flo, fhi = law_box_field(t + 273.15)
        plo, phi = min(plo, lo), max(phi, hi)
        print(f"  {name} T={t:6.2f} C  f=[{flo:.4f},{fhi:.4f}]  field-law box [{lo:.6f},{hi:.6f}]")
    site_fbox[name] = (plo, phi)
    print(f"  {name} FIELD-CALIBRATED LAW p-box [{plo:.6f}, {phi:.6f}]")

# ─── xorshift MC replica (valley-chain idiom; mirrors the .sio exactly) ──
# Validated below against the committed receipt (20.765 / 3.630 / 3.635 /
# 20.53 / 20.51 / stress 3.055 / 0.110) before being used for the new
# FIELD-CALIBRATED gate predictions.
M64 = (1 << 64) - 1
def _i64(v):
    v &= M64
    return v - (1 << 64) if v >= (1 << 63) else v

class Rng:
    def __init__(self, a=314159265, b=271828182):
        self.a, self.b = a, b
    def nxt(self):
        x, y = self.a, self.b
        self.a = y
        x = _i64(x ^ _i64(x << 23))
        x = _i64(x ^ (x >> 17))
        x = _i64(x ^ _i64(y ^ (y >> 26)))
        self.b = x
        r = _i64(x + y)
        return -r if r < 0 else r
    def u(self):
        return (self.nxt() % 10000) / 10000.0
    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0

def ss_ln(x):
    m, k = x, 0
    while m >= 2.0: m /= 2.0; k += 1
    while m < 1.0: m *= 2.0; k -= 1
    y = (m - 1.0) / (m + 1.0)
    y2 = y * y
    term, s = y, 0.0
    for i in range(12):
        s += term / (2.0 * i + 1.0)
        term *= y2
    return 2.0 * s + k * 0.6931471805599453

def ss_exp(x):
    if x > 700.0: return 1.0e200
    if x < -700.0: return 0.0
    kf = x / 0.6931471805599453
    k = int(kf)
    if kf < 0.0 and kf != float(k): k -= 1
    r = x - k * 0.6931471805599453
    s, term = 1.0, 1.0
    for n in range(1, 25):
        term *= r / n
        s += term
    for _ in range(abs(k)):
        s = s * 2.0 if k >= 0 else s / 2.0
    return s

def crf(r, n_yr):
    q = ss_exp(n_yr * ss_ln(1.0 + r))
    return r * q / (q - 1.0)

def prod_cost(p_elec, capex_m, e_spec, cf):
    a = 1500.0 * capex_m * crf(0.07, 25) + 20.0
    return e_spec * p_elec + a * e_spec / (cf * 8760.0)

def delivered(p_elec, capex_m, e_spec, cf, e_th, p_heat, ncyc, disp):
    return (prod_cost(p_elec, capex_m, e_spec, cf)
            + e_th * p_heat + 500.0 * crf(0.07, 25) / ncyc + disp)

def fs_of(l30_pct, tau):
    return 1.0 - l30_pct * tau / 3000.0

LO8 = [0.046, 0.8, 43.9, 0.55, 44.0, 0.005, 30.0, 0.50]
HI8 = [0.052, 1.2, 48.9, 0.80, 89.0, 0.02, 45.0, 1.50]

def _floor_i64(x):
    import math
    return math.floor(x)

def mc_hits_indep10(l30_lo, l30_hi, r_lo, r_hi, tau, n, seed=(314159265, 271828182)):
    rng = Rng(*seed)
    hits = 0
    ssum = 0.0
    ssq = 0.0
    for _ in range(n):
        # .sio stream order: 8 uniforms (v0..v4 kept), THEN the three
        # normals, THEN u5..u9 — drawing ph/ny/dp early shifts the stream.
        v = [LO8[c] + (HI8[c] - LO8[c]) * rng.u() for c in range(8)]
        e = v[0] + 0.002 * rng.normal()
        cc = v[3] + 0.02 * rng.normal()
        t = v[4] + 5.0 * rng.normal()
        ph = LO8[5] + (HI8[5] - LO8[5]) * rng.u()
        ny = LO8[6] + (HI8[6] - LO8[6]) * rng.u()
        dp = LO8[7] + (HI8[7] - LO8[7]) * rng.u()
        l30 = l30_lo + (l30_hi - l30_lo) * rng.u()
        rr = r_lo + (r_hi - r_lo) * rng.u()
        d = delivered(e, v[1], v[2], fs_of(l30, tau) * rr * cc, t, ph, ny, dp)
        ssum += d
        ssq += d * d
        if _floor_i64(d * 1000000.0 + 0.5) < 6000000:
            hits += 1
    return hits, ssum / n, (ssq / n - (ssum / n) ** 2) ** 0.5

def mc_hits_pinned_c(fs, fc, p_elec, capex_m, e_spec, cf, e_th, p_heat,
                     ncyc, disp, n, seed=(314159265, 271828182)):
    rng = Rng(*seed)
    hits = 0
    for _ in range(n):
        e = p_elec + 0.002 * rng.normal()
        c = cf + 0.02 * rng.normal()
        t = e_th + 5.0 * rng.normal()
        d = delivered(e, capex_m, e_spec, fs * fc * c, t, p_heat, ncyc, disp)
        if _floor_i64(d * 1000000.0 + 0.5) < 6000000:
            hits += 1
    return hits

def mc_hits_indep(n, seed=(314159265, 271828182)):
    rng = Rng(*seed)
    hits = 0
    for _ in range(n):
        # .sio stream order: 8 uniforms in the loop (v0..v4 kept), then the
        # three normals, then u5 (ph), u6 (ny), u7 (dp).
        v = [LO8[c] + (HI8[c] - LO8[c]) * rng.u() for c in range(8)]
        e = v[0] + 0.002 * rng.normal()
        cc = v[3] + 0.02 * rng.normal()
        t = v[4] + 5.0 * rng.normal()
        ph = LO8[5] + (HI8[5] - LO8[5]) * rng.u()
        ny = LO8[6] + (HI8[6] - LO8[6]) * rng.u()
        dp = LO8[7] + (HI8[7] - LO8[7]) * rng.u()
        d = delivered(e, v[1], v[2], cc, t, ph, ny, dp)
        if _floor_i64(d * 1000000.0 + 0.5) < 6000000:
            hits += 1
    return hits

print("== MC replica validation vs committed receipt (n = 20000) ==")
h, m, sd = mc_hits_indep10(0.0, 0.0, 0.0131, 0.9989, 1.0, 20000)
print(f"  S1 composed [{h/200:0.6f}] (committed 3.635000)  mean {m:.6f} (10.483580) std {sd:.6f} (9.212919)")
print(f"  no-coupling baseline [{mc_hits_indep(20000)/200:0.6f}] (committed 20.765000)")
h, _, _ = mc_hits_indep10(0.458073, 2.630814, 0.0131, 0.9989, 1.0, 20000)
print(f"  valley composed 25 C [{h/200:0.6f}] (committed 3.630000)")
h, _, _ = mc_hits_indep10(0.0, 0.0, 1.0, 1.0, 1.0, 20000)
print(f"  S1 subsurface-only [{h/200:0.6f}] (committed 20.530000)")
# stress-test committed values (annualized field extents)
leh_a_lo = 3.0 * 365.25 / 285.0
leh_a_hi = 3.2 * 365.25 / 285.0
lob_a_lo = 17.0 * 365.25 / 210.0
lob_a_hi = 31.5 * 365.25 / 210.0
h, _, _ = mc_hits_indep10(30.0 * leh_a_lo, 30.0 * leh_a_hi, 0.0131, 0.9989, 1.0, 20000)
print(f"  stress LEHEN-like [{h/200:0.6f}] (committed 3.055000)")
h, _, _ = mc_hits_indep10(30.0 * lob_a_lo, 30.0 * lob_a_hi, 0.0131, 0.9989, 1.0, 20000)
print(f"  stress LOBODICE-like [{h/200:0.6f}] (committed 0.110000)")

print("== FIELD-CALIBRATED gate predictions (n = 20000) ==")
for name, tlo_c, thi_c in FIELD_SITES:
    plo, phi = site_fbox[name]
    h, m, sd = mc_hits_indep10(plo, phi, 0.0131, 0.9989, 1.0, 20000)
    hsub, _, _ = mc_hits_indep10(plo, phi, 1.0, 1.0, 1.0, 20000)
    hb = mc_hits_pinned_c(fs_of(plo, 1.0), 0.9989, 0.046, 0.8, 43.9, 0.80,
                          44.0, 0.005, 45.0, 0.50, 20000)
    hw = mc_hits_pinned_c(fs_of(phi, 1.0), 0.0131, 0.052, 1.2, 48.9, 0.55,
                          89.0, 0.02, 30.0, 1.50, 20000)
    print(f"  {name} p-box [{plo:.6f},{phi:.6f}]  fs(1)=[{fs_of(phi,1.0):.6f},{fs_of(plo,1.0):.6f}]"
          f"  composed {h/200:.6f}  sub-only {hsub/200:.6f}  best {hb/200:.6f}  worst {hw/200:.6f}"
          f"  mean {m:.6f} std {sd:.6f}")
