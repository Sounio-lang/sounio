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
