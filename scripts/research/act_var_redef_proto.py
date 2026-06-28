#!/usr/bin/env python3
"""
Prototype: redefining Sounio's act_var into a genuinely CONCENTRATING,
nonstationarity-robust variance bonus (UCB-V / empirical-Bernstein), and
demonstrating it resolves the decay-vs-concentration tension.

We track ONE focal variable's per-conflict reward stream with a REGIME SHIFT:
  - reward r_t in [0,1] (bounded -> UCB-V valid)
  - true mean p=0.70 for t<2000, then p=0.20 for t>=2000 (the variable stops
    being conflict-relevant). A correct uncertainty bonus should:
      (a) SHRINK as evidence accrues within a stable regime (concentrate),
      (b) TRANSIENTLY RISE right after the shift (genuine new uncertainty),
      (c) stay BOUNDED (never the monotonically-growing pathology).

Three estimators of the per-variable variance bonus, scored identically as
  score = mean + beta*bonus :
  (1) GROWING  -- the current smt.sio act_var: sum of bump^2 with geometrically
                 growing bump (1/0.95 per conflict). Bonus = sqrt(act_var).
  (2) UCBV_STAT-- empirical-Bernstein over ALL history (stationary UCB-V).
  (3) UCBV_WIN -- empirical-Bernstein over a sliding WINDOW (the proposed fix).
"""
import math, random

random.seed(1234)
T = 4000
SHIFT = 2000
P_BEFORE, P_AFTER = 0.70, 0.20
ZETA = 1.2          # UCB-V log constant
B = 1.0             # reward range bound (rewards in [0,1])
W = 200             # sliding window for the proposed estimator

def true_p(t): return P_BEFORE if t < SHIFT else P_AFTER

# ---- estimator state ----
# (1) growing act_var (mirrors smt.sio: act_var += bump^2 * k ; bump *= 1/0.95)
grow_var = 0.0
bump = 1.0
# (2) stationary UCB-V: running n, sum, sumsq over all history
st_n = 0; st_sum = 0.0; st_sumsq = 0.0
# (3) windowed UCB-V: last W rewards
win = []
# (4) DISCOUNTED empirical-Bernstein (the PORTABLE form: O(1) storage = 3 floats/var,
#     drop-in for smt.sio's existing per-variable accumulators). gamma<1 ages out old
#     evidence so n_eff -> 1/(1-gamma) is bounded -> bonus concentrates to a floor AND
#     adapts to regime shifts, with NO ring buffer.
GAMMA = 0.99            # n_eff steady-state = 1/(1-gamma) = 100
dS0 = 0.0; dS1 = 0.0; dS2 = 0.0

def ucbv_bonus(mean, var, n, horizon):
    """UCB-V bias term (Audibert/Munos/Szepesvari 2009), b=B, c=1.
    `horizon` is the effective time used in the log factor: global t for the
    stationary estimator, but LOCAL min(t,W) for the windowed one (sliding-window
    UCB, Garivier-Moulines 2008) so the bonus concentrates instead of drifting up."""
    if n <= 0: return float('inf')
    E = ZETA * math.log(max(horizon, 2))
    return math.sqrt(2.0 * max(var, 0.0) * E / n) + 3.0 * B * E / n

rows = []
checkpoints = {100, 500, 1000, 1900, 1999, 2050, 2200, 2600, 4000}
for t in range(1, T + 1):
    r = 1.0 if random.random() < true_p(t) else 0.0

    # (1) growing act_var
    grow_var += (bump * bump) * 0.5      # *0.5 ~ (L-1)/L*div_mult lumped constant
    bump *= (1.0 / 0.95)
    grow_bonus = math.sqrt(grow_var)

    # (2) stationary UCB-V
    st_n += 1; st_sum += r; st_sumsq += r * r
    st_mean = st_sum / st_n
    st_var = max(st_sumsq / st_n - st_mean * st_mean, 0.0)
    st_bonus = ucbv_bonus(st_mean, st_var, st_n, t)          # global horizon t

    # (3) windowed UCB-V
    win.append(r)
    if len(win) > W: win.pop(0)
    wn = len(win)
    wmean = sum(win) / wn
    wvar = max(sum(x * x for x in win) / wn - wmean * wmean, 0.0)
    w_bonus = ucbv_bonus(wmean, wvar, wn, min(t, W))         # LOCAL horizon min(t,W)

    # (4) discounted empirical-Bernstein (O(1) storage)
    dS0 = GAMMA * dS0 + 1.0
    dS1 = GAMMA * dS1 + r
    dS2 = GAMMA * dS2 + r * r
    dmean = dS1 / dS0
    dvar = max(dS2 / dS0 - dmean * dmean, 0.0)
    d_bonus = ucbv_bonus(dmean, dvar, dS0, dS0)              # local horizon = n_eff

    if t in checkpoints:
        rows.append((t, true_p(t), grow_bonus, st_mean, st_bonus, wmean, w_bonus, dmean, dvar, d_bonus))

print(f"{'t':>5} {'p*':>4} | {'GROW bonus':>11} | {'STAT bon':>8} | {'WIN bon':>7} | {'DISC mean':>9} {'DISC var':>8} {'DISC bon':>8}")
print("-" * 86)
for (t, p, g, sm, sb, wm, wb, dm, dv, db) in rows:
    tag = "  <- shift" if t in (1999, 2050) else ""
    print(f"{t:>5} {p:>4.2f} | {g:>11.3g} | {sb:>8.4f} | {wb:>7.4f} | {dm:>9.3f} {dv:>8.4f} {db:>8.4f}{tag}")

def at(tt, idx): return next(r[idx] for r in rows if r[0] == tt)
# indices: 2=grow 4=stat 6=win 7=dmean 8=dvar 9=dbonus
print("\n=== VERDICTS (DISC = the O(1)-storage portable form for smt.sio) ===")
print(f"(1) GROWING act_var:  t100={at(100,2):.3g} -> t1900={at(1900,2):.3g}   "
      f"GROWS x{at(1900,2)/at(100,2):.1e}  -> PATHOLOGY (more evidence = MORE 'uncertain')")
print(f"(2) STAT UCB-V:  500={at(500,4):.4f} -> 1900={at(1900,4):.4f} (concentrates) but "
      f"shift-peak 2050={at(2050,4):.4f} ~ flat -> STALE (cannot adapt)")
print(f"(3) DISC (portable): mean tracks shift 0.69->{at(2600,7):.2f}; "
      f"bonus concentrates to a FLOOR (warmup t100={at(100,9):.4f} -> floor t1900={at(1900,9):.4f}); "
      f"transient RISE at shift-peak 2050={at(2050,9):.4f} (>floor: {at(2050,9)>at(1900,9)}); "
      f"re-concentrates 2600={at(2600,9):.4f} (<peak: {at(2600,9)<at(2050,9)})")
print(f"    BOUNDED: max DISC bonus = {max(r[9] for r in rows):.4f}   vs   GROW max = {max(r[2] for r in rows):.3g}")
print(f"    NOTE: floor>0 is CORRECT for nonstationarity — bounded n_eff=1/(1-gamma) keeps")
print(f"          irreducible uncertainty (the world may shift), unlike stationary UCB-V -> 0.")
ok = (at(1900,2) > at(100,2)*1000) \
     and (at(1900,4) < at(500,4)) and (abs(at(2050,4)-at(1900,4)) < 0.01) \
     and (at(100,9) > at(1900,9)) and (at(2050,9) > at(1900,9)) and (at(2600,9) < at(2050,9)) \
     and (max(r[9] for r in rows) < 1.0) and (abs(at(2600,7)-0.20) < 0.05)
print(f"\nRESULT: discounted empirical-Bernstein resolves the decay-vs-concentration nut")
print(f"  (bounded + concentrates within regime + transient-rise-then-reconcentrate at shift")
print(f"   + mean tracks, all with 3 floats/var = drop-in for smt.sio act_mean/act_var) = {ok}")
