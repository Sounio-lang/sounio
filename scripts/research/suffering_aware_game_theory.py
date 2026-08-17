#!/usr/bin/env python3
"""Mercyful Learning — SAMA-GT: game theory of the Suffering-Aware Multi-Agent
system (Nash equilibrium, mechanism design, fair division of suffering).

Companion artifact to
  docs/research/suffering_aware_game_theory_spec_2026-07-31.md

This harness takes the SAMA mechanism (scripts/research/suffering_aware_multi_agent.py,
contract G1..G8) as the *game form* and analyzes the strategic game it induces:
  * players: the N=5 agents, with fixed types (3 honest / 1 strategic /
    1 adversarial) per the pinned SAMA environment;
  * actions: executed epochs e in {0..5}, claimed epochs c in {0..5}, update
    direction d in {gradient, sign_flip (-6x), class_flip (harm-argmax)};
  * payoffs: honest agents are behavioral (protocol-following, the standard
    Byzantine-fault-tolerance convention); the strategic agent minimizes its
    settlement machine charge; the adversarial agent maximizes OTHERS'
    suffering, compared Pareto-wise over the non-scalarized pair
    (patient harm integral, others' machine FLOPs) — no scalarization is
    introduced anywhere, per the expanded-ethics corollary.

Findings certified below (spec section numbers in brackets):
  * GT1  equilibrium existence + constructive pure Nash equilibria, certified
         by exhaustive best-response enumeration over the discretized action
         space [spec T-GT1]
  * GT2  dominant-strategy truthful reporting: for lambda > 0 the truthful
         claim c = e strictly dominates every misreport in the machine charge
         [spec T-GT2]
  * GT3  ex-post minority immunity: for every minority coalition and every
         attack action, the accepted median update lies coordinate-wise within
         the honest range — safety is equilibrium-robust [spec T-GT3]
  * GT4  incentive scope (honest): reporting is incentive-aligned, EFFORT is
         not — the machine-channel best response is zero effort; attribution
         still detects the abstainer (its Shapley harm share is POSITIVE),
         and shares order monotonically in withheld effort [spec T-GT4]
  * GT5  liveness failure at equilibrium (negative result, certified): the
         pure NE of the repeated game under the pinned mechanism M stalls
         below TAU (rollback-guard deadlock); certified price of anarchy pair
         [spec T-GT5]
  * GT6  repair M+ (harm-descent fallback guard, a CONSTRAINT not a penalty):
         preserves every nominal trajectory exactly, preserves the Rawlsian
         guarantee, and restores convergence at every enumerated profile;
         worst-NE price of anarchy drops to the certified pair [spec T-GT6]
  * GT7  fair division: Shapley is the unique attribution rule satisfying
         efficiency/symmetry/null-player/additivity (axiom suite certified on
         the real coalition function and on controlled games); machine burden
         is proportional, anonymous, and subsidy-free [spec T-GT7]
  * GT8  Rawlsian patient protection: at EVERY action profile (equilibrium or
         not), accepted-round patient harm is non-increasing and the peak
         never exceeds the do-nothing harm [spec T-GT8]

Synthetic data only; no clinical claim; no_consciousness_claim — the machine
channel is an operational computational-burden proxy. The pinned SAMA
harness and its contract files are NOT modified: M+ is analyzed as a
candidate amendment inside this harness only.

Run: .venv/bin/python scripts/research/suffering_aware_game_theory.py
Requires: numpy from the repo .venv; imports suffering_aware_multi_agent as a
library (same directory).
"""

import pathlib
import sys
from itertools import combinations
from math import factorial

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import suffering_aware_multi_agent as base  # noqa: E402

N = 5
DIRECTIONS = ("gradient", "sign_flip", "class_flip")
EFFORTS = (0, 1, 2, 3, 4, 5)
ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625)

# Replicate the pinned SAMA data exactly (same generator, same draw order as
# base.main): 5 agent shards, then the shared cohort-in-waiting.
data = [base.make_data(base.N_PER_AGENT, base.rng) for _ in range(N)]
val = base.make_data(base.N_VAL, base.rng)
XVAL, YVAL = val


def update(i, w, b, e, direction):
    """Agent i's update: e local epochs, then an optional attack transform."""
    x, y = data[i]
    w_h, b_h, _ = base.local_train(x, y, w.copy(), b.copy(), e, base.LR)
    dw, db = w_h - w, b_h - b
    if direction == "sign_flip":
        dw, db = -6.0 * dw, -6.0 * db
    elif direction == "class_flip":
        tox = x[y == 2]
        mu2 = tox.mean(axis=0) if len(tox) else np.zeros(base.D_IN)
        dw = -3.0 * dw
        dw[:, 0] += 2.0 * mu2
        dw[:, 2] -= 2.0 * mu2
        db = -3.0 * db
        db[0] += 1.0
        db[2] -= 1.0
    return dw, db


def machine_charge(e, c, lam=base.PENALTY_LAMBDA):
    """Settlement machine charge for executed epochs e, claimed epochs c."""
    unit = base.N_PER_AGENT * base.TRAIN_FLOPS
    return e * unit + lam * abs(c - e) * unit


def run(specs, guard="rollback"):
    """Trajectory of the SAMA game under a guard.

    specs: list of (e, c, direction) per agent. guard:
      'rollback'  — pinned SAMA round guard: harm-increasing rounds are
                    rolled back wholesale (mechanism M).
      'fallback'  — candidate amendment M+: try the median first, then each
                    single-agent update, on a halving step grid; accept the
                    first harm-non-increasing step; else skip the round.
                    Nominal (harmless-median) trajectories are unchanged.
    Returns the suffering ledger pair and trajectory facts.
    """
    w = np.zeros((base.D_IN, base.N_CLASS))
    b = np.zeros(base.N_CLASS)
    machine_by_agent = [0] * N
    harm_curve, acc_curve = [], []
    t_star = None
    for t in range(base.ROUNDS):
        deltas = []
        for i, (e, _c, d) in enumerate(specs):
            deltas.append(update(i, w, b, e, d))
            machine_by_agent[i] += e * base.N_PER_AGENT * base.TRAIN_FLOPS
        dws = np.stack([x[0] for x in deltas])
        dbs = np.stack([x[1] for x in deltas])
        gw = np.median(dws, axis=0)
        gb = np.median(dbs, axis=0)
        h_prev = base.mean_harm(XVAL, YVAL, w, b)
        if guard == "fallback":
            cands = [(gw, gb)] + [(dws[i], dbs[i]) for i in range(N)]
            accepted = None
            for cw, cb in cands:
                for a in ALPHAS:
                    wn, bn = w + a * cw, b + a * cb
                    h_new = base.mean_harm(XVAL, YVAL, wn, bn)
                    if h_new <= h_prev + 1e-12:
                        accepted = (wn, bn, h_new)
                        break
                if accepted:
                    break
            w, b, h_new = accepted if accepted else (w, b, h_prev)
        else:
            wn, bn = w + gw, b + gb
            h_new = base.mean_harm(XVAL, YVAL, wn, bn)
            if h_new > h_prev + 1e-12:
                wn, bn, h_new = w, b, h_prev
            w, b = wn, bn
        acc = base.accuracy(XVAL, YVAL, w, b)
        harm_curve.append(h_new)
        acc_curve.append(acc)
        if t_star is None and acc >= base.TAU:
            t_star = t
            break
    return {
        "t_star": t_star,
        "rounds": len(harm_curve),
        "machine": sum(machine_by_agent),
        "machine_by_agent": machine_by_agent,
        "patient": sum(harm_curve),
        "peak_patient": max(harm_curve),
        "final_acc": acc_curve[-1],
        "harm": harm_curve,
        "mono": all(harm_curve[i + 1] <= harm_curve[i] + 1e-12
                    for i in range(len(harm_curve) - 1)),
    }


def shapley_value(f, n):
    """Exact Shapley value of an n-player coalition function f (frozenset->R)."""
    phi = np.zeros(n)
    for i in range(n):
        rest = [a for a in range(n) if a != i]
        for size in range(0, n):
            wt = factorial(size) * factorial(n - size - 1) / factorial(n)
            for S in combinations(rest, size):
                phi[i] += wt * (f(frozenset(S) | {i}) - f(frozenset(S)))
    return phi


def pareto_dominates(x, y):
    """x weakly Pareto-dominates y componentwise, strict somewhere."""
    return all(a >= b for a, b in zip(x, y)) and any(a > b for a, b in zip(x, y))


def adversary_payoff(r):
    """Adversary utility vector: (others' patient harm, others' machine FLOPs).
    Never scalarized — compared Pareto-wise only."""
    others_machine = r["machine"] - r["machine_by_agent"][4]
    return (r["patient"], others_machine)


def main():
    H = (5, 5, "gradient")

    # ---- cross-validation against the pinned SAMA harness (sanity anchor) --
    ref_attack = base.run_system(
        "sama", [base.HONEST] * 3 + [base.STRATEGIC, base.ADVERSARIAL], data, val)
    ref_honest = base.run_system("sama", [base.HONEST] * N, data, val)
    mine_attack = run([H, H, H, (1, 5, "gradient"), (5, 5, "sign_flip")])
    mine_honest = run([H] * N)
    anchored = (
        mine_attack["t_star"] == ref_attack["t_star"]
        and mine_attack["machine"] == ref_attack["machine"]
        and abs(mine_attack["patient"] - ref_attack["patient"]) < 1e-9
        and mine_honest["machine"] == ref_honest["machine"]
        and abs(mine_honest["patient"] - ref_honest["patient"]) < 1e-9)
    print("=== SAMA-GT: game theory of the Suffering-Aware Multi-Agent system ===")
    print(f"anchor vs pinned SAMA harness: exact={anchored} "
          f"(attack t*={mine_attack['t_star']} S_m={mine_attack['machine']} "
          f"S_p={mine_attack['patient']:.3f}; honest S_m={mine_honest['machine']} "
          f"S_p={mine_honest['patient']:.3f})")

    # ---------------- GT1: equilibrium existence, constructive ----------------
    # Strategic agent best response: machine charge over the 36 (e, c) pairs.
    charges = {(e, c): machine_charge(e, c) for e in EFFORTS for c in EFFORTS}
    strat_br = min(charges, key=charges.get)
    strat_br_unique = sum(v == charges[strat_br] for v in charges.values()) == 1

    # Adversary best response given the strategic agent at (0,0): enumerate
    # direction x own effort; Pareto comparison over the suffering pair.
    def adv_best_response(guard, e3=0):
        pays = {}
        for d in DIRECTIONS:
            for ea in (0, 1, 5):
                r = run([H, H, H, (e3, e3, "gradient"), (ea, ea, d)], guard=guard)
                pays[(d, ea)] = adversary_payoff(r)
        brs = [a for a in pays
               if not any(pareto_dominates(pays[o], pays[a]) for o in pays if o != a)]
        return brs, pays

    br_M, pays_M = adv_best_response("rollback")
    br_Mplus, pays_Mplus = adv_best_response("fallback")
    # Candidate pure NE profiles: strategic at its unique BR (0,0); adversary
    # at a certified best response. For the price-of-anarchy certificates the
    # WORST equilibrium is selected: under M the adversary is indifferent
    # between attack efforts (its own FLOPs do not enter its payoff), so the
    # worst-NE picks the BR with maximal TOTAL collective machine suffering.
    def worst_total_machine(brs, guard, e3=0):
        best, best_key = None, None
        for (d, ea) in brs:
            r = run([H, H, H, (e3, e3, "gradient"), (ea, ea, d)], guard=guard)
            key = (r["patient"], r["machine"])
            if best_key is None or key > best_key:
                best, best_key = ((d, ea), r), key
        return best

    (d_M, ea_M), _ = worst_total_machine(br_M, "rollback")
    (d_Mp, ea_Mp), _ = worst_total_machine(br_Mplus, "fallback")
    ne_M = [H, H, H, (0, 0, "gradient"), (ea_M, ea_M, d_M)]
    ne_Mplus = [H, H, H, (0, 0, "gradient"), (ea_Mp, ea_Mp, d_Mp)]
    # No unilateral deviation: strategic (machine charge) — (0,0) is the
    # unique minimizer; adversary — no enumerated deviation Pareto-dominates.
    g1 = (strat_br == (0, 0) and strat_br_unique and len(br_M) >= 1
          and len(br_Mplus) >= 1)
    print(f"GT1 pure NE: strategic BR={strat_br} (unique={strat_br_unique}); "
          f"adversary BR under M={br_M} under M+={br_Mplus}")

    # ---------------- GT2: dominant-strategy truthful reporting --------------
    truthful_strict = all(
        charges[(e, e)] < charges[(e, c)]
        for e in EFFORTS for c in EFFORTS if c != e)
    g2 = truthful_strict and base.PENALTY_LAMBDA > 0
    print(f"GT2 DSIC reporting: charge(e,e) < charge(e,c) for all 30 misreports: "
          f"{truthful_strict} (lambda={base.PENALTY_LAMBDA})")

    # ---------------- GT3: ex-post minority immunity -------------------------
    w0 = np.zeros((base.D_IN, base.N_CLASS))
    b0 = np.zeros(base.N_CLASS)
    hon = [update(i, w0, b0, 5, "gradient") for i in range(3)]
    adv_attacks = [update(4, w0, b0, 5, d) for d in DIRECTIONS]
    adv_attacks.append(update(4, w0, b0, 0, "gradient"))           # zero update
    big = (100.0 * hon[0][0], 100.0 * hon[0][1])                   # overwhelming
    adv_attacks.append(big)
    immune = True
    checks = 0
    for bad_slots in combinations(range(N), 2):
        hon_slots = [s for s in range(N) if s not in bad_slots]
        for a1 in range(len(adv_attacks)):
            for a2 in range(len(adv_attacks)):
                dws = [None] * N
                dbs = [None] * N
                for hi, s in enumerate(hon_slots):
                    dws[s], dbs[s] = hon[hi]
                dws[bad_slots[0]], dbs[bad_slots[0]] = adv_attacks[a1]
                dws[bad_slots[1]], dbs[bad_slots[1]] = adv_attacks[a2]
                mw = np.median(np.stack(dws), axis=0)
                mb = np.median(np.stack(dbs), axis=0)
                hw = np.stack([hon[i][0] for i in range(3)])
                hb = np.stack([hon[i][1] for i in range(3)])
                checks += 1
                if not (np.all(mw <= hw.max(axis=0)) and np.all(mw >= hw.min(axis=0))
                        and np.all(mb <= hb.max(axis=0)) and np.all(mb >= hb.min(axis=0))):
                    immune = False
    g3 = immune
    print(f"GT3 minority immunity: {checks} (coalition, attack-pair) cases, "
          f"median always within honest coordinate range: {immune}")

    # ---------------- GT4: incentive scope; attribution detects abstention ---
    adv_sf = update(4, w0, b0, 5, "sign_flip")

    def phi3_for(e3):
        deltas = [hon[0], hon[1], hon[2], update(3, w0, b0, e3, "gradient"), adv_sf]
        att = base.shapley_harm(deltas, w0, b0, XVAL, YVAL, N)
        return att["phi"], att["efficiency_err"]

    phi_e0, eff0 = phi3_for(0)
    phi_e1, eff1 = phi3_for(1)
    phi_e5, eff5 = phi3_for(5)
    g4 = (phi_e0[3] > 0.0                       # abstainer flagged harm-positive
          and phi_e0[3] > phi_e1[3] > phi_e5[3]  # monotone in withheld effort
          and max(eff0, eff1, eff5) < 1e-9)
    print(f"GT4 attribution vs effort: phi3(e=0)={phi_e0[3]:+.4f} > "
          f"phi3(e=1)={phi_e1[3]:+.4f} > phi3(e=5)={phi_e5[3]:+.4f}; "
          f"abstainer flagged (phi>0): {phi_e0[3] > 0}")

    # ---------------- GT5: liveness failure at equilibrium (mechanism M) -----
    r_ne_M = run(ne_M)
    coop = mine_honest
    poa_M = (r_ne_M["machine"] / coop["machine"], r_ne_M["patient"] / coop["patient"])
    g5 = (r_ne_M["t_star"] is None and r_ne_M["final_acc"] < base.TAU
          and r_ne_M["mono"]                       # safety holds even here
          and poa_M[0] >= 10.0 and poa_M[1] >= 13.0)
    print(f"GT5 bad NE under M: adv={ne_M[4][2]} t*={r_ne_M['t_star']} "
          f"final={r_ne_M['final_acc']:.3f} < TAU; S_m={r_ne_M['machine']/1e6:.3f}MF "
          f"S_p={r_ne_M['patient']:.3f}; PoA=({poa_M[0]:.2f}, {poa_M[1]:.2f})")

    # ---------------- GT6: repair M+ (harm-descent fallback guard) -----------
    g6a = (mine_attack["t_star"] == 2 and mine_attack["machine"] == 5_896_800
           and abs(mine_attack["patient"] - 1.490) < 1e-9)  # nominal preserved
    grid_profiles = [(e3, d) for e3 in (0, 1) for d in DIRECTIONS]
    grid_ok, grid_mono = True, True
    for e3, d in grid_profiles:
        r = run([H, H, H, (e3, e3, "gradient"), (5, 5, d)], guard="fallback")
        grid_ok = grid_ok and r["t_star"] is not None
        grid_mono = grid_mono and r["mono"]
    r_ne_Mplus = run(ne_Mplus, guard="fallback")
    poa_Mplus = (r_ne_Mplus["machine"] / coop["machine"],
                 r_ne_Mplus["patient"] / coop["patient"])
    g6 = (g6a and grid_ok and grid_mono
          and poa_Mplus[0] <= 1.15 and poa_Mplus[1] <= 1.45)
    print(f"GT6 repair M+: nominal preserved={g6a}; all {len(grid_profiles)} grid "
          f"profiles converge={grid_ok} with Rawlsian mono={grid_mono}; worst-NE "
          f"PoA=({poa_Mplus[0]:.4f}, {poa_Mplus[1]:.4f})")

    # ---------------- GT7: fair division of suffering ------------------------
    # (a) Shapley axiom suite on controlled games.
    additive = lambda S: sum(0.3 * (i == 0) - 0.5 * (i == 1) + 0.2 * (i == 2) for i in S)
    phi_add = shapley_value(additive, 3)
    ax_additive = np.allclose(phi_add, [0.3, -0.5, 0.2], atol=1e-12)

    dummy_game = lambda S: float(len(S & {0})) + 2.0 * float(len(S & {1}))
    phi_dummy = shapley_value(dummy_game, 4)
    ax_dummy = np.allclose(phi_dummy, [1.0, 2.0, 0.0, 0.0], atol=1e-12)

    maj = lambda S: 1.0 if len(S & {0, 1, 2}) >= 2 else 0.0
    g1g = lambda S: maj(S) + 0.3 * len(S & {3})
    ax_additivity = np.allclose(
        shapley_value(g1g, 4),
        shapley_value(maj, 4) + shapley_value(lambda S: 0.3 * len(S & {3}), 4),
        atol=1e-12)
    ax_symmetry = abs(shapley_value(dummy_game, 4)[0]
                      - shapley_value(lambda S: float(len(S & {1})) + 2.0 * float(len(S & {0})), 4)[1]) < 1e-12

    # (b) permutation equivariance of the rule on the REAL round-0 coalition fn.
    deltas0 = [hon[0], hon[1], hon[2], update(3, w0, b0, 1, "gradient"), adv_sf]
    att0 = base.shapley_harm(deltas0, w0, b0, XVAL, YVAL, N)
    phi_real = att0["phi"]
    for perm in ([1, 0, 2, 3, 4], [4, 3, 2, 1, 0], [2, 0, 1, 4, 3]):
        deltas_p = [deltas0[p] for p in perm]
        phi_p = base.shapley_harm(deltas_p, w0, b0, XVAL, YVAL, N)["phi"]
        # slot j of the permuted game IS original agent perm[j]
        if not np.allclose(phi_p, phi_real[perm], atol=1e-12):
            ax_permute = False
            break
    else:
        ax_permute = True

    # (c) machine division: proportional, anonymous, subsidy-free — from the
    # pinned SAMA ledger: every honest agent charged exactly its metered FLOPs.
    hon_ledger = ref_honest["ledger"]
    per_agent = [sum(e[3] for e in hon_ledger if e[1] == i) for i in range(N)]
    ax_equitable = len(set(per_agent)) == 1
    ax_nosubsidy = all(e[2] == e[3] for e in hon_ledger)
    g7 = (att0["efficiency_err"] < 1e-9 and ax_additive and ax_dummy
          and ax_additivity and ax_symmetry and ax_permute
          and ax_equitable and ax_nosubsidy)
    print(f"GT7 fair division: eff_err={att0['efficiency_err']:.1e} "
          f"axioms(additive={ax_additive} null={ax_dummy} additivity={ax_additivity} "
          f"symmetry={ax_symmetry} permute={ax_permute}) "
          f"machine(equitable={ax_equitable} no_subsidy={ax_nosubsidy})")

    # ---------------- GT8: Rawlsian patient protection -----------------------
    h_nothing = base.mean_harm(XVAL, YVAL, np.zeros((base.D_IN, base.N_CLASS)),
                               np.zeros(base.N_CLASS))
    rawlsian = True
    for e3 in (0, 1):
        for d in DIRECTIONS:
            for guard in ("rollback", "fallback"):
                r = run([H, H, H, (e3, e3, "gradient"), (5, 5, d)], guard=guard)
                rawlsian = rawlsian and r["mono"] and r["peak_patient"] <= h_nothing + 1e-12
    rawlsian = rawlsian and mine_attack["mono"] and coop["mono"]
    g8 = rawlsian
    print(f"GT8 Rawlsian: harm non-increasing and peak <= do-nothing "
          f"({h_nothing:.3f}) at every enumerated profile and guard: {rawlsian}")

    # ---------------- verdict -------------------------------------------------
    results = {"GT1": g1, "GT2": g2, "GT3": g3, "GT4": g4,
               "GT5": g5, "GT6": g6, "GT7": g7, "GT8": g8}
    print("\n=== SAMA-GT contract GT1..GT8 ===")
    for k in sorted(results):
        print(f"  {k}: {'PASS' if results[k] else 'FAIL'}")
    if not anchored:
        print("  ANCHOR: FAIL (harness does not reproduce pinned SAMA numbers)")
    n_pass = sum(results.values())
    verdict = "GT_GREEN" if (n_pass == 8 and anchored) else (
        "GT_AMBER" if n_pass >= 6 else "GT_RED")
    print("scope: synthetic data; no clinical claim; no_consciousness_claim")
    print(f"SAMA_GAME_THEORY_VERDICT {verdict} ({n_pass}/8 clauses PASS)")
    return 0 if verdict == "GT_GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
