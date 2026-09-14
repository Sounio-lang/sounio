#!/usr/bin/env python3
"""Exhaustive check, on every state of a periodic 4x4 lattice, that dimer
adsorption/desorption without diffusion IS Kawasaki spin-exchange dynamics of
an Ising ferromagnet, under the gauge transform tau_i = s_i * (-1)^(r+c),
s_i = 2*sigma_i - 1.

What is checked, for all 65,536 states and all 32 bonds:
  1. a dimer move is allowed on a bond  <=>  tau is antiparallel on it;
  2. the state after the dimer move equals tau with the two spins exchanged;
  3. 2*Q = sum(tau): the conserved sublattice charge is the tau magnetisation;
  4. sum s_i s_j = 4*pairs - 8*occ + 2N and sum tau_i tau_j = -sum s_i s_j,
     so K^(occ/2) w^pairs = const * exp(-(ln w / 4) sum tau tau)
                                  * exp((ln K / 2 + 2 ln w) occ):
     a ferromagnet in tau with beta*J = ln(1/w)/4, and a uniform field on occ
     (a staggered field on tau) that vanishes exactly at K = w^-4;
  5. no single-adatom hop conserves sum(tau): hops are non-conserving pair
     flips of tau, so adding them breaks model-B dynamics.

Consequence for this directory: the no-diffusion c(2x2) coarsening in
kmc_c2x2_growth.sio / kmc_c2x2_production.sio is 2D Kawasaki Ising coarsening
(Huse 1986; Amar, Sullivan, Mountain 1988), not a new universality class, and
the Q=0 equilibrium question is the fixed-magnetisation Ising model. The
conservation law itself is Barma, Grynberg, Stinchcombe (1993).
"""
L = 4
N = L * L


def main():
    eps = [1 if (i // L + i % L) % 2 == 0 else -1 for i in range(N)]
    bonds = [(r * L + c, r * L + (c + 1) % L) for r in range(L) for c in range(L)] + \
            [(r * L + c, ((r + 1) % L) * L + c) for r in range(L) for c in range(L)]
    fails = {k: 0 for k in ("allowed", "result", "charge", "energy", "hop")}
    for s in range(1 << N):
        sig = [(s >> i) & 1 for i in range(N)]
        spin = [2 * x - 1 for x in sig]
        tau = [spin[i] * eps[i] for i in range(N)]
        occ = sum(sig)
        pairs = sum(sig[i] * sig[j] for i, j in bonds)
        ss = sum(spin[i] * spin[j] for i, j in bonds)
        tt = sum(tau[i] * tau[j] for i, j in bonds)
        if 2 * sum(sig[i] * eps[i] for i in range(N)) != sum(tau):
            fails["charge"] += 1
        if ss != 4 * pairs - 8 * occ + 2 * N or tt != -ss:
            fails["energy"] += 1
        for i, j in bonds:
            dimer = sig[i] == sig[j]
            if dimer != (tau[i] != tau[j]):
                fails["allowed"] += 1
            if dimer:
                exch = tau[:]
                exch[i], exch[j] = tau[j], tau[i]
                after = sig[:]
                after[i] ^= 1
                after[j] ^= 1
                if exch != [(2 * after[k] - 1) * eps[k] for k in range(N)]:
                    fails["result"] += 1
            else:
                hop = sig[:]
                hop[i], hop[j] = sig[j], sig[i]
                if sum((2 * hop[k] - 1) * eps[k] for k in range(N)) == sum(tau):
                    fails["hop"] += 1
    for k, v in fails.items():
        print(f"{k:>8}: {v} failures")
    ok = all(v == 0 for v in fails.values())
    print("KAWASAKI_MAPPING_OK" if ok else "KAWASAKI_MAPPING_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
