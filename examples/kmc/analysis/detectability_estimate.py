#!/usr/bin/env python3
"""Why existing Video-LEED data cannot test the half-order spot prediction, and
what an experiment needs. Kinematic order-of-magnitude estimate.

Parseval: for the sublattice sign field, the diffuse intensity averaged over
the Brillouin zone is the coverage theta per site. A perfectly ordered c(2x2)
or p(2x2) layer at the same coverage puts a fraction theta of that total into
one Bragg mode. Immobile, uncorrelated pairs put S_centre = theta * R per site
at the spot centre (R = 2 P_same, the prediction card's ratio), spread
smoothly, so an instrument resolution element of area fraction f of the zone
collects R * f of the total. Hence

    diffuse (in one resolution element) / ordered Bragg spot  =  R f / theta,

with f = pi sigma_q^2 / (2 pi)^2 for a resolution disk of radius sigma_q in
reciprocal lattice units, sigma_q = 2 pi / l_coh, l_coh in lattice constants.

Evidence this addresses: Wang, Ames Lab thesis IS-T-1519 (1990), p. 24:
"patterns begin to be apparent at about 130 K; nothing appears at lower
temperature ... no patterns are visible upon adsorption at 80 K" (O2 on
Ni(100), Video-LEED). With immobile O there is no ordered phase, and the
diffuse signal the prediction concerns sits orders of magnitude below an
ordered spot, so its absence says nothing about pair geometry. It does say
O on Ni(100) stays disordered below ~130 K, i.e. an immobility window.
"""
import math

A_PD = 2.75  # Pd(100) surface lattice constant, Angstrom; Ni(100) is 2.49


def ratio(R, theta, l_coh_A, a=A_PD):
    l = l_coh_A / a
    f = math.pi * (2 * math.pi / l) ** 2 / (2 * math.pi) ** 2
    return R * f / theta, f


def main():
    print("diffuse-in-one-resolution-element / fully ordered Bragg spot, same coverage")
    print(f"{'instrument':<28} {'l_coh (A)':>9} {'f':>10}   R=1.19 (Lin-Jiang)        R=2.00 (Bukas-Reuter)")
    print(f"{'':<28} {'':>9} {'':>10}   theta=0.01  0.05  0.10     theta=0.01  0.05  0.10")
    for name, l in (("conventional / Video-LEED", 150), ("good LEED", 400), ("SPA-LEED", 2000)):
        cells = []
        for R in (1.19, 2.00):
            vals = [ratio(R, th, l)[0] for th in (0.01, 0.05, 0.10)]
            cells.append("  ".join(f"{v:9.1e}" for v in vals))
        f = ratio(1, 1, l)[1]
        print(f"{name:<28} {l:>9} {f:>10.2e}   {cells[0]}   {cells[1]}")
    print("\nThe diffuse centre is 1e-1 (coarse resolution, 0.01 ML) down to 1e-4 of what a")
    print("fully ordered layer at the same coverage would give -- and at low temperature there")
    print("is no ordered layer, so it sits on nothing brighter. Not something to spot by eye on")
    print("a Video-LEED screen; measurable with a counting SPA-LEED (dynamic range ~1e6) or")
    print("helium-atom scattering, provided substrate thermal diffuse scattering at the")
    print("half-order position is below it -- which favours the lowest temperatures.")
    print("The discriminating information lives at |q| ~ 1/|D| ~ 0.3-1 rad, so the")
    print("resolution needed is modest (l_coh of a few tens of lattice constants).")


if __name__ == "__main__":
    main()
