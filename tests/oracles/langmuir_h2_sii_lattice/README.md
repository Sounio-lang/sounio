# langmuir_h2_sii_lattice

Oracle for `tests/run-pass/langmuir_h2_sii_lattice_uq.sio`.

- `sII.hpp` builds the explicit sII oxygen lattice by closing eight generators of Fd-3m
  (origin choice 2; group order 192) over the 8a/32e/96g orbits, with the fractional
  coordinates of empty sII (ice XVI, Falenty et al. 2014, read from GenIce). The 136 oxygens
  reproduce that structure to 3e-15 A up to the choice of origin; every O has four neighbours
  at 2.78-2.80 A, and the 16c/8b cages have 20/28 oxygens at 3.90/4.68 A (a = 17.31 A).
- `lattice_oracle.cpp expect` evaluates, in long double, exactly the discrete sum the witness
  takes (10 radial panels x 4 Gauss-Legendre, 8 Gauss-Legendre in cos(theta), 16 midpoint in
  phi, Voronoi domain of the cage), with the exact gradient in (eps_H, sigma_H), and checks
  that gradient against a central finite difference (agreement to 1e-11).
- `gen_witness.cpp` writes the site and cage-centre tables of the witness from the same
  lattice, so witness and oracle read one geometry.

Build: `g++ -std=c++23 -O2 lattice_oracle.cpp -o lattice_oracle && ./lattice_oracle expect`.
