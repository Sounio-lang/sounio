Tests: the Langmuir constant of H2 in the two sII cages, as a witness that first- and second-order GUM is exact through a quadrature loop with nested helpers. The model is a Lennard-Jones–Devonshire cell with a single-site LJ H2 guest, the O-site LJ of SPC/E, TIP4P/Ice and TIP5P, Lorentz–Berthelot combining and quadratic Feynman–Hibbs.

`langmuir_h2_sii_ljd.sio` checks 54 values: derivatives and Hessian entries of ln C.

`langmuir_h2_sii_uq.sio` checks 50 values. Its uncertainty comes from the H2 parameters, with the covariance of four published one-site sets. It checks the variance, the second-order mean, the variance of the difference between two water models (the shared term cancels) and the mixed Hessian entries behind the uncertainty of the enclathration heat and of the lattice sensitivity.

Oracles are in `tests/oracles/langmuir_h2_sii/`: C++23 second-order jets plus a Monte Carlo check.
