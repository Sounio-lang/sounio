# stdlib/ode

ODE solvers.

## Types
- Euler, midpoint, RK4, RK45 (adaptive), BDF (stiff)
- Epistemic variants
- PBPK models (14-compartment, 3-compartment)
## Checked foundation

See [checked solver contracts](CHECKED_SOLVERS.md) for repaired scalar/PK termination
and transitional generic typed64-component DP54 and velocity Verlet. These are
source-prepared pending centralized Slurm validation, not a dynamic heap API or
a certificate for the legacy PBPK/epistemic solvers.
