# Vacuity positive-control seed

Deliberately empty fixture directory used only by
`scripts/ci/fixtures/vacuity_positive_control/vacuous_seed_gate.sh`.

No `*.sio` files live here on purpose. The vacuous-fixture sweep must flag
that gate with `cases=0`. If the sweep reports zero vacuous gates while this
seed is present, the instrument is broken.
