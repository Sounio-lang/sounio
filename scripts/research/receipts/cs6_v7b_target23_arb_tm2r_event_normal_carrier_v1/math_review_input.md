# Hostile math review: event-normal TM2R carrier

Review the mathematical soundness of the implementation and receipt in these
files:

- `scripts/research/cs6_v7b_target23_arb_tm2r_event_normal_carrier_worker.py`
- `scripts/research/cs6_v7b_target23_arb_tm2r_event_normal_carrier_verify.py`
- `scripts/research/cs6_v7b_target23_arb_tm2r_event_normal_carrier_contract_v1.txt`
- `scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_normal_carrier_v1/preflight.json`
- `scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_normal_carrier_v1/result_report.md`

The representation preserves the original degree-2 Taylor model in six
variables and appends four independent carrier variables. At each
reconditioning, coefficient radii, sigma-bearing monomials, and component
remainders are projected through an exact rational inverse. The first basis
column is transverse to the exact rational event covector and the remaining
three columns lie exactly in its kernel. Arb containment checks require the
reconstructed carrier enclosure to contain every original generator.

The one-step preflight compares the carrier against a lineage-preserving
control reconstructed from the same frozen raw event projection. It reports a
strict but tiny event-derivative width reduction:

- doubleton: 9.704881887288939e-05 to 9.704823469292023e-05;
  factor 1.0000060194806326
- tripleton: 9.704881887288939e-05 to 9.704823457923340e-05;
  factor 1.0000060206520864

Attack these questions:

1. Does the generator projection and reconstruction argument rigorously enclose
   the original set, or is there a direction, dependency, or double-counting
   error?
2. Is preserving all monomials supported on the original six variables while
   reconditioning every sigma-bearing monomial mathematically coherent across
   multiple flow steps?
3. Is the one-step comparison genuinely like-for-like, or can the tiny gain be
   caused by asymmetric rounding, representation, or discarded dependence?
4. Do the exact event-kernel conditions support the stated interpretation, and
   are they sufficient only for a candidate rather than transversality?
5. Identify any BLOCKER or MAJOR issue that must be fixed before the full
   Foundry/Slurm transport. Be explicit if the full transport is merely the next
   falsifier and no theorem follows from the preflight.
