<!-- docs:meta
topic_id: repo.docs.audit.madaros-imported-ep-var-preserve-2026-08-06
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-imported-ep-var-preserve-2026-08-06
-->

# Madaros C1 — imported Epistemic variance preservation (2026-08-06)

## Claim

Under default Madaros native multimodule import, Epistemic **variance** returned
from `sm_params::mass_bottom` and from `nonunitary_amp::h_bb_yukawa_amplitude_nu`
(H pole) is **bit-identical** (scaled i64 ×1e18) to lean_single.

Classification: **TRUSTWORTHY** (trust map row + Section A gate).

## Measurement (shipped Madaros, 2026-08-06)

| Quantity | scale18 i64 | Meaning |
|---|---:|---|
| `ep_variance(mass_bottom())` | `900000000000000` | PDG σ=0.03 → var=0.0009 |
| `ep_variance(h_bb…amp_sq)` at pole | `1354` | ≈1.354×10⁻¹⁵ |

Both engines print the same integers. Fixture sentinel:
`MADAROS_IMPORTED_EP_VAR_PRESERVE_OK`.

## Non-finding (important)

Particle EXP18/19 sometimes printed `AMP_VAR 0.000000` under Madaros while
lean_single showed `~1e-15`. That was **`print_f64` display rounding**, not
loss of GUM variance on the imported path. The preserve gate reads variance via
`ep_variance` + integer scale, avoiding #862/`print_f64`.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/ci/madaros_imported_ep_var_preserve_gate.sh
# expect: MADAROS_IMPORTED_EP_VAR_PRESERVE_GATE_OK
```

Receipt: `artifacts/compiler/madaros_imported_ep_var_preserve_receipt.v1.json`

## Related

- Trust map: `docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md`
- Parent gate: `scripts/epistemic_trust_gate.sh` (Section A hooks this gate)
