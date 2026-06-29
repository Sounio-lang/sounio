<!-- docs:meta
topic_id: repo.docs.dissertation.results.pbpk14-midazolam-ddi-canonical-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.pbpk14-midazolam-ddi-canonical-v1
-->

# PBPK14 Midazolam — CYP3A Drug–Drug Interaction (Fifth Canonical Drug)

## Scope

Midazolam is the **fifth canonical drug** of the PBPK dissertation suite and the
**first to model a drug–drug interaction (DDI)**. The four prior drugs each added
a distinct modeling surface — rapamycin (PBPK28 + TMDD + neointima PD),
semaglutide (PBPK28 + TMDD + Bergman PD), venlafaxine (parent + ODV + matrix
release), haloperidol (PBPK14 + BBB Kpuu + D2 occupancy). None modeled enzyme
inhibition. Midazolam adds it: **oral midazolam co-administered with
ketoconazole**, the textbook CYP3A victim/perpetrator pair, whose iconic readout
is oral AUC rising ~15× under a strong CYP3A inhibitor (Olkkola 1994).

As with the prior four drugs, the contribution is a **Sounio↔Node 0.0000% parity
surface**: a self-contained Sounio reference and the pure-JS engine the
dissertation viewer runs produce byte-identical trajectories under a fixed-step
RK4 integrator. Parity gate cases **17–19** were added; the full gate (cases
1–19) is green with no regression.

## The mechanism (and the honesty it required)

The DDI is modeled with the **standard well-stirred oral first-pass** formulation,
parameterized by *intrinsic* clearances rather than lumped quantities:

| Quantity | Expression |
|---|---|
| Gut-wall survival | `FG  = Qg / (Qg + fu_g·CLint_g)` |
| Hepatic first-pass survival | `FH  = Qh / (Qh + fu·CLint_h)` |
| Systemic clearance | `CL_h = Qh·fu·CLint_h / (Qh + fu·CLint_h)` |
| Oral bioavailability | `F = Fa · FG · FH` |
| Oral exposure | `AUC = F·Dose / CL_h` |
| Competitive inhibition | `CLint → CLint / (1 + I/Ki)` (gut and hepatic, same enzyme) |

**The avoided trap.** A naïve implementation would scale a lumped `F` *up* and a
separate lumped `CL` *down* under inhibition — double-counting hepatic CYP3A and
**fabricating** the 15×. Parity cannot catch this (both sides compute the same
wrong number). The fix is that a **single** hepatic intrinsic clearance `CLint_h`
drives *both* `FH` and `CL_h`; inhibiting that one quantity raises `FH` and lowers
`CL_h` consistently, so the ~15× emerges with **no double-count**. This is the
well-stirred model's correct closed form, not an approximation, and required no
portal-vein topology change.

**Gut matters.** With the gut term, the model lands AUCR = **14.99×** at
`I/Ki = 8`; hepatic-only inhibition reaches only **9.0×**. The gut-wall CYP3A
contribution (`FG'/FG = 1.67×`) is exactly why oral midazolam is *the* CYP3A
probe (Thummel 1996). This is asserted in `validation/midazolam_ddi.sio` TEST 4.

## Parity surface (gate cases 17–19)

| Case | Series | Result |
|---|---|---|
| 17 | Solo oral PK (5 mg), 12 samples | `MIDAZOLAM_SOLO_PK_PARITY_PASS` — 0.0000% RMSE |
| 18 | Ketoconazole-inhibited oral PK, 12 samples | `MIDAZOLAM_DDI_PK_PARITY_PASS` — 0.0000% RMSE |
| 19 | DDI inhibition curve (F, CL_h, AUCR across I/Ki), 8 points | `MIDAZOLAM_AUCR_PARITY_PASS` — 0.0000% RMSE |

All three series are **byte-identical** between the Sounio reference
(`tests/run-pass/dissertation_pbpk28_parity_ref_midazolam.sio`) and the Node
runner (`scripts/dissertation/run_midazolam_node.mjs`, consuming
`website/src/lib/pbpk28_core.mjs runMidazolamScenario`).

## Locked parameters

```
Qh = 90 L/h   Qg = 18 L/h (Qgut model)   fu = 0.03   fu_g = 1.0   Fa = 0.95
CLint_h = 1090 L/h   CLint_g = 14.7 L/h     ka = 3.0 /h   tlag = 0
Ketoconazole: Ki = 0.008 mg/L, I_ss(unbound) = 0.064 mg/L  -> I/Ki = 8, R = 9
=> solo: FH 0.733, FG 0.551, CL_h 23.99 L/h, F 0.384  (oral F in Heizmann 1984 range)
=> +keto: FH 0.961, FG 0.917, CL_h 3.49 L/h, F 0.837   -> AUCR 14.99×
```

Pharmacogenomics (`pgx/cyp3a45_midazolam.sio`): CYP3A5*3 expresser vs
non-expresser and CYP3A4*22 carrier scale `CLint_h` (and hence `CL_h`); validated
ordering — expresser CL_h 28.6 > non-expresser 23.99 > *22 carrier.

## Honest framing (what this does and does not claim)

- **Parity proves Sounio == Node to f64**, not ground-truth accuracy. The ~15× is
  a **model output**, validated against literature (Olkkola 1994; band 12–18×) in
  `validation/midazolam_ddi.sio`, not asserted as a measured constant.
- **Well-stirred first-pass** is assumed; the gut `Qgut` is the Yang 2007 hybrid
  villous flow, distinct from splanchnic perfusion.
- **Static inhibitor.** Ketoconazole is held at its steady-state unbound
  concentration (as clinical DDI studies pre-dose the perpetrator) — this is *not*
  a dynamic two-drug PK simulation.
- **Fixed-step RK4 (parity) vs adaptive Tsit5 (production).** Bit-for-bit parity
  on an adaptive controller is intractable, so both the reference and the JS
  engine use a fixed-step RK4 — parity validates exactly what the viewer runs. The
  production scenario (`scenarios/oral_midazolam_ddi.sio`) uses Tsit5 and
  independently reproduces AUCR ≈ 15.2× as a cross-check.
- **Nominal distribution Kp.** Systemic tissue partition coefficients are
  midazolam-plausible (Vss ≈ 1.4 L/kg) but nominal; the *validated* DDI quantities
  (F, CL_h, AUCR) are distribution-independent.

## Artifacts

- `tests/run-pass/dissertation_pbpk28_parity_ref_midazolam.sio` — Sounio parity ref
- `scripts/dissertation/run_midazolam_node.mjs` — Node runner
- `website/src/lib/pbpk28_core.mjs` — `runMidazolamScenario` / `runMidazolamDDIResponse`
- `stdlib/darwin_pbpk/ddi/cyp3a_competitive.sio` — competitive-inhibition + first-pass primitive
- `stdlib/darwin_pbpk/drugs/midazolam.sio` — drug module
- `stdlib/darwin_pbpk/pgx/cyp3a45_midazolam.sio` — CYP3A5 / CYP3A4*22 PGx
- `stdlib/darwin_pbpk/scenarios/oral_midazolam_ddi.sio` — production Tsit5 scenario
- `stdlib/darwin_pbpk/validation/midazolam_ddi.sio` — literature-anchored validation (6/6)
- `scripts/ci/dissertation_pbpk28_parity_gate.sh` — gate cases 17–19

## References

- Olkkola KT et al. (1994) *Clin Pharmacol Ther* 55:481 — oral midazolam +
  ketoconazole AUC ~15×.
- Heizmann P et al. (1984) *Br J Anaesth* 56:1215 — oral midazolam bioavailability.
- Thummel KE et al. (1996) *Clin Pharmacol Ther* 59:491 — gut + hepatic first-pass.
- Gorski JC et al. (1998) *Clin Pharmacol Ther* 64:133 — CYP3A inhibition.
- Yang J et al. (2007) *Curr Drug Metab* 8:676 — Qgut model.
- Kuehl P et al. (2001) *Nat Genet* 27:383 — CYP3A5*3; Wang D et al. (2011)
  *Pharmacogenomics J* 11:274 — CYP3A4*22.
