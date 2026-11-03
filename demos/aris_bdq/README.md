# Bedaquiline lung PBPK in Sounio

This demo reimplements the human lung predictions of Karakitsios, Della Pasqua & Dokoumetzidis (2025, *Br J Clin Pharmacol* 91:3167, doi:10.1002/bcp.70163). It works from the paper and its Supporting Information alone, with the human plasma model of Svensson, Dosne & Karlsson (2016, *CPT:PSP* 5:682).

It is written for the authors. It shows what the language does on a model they know, not on one we chose.

## What runs

| File | What it shows | Command |
|---|---|---|
| `bdq_lung.sio` | The full model:<br>• Svensson plasma driver<br>• healthy lung (PB/EW/IW/lysosome)<br>• cellular lesion<br>• outer caseum<br>• caseum chain 1–6 with the fractal rate k(t)<br><br>Integrated with TR-BDF2 over the 24-week regimen. It also proves two mass ledgers, then propagates the uncertainty both papers report (first-order GUM, 30 extra simulations). | `./bin/souc run demos/aris_bdq/bdq_lung.sio` |
| `ref/bdq.cpp` | An independent C++23 oracle of the same equations. It also contains the Monte Carlo study, step refinement, and the three readings of the ionisation formula. | `g++ -std=c++23 -O3 ref/bdq.cpp -o bdq && ./bdq conv` |
| `parity.sh` | Gate: every endpoint and both GUM uncertainties must agree between Sounio and C++ to a relative 1e-5. | `bash demos/aris_bdq/parity.sh` |
| `bdq_mic_claim_honest.sio` | The paper's central claim (unbound lysosomal bedaquiline > MIC50), written as a typed claim at the confidence its inputs support. | `./bin/souc run …honest.sio` |
| `bdq_mic_claim_overclaim.sio` | The same claim declared at ε ≥ 0.80. **The compiler refuses to build it.** It names the two inputs that do not support it: fu_pls and lysosomal pH. | `./bin/souc run …overclaim.sio` → `error[E215] EpistemicComplete violation` (×2) |

Measured on 2026-11-03 with the committed Madaros ELF (`3.0.0-rc.1`, madaros 8ad309f6):
- All parity lines agree to the six printed decimals.
- Plasma ledger residual: 1.6e-12. Caseum ledger residual: 6.5e-13.
- Sounio run time: about 2.2 s per 24-week simulation.

## Reproduction against the published figures

| Quantity | Fig. 7/8 | Sounio |
|---|---|---|
| Unbound plasma, peak during loading | ~1.3 ng/mL | 1.22 |
| Macrophage lysosome (unbound), peak during loading | ~230 ng/g | 243 |
| Caseum-2 / caseum-6 at week 24 | ~2930 / ~980 ng/g | 2918 / 963 |

## Findings for the authors

1. **The ionisation formula as printed in S1 cannot be the one that was run.**
   - It contains the acid term 10^(pH−pKa2).
   - Table S2 and the text swap the pKa labels.
   - Under either labelling, the printed formula moves the lysosomal endpoint by 2–5 orders of magnitude.
   - The diprotic-base formula reproduces the figures.
2. **Eq. S12: back-diffusion from cas2 should enter cas1 with a + sign.** The printed sign removes drug from the chain; the conserving sign matches Fig. 8 better. There are also two smaller typos: the missing (t) in S13 and the duplicated "Caseum 4" label in S14.
3. **The efficacy conclusion does not depend on the lung parameters fitted in mice.**
   - At quasi-equilibrium, Cu_LYS = fu_pls · C_plasma · fni(pH_EW)/fni(pH_LYS).
   - With the reported standard errors propagated, the trough minimum is 63.8 ± 2.1 ng/g, against an MIC50 of 60.
   - 100 % of that variance comes from the plasma model; the lung-fitted parameters contribute 0 %.
   - What the conclusion actually rests on is fu_pls (0.0005) and pH_LYS (5.0), both fixed without uncertainty.
   - Giving those two inputs illustrative spreads (oracle, budget 1) moves P(> MIC50) from 0.96 to about 0.47.
   - Point 3 is what `bdq_mic_claim_overclaim.sio` turns into a compile-time refusal.

## What Sounio adds here — stated narrowly

- **Claims carry their evidence.** `Knowledge[f64]` carries a value, ε and provenance. `with Epistemic(N)` rejects any input below N at compile time. The refusal points at the exact source spans: the pH_LYS and fu_pls declarations.
- **Models carry their own audits.** Mass ledgers are part of the program, not a separate check. Parity against an independent oracle is a script anyone can rerun.
- **Uncertainty is propagated in the same language and run as the model.**
- **Physical inputs are checked for dimension.** `units_audit` recomputes the permeability–surface product (cm/h × cm² → mL/h), the extracellular capacity (mL) and k10 (L/h ÷ L) from unit-typed literals and requires equality with what the f64 core uses (`BDQ_UNITS_AUDIT_PASS`). Feeding an area where a volume belongs, or mL where L belongs, does not compile (`units_refuse.sio`, E041).

Not claimed:
- Superiority over NONMEM, Stan or Monolix.
- Automatic propagation of every source of uncertainty: the GUM here is explicit central differences, diagonal.
- Clinical validity of either the paper's model or this reimplementation.
- The ε values are illustrative and meant to be argued with.
- The epistemic gate is checked where a `Knowledge` value is opened, not across calls: a helper declared `with Epistemic` at a lower threshold can feed a stricter caller unnoticed. The full-model claim (`lysosomal_claim_mic50`) opens fu_pls and pH_LYS inside the gated function for that reason.
- The core is f64, not unit-typed end to end. Casting an f64 into a unit type returns 0 on the committed compiler ELF (fixed in source by PR #3390; the shipped binary needs a refresh). The audit uses typed literals and unit→f64 casts only, which work on both.

## Open questions we would like to work on together

- Are fu_pls and lysosomal pH identifiable from any available data? Or which measurement would collapse the P(> MIC50) ambiguity?
- What are the covariances of the mouse fits (S8, S11, S13)? The GUM here assumes independence.
- How does the k(t) = k_cas/(t+τ)^h caseum model compare with a diffusion PDE on the same LCM data (Table S14 is public)?
