# Direction 3 — Drug-state EEG candidates for the 7-class partition test

## Hypothesis

The 7-class linear partition of 168 ZD pairs defines seven "algebraic gating
regimes." If pharmacological state modulates cortical dynamics in a way that
is *algebraically structured*, drug vs placebo EEG should separate cleanly
across these 7 classes — i.e. per-class MSE means should shift with drug
state, and the separation should be larger for classes predicted to be
sensitive to global-regime change (Regime II 7-class refinement classes)
than for the stable bulk $L_2$ class.

This is a **falsifiable drug-signature test** of the algebraic framework.
Null: all 7 classes shift by the same multiplicative factor → the partition
is redundant with scalar MSE. Alternative: a non-uniform shift → the
partition carries drug-specific information.

## Public candidate datasets

| dataset | drug | n | why | access |
|---|---|---|---|---|
| OpenNeuro ds003620 | ketamine (0.5 mg/kg IV) | 30 | double-blind crossover, 64ch EEG, placebo-control | BIDS-EEG, free |
| OpenNeuro ds004504 | anesthesia (propofol) | 20 | continuous induction → emergence | free |
| Timmermann 2019 (Imperial) | psilocybin (20 mg) | 24 | closed-eyes rest pre/post, 64ch | on request via Imperial |
| SPIS Resting-State | caffeine (200 mg) | 10 | small but fast, public | free |
| OpenNeuro ds002778 | LSD | 15 | closed-eyes, high-density | free |

**Primary candidate: ds003620 (ketamine).** Crossover design, placebo-controlled,
large effect size, 64 channels. If the 7-class partition is drug-sensitive,
ketamine should produce the largest signal given its NMDA-driven dissociative
state.

## Protocol sketch (ds003620)

1. For each subject × condition (drug, placebo), extract 80-sample windows
   from resting-state blocks (eyes-closed).
2. Per window: reduce 64 channels to 16 via PC1..PC16 (match existing
   sedenion_ssm pipeline).
3. Per (subject, condition, window): 168-ZD-pair sweep, compute 7-class
   mean MSE vector $\mu \in \mathbb{R}^7$.
4. **Within-subject contrast** $\Delta\mu = \mu_{drug} - \mu_{placebo}$ in
   $\mathbb{R}^7$. Test:
   - Is $\Delta\mu$ significantly non-zero in any class (FDR-corrected over 7)?
   - Is the direction of $\Delta\mu$ coherent across subjects (sign test)?
   - Is $\Delta\mu$ aligned with a predicted direction (e.g., Regime II
     classes carry more drug signal than Regime I)?

## Sanity gates

- **Scramble control.** Shuffle the 168 ZD pair assignments into random
  7-partition; re-run. Drug $\Delta\mu$ should VANISH on scrambled partition.
- **Regime I-only baseline.** Restrict to Regime I (5 classes). If drug
  signal equally present here, the 7-class Regime II structure is not
  carrying extra info.
- **Per-channel scramble.** Randomly permute the 16 channels before PC
  reduction. Drug signal should degrade — confirms spatial structure matters.

## Status

- [ ] pick primary dataset (pending user call)
- [ ] download + BIDS parse
- [ ] adapt sedenion_ssm_connectome_orbit.sio to accept generic 16-channel
  window input (already close)
- [ ] run sweep on one subject both conditions as smoke test
- [ ] full sweep, within-subject paired test

Estimated compute: ~30 min per subject × 2 conditions on beagle-sounio.
For n=20 subjects: ~20 hours single-node; parallelizable.

## Why this matters

Closes the biology loop with a *prediction from the algebra*, not a
post-hoc correlation. The 7-class partition was derived purely from
PSL(2,7) / Fano-incidence structure — it has no free parameters tuned
to drug state. If it separates ketamine from placebo, the sedenion
algebra is doing genuine pharmacological work; if not, the clean
theoretical taxonomy stops at "measurement of stationary signal
statistics" and we've bounded its biological reach honestly.

Either outcome publishable.
