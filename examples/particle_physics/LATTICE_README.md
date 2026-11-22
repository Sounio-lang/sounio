# Lattice gauge code (SU(2) / SU(3)): what it is and what it is not

This directory holds the drivers for `stdlib/particle_physics/lattice_gauge.sio`
(pure SU(2) Wilson), `lattice_su3.sio` (pure SU(3) Wilson) and
`lattice_prec.sio` (an f64 vs Dd64 check of the arithmetic). The code is a
typed, seeded, reproducible lattice Monte Carlo whose gates refuse to report
more than the data supports. **It is not a physics result**, and nothing here
bears on the Clay Millennium Yang–Mills mass-gap problem.

## Provenance

These files existed only on side branches until 2026-10-06. They were first
committed on 2026-08-14 in `f8c158ebb` ("checkpoint: trabalho em andamento das
10 lanes"), which about 45 branches carry. They were taken from the newest copy,
`origin/wip/loom-codex-1-uncommitted-20260921`. Every result JSON under
`results/*_20260813.json` was written on 2026-08-13 by the lean_single engine
and comes from that same branch, unedited.

Changes made while rescuing the code (no change to any computed number):

- Every `.sio` file got a `PROVENANCE:` comment in its header.
- `lattice_gauge.sio::glueball_n_valid_meff` and
  `lattice_prec.sio::prec_gate_ok` now declare `with Mut`. lean_single
  refused both with E035 when the library was checked on its own.
- `su2_meff_plateau.sio` uses a driver-local copy of `glueball_meff_best_tau`.
  When the function is imported, lean_single gives the middle element of its
  `(f64, i64, i64)` result the type f64 and refuses the program (E001). The
  same function defined in the driver checks on both engines.
- Main's `stdlib/particle_physics/README.md` and `mod.sio` were left as they
  are. The branch copies of those files, and of the other 30 or so
  `particle_physics` files that differ, are older than main.

## Scope of the lattices

- Most drivers use **L=4** (4⁴ = 256 sites) or **L=6** (6⁴ = 1296 sites).
- Four SU(2) drivers also run an **L=8** point: `su2_sommer_production`,
  `su2_fs_gevp`, `mass_claim_plateau` and `mass_claim_gevp_plateau`.
- SU(3) runs only at L ∈ {4, 6}.
- The time extent is T = L, so a correlator has at most 2–4 usable τ values.

At these volumes and statistics:

- The **m/√σ ratios are contact-UV scale ratios at best, not spectral gaps.**
  Nearly every ratio is built from m_eff(0) = ln C(0)/C(1), the contact
  estimator. `mass_claim_plateau.sio` shows that no single-operator channel
  (SU(2) L=6, SU(2) L=8, SU(3) L=6) reaches a multi-τ plateau. Wherever that
  gate applies, the mass-gap ratio is set to 0 (`MASS_GAP_CLAIM_KILLED` or
  `CONTACT_UV_ONLY`).
- `mass_claim_gevp_plateau.sio` gets `PLATEAU_MASS` from a 2×2 GEVP at L=6 and
  L=8, but only from a two-sink "plateau". It gives m/√σ ≈ 2.49 at L=6 and
  ≈ 1.91 at L=8, which do not agree with each other and sit below the
  literature value of about 3.5–4. The JSON calls this fragile.
- "Literature window" PASS gates (`lit_window`, `*_lit`) only check that a
  number falls inside a band. They are not agreement with the literature.
- No continuum limit is taken, there is no infinite-volume extrapolation, and
  there are no quarks.

## Multi-β results stay FAIL_HONEST

The multi-β "continuum sketches" quote per-β standard errors. These errors are
far too small for the β points to agree with each other, and the code reports
that rather than hiding it:

| driver | quantity | value | gate |
|---|---|---|---|
| `su2_jackknife_continuum` | χ²/dof across β ∈ {2.2, 2.3, 2.4, 2.5}, L=6 | **790.6** in the 2026-08-13 JSON; **14.57** from today's code, which floors the SEMs (see below) | `multi_beta_chi2: FAIL_HONEST` either way |
| `su3_jackknife_continuum` | χ²/dof across β ∈ {5.5, 5.7, 5.9, 6.1}, L=6 | **13.4** in its JSON (not reproduced by today's driver; see below) | FAIL_HONEST |
| `dual_ym_scoreboard` | χ²/dof SU(2) / SU(3), with 8 % m / 5 % σ floors | 17.5 / 13.4 | `SU2_chi2_honest`, `SU3_chi2_honest: FAIL_HONEST` |

`su2_jackknife_continuum` also prints a weighted mean ⟨m/√σ⟩_w ≈ 3.49 that
falls inside the literature band. Its own JSON says this happens "by accident
of weighting toward β=2.3", and the β-spread gates (78 %) FAIL_HONEST.

Every `not_claimed` field in the JSONs, and every Clay disclaimer in the
drivers, is kept as written.

## What the gates refuse

Gate verdicts are `PASS`, `FAIL` or `FAIL_HONEST`. `FAIL_HONEST` means the
check ran and the data does not support the statement, so the statement is not
made.

| gate family | refuses |
|---|---|
| `correlator_decays`, `*_C_decays_*` | an m_eff when C(1) ≥ C(0) or C(0) ≤ 0, since there is no decay to read a mass from |
| `spectral_meff_positive`, `meff*_positive` | a mass from a non-positive or invalid log ratio |
| `creutz_confinement`, `creutz_chi*` | a string tension when the Creutz ratio χ(R,R) is not positive (no area law seen) |
| `glueball_mass_claim` (status 0/1/2) + `mass_gap_ratio_or_kill` | any m/√σ "gap" without a multi-τ plateau. Status 0 or 1 forces the ratio to 0 |
| `meff_plateau`, `gevp_tau_consistency`, `pair_agreement`, `pair_mass_spread` | a mass whose m_eff(τ) drifts with τ, or that depends on which operator pair is used |
| `*_beta_spread*`, `multi_beta_chi2`, `*_chi2_honest` | a continuum statement when the β points disagree beyond their errors |
| `finite_size_mild`, `FS_*` | a volume-independent value when L4→L6 or L6→L8 moves it beyond tolerance |
| `sommer_r0_interpolated`, `m_gevp_times_r0_F` | a force-based Sommer r0 when F(r)·r² never crosses 1.65 inside the lattice (L=6) |
| `*_lit` (e.g. `gevp3_ratio_lit`) | agreement with the literature 0++ ratio when the number is outside the 3.5–4 band |
| `lattice_prec` gates | any observable whose f64 and Dd64 values differ by more than max(ε_abs, κ·SEM). This rules out float error as the source of a signal |

## Engines and runtimes

On 2026-10-06 every library and driver passes `souc check` on lean_single, on
main's shipped Madaros, and on a Madaros built from main + #2771 + #2737.
**Running** the drivers is a different matter:

- **lean_single** (`SOUNIO_SOUC_ENGINE=lean_single`) computes them correctly.
  This is the engine that wrote the 2026-08-13 JSONs.
- **Madaros as shipped on main** miscomputes them and then aborts:
  - the f64 deref-in-arithmetic miscompile is fixed by **#2771**;
  - the abort is `madaros: handles full`, fixed by **#2737** (region
    reclamation).
- **Madaros with #2771 + #2737** reproduces lean_single.
  `tests/run-pass/lattice_su2_glueball_parity_short.sio` is the witness: a
  short form of `su2_glueball_mass_gap.sio` whose full stdout is asserted. Its
  output is byte-identical on lean_single and on the combo Madaros, and it
  fails on main's shipped Madaros.

Measured wall-clock times on 2026-10-06, on an 8-CPU workspace with load average
around 10 and 3–7 runs in parallel, so these are rough. "co" is the combo Madaros
(main + #2771 + #2737), timing the run of the compiled ELF; compiling took
12–23 s per driver. "ls" is lean_single (`souc run`).

| driver | co run | ls run |
|---|---|---|
| su2_mass_gap_probe | 8 s | 75 s |
| su2_glueball_mass_gap | 14 s | 143 s |
| su2_glueball_smeared | 27 s | 205 s |
| su3_glueball_mass_gap | 24 s | 58 s |
| su3_mass_gap_probe | 24 s | 50 s |
| su2_continuum_sketch | 53 s | 441 s |
| su2_glueball_L6 | 54 s | 428 s |
| su3_continuum_sketch | 55 s | 184 s |
| heatbath_vs_metropolis | 81 s | finished, time not kept |
| su2_meff_plateau | 169 s | — |
| su2_precision_witness | 178 s | — |
| su2_variational_glueball | 185 s | — |
| su2_scale_gevp | 214 s | — |
| su2_variational_stability | 226 s | — |
| su2_gevp3 | 267 s | — |
| su2_continuum_heatbath | 276 s | finished, time not kept |
| mass_gap_heatbath_production | 377 s | — |
| su2_sommer_r0 | 414 s | — |
| su3_fs_probe | 1041 s | — |
| su2_jackknife_continuum | 1059 s | > 3600 s (timed out after printing β=2.2) |
| mass_claim_gevp_plateau | 1207 s | > 3600 s (timed out) |
| su2_contact_continuum | 1678 s | > 3600 s (timed out) |
| su2_fs_gevp | 2175 s | — |
| su2_sommer_production | 2265 s | — |
| mass_claim_plateau | 2848 s | — |
| su3_l6_production | 3860 s | — |
| su3_jackknife_continuum | > 3600 s (timed out after 3 of 4 β) | — |
| dual_ym_scoreboard | > 3600 s (timed out during the SU(3) half) | — |

A dash means the driver was not timed on that engine. lean_single was about
3–10× slower than the combo Madaros on every driver timed on both.

Every driver except `su3_glueball_mass_gap` and `su3_mass_gap_probe` takes more than about 60 s on lean_single, so those 26 carry `//@ requires: slow` and `//@ timeout: 7200`. The harness
does not scan `examples/`, so this only documents cost.

## Reproduced numbers

Checked against the 2026-08-13 JSONs:

- **su2_glueball_mass_gap** (full run, n_therm=50, n_meas=40). Output is
  identical on lean_single and the combo Madaros: C(0)=30.295474,
  C(1)=0.197919, C(2)=−14.575211, m_eff=5.030896±0.041296,
  ⟨P⟩=0.608851±0.001475, Creutz χ(2,2)=0.191461±0.008547 (16/16 valid),
  m/√σ=11.497543, and all four gates PASS.
  Every number matches the JSON except the plaquette SEM, which the JSON
  records as 0.001626. The program prints 0.001475.
- **su2_jackknife_continuum** (combo Madaros). The β=2.2 point reproduces
  m=1.916039±0.075801, σa²=0.423205±0.007726 and **m/√σ=2.945294**. All four
  β points reproduce m and σa² digit for digit. The ratio *SEMs* do not
  match: the library's `mass_over_sqrt_sigma_ep` now applies SEM floors
  (8 % on m, 5 % on σ), and the 2026-08-13 JSON was written before they
  existed. With the floors, the code prints χ²/dof=**14.57** (the JSON says
  790.6), ⟨m/√σ⟩_w=3.38±0.15 (JSON 3.49±0.02) and spread 0.81 (JSON 0.78).
  `multi_beta_chi2` and both spread gates are still **FAIL_HONEST**. The JSON
  is kept as recorded; the two disagree only in the ratio error bars.
- **su2_jackknife_continuum on lean_single** printed the same β=2.2 value,
  m/√σ=2.945294±0.246861, before its 3600 s cap.
- **dual_ym_scoreboard** (combo Madaros, stopped at 3600 s). The whole SU(2)
  half reproduces the JSON: the four ratios 2.712979 / 2.592866 / 5.494584 /
  7.043530 with their floored SEMs, ⟨m/√σ⟩=3.191121±0.177328 and
  **χ²/dof=17.543336** (FAIL_HONEST). In the SU(3) half, the first two β
  points reproduce the JSON (1.989206±0.205043 and 2.556426±0.263510). The
  run hit the cap before printing the SU(3) χ²/dof (13.41 in the JSON).
- **su3_jackknife_continuum does not reproduce its JSON.** Its 28-line JSON
  lists the same per-β ratios as dual_ym's SU(3) half (1.989206, 2.556426,
  5.035487, 3.800053; χ²/dof 13.41). Today's driver prints 0.810421,
  3.740182 and 5.371315 for its first three β points. The driver changed
  after the JSON was written, or the JSON was copied from dual_ym. Either
  way, **the SU(3) χ²/dof ≈ 13 is backed by dual_ym_scoreboard, not by this
  driver**, until someone reruns it to the end (more than an hour).
- The 30-line parity witness gives the same stdout on lean_single and the
  combo Madaros, and fails on main's Madaros (see above).
