<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-172.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-172.readme
-->

# SIO-172: joint uncertainty in checked lattice transport

Base: remote main `c21b0e139830ca4949f80cba62887a8922a972d7` after a coordinated rebase from `a2a44df195bb8caf3fb07cbc9517275b29325e94`. Branch: `devpass/SIO-172-phonon-factor-transport`. This package changes the private phonon/PBPK kernels and introduces an explicit checked public interface. It does not choose an organ calibration, blood/plasma conversion convention, or cross-sectional area.

## Scientific contract

`phonon_from_factor` consumes caller-owned nominal and joint-factor `NumericBuffer`s. For n sites and k latent sources, the factor has `(n+15)*k` entries in row-major order. Rows 0 through n-1 are the initial concentration sensitivities; rows n through n+14 are parameter sensitivities in the order below. Columns are the same latent sources for every row. The caller declares the complete covariance as `Sigma = L L^T`; signed columns preserve negative correlations, rank deficiency, initial/parameter correlations, and parameter/parameter correlations. A factor is not silently diagonalized or projected.

The checked linear state owns both buffers and keeps its nominal values, factors, parameters, dimensions, and time private. Callers cannot update a marginal variance while retaining stale covariance. `phonon_from_independent_initial` is explicitly a separate constructor: it declares independent initial concentrations and exact parameters. It initializes every factor entry that integration will read. Parameter uncertainty uses the general constructor.

The augmented system is `y' = f(y,theta,t)` and `S' = A_y S + A_theta L_theta`, integrated through the same four nominal RK4 stages. Parameter factor rows remain unchanged. Output covariance is `S S^T`. `uncertainty_model=2` and `first_order=true` identify supplied joint factors and local first-order propagation; they do not certify that a nonlinear uncertainty distribution is accurately linearized. Process noise mode zero is deterministic evolution of these uncertain inputs. Every nonzero mode returns status 7 before mutation; no arbitrary noise or 10% boundary CV is injected.

## Input units and parameter order

All checked core concentrations are kg/m³ and time is seconds. Every factor entry has the units of its corresponding input row per standardized latent coordinate.

| Offset after n | Parameter | Unit |
|---|---|---|
| 0 | d_eff | m²/s |
| 1 | dx | m |
| 2 | k_release | 1/s |
| 3 | cl_site | 1/s |
| 4 | fu | dimensionless |
| 5 | kp | dimensionless |
| 6 | c_portal | kg/m³ |
| 7 | c_ha | kg/m³ |
| 8 | f_heart | Hz |
| 9 | a_portal | dimensionless |
| 10 | a_ha | dimensionless |
| 11 | phi_ha | radians |
| 12 | c_tissue | kg/m³ |
| 13 | v_blood | m/s |
| 14 | zonation | dimensionless |

`phonon_from_legacy_factor` explicitly converts ng/mL nominal concentrations and the corresponding state/c_portal/c_ha/c_tissue factor rows by 1e-6. Other rows already use SI units and retain their factors. This scales the entire declared covariance consistently. The retained flat rapamycin defaults are input presets, not a validation or calibration claim. The units parity test imports the real `units::lib` Quantity/UnitDim API and both production constructors.

The domain requires nonnegative diffusion, rates, velocity, boundary/tissue concentrations and frequency; positive dx, kp and zonation; fu in [0,1]; and pulsatility amplitudes in [-1,1]. The directional transport operator supports nonnegative velocity. Boundary phase is checked within absolute 1e6 radians before bounded-cost reduction and canonical pure sine/cosine calls. Unsupported or unrepresentable arithmetic is explicit failure.

## Integration, balance, and cycle averages

`phonon_workspace(n,k)` validates caps before integer products: 1≤n≤64, 0≤k≤min(79,n+15). Its sizes are the exact application buffer payload, excluding allocator/compiler overhead:

| Shape | Nominal values | Joint factor | RK4 scratch | Total payload |
|---|---:|---:|---:|---:|
| n=1,k=0 | 1 | 0 | 5 | 48 bytes |
| n=64,k=1 | 64 | 79 | 640 | 6,264 bytes |
| n=64,k=79 | 64 | 6,241 | 25,600 | 255,240 bytes |

Scratch length is `5*n*(k+1)` and is supplied once by the caller. Parameter rows are not copied into every RK stage. The core allocates no new application buffers in its stepping loop; native RSS receipts qualify only the measured runs, not all compiler/runtime allocation behavior.

The integrator uses a conservative outgoing-rate step bound, validates complete candidates before committing state, and rejects negative candidates rather than clamping concentrations or variances. It checks representable forward time progress, a distinct represented midpoint, finite endpoints, positive dt and a bounded step budget. RK4 stage weights use the actual represented span `(start+h)-start`, and that span is checked again against the stability bound. The recorded endpoint is the endpoint actually integrated. A final one-ULP trailing sliver may be fused before integration only when the full requested target is exactly reached by the predicted endpoint, its midpoint is distinct, and its actual full span remains within the stability bound. No endpoint is stamped after integration. A failed step leaves the protected state unchanged; a simulation that exhausts its budget retains its successful accepted steps and reports the reached time. Getter failures return status with zero unavailable value.

`phonon_balance_checked` reports minimum concentration, length-integrated concentration `dx*sum(y)`, its derivative, boundary flux, reaction contribution and the telescoping balance residual. The inventory is a one-dimensional concentration integral; calling it organ mass would require geometry absent from this model. It reports the residual and does not hide it with a conservation correction.

`phonon_liver_cycle_checked` integrates outlet concentration and every shared factor coordinate over the same RK4 stages. It then forms the uncertainty of the integral using those shared factors, retaining cross-time covariance. The period is T=1/f_heart; frequency uncertainty includes both the endpoint and normalization terms. Frequency zero is explicitly invalid rather than selecting an invented default period. For deterministic frequency, a rounded period window is integrated, reported and normalized using the actual `target-start`; quadrature uses each actual accepted step span. If a frequency factor is nonzero and the represented window differs from the requested `1/f_heart`, preflight returns status 6 before state or average-buffer mutation, because substituting a different period sensitivity would be inconsistent. State factors after a successful cycle are sensitivities at the fixed nominal endpoint; the returned average factors additionally include the period sensitivity.

| Status | Meaning |
|---|---|
| 0 | success |
| 1 | nonfinite input |
| 2 | invalid physical/step domain or negative candidate |
| 3 | shape/index/workspace mismatch |
| 4 | arithmetic, phase-range, or representability failure |
| 5 | accepted-step budget exhausted |
| 6 | no representable forward time progress |
| 7 | process-noise mode unsupported |

## Verification and evidence boundaries

`tests/reference/phonon_factor_reference.cpp` is an independent C++23 decimal100 reference. Its forward dual arithmetic differentiates the value equations independently of the production analytic Jacobian. A decimal100 centered-difference control checks all 45 parameter derivatives at three sites; analytic decay and a two-site matrix exponential control the complete time path. It reads actual imported Sounio function outputs, rejects missing/duplicate/nonfinite records, and has deliberate corruption controls.

Qualification job 17234 (`qual20`) passed core 139 numerical records, SI parity 30 records, and 11 numerical guard values plus 31 status codes. The private-field fixture was rejected with E259. The core included all 15 integrated parameter sensitivities, signed initial/parameter cancellation, two-site covariance, 64 sites with both shared rank-one and full 79-column factors, and correlated cycle averages including uncertain period. Core elapsed 0.46 s / max RSS 1,080 KiB; SI elapsed 0.01 s / max RSS 1,580 KiB. These are bounded fixture observations, not broad performance claims.

The first full gate job **17237** (`final22/`) passed: **145 core values, 30 SI values, 11 guard values and 31 status codes**, six missing/corrupted-output refusal controls, the private-field E259 rejection, and the original variance defect witness. The six inventory/boundary/reaction balance controls are included in the 145 core values. Final measured core elapsed 0.45 s / max RSS 1,648 KiB, SI 0.01 s / 1,604 KiB, and status 0.00 s / 1,080 KiB. The frozen source packet is 436 KiB; retained native artifacts are 1.9 MiB, with text/hash receipts committed and executable custody retained remotely. Run it on a Slurm worker with UID1000 and a verified absolute `MADAROS_RAW_BIN`, setting `PHONON_GATE_OUTPUT` to a new absent directory. The source packet contains only required launchers, library closure and fixtures, not bundled compiler archives.

The superseding time-consistency gate job **17244** (`time25/`) passed **164 core values**, the original 30 SI values and 11 guard values / 31 status codes, all six missing/corrupted-output controls, privacy and the old variance witness. The original 145 numerical controls remain in the expanded core fixture. New controls cover a refused one-ULP midpoint at nonzero start, an accepted two-ULP interval using its actual span, a rounded span that exceeds the stability bound, an exactly constant nonzero-start cycle normalized by its actual elapsed window, and atomic refusal of a nonrepresentable uncertain period. A separate native program imports the frozen c9a91032 physics module and reproduces the exact old time bug; the decimal100 reference confirms the old requested-step output differs by more than 0.01 from the actual-span RK4 control. Its three before records pass as a defect witness, not a corrected numerical result. The current core elapsed 0.45 s / max RSS 1,584 KiB; source packet 472 KiB. The compiler hash was verified again after both native runs.

Current compiler pin: `/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready`, SHA256 `a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`. Its custody is artifact 12580970584 / workflow run 40144792765 / witness 121024346990, independently verified by the parent lane. This is separate from package CI and human PR review.

The original private module is preserved byte-for-byte in `before_variance/phonon_original.sio`. Its native witness uses that same RHS in its defining compilation unit, with only effect annotations widened to satisfy the current checker and an appended driver. Pure decay has y'=-y, so an initial variance of one requires variance'=-2. The old RHS instead returns +1. The new factor-based analytic decay controls verify the corrected decay without inventing independent noise.

`compiler_context/` preserves an unrelated contextual native failure: an inline final commit in the larger step function fails, while extracting the same commit into a short helper passes both the retained and current compiler. Job 17197 replayed exact frozen before and after sources on the current pin. Compact IR segments and instruction/register counts are retained. This is evidence of a context-dependent compilation defect, not an identified compiler root cause; no compiler source is changed in this package. The exact original imported module and driver are copied into `compiler_context/frozen_before/`. A reduced standalone control retains the exact step body while stubbing the RHS/trial/limit bodies; it passes on the current compiler (job 17241), so local register counts alone do not establish the failing cause.

The earlier qual18 SI trap was a fixture error: it called nonexistent `dim_concentration`. The compiler accepted that unresolved import and the resulting executable trapped before constructing a physics state. `probe19/` preserves that exact diagnostic. Replacing the fixture with existing UnitDim/quantity_new APIs makes the production SI comparison pass; this trap is separate from the inline-commit issue.
