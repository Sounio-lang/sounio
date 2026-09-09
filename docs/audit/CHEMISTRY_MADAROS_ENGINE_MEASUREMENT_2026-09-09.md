<!-- docs:meta
topic_id: repo.docs.audit.chemistry-madaros-engine-measurement-2026-09-09
authority: repo_only
audience: users
last_validated: 2026-09-09
validated_by: chemistry-surface-microkinetics
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.chemistry-madaros-engine-measurement-2026-09-09
-->

# The chemistry stdlib, measured against Madaros (2026-09-09)

**What this is.** `stdlib/chemistry/catalysis.sio` and `kinetics.sio` both
carried `//@ check-only` test drivers, justified in their headers by a native
"thin-link" limitation. Those headers were written against the **lean_single**
engine. This document is what happens when the same files are pointed at
**Madaros**, which `bin/souc` has resolved to by default since the wrapper lane
merged (`docs/MADAROS_STATUS.md`).

The short version: **one of the two limitations does not exist on Madaros and
the driver is now a real, executing test; the other file is blocked by a
different and previously unrecorded engine incompatibility.**

## Provenance of every number below

| what | value |
|---|---|
| engine | Madaros v0.80.0, `bin/madaros-linux-x86_64` |
| md5 | `92f2c5664c6c6a8d8ba7c36bc0d503b8` |
| receipt | `artifacts/self-hosted/madaros.gate-receipt`, `gate_result=pass`, `source_commit=67caef4b74` |
| comparison engine | lean_single, `bin/souc-lean-single-x86_64`, via `SOUNIO_SOUC_ENGINE=lean_single` |
| tree | `origin/main` @ `ed1fa84234` |
| date | 2026-09-09 |

The ELF is the **committed, gate-receipted** binary, not a local build. Madaros
says so itself on every invocation ("this ELF is the COMMITTED binary; it is not
built from the tree above"). For a claim about compiler *behaviour* that is a
weaker artifact than a source build — so the claims below are scoped to what
this receipted binary does, which is exactly what a user of this repository
gets.

## 1. The thin-link limitation is a lean_single property, not a language one

`catalysis.sio`'s header records, as a standing and deliberately un-worked-around
limitation, that a function taking a `&![T;N]` mutable-array-reference parameter
"breaks the same thin-link stage when called from any function OTHER than
main()", and concludes that `catalysis.sio` "cannot currently be executed as ONE
linked binary through `bin/souc run`".

Measured on Madaros: **it can.** `catalysis.sio` compiles to a single 131 KB
native ELF, and each of the ten `pub fn test_*` in it runs and passes when
driven individually:

```
test_mass_action_exact                 rc=0  OK
test_mm_matches_baseline               rc=0  OK
test_mm_saturation_is_kcat             rc=0  OK
test_hill_identities                   rc=0  OK
test_langmuir_hinshelwood_asymptotics  rc=0  OK
test_enzyme_conservation               rc=0  OK
test_catalyst_conserved_cycle          rc=0  OK
test_qssa_checkpoint                   rc=0  OK
test_turnover_bookkeeping              rc=0  OK      <-- the &![f64;16] caller
test_turnover_carbonic_anhydrase       rc=0  OK
```

`test_turnover_bookkeeping` is the specific test the limitation named. Nothing
in `catalysis.sio` was changed to obtain this.

`tests/stdlib/chemistry/test_catalysis_stdlib.sio` is therefore **no longer
`//@ check-only`**.

## 2. What DOES bound it: a measured allocation budget, not a link failure

Running all ten in one binary fails — but on a different mechanism, and the
distinction matters:

```
madaros: arena full     (exit 181)
```

Madaros's generated runtime bump-allocates a 2 GiB arena and never reclaims
(`self-hosted/native/codegen_x86_linux.sio:4177`, "Its arena never refills").
The cost driver is not the integrator, it is **`MatNM`**, which carries
`data: [f64; 4096]` — 32 KiB per value, regardless of the declared `rows`/`cols`.
`catalysis.sio`'s `mechanistic_dc` allocates about twenty of them per call
(`matnm_new` + 16 × `matnm_set` + `matnm_mul` + the by-value `nu` parameter),
and RK4 calls it four times per step.

Bisected, on `simulate_mechanistic_crn` with the enzyme cycle:

| steps | result |
|---|---|
| 300 | `P=0.154055` |
| 600 | `P=0.311292` |
| 750 | `P=0.388037` |
| 850 | `madaros: arena full` |

So **~2.7 MiB per RK4 step, ~800 steps per process.** Cumulatively, the first
eight tests fit in one binary and the ninth does not:

| tests in one binary | result |
|---|---|
| 2, 4, 6, 8 | pass |
| 9, 10 | `madaros: arena full` |

The suite is therefore split across two drivers —
`test_catalysis_stdlib.sio` (rate laws and invariants) and
`test_catalysis_turnover_stdlib.sio` (turnover) — and both run green. The split
is recorded in both headers with this measurement, not presented as a stylistic
choice.

**This is a stdlib-efficiency finding as much as a compiler one.** A 32 KiB
fixed-capacity matrix copied twenty times per derivative evaluation is a design
cost that any new chemistry module should avoid rather than inherit.

## 3. A real stdlib visibility defect that only Madaros surfaces

`kinetics.sio` type-checks clean under lean_single (rc=0, 0 errors) and fails
under Madaros with **61 errors across 17 modules**. Sixteen of those are one
defect, in this class:

```
error[E175] in chemistry/kinetics::r_const: function is private in its defining module
   = callee constants/physical::gas_constant_approx
```

`stdlib/constants/physical.sio` declares **23 functions and zero of them `pub`**,
while six stdlib modules across `chemistry/` and `physics/` import and call it.
lean_single downgrades this to a warning ("cannot call non-pub function from
imported module"); Madaros is correct to reject it. A constants module with no
public surface is a module nobody can legally use.

Fixed here — `pub` on all 23 in `constants/physical.sio`, and on the seven
further cross-module callees Madaros named by hand
(`epistemic/ode::estate_new`, `estate_set`, `ode_params_new`, `ode_params_set`;
`plot/epistemic::error_bar_entry`, `error_bar_chart`; `linalg/matrix::mat3_new`).
Every `pub` added removes a measured error; none was added speculatively.

Result: **61 → 43 errors**, the entire E175 class gone.

Regression-checked against the pristine tree, both engines, on every module
touched or importing a touched module: `physics/thermo.sio`, `physics/sr.sio`,
`physics/em.sio`, `chemistry/equilibrium.sio`, `linalg/matrix.sio`,
`epistemic/ode.sio`, `plot/epistemic.sio`. Every exit code is **identical
before and after**. The pre-existing failures among them are pre-existing.

## 4. What still blocks kinetics.sio on Madaros, and why it is not a small fix

The remaining 43 are two classes, both in `chemistry/ontology.sio` and its
callers:

- **32 × `error[E004]`: `expected &str, found &string`.**
- **11 × `error[E009]`: argument type does not match parameter**, including
  function-pointer identity (`expected fn#148, found fn#36`).

The E004 site carries its own explanation, and it is the interesting part:

```sio
// Bare string literals compared against a `&str` parameter fail lean_single's
// checker ("comparison operands must have the same type" -- a bare literal
// infers as `string`, not `&str`); bind each to a local `string` first and
// compare against its reference, matching the workaround already used
// elsewhere in this stdlib wave for the same class of bug.
```

**The code contains a workaround written for lean_single, and that workaround is
what Madaros rejects.** The two engines disagree on whether `&str` and `&string`
are the same type, and the stdlib is written to one of them. This is not a typo
to patch; it is an engine-level incompatibility, and satisfying both from one
source is not obviously possible. It is recorded here rather than worked around
silently.

**Consequence, stated plainly:** `tests/stdlib/chemistry/test_kinetics_core.sio`
stays `//@ check-only`, and its header now says *this* is why, rather than
citing the thin-link reason that section 1 disproves.

## 5. What this measurement was worth, concretely

Extracting `kinetics.sio`'s own two enzyme integrators verbatim into standalone
probes — the only way to execute them, given section 4 — and running them under
Madaros produced:

```
direct  (enzyme_rhs + rk4_step):       E=0.016743 S=1.812689 ES=0.033257 P=0.154055
general (compute_rates_general path):  E=0.016743 S=1.812689 ES=0.033257 P=0.154055
mass balance S+ES+P = 2.000000
```

`kinetics.sio::test_enzyme_crn` asserted **P ≈ 0.7** on the first path and
**P > 0.4** on the second. Both are wrong by a factor of about 4.5, and
Michaelis-Menten bounds it analytically at `P ≤ ~0.165`. The test also had a
second defect that explains the first's survival: the direct-path check sat in
statement position as dead code, so the function returned only the general-path
line. Corrected in this change, with the full derivation in the function's own
comment.

**A wrong assertion sat in merged, trusted stdlib code because its driver was
`check-only`.** That is the cost of a test that type-checks and never runs, and
it is the reason section 1 was worth measuring.
