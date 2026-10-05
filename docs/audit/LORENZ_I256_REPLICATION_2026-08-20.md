<!-- docs:meta
topic_id: repo.docs.audit.lorenz-i256-replication-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-08-20
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lorenz-i256-replication-2026-08-20
-->

# Independent replica — magnitude of Lorenz certificate i256 arithmetic

**Status:** measured. `stdlib/systems` is not patched. i256 annotations
are not changed. i256 is not implemented. No certificate *conclusion*
is asserted false: a wrapping product can still fall on the passing
side of a comparison.

**Verdict: EXCEEDS THE I64**

The signed i64 maximum is `9223372036854775807` (`2^63 - 1`).
The largest intermediate this replica observed, on a path that the
from-source Madaros actually ran, is

```text
8007432506888905229835698176
```

That is `y_lte_source * den` at
`stdlib/systems/lorenz_i256_cert_step5.sio:2310` inside
`lorenz_i256_step5_taylor2_remainder_obligation_check`.
Bit-width of the absolute value: **93**. Exact identity:

```text
y_lte_source = 217041893
den          = 2 * source_scale * source_scale
source_scale = 4294967296 = 2^32
den          = 2^65
product      = 217041893 * 2^65
             = 868167572 * 2^63
```

It is **not** an integer multiple of `2^63 - 1`. The integer division
`product / (2^63 - 1)` is `868167572` remainder `868167572`.

Worktree: `/workspace/.wt/lorenz-i256-replica-20260820`
Branch: `lane/grok-cli1/lorenz-i256-replica-20260820`
Source SHA: `67aa2aec12` (`origin/main` at replica start).
The branch was not switched while the from-source build ran.

---

## Semantic declaration

```text
Semantic-Lane-ID: lorenz-i256-replica-20260820
Owner: grok-cli1
Concept-IDs: none
Intent-Preserved: magnitude of executed Lorenz i256 arithmetic versus
  signed i64, measured independently of any prior receipt.
Transformation: none.
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the executed remainder-obligation comparison at
  step5.sio:2310 reaches 217041893 * 2^65, which does not fit in i64.
Claims-Forbidden: Madaros is fixed-point-verified; the Lorenz
  certificate is therefore false; i256 should be implemented this turn.
Assumptions: Python int is an exact accumulator (no float). A function
  invoked by a from-source Madaros test with rc=0 was executed.
Write-Set: docs/audit/LORENZ_I256_REPLICATION_2026-08-20.md
Read-Set: stdlib/systems/lorenz_i256_*.sio, stdlib/theorem/div_witness.sio,
  tests/run-pass/lorenz_i256_*.sio, scripts/dev/souc-build-remote.sh
Positive-Witness: detector fires on 2^63 before any Lorenz number is
  reported; from-source ELF runs the step-5 remainder fixture at rc=0.
Negative-Witness: a detector that never fired; a peak taken from a
  function that was only grepped.
Acceptance-Gate: control must fire; peak site must be on a runtime path.
Integration-Target: origin/main (docs only)
Authoritative-Only-If: the peak is named with file:line:expression,
  the runtime fixture, and an integer identity with no float.
```

---

## Method, written before the numbers

1. Build Madaros from this tree via `scripts/dev/souc-build-remote.sh`
   (Slurm, not the workspace pod). Do not change the branch while it
   packs the live tree.
2. Exact accumulator: Python `int`. No `float`, no `f64`, no decimal
   approximation of any product.
3. **Positive control first.** Feed the detector `2^63` (which is
   `I64_MAX + 1`) and `5000000000000000000 + 5000000000000000000`.
   If either is not flagged, the replica is void.
4. Name coverage. A `_check` function with a `tests/run-pass` caller
   is a *candidate*. Only functions invoked on the from-source ELF
   this turn are **runtime**. Source-replay of the others is declared
   as such and is not counted as execution.
5. Static inspection of an uncalled body is **NOT EXECUTABLE**.

The existing receipt
`docs/audit/LORENZ_I256_PRODUCT_MAGNITUDE_2026-08-20.md` / PR #2046
was not read until after the verdict below was in hand.

---

## Positive control

Detector: `value > 9223372036854775807 or value < -9223372036854775808`.

```text
CONTROL_PASS
  known 2**63 = 9223372036854775808 > I64_MAX  flagged=True
  5e18+5e18   = 10000000000000000000           flagged=True
```

The detector fired before any Lorenz expression was reported. The
instrument is not broken.

---

## From-source Madaros

Mandated remote build (`souc-build-remote.sh`, partition `all`, node
`gpuorangefs-r770-proxmox`, 32 CPUs):

```text
REMOTE: host=gpuorangefs-r770-proxmox nproc=32 unpacked=70M
REMOTE: build rc=0 elapsed=229s
REMOTE: elf bytes=100562528
```

That script deletes the ELF. A second from-source build of the same
tree was kept at
`/orangefs/training/lorenz-i256-replica-20260820T1538Z/madaros.elf`.

```text
BUILD_RC=0 elapsed=226s
bytes    100562528
SHA-256  0b2f7e21f7a9260e85cbd13e121bfd7537b3ef273148db2192abe2e241bf2769
compile: fns=10951
```

`fn i256_*` occurs **zero** times under `stdlib/`. `: i256` annotations
in `stdlib/systems/` count **733**.

On this ELF, not in dispute and re-measured:

```text
let a: i256 = 5000000000000000000 as i256
let s: i256 = a + a
print_int(s as i64)
→ -8446744073709551616    WRAP_RUN_RC=0
```

That is the exact two's-complement wrap of `10000000000000000000` in
signed i64. i256 is i64 on this compiler. The magnitude question is
therefore the *mathematical* product, not the stored word.

---

## Runtime paths this replica actually ran

From-source ELF, `SOUNIO_STDLIB_PATH` pinned to the staged tree.

| fixture | build rc | run rc | what it proves |
|---|---:|---:|---|
| `i256_wrap.sio` (local probe) | 0 | 0 | i256 addition wraps at i64 |
| `tests/run-pass/lorenz_i256_step5_taylor2_remainder_obligation_imported.sio` | 0 | **0** | `lorenz_i256_step5_taylor2_remainder_obligation_check` ran and returned fingerprint `69995750` |
| `tests/run-pass/lorenz_i256_step2_taylor2_local_flowpipe_seed_imported.sio` | 0 | **0** | `lorenz_i256_step2_taylor2_local_flowpipe_seed_check` ran and returned fingerprint `633359277` |

The step-5 fixture calls the peak function with
`(40000, 55000, 10000, 0)`. Those arguments are ppm caps; the i256
lets that form `den` and `y_lte_source` are literals inside the
function. The comparison at line 2310 is not behind a dead branch:

```sounio
let den: i256 = (2 as i256) * source_scale * source_scale
…
if y_lte_source * den < y_second * dt2 { source_lte_ok = 0 }
```

Both products are evaluated on every call. rc=0 means the function
returned the expected fingerprint, so that comparison ran.

Exact arithmetic of the same comparison (not the wrapped machine
word) does **not** take the failing branch:

```text
y_lte_source * den     = 8007432506888905229835698176
y_second * dt2         = 8007432477754892763381441088
y_lte_source * den < y_second * dt2          → false
(y_lte_source-1)*den >= y_second * dt2       → false
```

`source_lte_ok` stays 1 at full width. A wrapping implementation can
pass for a different reason. That is a different question, and this
replica does not answer it.

The step-2 seed path, also runtime, forms

```text
xy_num = x1 * y1
       = 4294967296 * 5411658768
       = 23242897425671651328          (65 bits, exceeds i64)
dy_num = x1 * ((28 as i256)*scale - z1)
       = 498369535714984460288         (69 bits, exceeds i64)
```

at `stdlib/systems/lorenz_i256_cert_step2.sio:286–285`. Smaller than
the peak, independently exceeding i64, and on a second executed path.

`div_witness_check_i256` (`stdlib/theorem/div_witness.sio:27–34`) is
a callee of the seed check. It multiplies `quotient * denominator`.
For `xy_num` that is again `5411658768 * 4294967296` (65 bits).

---

## Exact replay of other `_check` bodies

`tests/run-pass` contains 710 `lorenz_i256*` fixtures. They mention
204 distinct `lorenz_i256_*_check` functions. Every one of those 204
is defined under `stdlib/systems/`. An exact replay of their i256
`let`s and in-line products (literals and earlier lets only) reports
110 intermediates above i64, with the same peak at step5:2310.

That replay is **not runtime**. Per the method, those 202 functions
not invoked on the from-source ELF this turn are

**NOT EXECUTABLE (this replica)** — a test file exists; this turn
did not run it.

Files in `stdlib/systems/` with no `_check` function:
`ball_fixed.sio`, `gpu.sio`, `lib.sio`, `mod.sio`, and the façade
`lorenz_i256_cert.sio` (`pub use` only). Not executable as
certificate arithmetic.

Uninterpreted lets in the replay: 258. Of those, 224 are `… as i256`
casts from i64 names (cannot exceed i64). 34 are subtractions or
products whose operands were not in the literal environment
(`dt_q * dx_rad` and similar). None of those 34 can be claimed as a
larger peak. They remain **NOT EXECUTABLE / not interpreted**.

---

## Peak, named

| field | value |
|---|---|
| absolute value | `8007432506888905229835698176` |
| bits | 93 |
| file:line | `stdlib/systems/lorenz_i256_cert_step5.sio:2310` |
| function | `lorenz_i256_step5_taylor2_remainder_obligation_check` |
| expression | `y_lte_source * den` |
| `y_lte_source` | `217041893` (line 2220) |
| `den` | `(2 as i256) * source_scale * source_scale` (line 2225) |
| `source_scale` | `4294967296` (line 2199) |
| runtime fixture | `tests/run-pass/lorenz_i256_step5_taylor2_remainder_obligation_imported.sio` |
| fixture result | build rc=0, run rc=0 |

Compare with `9223372036854775807`: the product is larger.

---

## Coverage map

**Runtime this turn**

- wrap probe (i256 is i64)
- step-5 remainder obligation (peak)
- step-2 local-flowpipe seed (`xy_num`, `dy_num`)
- `div_witness_check_i256` as callee of the seed path

**NOT EXECUTABLE this turn** (test exists or body exists; no
from-source invocation here)

- remainder / centre / radius `_check` for steps 1, 3, 4, 6
- trajectory-5 certificate
- child-0 through child-4 bundles
- bridge families (`scaled_product_bridge`, `beta_z_bridge`, …)
- long loops, GPU, `ball_fixed`
- the 34 uninterpreted lets

Partial declared coverage is the result. It is enough for the
verdict because the peak sits on a runtime path.

---

## Verdict

**EXCEEDS THE I64**

Because a product on an executed certificate comparison is
`217041893 * 2^65`, which does not fit in signed i64. The compiler
stores i256 as i64; that wrap was re-measured. Whether the Lorenz
*claim* is still true under wrap is not this question.

---

## Comparison with #2046 (read last)

#2046 (`audit: measure Lorenz i256 product magnitudes`, head
`codex-2/lorenz-i256-magnitude-20260820`) was opened on the same
source SHA `67aa2aec12`. Its receipt was not read until the numbers
above were already in hand.

**Agrees**

- Verdict: EXCEEDS I64.
- Peak value: `8007432506888905229835698176`.
- Site: `lorenz_i256_cert_step5.sio:2310`, expression
  `y_lte_source * den`, function
  `lorenz_i256_step5_taylor2_remainder_obligation_check`.
- From-source ELF size `100562528`, `compile: fns=10951`.
- ELF SHA-256 of this replica's kept blob
  `0b2f7e21f7a9260e85cbd13e121bfd7537b3ef273148db2192abe2e241bf2769`
  matches the SHA #2046 quoted (they omitted the last hex nibble in
  the PR body; the 64-hex form here is the `sha256sum` of the ELF
  this turn built). Same compiler, same tree.

**Disagrees, and why**

1. **Ratio to the i64 bound.** #2046 says the product is “exactly
   868,167,572 times that bound”, and names the bound as
   `9,223,372,036,854,775,807` (`2^63 - 1`). Integer division of the
   shared peak by `2^63 - 1` is `868167572` remainder `868167572`, so
   it is *not* an integer multiple of the signed maximum. The exact
   identity is `868167572 * 2^63` = `217041893 * 2^65`. The integer
   868167572 is right; the word “exactly … times [`2^63 - 1`]” is
   not.
2. **Positive control.** #2046 forced `10^30`. This replica forced
   `2^63` and `10^19`. Both fire a working detector. `2^63` is the
   tightest value that must be flagged. Not a disagreement of
   outcome.
3. **Runtime breadth.** #2046 ran 25 fixtures (centre, radius,
   remainder for steps 1–6, plus others). This replica ran two
   certificate fixtures plus the wrap probe. The peak function *was*
   among those two. Broader runtime in #2046; independent exact
   replay here of 204 `_check` bodies that have tests, of which 202
   are still **NOT EXECUTABLE** this turn. Different coverage, same
   peak.
4. **Build host.** #2046 used `cpuops-t560-proxmox`. This replica
   used `gpuorangefs-r770-proxmox`, which is what
   `souc-build-remote.sh` targets (t560 is refused as the
   control-plane). Irrelevant to the integer.

**Measures a different thing**

#2046's 933 intermediates versus this replica's 461 lets + 130
in-line products is a counting difference (callee
`div_witness_check_i256` expansions, comparison products, how many
functions were replayed). Not a different peak.

No Lorenz annotation was changed. No certificate conclusion was
overturned.
