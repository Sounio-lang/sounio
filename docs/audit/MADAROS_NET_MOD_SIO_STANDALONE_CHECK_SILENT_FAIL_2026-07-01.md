<!-- docs:meta
topic_id: repo.docs.audit.madaros-net-mod-sio-standalone-check-silent-fail-2026-07-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-net-mod-sio-standalone-check-silent-fail-2026-07-01
-->

# Madaros checker: `souc check` on stdlib/net/{addr,tcp,udp}.sio fails silently once net/mod.sio exists (2026-07-01)

Branch `codex/stdlib-networking-wave-20260701` off `main` @ `53169eb4b`.

## Symptom

Before this branch's changes, `stdlib/net/` had no `mod.sio`. Adding one (per the
`mod.sio` public-surface convention already used by `stdlib/http`, `stdlib/distributed`
etc.) and marking `IPv4Addr`/`SocketAddr`/`TcpConfig`/`UdpConfig` and their constructor
and impl methods `pub` (required for `mod.sio`'s `pub use` re-export to resolve) causes:

```
$ souc check stdlib/net/addr.sio   # and tcp.sio, udp.sio
Madares v0.80.0 -- ...
(no diagnostic printed)
$ echo $?
1
```

Reproduced consistently (9/9 runs) with `SOUNIO_STDLIB_PATH` set correctly and the repo
otherwise clean. No error code, no message — the checker exits 1 with empty diagnostic
output.

## What does NOT trigger it

- `souc check stdlib/net/mod.sio` — OK.
- `souc check stdlib/net/lib.sio` — OK (tried both the `pub use net::{addr,tcp,udp}::*;`
  re-export form and a re-export-free comment-only form — **no difference**, both still
  make `addr.sio` itself fail standalone; this rules out "double re-export" as the cause,
  despite an earlier single non-reproducing run that suggested otherwise — see below).
- `souc check tests/stdlib/net/test_net_core.sio` (the real consumer, `use net::...`
  qualified calls) — OK.
- `souc check tests/stdlib/net/test_addr_e2e.sio` (the real consumer, `use net::addr::*`)
  — OK.
- `souc check stdlib/http/http.sio` with its own sibling `mod.sio` + `lib.sio` both
  re-exporting the same symbols — OK (this module has no `impl` blocks).
- `souc check stdlib/distributed/pure/types.sio` with its own sibling `mod.sio` +
  `lib.sio` both re-exporting the same symbols — OK (this module also has no `impl`
  blocks on the types re-exported at this level).
- Removing `stdlib/net/mod.sio` entirely — `addr.sio` then checks OK again.

## Working hypothesis (unconfirmed)

The common factor across the two failing files (`addr.sio`: `impl IPv4Addr`,
`impl SocketAddr`; `tcp.sio`: `impl TcpConfig`; `udp.sio`: `impl UdpConfig`) vs. the two
non-failing analogues (`http/http.sio`, `distributed/pure/types.sio`) is the presence of
`impl Type { pub fn ... }` blocks on a type that a sibling `mod.sio` also re-exports by
name. One isolated probe (a minimal `pub struct Foo` + free `pub fn` + `impl Foo { pub fn
get_a }`, no sibling `mod.sio`) checked fine standalone — so a bare `impl` block is not
sufficient on its own; the interaction with a sibling `mod.sio`'s re-export of the same
type name is likely required. Not fully isolated — see follow-up needed below.

## Why this is not fixed here

This is a `self-hosted/` checker behaviour, not a `stdlib/` authoring mistake. Per
CLAUDE.md §8, compiler bugs go through the forensic-dispatch protocol, not ad hoc
patches from a stdlib PR. Filed here for triage.

## Practical impact (measured)

No CI gate or script checks `stdlib/net/{addr,tcp,udp}.sio` directly as a standalone
target (`grep -rl "net/addr\.sio\|net/tcp\.sio\|net/udp\.sio" scripts/ tests/ docs/
Makefile` returns only the E2E test files that consume them, not the files themselves).
The actual consumers (`tests/stdlib/net/test_net_core.sio`,
`tests/stdlib/net/test_addr_e2e.sio`) both check clean. Net effect: zero observed impact
on any real gate today, but it is a latent trap for anyone who runs `souc check` on a
`stdlib/net/*.sio` sibling file directly (a natural thing to do while editing).

## Follow-up needed (not done here)

- Isolate with a minimal 2-file repro (`mod.sio` + one `impl`-bearing sibling, no other
  content) to confirm/refute the impl+mod.sio-reexport hypothesis without stdlib noise.
- If confirmed, the fix likely belongs in the checker's package/module symbol-table
  construction (self-hosted/check/), where a same-directory sibling file's own top-level
  symbols may be getting merged with mod.sio's re-exports of those same symbols in a way
  that a `impl`-Type's method table does not tolerate.

## Update 2026-07-02: two more lean_single (not Madaros) bugs found via CI, one fixed pragmatically

This dispatch was written after testing exclusively against the locally-prebuilt Madaros
binary (`bin/madaros-linux-x86_64`), because `artifacts/self-hosted/madaros` did not exist
in this worktree and `bin/souc` silently fell back to whatever engine `_resolve_madaros`
found — nothing in the session flagged that CI's "Full Test Suite" job (`souc-stage2`)
uses a **freshly-bootstrapped `lean_single`**, a different engine entirely
(`self-hosted/compiler/lean_single.sio`, not `self-hosted/compiler/main.sio`). PR #565's
new `tests/stdlib/net/test_net_core.sio` passed every local check under Madaros and still
failed CI. Rebuilding `lean_single` locally via `scripts/dev/souc-build-lock.sh
./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fresh.elf`
and testing against *that* reproduced the CI failure and two distinct, previously-unknown
lean_single bugs (isolated with minimal repros outside stdlib, both independent of the
Madaros bug above):

**Bug A — inline fully-qualified paths need ≥3 segments.** `pkg::file::symbol()` (a
2-segment path into a sibling file directly under the package root, e.g.
`net::addr::ipv4_loopback()`) fails to resolve (`error: unknown identifier`), regardless
of whether the target file has an explicit `module pkg::file` header. `pkg::subdir::file::
symbol()` (3+ segments, e.g. `distributed::pure::types::registry_new()`) resolves fine.
`use pkg::file::*;` followed by a *bare* call (e.g. `use net::addr::*; ipv4_loopback()`)
works for 2-segment packages too — this is the workaround used in
`tests/stdlib/net/test_addr_e2e.sio` (pre-existing) and now in the rewritten
`tests/stdlib/net/test_net_core.sio`. This also affects `stdlib/http`/`stdlib/web`
(`http::http::X`, `web::http::X` are 2-segment) — confirmed pre-existing on `main` before
this branch (traced via `git show main:...`), not introduced here.
`tests/stdlib/web/test_web.sio` is separately listed in
`tests/known_failures/hardened_diagnostics_full_suite.txt` and is silently skipped by the
CI harness rather than failing — that registration predates this branch too.

**Bug B — qualified-path parameter type + co-scope glob import → false arity mismatch.**
A function parameter typed as a qualified path (`fn f(x: &pkg::mod::Type)`) reports
`error: arity mismatch` at *every* call site that also has `use pkg::mod::*` in scope,
even when the call's argument count is correct. Reproduced in an isolated 2-file repro
(package root file `pkg/addr.sio` exporting `Thing`/`thing_new`; nested `pkg/sub/wrapper.sio`
exporting `take_thing(target: &pkg::addr::Thing)`; caller with both `use pkg::addr::*;
use pkg::sub::wrapper::*;` in scope). Renaming the parameter away from a name that
collides with the type path's own segment text (`addr: &pkg::addr::Thing` →
`target: &pkg::addr::Thing`) fixes an *additional* co-occurring "duplicate parameter name"
false positive but does not fix the arity mismatch itself.

**Disposition:** Bug A and Bug B are both self-hosted/check checker bugs, not fixed here
(§8 applies the same way as the impl+mod.sio bug above). Worked around pragmatically in
`stdlib/net/ffi/wrapper.sio` by changing `tcp_connect`/`udp_send` to take raw IPv4 octets
+ port (`i64` primitives) instead of `&net::addr::SocketAddr` — arguably better FFI-stub
design anyway (a real syscall boundary needs primitives, not an opaque struct reference),
and `tests/stdlib/net/test_net_core.sio` was rewritten to use `use pkg::mod::*` + bare
calls throughout instead of inline fully-qualified paths. Verified against a freshly
rebuilt `lean_single` (matching CI's `souc-stage2` build path) via
`scripts/run_sio_test_suite.sh --filter test_net_core` (and no regression on
`test_addr_e2e`, `test_distributed_core`, `test_http_e2e`).

**Process lesson:** local dev testing in this worktree defaulted to Madaros
(`bin/madaros-linux-x86_64`, prebuilt, present from earlier unrelated work) without any
visible fallback notice, while CI's authoritative gate uses a freshly-bootstrapped
`lean_single`. The two engines disagree on real, checker-level behavior (not just
performance). Before trusting a local `souc check` result as CI-equivalent, confirm which
engine `bin/souc info` (or `_resolve_madaros`) actually resolved to, or rebuild
`lean_single` from source and test against that directly.

## Update 2026-07-02 (continued): Bug D — qualified call nested as an argument miscounts the outer call's arity

Found while fixing issue #569 (the "arity mismatch" cluster: `tests/stdlib/dataframe/
test_dataframe_core.sio`, `tests/stdlib/audio/test_audio_core.sio`, `tests/stdlib/geo/
test_geo_core.sio`, `tests/stdlib/simulation/test_simulation_core.sio`). Issue #569 had
speculated this cluster might share Bug B's root cause (qualified-path parameter type +
co-scope glob import) — checked and **ruled out**: none of the 4 files' failing calls
have a qualified-path parameter type anywhere in scope. The real, distinct trigger,
isolated with a minimal 2-file repro outside stdlib:

```sio
// stdlib/pkg/sub/inner.sio
module pkg::sub::inner
pub fn two_args(a: i64, b: i64) -> i64 { a + b }

// main.sio
fn outer(x: i64, y: i64) -> bool { x == y }

fn main() -> i64 {
    let r1 = outer(pkg::sub::inner::two_args(1, 2), 3)   // error: arity mismatch (on outer!)
    let r2 = pkg::sub::inner::two_args(1, 2)              // fine — same call, standalone
    0
}
```

A fully-qualified call (`pkg::subdir::file::symbol(...)`, 3+ segments — already known to
resolve per Bug A) used as an **argument expression nested inside another call**
(`outer(qualified_call(...), other_arg)`) produces a false `arity mismatch` on the
*outer* call, even though both calls individually have the correct argument count. The
identical call used as its own standalone statement (not nested) does not trigger it.
Madaros does not reproduce this (checks OK); confirmed lean_single-only, same as Bugs A
and B.

All 4 affected files had the exact same shape: `near(pkg::pure::types::fn(args), literal)`
— a qualified accessor call nested directly inside a `near()` (approx-equality) helper
call. Fixed by rewriting to `use pkg::pure::types::*;` (matching the established Bug A
workaround) *and* binding each nested call's result to a local `let` first, e.g.
`let v = fn(args); if !near(v, literal) { ... }` — the local-binding step was kept even
though `use` already de-qualifies the call, since Bug D was isolated for *qualified*
nested calls specifically and a bare nested call was not independently verified safe;
binding first avoids the ambiguity regardless. `tests/stdlib/simulation/
test_simulation_core.sio` also had a separate `error: unknown field access` on
`ts.n_points` (a `pub` field, so not a visibility issue) that resolved once `ts` was
constructed via the `use`-imported bare `timeseries_new()` instead of a fully-qualified
call — consistent with Bug A/D also disrupting return-type tracking through a qualified
call chain, not just direct symbol/arity resolution.

Verified against a freshly rebuilt `lean_single` via `scripts/run_sio_test_suite.sh
--filter`, no regression on the full set of tests already fixed in this dispatch's
earlier updates.

## Update 2026-07-02 (continued): closing most of issue #567 (Bug A cluster), plus a Madaros-only regression and a "halt, don't fix" call on stdlib/chemistry/kinetics.sio

Fixed 6 of #567's 7 files (`test_equilibrium_acids`, `test_serialization`,
`test_thermo_sr_e2e`, `test_viz_core`, `test_particle_physics_core`,
`test_graphics_core`) the same way as the Bug A/D fixes above: `use pkg::mod::*;` +
bare calls, plus un-nesting a triple-nested qualified call in `test_graphics_core.sio`
(same Bug D shape). Two incidental content bugs surfaced and were fixed at the root:

- `stdlib/chemistry/equilibrium.sio` and `stdlib/chemistry/acids.sio` each had an
  **unused** `use epistemic::knowledge::Epistemic;` (the named-import Bug C shape,
  documented in issue #568) that was silently dead code — deleted rather than converted,
  since nothing in either file referenced `Epistemic`.
- `stdlib/chemistry/equilibrium.sio`, `stdlib/chemistry/kinetics.sio`, and
  `stdlib/physics/sr.sio` each had `use constants::physical::X as get_R;` /
  `use constants::physical::speed_of_light_approx;` (Bug C shapes) — converted to
  `use constants::physical::*;` and renamed the 1-2 call sites per file back to the
  real name (`get_R()` → `gas_constant_approx()`).
- `stdlib/chemistry/kinetics.sio` additionally had `use plot::epistemic as epi_plot;`
  (a *module-alias* variant of Bug C) used as a namespace prefix at 6 call sites —
  converted to `use plot::epistemic::*;` and stripped the `epi_plot::` prefix at each
  site — and `use special::caputo;` + `caputo::fn(...)` qualified-via-bare-module-alias
  calls (4 sites) — converted to `use special::caputo::*;` + bare calls.
- `stdlib/autodiff/grad.sio`'s `struct Dual { val: f64, dot: f64 }` had neither the
  struct nor its fields marked `pub` — `stdlib/chemistry/kinetics.sio` constructs `Dual`
  literals directly once `Dual` became reachable via `use autodiff::grad::*;`, which
  needs `pub` on both the struct and its fields (same shape as the `stdlib/web`
  `pub`-visibility fixes from the networking wave). Fixed.
- `tests/stdlib/viz/test_viz_core.sio` itself asserted `scene.data_count != 0` on a
  field that **does not exist** on `VizScene` (confirmed via `grep`) — a pre-existing
  test bug, masked until now by the qualified-call failure occurring first. Replaced
  with the real field `series_count` (also zero-initialized by `viz_scene_new()`),
  preserving the test's evident intent (assert a fresh scene has no data yet).

**Madaris-only regression, documented not fixed:** all 6 fixed files now fail standalone
`souc check` under Madaros (confirmed via `git show HEAD:<file>` that the *originals*
passed Madaros cleanly before this change) despite passing cleanly under a freshly
rebuilt `lean_single`. This is the same class of engine divergence as this dispatch's
original finding (Madaros and lean_single disagree on `use module::*;`-adjacent checker
behavior) — not investigated further per the same "flag, don't fix Madaros-only checker
quirks" reasoning as above. No CI gate exercises Madaros for these files.

**Halt call: `stdlib/chemistry/kinetics.sio` is out of scope for a quick fix.** The 7th
file, `tests/stdlib/chemistry/test_kinetics_core.sio`, remains failing after the same
import fixes above (which did land real, verified improvements — Bug C's imports now
resolve, `Dual` literals now construct). Once the qualified-path/import layer was fixed,
`souc check` under lean_single surfaced roughly 100+ *additional*, independent errors
throughout kinetics.sio's ~2000 lines: dozens of "arity mismatch" (the Bug D shape,
recurring at ~28 distinct call sites, not just the 1-2 seen elsewhere), "unknown
identifier `[`" (a parser/checker issue with array literals, ~35 occurrences),
"private struct field access [struct=ODESolution]" (another un-`pub`bed struct, same
shape as `Dual` above but a different type), 6 distinct "tail type mismatch" sites, and
a **genuinely nonexistent** imported symbol (`use epistemic::ode::{..., 
ode_system_chem_simple, ...};` — confirmed via `grep -n ode_system_chem_simple
stdlib/epistemic/ode.sio` returning nothing). This is consistent with kinetics.sio never
having been fully type-checked by any compiler before — it was always accessed via
inline fully-qualified calls that never resolved under lean_single at all (Bug A), so no
one has run the whole file through lean_single's checker until this investigation.
Fixing kinetics.sio properly is a dedicated stdlib-content triage task, not a follow-on
to a qualified-path bug fix — filed as a separate issue rather than attempted here.

Verified via `scripts/run_sio_test_suite.sh --filter` against a freshly rebuilt
`lean_single`: all 6 fixed files pass, no regression on the full set of tests fixed
earlier this week (`test_net_core`, `test_env_present`, `test_classical_e2e`,
`test_dataframe_core`, `test_audio_core`, `test_geo_core`, `test_simulation_core`,
`test_distributed_core`, `test_http_e2e`).

## Update 2026-07-02 (continued): issue #566 closed as a kinetics.sio duplicate; issue #568's two files split — one fixed, one also blocked by kinetics.sio

**Issue #566** (`test_pbpk_ontology.sio` failing both CI jobs) was investigated next.
Converting its inline fully-qualified calls to `use chemistry::kinetics::*;` (the
standard Bug A fix) immediately surfaced ~274 of the same pre-existing errors already
inventoried in issue #575 — this is the identical `stdlib/chemistry/kinetics.sio`
content debt, reached through a different test file that happens to call a different
(also-broken) subset of the module's functions. Reverted the test-file edit (trading 4
clear "unknown identifier" errors for 274 confusing ones fixes nothing) and closed #566
as a duplicate of #575 rather than tracking the same root cause twice.

**Issue #568**'s two files split into one real fix and one partial, both via the
established Bug C workaround:

- `tests/stdlib/physics/test_em_e2e.sio`: root cause was in `stdlib/physics/em.sio`
  itself (`use constants::physical::speed_of_light_approx;`, 1 usage site) — converted
  to `use constants::physical::*;`. Unlike kinetics.sio, `em.sio` is a normal-sized
  (419-line) file with no other latent debt — fixed cleanly, verified passing.
- `tests/stdlib/chemistry/test_lib_surface.sio` imports one named symbol each from THREE
  modules (`chemistry::kinetics::test_arrhenius`, `chemistry::equilibrium::test_k_dg`,
  `chemistry::acids::test_ph`) but never calls any of them — it's a pure "does this
  import surface resolve" smoke test. The `equilibrium`/`acids` imports fixed cleanly
  (glob, matching the already-verified #567 fixes to those two files). The `kinetics`
  import hits the exact same #575 wall as #566 — reverted to its original (still Bug-C-
  broken) form rather than glob-importing and drowning the signal in 274 unrelated
  errors. Net effect: this file's own error count dropped from 3 to 1, but it does
  **not** flip to passing — it remains blocked on #575, same as #566 was.

Verified via `scripts/run_sio_test_suite.sh --filter`: `test_em_e2e` passes;
`test_lib_surface` still fails (expected, tracked against #575); no regression on the
full set of tests fixed earlier this week.

## Update 2026-07-03: issue #570 (unknown-field-access / type-mismatch cluster) — real bug, not stdlib content, same shape as Bug A/D's return-type-tracking gap

Issue #570 was filed at low confidence — it was unclear whether `test_mesh_core.sio` and
`test_serial_core.sio` had genuine content bugs or were hitting a checker issue.
Confirmed **checker issue**, not content: `stdlib/mesh/pure/types.sio` (`Mesh`) and
`stdlib/serial/pure/types.sio` (`SerialConfig`/`SerialBuffer`) all already had every
field correctly `pub`, and `mesh_get_vertex` already correctly returns `[f64; 3]`. Both
test files call their target module functions via *3-segment* fully-qualified paths
(`mesh::pure::types::mesh_get_vertex(...)`, `serial::pure::types::serial_config_new(...)`)
— already known to *resolve* fine per Bug A (3+ segments), so this isn't Bug A. Instead:
the checker fails to track the **return type** of a 3-segment fully-qualified call
correctly for later use — indexing the result (`v[0]` on a `mesh_get_vertex(...)`
result) or field-accessing it (`cfg.baud_rate` on a `serial_config_new(...)` result)
then fails with a type/field error, even though the value's real type is completely
correct.

This is the same underlying gap already noted for `tests/stdlib/simulation/
test_simulation_core.sio`'s `ts.n_points` fix two updates ago in this dispatch ("Bug
A/D also disrupting return-type tracking through a qualified call chain, not just direct
symbol/arity resolution") — confirmed here to also apply at 3 segments, not just 2, and
to indexing as well as field access. Fixed both files the same way: `use pkg::mod::*;`
+ bare calls (binding the qualified call's result to a `let`/`var` first, then indexing/
field-accessing the *locally-typed* value, resolves cleanly). No stdlib source files
needed changes this time — `mesh/pure/types.sio` and `serial/pure/types.sio` were
already correct.

Verified against both Madaros and a freshly rebuilt `lean_single`: both files pass
cleanly on both engines (no Madaros-only regression this time, unlike the #567/#568
fixes). No regression on the full set of tests fixed earlier this week.

## Update 2026-07-03: issue #575 (kinetics.sio full audit) — two new bugs found and fixed, test now passes

Picked up issue #575 (`stdlib/chemistry/kinetics.sio` had ~100+ pre-existing errors,
never fully type-checked before this week's investigation). Two more previously-unknown
`lean_single` checker bugs found and isolated with minimal repros outside stdlib —
fixing both, plus the two isolated content bugs already flagged in #575's filing,
brought `tests/stdlib/chemistry/test_kinetics_core.sio` from 274 errors (rc=1, blocking)
down to a passing state.

**Bug E — `&[literal, ...]` (a reference to an inline array literal) as a call argument
fails to parse.** Isolated:
```sio
fn takes_arr(a: &[f64; 3]) -> f64 { a[0] }
fn main() -> f64 {
    takes_arr(&[1.0, 0.0, 0.0])   // error: unknown identifier `[` + arity mismatch
}
```
`&variable_name` (a reference to an already-bound local) works fine; only the inline
literal form breaks. Workaround: bind the literal to a local `let` first (with an
explicit `[T; N]` type annotation), then pass `&local_var`. This was by far the largest
contributor to kinetics.sio's error count — a subagent applied the fix at 25 call sites
across the file (many were the same repeated 8-element "demo initial concentration"
vector reused across different test functions), reducing errors from 274 → 81. Also
affects string literals passed where `&str` is expected (`fn f(s: &str)`, called as
`f("literal")`) — same root cause, same workaround
(`let s: string = "literal"; f(&s)`), used to fix 3 more sites in kinetics.sio.

**Bug F — a module-level `let CONST: f64 = some_fn();` (initialized from a function
call, not a literal) doesn't propagate its declared type to later arithmetic.**
Isolated:
```sio
fn get_const() -> f64 { 8.314 }
let R: f64 = get_const()          // module-level, from a function call
fn f(t: f64) -> f64 { R * t }     // error: arithmetic operands must have matching numeric types
```
A module-level `let` initialized from a *literal* (`let R: f64 = 8.314`) does NOT
trigger this — only the function-call-initializer form does. Workaround: replace the
module-level `let` with a zero-arg function (`fn r_const() -> f64 { get_const() }`) and
call it at each use site instead of referencing a bare constant. Fixed in both
`stdlib/chemistry/kinetics.sio` (1 site) and `stdlib/chemistry/equilibrium.sio` (6
sites) — the latter's existing `let R: f64 = gas_constant_approx();` was the direct
cause of the "44 arithmetic operand" errors tolerated as non-fatal (marker-present) in
the #567 update above; now genuinely fixed rather than merely tolerated.

**Also fixed (content bugs, not checker bugs, per #575's original filing):**
- `stdlib/epistemic/ode.sio`: added the missing `ode_system_chem_simple()` preset
  (confirmed via `grep` to not exist anywhere in stdlib before this) — `system_id: 5,
  n_dims: 4`, matching the sibling presets' pattern (`ode_system_exp_decay`,
  `ode_system_pbpk_14`, etc.) and the first call site's `EState` dimensionality.
- `stdlib/epistemic/ode.sio`: `ODESolution` struct + its `n_steps` field needed `pub`
  (same shape as the `Dual` fix in #567's PR) — `kinetics.sio` reads `sol.n_steps`
  cross-module.
- `stdlib/chemistry/kinetics.sio`: converted the bare-module-import qualified-call
  pattern `use chemistry::ontology; ontology::fn(...)` to
  `use chemistry::ontology::*; fn(...)` (Bug C's bare-module-alias variant, same shape
  as the `caputo::`/`epi_plot::` fixes from #567's PR) — 24 call/type-reference sites.

**Result:** `tests/stdlib/chemistry/test_kinetics_core.sio` now passes (rc=0,
`compile: fns=785`). Re-testing the two other tests blocked on kinetics.sio's debt
(`test_pbpk_ontology.sio` from the closed-duplicate #566, and `test_lib_surface.sio`'s
remaining scope from #568) — both **also now pass** once their Bug-A-broken
`chemistry::kinetics::*` references were re-converted to `use chemistry::kinetics::*;`
(previously reverted in both cases specifically because the underlying kinetics.sio
wall made this conversion counterproductive; the wall is now gone).

**Residual, separately-tracked debt (not fixed here, filed as new issues):** kinetics.sio
transitively pulls in `stdlib/chemistry/ontology.sio` (24 of its own pre-existing
errors: 16 "comparison operands must have the same type", 4 E001, 2 "unknown identifier
`[`" — likely Bug E recurring, not yet isolated per-site, 1 tail-type-mismatch, 1
ordered-comparison) and `stdlib/epistemic/ode.sio` (14 of its own: 10 "arithmetic
operands", 4 E001) — both **non-fatal** per the harness's marker-based pass/fail logic
(same tolerance pattern as equilibrium.sio's residual errors in the #567 update above),
so they do not block any test today, but represent the same kind of "never fully
type-checked" content debt kinetics.sio had. Filed separately rather than fixed in this
pass, matching the precedent set for kinetics.sio itself.

Verified against a freshly rebuilt `lean_single` via `scripts/run_sio_test_suite.sh
--filter` for all 3 previously-failing tests plus the full regression set from earlier
this week (21 tests total, all pass). Also verified Madaros: the 3 kinetics-cluster
tests show the same pre-existing "no method named for this type" (E011) Madaros-only
divergence already documented for `equilibrium.sio`/`acids.sio` in the #567 update — not
a new regression, inherited from those two modules.

## Update 2026-07-03 (continued): issue #579 (ontology.sio audit) — one more new bug found, file down to zero errors

Picked up issue #579 (`stdlib/chemistry/ontology.sio` had ~24 pre-existing, non-fatal
errors). All 24 fixed; the file now checks with zero errors. Breakdown:

**16 "comparison operands must have the same type"** — all the same root cause as the
string-literal-argument issue from #575's PR, but for `==` comparisons instead of call
arguments: `mechanism == "big_crn"` (comparing a `&str` parameter against a bare string
literal, which infers as `string` not `&str`) fails. Same workaround: bind the literal
to a local `string` first, compare against its reference (`let lit: string = "big_crn";
mechanism == &lit`). Fixed at every comparison site across `species_to_chebi_iri` (12
distinct literals) and `attach_chebi_to_crn_output` (4 literals, reused from the first
function's set since both take a `mechanism: &str` parameter with the same literal
vocabulary).

**4 E001 + 2 "unknown identifier `[`" + 1 arity mismatch** — the already-known Bug E
(inline array/string literal as a call argument) recurring in this file's own test
functions (`species_to_chebi_iri(5, "big_crn")`, `attach_chebi_to_crn_output(&[...],
&[...], "big_crn")`) — same fix, bind to locals first.

**3 "effect not declared in function signature"** — `attach_chebi_to_crn_output` (array
mutation via `iris[i] = ...`, needed `with Mut`, never declared) and its 2 transitive
callers needed the same effect propagated up their own signatures — ordinary
test-authoring omissions, same shape as issue #571's fix, not checker bugs.

**Bug G — a struct type in the LAST position of a 3-tuple confuses lean_single's type
inference for the tuple's *middle* element.** Isolated with a minimal repro outside
stdlib:
```sio
struct Foo { id: i64 }
fn f(val: f64, u: f64) -> (f64, f64, Foo) { (val, u, Foo { id: 1 }) }
fn main() -> bool {
    let (v, u, iri) = f(0.18, 0.02);
    u > 0.0   // error: ordered comparison requires matching numeric operands
}
```
Isolated further: `(f64, f64, f64)` (all-numeric) works fine; `(Foo, f64, f64)` (struct
*first*) works fine; only a struct in position 2-of-2 (i.e. last, with something before
it) corrupts the *middle* f64's inferred type — `let u2: f64 = u;` in the repro above
reports `expected f64, got i64`, i.e. the checker doesn't just lose the type, it assigns
the wrong concrete one. Access via `.1` field syntax instead of destructuring hits the
identical error, so this isn't specific to `let (a, b, c) = ...` pattern-matching syntax
— it's the tuple type's own internal element-type bookkeeping. `stdlib/chemistry/
ontology.sio`'s `epistemic_water_conc(val, u) -> (f64, f64, IRI)` matched this exact
shape (struct last, 2 f64s before it) and had exactly 2 real call sites (one in this
file's own test, one in `stdlib/chemistry/kinetics.sio`). Fixed by reordering the
return tuple to `(IRI, f64, f64)` (struct first) and updating both call sites'
destructuring order to match — this is a public-API change to `epistemic_water_conc`,
justified as a workaround for a confirmed checker bug with a fully enumerated, tiny
blast radius (grepped for all callers before changing).

**Result:** `stdlib/chemistry/ontology.sio` now checks with zero errors (previously 24).
`stdlib/chemistry/kinetics.sio`'s own remaining error count also dropped by one (the
`epistemic_water_conc` tail-type-mismatch this file had been carrying) — its residual
14 errors (separately tracked, non-fatal, same file this whole dispatch update-chain has
been about) are otherwise unchanged; `stdlib/epistemic/ode.sio`'s 14 (issue #580) are
untouched, out of this issue's scope.

Verified against a freshly rebuilt `lean_single`: all 3 kinetics-cluster tests plus the
other 18 tests fixed this week still pass (21 total, zero regressions). Madaros shows
the same pre-existing E011 divergence already documented above — not new.

## Update 2026-07-03 (continued): issue #580 (ode.sio audit) — one more severe new bug found, this week's audit chain closed

Picked up issue #580 (`stdlib/epistemic/ode.sio` had ~14 pre-existing, non-fatal
errors — the last item in this whole audit chain). All fixed; the file now checks with
zero real errors (only the expected "no main" library-file artifact).

**5 "effect not declared in function signature"** — `estate_set` (mutates `s.values`/
`s.variances`, needed `Mut`, never declared), `ode_params_set` (same shape), and
`print_uncertainty_budget` (needed `Mut` alongside its existing `with IO` — the
"missing" effect wasn't obvious from the error text alone; found by testing `Mut`
directly, matching this week's most common gap). Ordinary test-authoring omissions
(#571's shape), not checker bugs.

**Bug H — a `*ref = ...` dereference-assignment lexically following an earlier
function-call statement in the same function corrupts lean_single's type inference for
that assignment, even for a pure literal write with no arithmetic.** By far the most
severe bug found this week — it isn't specific to arithmetic, tuples, qualified paths,
or literals; it's about control flow shape. Isolated progressively with minimal repros
outside stdlib:
```sio
fn noop() -> i64 { 0 }
fn write_only(n: &!i64) with Mut {
    noop()
    *n = 999          // error: arithmetic operands must have matching numeric types
}                      // (a LITERAL write, no arithmetic operator anywhere in source)
```
`*n = 999` as the function's *first* statement (no preceding call) works fine. Reading
`*n` into a local *before* any call, then using the local, works fine. A plain `var`
counter (not a dereferenced reference) increments correctly after any number of
preceding calls. Only a dereference EXPRESSION (`*ref`, read or write) appearing
textually after an earlier call statement, in the same function, is corrupted — and the
call itself needn't touch the reference in question, or take any arguments at all
(`noop()` takes none). The fix: extract the dereference-assignment into its own tiny
function, called as an ordinary statement from the outer function:
```sio
fn bump_n(n: &!i64) with Mut { *n = *n + 1 }     // safe: this IS its first statement
fn outer(n: &!i64) with Mut {
    noop()
    bump_n(n)        // fine — the deref lives inside bump_n, not here
}
```
Applied this exact pattern to `stdlib/epistemic/ode.sio`'s `rk4_step` and `rk45_step`,
which call `ode_rhs_dispatch(...)` 4-6 times each, incrementing a `n_evals: &!i64`
counter inline after every call (`*n_evals = *n_evals + 1`, 10 total sites across both
functions) — extracted to a shared `bump_n_evals(n: &!i64)` helper, called instead of
the inline dereference at each site.

**Why this didn't already break `compute_jacobian_state`/`compute_jacobian_params`**,
which use the identical `*n_evals = *n_evals + 2` pattern and share the same parameter
name: their increment sites are the *first* dereference of `n_evals` reached inside a
nested loop body that itself contains the preceding `ode_rhs_dispatch`-family calls —
still technically "after a call", so this is likely a difference in exactly which
prior-call/prior-deref sequences trigger the corruption rather than a clean rule; not
fully characterized. Left untouched since they don't currently error — flagging here so
a future compiler-side fix attempt knows this pair exists as a "control" case that
currently passes.

**Result:** `stdlib/epistemic/ode.sio` now checks with zero errors (previously 14, plus
5 more effect-annotation gaps found once those were fixed first and the arithmetic
errors could be isolated cleanly). `stdlib/chemistry/kinetics.sio`'s own residual error
count (tracked separately, non-fatal, unrelated to this issue) is unchanged by this fix.

This closes out the full audit chain opened by issue #575: every file transitively
reachable from `tests/stdlib/chemistry/test_kinetics_core.sio` now checks clean, and
`Full Test Suite` CI reached 0 failures for the first time this week (1310 pass) as of
this issue's merge — `Contracts` (previously failing since before this week's work
began, via an unrelated specialized ontology-validation driver) also went green,
apparently as a side effect of `ontology.sio`'s cleanup in the #579 update above.

Verified against a freshly rebuilt `lean_single`: all 21 tests fixed this week still
pass, zero regressions. Madaros shows the same pre-existing E011 divergence already
documented above — not new.
