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
