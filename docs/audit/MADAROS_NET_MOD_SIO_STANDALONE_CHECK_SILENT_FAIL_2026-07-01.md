<!-- docs:meta
topic_id: repo.docs.audit.madaros-net-mod-sio-standalone-check-silent-fail-2026-07-01
authority: repo_only
audience: users
last_validated: 2026-07-01
validated_by: Claude
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
