<!-- docs:meta
topic_id: repo.docs.audit.user-frontier-wsb2-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.user-frontier-wsb2-2026-08-19
-->

# WS-B2 — user frontier (2026-08-19)

Status: measured. Method is above the numbers. Engine: canonical Madaros (`bin/souc` default). No `souc main.sio`. No lean_single unless a probe names it.
Engine: canonical Madaros (`bin/souc` default). No `souc main.sio`. No lean_single unless a probe names it.

## Semantic declaration

A compiler is a binary that typechecks the tree we wrote.
A language is a front door a stranger can walk: they read a public guide, write a new file, invoke the documented commands, and either get a running program or a message that names the next step.

This receipt measures the second thing. It does not measure whether we have a compiler.

`examples/` is our corpus. We wrote those files. A count of `examples/` that check under Madaros is a statement about our tree, not about external users. **Claims-Forbidden:** "users can write Sounio because N examples compile."

The proxy for "someone who is not us" is a program transcribed from the public front door (README Quick taste, README Get started, `docs/guide/LLM_PROGRAMMING_GUIDE.md` §1), plus the documented CLI sequence on a file that did not exist in the tree before this measurement.

Green without a diagnostic is not a measurement. `souc check` that prints `typecheck: failed` and exits 0 is empty-green — the same class this lane already named on CI gates. The instrument records both `rc` and whether the log contains a failure token.

## Method (read before the counts)

### Q1 — new program from the guide

- Front door, in this order: README Get started (clone + two exports + commands), README Quick taste (Uncertainty snippet as a complete file), LLM Programming Guide §1 Hello (exact).
- Not used as the "new program": `examples/hello.sio` (we wrote it). It is only the Get-started path probe.
- Environment **documented**: `SOUC=$(pwd)/bin/souc`, `SOUNIO_STDLIB_PATH=$(pwd)/stdlib`, `MADAROS_STACK_KB=524288`, `ulimit -s 1048576`, inherited `SOUC_BIN` / `SOUNIO_SOUC_ENGINE` unset. This is what README tells a clone to do.
- Environment **forgot-stdlib**: same, `SOUNIO_STDLIB_PATH` unset. A stranger who copied only the first export.
- For each source × environment: `souc check`, then `souc run` only if check is `pass-measured`.
- First failure = first diagnostic line or first dishonest exit, whichever comes first.
- Next-action score on that message: `NEXT` (names a command, flag, path, or concrete fix), `ERROR_ONLY` (names the defect, not a next step), `DISHONEST` (failure token with `rc=0`, or success token with `rc≠0`).

### Q2 — `examples/` under canonical Madaros

- Universe: `git ls-files 'examples/**/*.sio'` on the measurement SHA.
- Instrument: `bin/souc check` only. Not `run`. Not a 468-gate count.
- Cap: 20 s per file (`timeout` SIGTERM). A cap is not a measurement.
- Pass = `rc=0` and no failure token (`typecheck: failed`, `error[`, `error:`) in the log.
- Empty-green = `rc=0` and a failure token.
- Fail families (first match in the log): `parse`, `effects` (E035), `unresolved` (E137 / undeclared), `import` (unreadable import), `no_main`, `type`, `limit` (E182/E218/capacity), `timeout`, `other`.
- Split reported: files with a `^fn main` line vs the rest. A library that dies `no_main` is that family, not a user-frontier claim.
- This number is about `examples/`. It is not about strangers.

### Q3 — entry path

- Sequence on a new file written from Guide §1: `souc --version`, `souc check <file>`, `souc run <file>`.
- Extra dishonest-exit probes: `souc` with no args, `souc check` missing file, `souc check` on a file with a known syntax error (`let x = 5;`), `souc init` with no name (rc captured without a pipe).
- README promises `$SOUC --version` prints `souc 1.0.0-beta.6`. The measured string is recorded; a mismatch is a front-door lie, not a compiler crash.
- `souc init <name>` in `/tmp` then `souc check` / `souc run` in that project, as README Get started lists.

## Claims-Forbidden

- Extrapolating Q2 onto "external users".
- Quoting the May 2026 PL adoption audit as this measurement. That audit is a prior; this run is live.
- Treating `rc=0` as pass without reading the log.
- Wiring any of this into `ci.yml` in this lane.

## Numbers

Measurement SHA: `6f23dfe1dac1` (`origin/main` at the start of this run). Engine: `bin/souc` → Madaros v0.80.0. Logs: `/tmp/user-frontier-probes/`. Probes were written under `/tmp`, not into `examples/`.

### Q1 — new program from the guide

The Guide §1 hello, transcribed exactly, **checks and runs**. `souc run` prints `Hello, Sounio!`. The same file still checks when `SOUNIO_STDLIB_PATH` is unset.

The README Quick taste snippet (the first program a scientist copies) **does not parse**. Isolated constructs:

| Construct | `souc check` |
|---|---|
| `Knowledge[f64] = Knowledge(15.0, ε=0.92, prov=…)` | OK |
| `a * (b / c)` on those values | OK |
| `let conf = x.ε` | parse fail, rc=1 |

**What fails first:** field access `.ε`. The message is `run_check_mode: module failed to parse` plus a note that the closure walk stopped. It does not name the line, the character, or `.epsilon`. Next-action: `ERROR_ONLY`.

That is the stranger-proxy result. It is not a statement about `examples/`.

### Q2 — `examples/` under canonical Madaros

Universe: 571 tracked `examples/**/*.sio`. Instrument: `souc check` only, 20 s cap, 3 workers. Pass = rc=0 and no failure token. Empty-green = 0. First-pass family regex treated `E00x` as parse; reds were reclassified from the first diagnostic (`parse` = parse-error / `failed to parse` only).

| family | n | meaning |
|---|---:|---|
| pass | 245 | check measured OK |
| unresolved | 168 | first diagnostic E137 / undeclared |
| parse | 100 | parse error or `failed to parse` |
| type | 32 | first diagnostic other `error[E0…]` (E004/E019 dominate the spot-check) |
| effects | 25 | first diagnostic E035 |
| crash | 1 | `examples/render/cube_wireframe.sio` rc=139 during check |

Of 555 files with `^fn main`, 231 pass. Of 16 without, 14 pass and 2 parse-fail.

Parse messages in this tree look like `expected token at line 7:31 expected=131 actual=-7889…` — token ids, not names. E035 names the missing effect (`Div`, `Mut`). E137 says `use of undeclared variable` with a span, not the identifier.

**Claims-Forbidden:** this table is about files we wrote. It does not say what a stranger can compile.

### Q3 — entry path

Full rows: `docs/audit/USER_FRONTIER_ENTRY_PATH_2026-08-19.tsv`.

Steps where the message does **not** say what to do next, or says the wrong next step:

1. README Quick taste / `.ε` — parse fail, no token, no fix (Q1).
2. `souc check` on a missing `*.sio` — `madaros check requires <source.sio>` rather than "file does not exist".
3. Bare `souc` — madaros usage. The stranger typed `souc`. `souc --help` is the matching surface; bare invocation is not.
4. README Get started promises `$SOUC --version` prints `souc 1.0.0-beta.6`. Measured: `Madaros v0.80.0`. Not a crash; a front-door lie.

Steps that do name a next action: `souc --help`, `souc info`, `souc init` without a name (`requires <name>`), E001's `use as` note, Guide hello check/run, Get-started `examples/hello.sio`, `souc init <dir>` then check/run (prints `42`).

The Guide's "no semicolons" rule is not enforced: `let x = 5;` checks and runs. That is a doc/compiler mismatch, not a first failure.

### What this is not

- Not a 468-gate count.
- Not "almost half the language is empty-green". Q2 empty-green is 0; Q2 red is 326 of 571 of **our** examples.
- Not evidence that an external user can or cannot grow a multi-file project beyond `souc init`. That path was one generated project in `/tmp`.

