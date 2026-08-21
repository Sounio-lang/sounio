---
name: sounio-compiler
description: Use when changing, measuring, or debugging the Sounio compiler (self-hosted/, bin/souc, Madaros, lean_single), or when building a CI gate or witness about compiler behaviour. Encodes the failure modes that have cost this project whole sessions.
---

# Working on the Sounio compiler

Every rule below was written after a measured failure, most of them more than
once. They are ordered by how much time they have cost.

## 1. Validate the instrument before you believe it

The dominant failure in this repository is not bad reasoning. It is **good
reasoning on top of an instrument that was never checked**. Before any claim:

- **State the refutation criterion first.** What result would prove you wrong?
  If nothing would, you are not measuring.
- **Run the negative control before the positive one.** Sabotage the thing you
  are testing and confirm your probe goes red. A probe that has never failed has
  not measured anything.
- **Check the control is not pre-satisfied.** Three times this session a control
  "passed" for an unrelated reason: an artifact already existed from an earlier
  run; a `sed` sabotage silently matched nothing; a compile-fail witness "passed
  its control" because a compile-fail witness never builds, sabotaged or not.

## 2. Probes that return green and prove nothing

| probe | why it is void |
|---|---|
| `fn f(x: T)` with no caller | passes for any `T`, invented ones included |
| `with <invented effect>` | accepted on both engines, propagates nothing |
| an invented field beside a valid one in a `Knowledge` literal | one recognised name (`value`) licenses all the others |
| a top-level block with invented syntax | `lean_single` skips it in silence and still emits an ELF |
| `./bin/souc check` after editing `self-hosted/` | **`bin/souc` is PREBUILT.** Source edits do not change it. Build first. |
| a parameter-position type "arriving as its own kind" | it arrives as `TypeNamed`. Adding match arms looks like the fix and is not. |
| `grep -c` under `set -euo pipefail` | prints 0 **and** returns 1 on no match; `\|\| echo 0` then appends a second line and the arithmetic test never fires |
| `gh run view --log` | can come back empty where `gh api .../jobs/<id>/logs` has the answer |
| `cut -c1-150` on an error line | truncates exactly the part that identifies the error |
| a cancelled CI run | reads as a current result; check `status`, not just `conclusion` |
| a PR with conflicts | gets **no CI run at all** — the suite looks absent, not red |

## 3. Two engines, and they disagree

`bin/souc` is **Madaros** (default). `SOUNIO_SOUC_ENGINE=lean_single` is the
bootstrap seed, and **it is what CI actually runs**. They diverge on accept /
reject, on `~`, on ε polarity, on printed variance. **Never quote a green without
naming the engine.** `lean_single` mutes 35+ diagnostic classes for `from_import`
functions — build exit 0 is not typecheck clean.

## 4. Building

```bash
# correct
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
# better -- off the pod entirely
bash scripts/dev/souc-build-remote.sh
```

- **Never** wrap `build_modular_madaros.sh` in `souc-build-lock.sh`. It takes the
  lock itself, twice. The deadlock is **silent** and has blocked other agents for
  half an hour.
- A 9–15 minute log with no `Madaros ready` is still **running**, not hung.
- `souc-build-remote.sh` streams the **live tree** into `tar`. Switching branches
  during a run poisons it, and the poisoned read **passes** instead of failing.
  Use a second worktree for concurrent work.
- Raise the soft stack to 524288. `madaros run` needs 4× the stack of
  compile-then-execute; CI reports that crash as a 30s timeout.

## 5. Where things run

**Do not run heavy work on the pod.** The k8s liveness probe recycles it under
CPU saturation — measured twice. Use Slurm via `souc-build-remote.sh`, or the
build lock for anything local and heavy. Cheap `souc check <file>` is fine.

## 6. Measure the blast radius before you change a table

Before recognising a name, adding an effect id, or tightening a rule: **count the
sites that will start being refused, and count them in `self-hosted/` separately.**
A rule the compiler's own source violates is a self-inflicted cascade that kills
the fixed point. This is a `grep` and takes seconds; skipping it has cost a full
build cycle more than once.

## 7. Gates and ratchets

- A gate that no workflow names **is not a gate**. Check: `grep -rn <gate> .github/`.
  Measured 2026-08-20: `SOUNIO_WITNESS_GLOB` was named in no workflow at all, so
  the witness gate had never run in CI.
- Gate artifacts go in `artifacts/gates/`, never loose in `scripts/ci/` — a gate
  that dirties the tree fails the worktree governance gate **in the same job**.
- Emit `status=pass` plus `metrics {total,passed,failed,not_run}`.
- **Ship the negative control inside the workflow**, not just in your session, so
  the gate cannot rot into a green that proves nothing.
- A ratchet freezes a measured count and fails the next increment. It may only
  shrink. Record tolerated baselines with the reason.
- Adding a gate script without naming it in a workflow raises the unnamed-gate
  ratchet and blocks every open PR.

## 8. Cross-check with another agent — this is not optional

**Founder ruling 2026-08-21: compiler work is critical, and every substantive
finding is confirmed by a second agent before it is reported or landed.**

It has repeatedly paid for itself in both directions:

- A consulting agent refuted two of my hypotheses cleanly and supplied the one
  that held.
- The same agent then **gave up its own claim** when its measurement came back
  vacuous — the corpus it proposed (`main.sio`) is saturated, so inferred and
  declared coincide everywhere by construction.
- And an agent's confident diagnosis has also been wrong. Do not take it at face
  value; re-measure the load-bearing claim yourself. I have twice repeated a
  claim from a lane and had to correct it.

Give the consulted agent the measurements, the framing you doubt, and explicit
licence to contradict you. Ask for a position, not a survey.

## 9. Reporting

- Name the command, the path, the engine, and the exit code. No adjectives.
- **Correct your own inflation out loud.** A wrong number that changes a decision
  is worse than a gap. Twice today a claim of mine was directionally right and
  mechanically wrong — e.g. "the compiler infers `Mod` from the body": what it
  actually does is propagate the callee's *declared* effect. Different fact,
  different plan.
- Before writing "nobody has this", name the languages that come closest and say
  what each ships. `i<N>` parametric widths are Zig, not novel.

## 10. Commits and PRs

- Atomic: one logical change per commit.
- **No `Co-Authored-By` trailer** (founder directive 2026-06-30).
- Spec, audits, PR bodies and commit messages in **English (EN-UK)**.
- Backticks inside `git commit -m "..."` are command substitution; `<word>` is a
  redirect. Both have corrupted messages here.
- A PR gone `DIRTY`: **rebuild on fresh main, do not rebase.** The registry
  resync commits make later rebases fail on their own replay. When rebuilding,
  never take a shared file (`ci.yml`) wholesale from the branch — reapply only
  the lines the branch added, or you delete what landed meanwhile.
