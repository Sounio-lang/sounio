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
| `with <invented effect>` | accepted **by Madaros** (`check: OK`) and propagates nothing. **Not both engines** — `lean_single` refuses it: `error: effect not declared in function signature`, rc=1. Measured 2026-08-21 with `with Zorblex`. lean_single is the stricter engine here, so a probe that looks void under Madaros may be caught by the seed. |
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
bootstrap seed, and **it is what CI actually runs** — the Full Test Suite job
prints `SOUNIO_TEST_SOUC_BIN: /tmp/souc-stage2`, and `selfhost_host_gate.sh`
builds that stage2 from `lean_single.sio`. They diverge on accept /
reject, on `~`, on ε polarity, on printed variance. **Never quote a green without
naming the engine.** `lean_single` mutes diagnostic classes for `from_import`
functions — build exit 0 is not typecheck clean. (The "35+" figure comes from an
earlier session's note, not from a measurement made here; treat the direction as
solid and the number as needing a re-count before you quote it.)

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
- Raise the soft stack to 524288. `madaros run` needs more stack than
  compile-then-execute, and CI has reported that crash as a timeout rather than a
  segfault. (The "4×" and "30s" come from an earlier session's note; the practice
  — raise the stack — is what matters, the figures need re-measuring.)

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

**But the grep is a floor, not the radius.** A literal `with X` count finds
*declaration sites*; effects also reach a file transitively through call chains,
and those sites carry no text to grep for. The only honest radius is a corpus run
under the engine you changed. Measured 2026-08-21: lifting the Mod hold showed
zero new failures in CI's 3016-test suite — because that suite runs lean_single,
which never saw the change — while `--gate corpus` under a freshly built Madaros
came back **red**, with the gate saying so itself: *"These pass on the baseline
and fail on this change. CI's full-test-suite runs lean_single and will not show
this."*

## 7. Gates and ratchets

- A gate that no workflow names **is not a gate**. Check it with
  `bash scripts/ci/gate_workflow_reference_ratchet.sh`, **not** with
  `grep -rn <gate> .github/` — that grep counts a mention in a YAML comment as
  wiring, and this skill shipped that advice and then failed its own example on
  it: after adding a workflow whose *comment* names `SOUNIO_WITNESS_GLOB`, the
  grep went green while the gate keyed on it was still run by nothing.
  The ratchet had the same hole and it is now closed — it strips comments before
  extracting. Verified both ways: a comment no longer moves the count, a real
  `run:` line still does.
- Not everything belongs in GitHub CI. `souc-build-remote.sh --gate witness`
  needs a freshly built Madaros and runs on Slurm; what is CI-reachable is the
  ratchet over the *declared witness list*, not the witness run itself. Say which
  of the two you mean.
- Gate artifacts go in `artifacts/gates/`, never loose in `scripts/ci/` — a gate
  that dirties the tree fails the worktree governance gate **in the same job**.
- Emit `status=pass` plus `metrics {total,passed,failed,not_run}`.
- **Ship the negative control inside the workflow**, not just in your session, so
  the gate cannot rot into a green that proves nothing.
- A ratchet freezes a measured count and fails the next increment. It may only
  shrink. Record tolerated baselines with the reason.
- Adding a gate script without naming it in a workflow raises the unnamed-gate
  ratchet and blocks every open PR.

## 7b. A gate can lie about scale before it lies about semantics

Two failure modes of the harness itself, both measured 2026-08-21, both of
which produced a confident and wrong story before anyone questioned the tool:

- **Truncated reporting.** `souc-build-remote.sh` piped the corpus gate through
  `tail -25`. A regression list arrived cut, with nothing saying it had been cut,
  and the visible slice ran alphabetically from `uncertain_` to `zero_`. Reading
  the count off that slice gives 21 where the number is 1510.
- **Oversubscription.** The same script ran the corpus `$(nproc)`-way parallel.
  On a 128-core node that reported **1510 regressions against a 284-entry
  baseline** — more than half the corpus — and a sampled "regressed" file built
  and ran clean (`rc=0`) when compiled by itself from the same tree. Madaros
  lowering holds ~900 KB RSS per function with no reclamation, so wide
  parallelism becomes mass false failure.

The tell in the second case was there before the diagnosis: the run with
`SOUNIO_EFFECT_INFER=1` returned **exactly** the same 1510. A number that does
not move when you turn a knob is the signature of an inert knob or a wrong
cause — never of a confirmed finding. Check the count against the baseline size
and spot-check one item alone before believing any mass regression.

## 7c. Four instruments lied in one day, and none of them was broken

Every one produced a confident, round number. None threw an error. The failure
was always the same: the tool measured less than it claimed to.

| what happened | what it looked like |
|---|---|
| the corpus gate ran the raw ELF without raising the stack | 1510 semantic regressions, identical across three configurations |
| `gh api .../logs` returned an **empty** file | `grep -c 'Segmentation fault'` said **0**, read as "no crashes" |
| a CI job hit the 60-minute timeout | `gh pr checks` printed **fail**, read as a regression |
| the witness gate never compared `//@ expect-stdout:` | a test that exits 0 and prints the wrong number **passed** |

Three checks that would have caught all four, and cost seconds:

- **Does the number move when you turn a knob?** A count *identical* across a
  parallelism change, a flag change and a branch change is a deterministic
  fault, not a finding.
- **Is the file you just parsed non-empty?** `grep -c` on an empty file returns
  0 and exits 0. Check the line count before believing the count.
- **Is the run `completed` or `cancelled`?** Read `status`, not only
  `conclusion`. A timeout and a failure print the same word downstream.

## 7d. Pin the base before you believe a comparison

A control taken this morning is not a control this afternoon. Seven PRs landed
between one corpus run and the next, and the earlier number was used all day as
if the base had held.

Three ways the base moved without complaint, all measured 2026-08-21, all in
one session:

- **A stale `origin/main` in a second worktree.** The fetch had been run in the
  *other* worktree.
- **A `git checkout` that never ran**, killed by a `pkill` earlier in the same
  command chain. The next `git log` reported the old branch, confidently, as
  though it had succeeded.
- **A dirty worktree refusing the switch.** `git checkout` printed *"Please
  commit your changes"* and aborted; the verification line that followed read
  the unchanged tree and reported it as the new base.

After any checkout, print `git rev-parse --short HEAD`, the branch name, **and
one content fact you expect to differ** — a grep count of the thing under test.
The SHA alone does not tell you the file you care about changed. This paragraph
was itself written on the third such failure of the session, one command after
being drafted.

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
