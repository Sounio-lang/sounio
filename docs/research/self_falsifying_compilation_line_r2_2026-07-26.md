<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r2-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r2-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R2 — verdict-token binding: the compiler now checks the proposition, not just the exit code

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `TOKEN_BINDING_IMPLEMENTED__CATCHES_DRIFT_NOT_MISINTERPRETATION`
**Parents:** `self_falsifying_compilation_line_2026-07-26.md` (R0: the drift / shared-misinterpretation distinction), `self_falsifying_compilation_line_r1_2026-07-26.md` (R1: corpus binding, module-closure wall)
**Harness:** `scripts/research/self_falsifying_compilation_line_r2_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r2_gate.sh`
**Compiler change:** `self-hosted/compiler/claim_executor.sio`

---

## 0. What this rung did

R0's finding was that exit-code gating binds a build artifact to a
**computation**, not to a **proposition**: a gate exiting 0 says the check ran,
not that the check establishes what the claim asserts. R2 closes the part of
that gap which is closeable.

A claim may now declare `verdict_token`. The compiler captures the gate's
stdout, extracts the token the gate actually emitted, and **refuses to emit
code** if the two disagree — or if the gate emitted none at all.

Measured on a compiler built from this exact executor source:

| Probe | Gate behaviour | Result |
|---|---|---|
| `self_falsifying_token_pass.sio` | exits 0, emits `TOKEN_ALPHA` (declared) | `CLAIM_PASS … verdict_token=TOKEN_ALPHA`, exit 0, **ELF emitted** |
| `self_falsifying_token_drift.sio` | **exits 0**, emits `TOKEN_BETA` | `CLAIM_TOKEN_MISMATCH … declared=TOKEN_ALPHA emitted=TOKEN_BETA`, exit 1, **no ELF** |
| `self_falsifying_token_absent.sio` | **exits 0**, emits no token | `CLAIM_TOKEN_ABSENT … declared=TOKEN_ALPHA`, exit 1, **no ELF** |
| `self_falsifying_claims_pass_only.sio` | no `verdict_token` declared | unchanged: `VERIFY_CLAIMS_OK`, ELF emitted |

Every probe gate **exits 0**. That is the point: if any exited non-zero it would
be caught by the pre-existing mechanism and would prove nothing about token
binding.

---

## 1. Results

| Clause | Result |
|---|---|
| `T1_EXECUTOR_SURFACE` | `verdict_token` field, capture, extraction, both outcomes present and counted as failures |
| `T2_FIXTURES` | 3 probes discriminate match / drift / absent, all exiting 0 |
| `T3_NO_SHELL_STRING` | capture is `open`+`dup2` with fixed argv — no command interpolation |
| `T4_REACH` | **25/270** specs (9.3 %) declare a parseable verdict token |
| `T5_BEHAVIOUR_RECEIPT` | D1–D4 observed on a compiler built from this executor's exact hash |
| `D1_MATCH_PASSES` / `D2_DRIFT_BLOCKS` / `D3_ABSENT_BLOCKS` / `D4_BACKWARD_COMPAT` | all PASS |

Verdict: `SELF_FALSIFYING_R2_VERDICT TOKEN_BINDING_IMPLEMENTED__CATCHES_DRIFT_NOT_MISINTERPRETATION`.

### 1.1 Why `T5_BEHAVIOUR_RECEIPT` exists

Because this rung's own contract overclaimed, in exactly the way the line
studies.

`T1_EXECUTOR_SURFACE` passed — the field, the capture, the extraction and both
outcomes were all present in the source — while the compiler built from that
source **SIGSEGV'd on every claim**, including claims using none of the new
machinery. A contract that certifies "implemented" from source text alone is
checking the computation, not the proposition.

So the contract now refuses to emit `TOKEN_BINDING_IMPLEMENTED` without a
receipt recording that D1–D4 were actually observed, bound to the executor's
content hash. Edit the executor and the receipt goes stale. No run, no claim.
This is the R0 §3 lesson applied to the tooling that reports on R0 §3.

---

## 2. What it catches, and what it provably cannot

**Catches — drift.** A check and a claim that disagree. The `drift` probe is
exactly this: the computation still succeeds, and the proposition it reports is
no longer the one the claim declares. Under exit-code gating this is invisible.

**Does not catch — shared misinterpretation.** If claim and check were authored
together from the same misunderstanding, they agree, and token binding sees
agreement. R0 §3's proposition says why no compile-time procedure whose only
evidence is the claim's own check can do better.

**This is not a hedge, it is the measured result.** R0 §2 audited three real
self-corrections in this repository: at the commit where each claim was false,
the spec's token and the harness's token **agreed**. Token binding would have
caught **none of the three**. What R2 adds is a guard against a failure mode
that has not yet damaged this corpus, while the one that has remains out of
reach. R3 (executable falsifiers) is the next attempt at the harder half.

---

## 3. Three implementation hazards, recorded because each looked fine

The mechanism took four compiler builds. Each failure was silent in a different
way, and none was visible from the source.

1. **Runtime string building in the executor SIGSEGVs.** The capture path was
   first built as `/tmp/sounio_claim_gate_<pid>.out` via `getpid` plus
   digit-wise concatenation. The compiler segfaulted before the claim loop
   printed anything — *including* for claims with no `verdict_token`. Replaced
   with a fixed path (see §5).
2. **Assigning back into `outcome` does not stick.** The natural code —
   `outcome = <token verdict>` inside the `if outcome == CLAIM_GATE_PASS`
   block — had no effect: capture and extraction worked, the emitted token
   printed correctly, and the build passed anyway. Hoisting the decision into a
   function did **not** help, so nesting was not the cause. Writing into a
   fresh `decided` variable and branching on *that*, never writing back into
   `outcome`, is what worked. An intermediate version that assigned
   `outcome = decided` was worse than either: it made even a plainly passing
   gate with no token report `CLAIM_FAIL`.
3. **Capture and extraction were right the whole time.** Worth stating, because
   for two builds the visible symptom ("drifted token still passes") pointed at
   the parsing, and the diagnostic print showed the extraction had been correct
   from the first build. The bug was always downstream.

**What is *not* claimed:** the two new outcome codes are written as integer
literals rather than `pub let` module constants. They were module constants
during the failing builds and were inlined in the same pass as the real fix, so
no build isolates them as a cause. They stay literals because that form is
proven working here — not because module constants are known bad. Saying more
would be the overclaim this line exists to study.

---

## 4. Reach: the guard covers a small minority of the corpus

`25 of 270` research specs (**9.3 %**) declare a machine-parseable verdict
token — this document is one of them. `190` carry no `**Status:**` line at all and need the convention
introduced before anything can be bound to them.

The verdict token above deliberately does **not** embed that count. Counts move;
a token carrying one drifts without the claim changing, which is the sub-token
failure mode R0 §1 recorded and R0 §5 was amended to avoid. Reach is reported as
a metric, by `T4_REACH`, and re-derived rather than quoted.

---

## 5. Limitations

- **The capture path is fixed** (`/tmp/sounio_claim_gate_capture.out`), not
  per-process. Two `--verify-claims` compiles running concurrently in the same
  container would clobber each other's captures. This workspace does run several
  agents at once, so the hazard is real, not theoretical; the per-process
  version segfaulted (§3.1) and correctness of the common case was preferred to
  a broken compiler. Unresolved.
- **Extraction follows one convention**: the word after the **last**
  `_VERDICT ` in the gate's output. A gate that reports its verdict any other
  way is treated as emitting none, which fails closed (`CLAIM_TOKEN_ABSENT`)
  rather than passing silently.
- **Only the main source file's claims run** — R1's module-closure wall is
  untouched by this rung.
- **`verdict_token` is opt-in.** Claims without it behave exactly as before
  (`D4_BACKWARD_COMPAT`).

---

## 6. What this is NOT

- **Not a fix for the failure mode that actually damaged this corpus.** See §2.
- **Not corpus coverage.** 9.3 % could be bound; none of the 15 claims R1 bound
  declares a token yet.
- **Not a security regression.** `T3_NO_SHELL_STRING` checks that the capture
  did not trade the mechanism's fixed-argv property for a shell redirect.

---

## 7. Reproduce

```bash
# static clauses only — T5 will FAIL until the compile arm has run, on purpose
python3 scripts/research/self_falsifying_compilation_line_r2_contract.py

# full: runs the four probes, writes the receipt, then evaluates the contract
SFCL_R2_RUN_COMPILE=1 bash scripts/ci/self_falsifying_compilation_line_r2_gate.sh
# expect: D1..D4 PASS, T1..T5 PASS, SELF_FALSIFYING_COMPILATION_LINE_R2_GATE_OK
```

Directly, with the token-binding compiler:

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib && ulimit -s unlimited
RAW=artifacts/self-hosted/madaros-token-binding

# a gate that exits 0 but states a different verdict: build refused
$RAW run scripts/ci/fixtures/self_falsifying_token_drift.sio -o /tmp/d.elf --verify-claims
```

Rebuild the compiler with
`bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros-token-binding`
(serialised through the shared build lock; it can queue for a long time behind
other agents).

---

## 8. AI disclosure

Compiler change, fixtures, harness, gate and spec drafted under human direction
(2026-07-26). All behavioural results come from real runs of a compiler built
from the recorded executor hash and are re-measurable via the gate's compile
arm. No clinical content. GAIDeT-ICMJE 2025.
