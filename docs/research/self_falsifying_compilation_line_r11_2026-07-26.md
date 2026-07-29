<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r11-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r11-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R11 — widening the probe: still no new corroboration, and five hazards with one answer

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `WIDER_PROBE__NO_NEW_PREEXISTING_CORROBORATION`
**Parents:** `self_falsifying_compilation_line_r10_2026-07-26.md` (the procedure, and its one-family coverage), `self_falsifying_compilation_line_r8_2026-07-26.md` (the kernel/wrapper collapse this rung runs into again)
**Harness:** `scripts/research/self_falsifying_compilation_line_r11_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r11_gate.sh`

---

## 0. What R10 left, and what happened when it was picked up

R10's negative — *no latent corroboration beyond `cds`/`cd_sigma`* — covered one
signature family: functions taking 2–3 integers and returning a scalar. Every
function it accepted computed the same thing. R11 widens the probe to array-,
set-, dict- and float-valued kernels.

The widening worked, found nothing new, and **cost five separate failures — each
a hazard this technique has to defend against, none of them obvious in advance,
and all five with the same root cause and the same one-line answer (§3).**

---

## 1. Result

| | R10 | R11 |
|---|---:|---:|
| probeable functions | 31 | **35** |
| distinct behaviour classes | 1 | **4** |
| pre-existing corroborations | 1 | **1** (new: **0**) |
| corroborations introduced by this line's own oracle | — | 0 |

Verdict: `SELF_FALSIFYING_R11_VERDICT WIDER_PROBE__NO_NEW_PREEXISTING_CORROBORATION`.

Self-reference is discounted rather than hidden: this line's harnesses now live
in `scripts/research/` and carry a deliberately independent Cayley–Dickson
oracle. A corroboration between that oracle and a corpus kernel would be real
evidence, but evidence **this work introduced**, not evidence the corpus had.
Counted separately; the verdict depends only on the pre-existing column.

---

## 2. Why the array kernels were still not reached — and why that is fine

The widened probe exercised only the integer families. **No array family ran at
all.** The reason is not signatures, and it is worth stating precisely because
the obvious diagnosis is wrong:

```
omul(np.eye(8)[1], np.eye(8)[2])
  -> NameError: name 'cds' is not defined
```

`omul`, `mul` and `o` fail **in isolation**, because they call `cds` and the
probe never imports their module. The isolation that makes the probe safe and
independent is exactly what withholds their dependencies.

So the probe can only reach **self-contained** functions — which are precisely
R8's *irreducible kernels*. Wrappers are structurally unprobeable under
isolation.

**And by R8's own collapse, probing them would add nothing.** A wrapper's
evidence *is* its kernel's evidence: `omul` corroborates exactly what `cds`
corroborates, plus a loop. R8 established that to make auditing affordable; the
same fact makes this coverage gap harmless. The two results agree, from
opposite directions.

---

## 3. Five hazards, discovered by hitting each one — and one answer

A tool that discovers corroboration by **calling unknown functions on synthetic
inputs** must defend against all of these. Each cost a run, and none announces
its cause.

1. **Unbounded return values.** `MemoryError` inside a nested dict — a census
   routine handed a plausible level returns a structure large enough to exhaust
   memory. *Defence:* cap every container (4 096 elements) and the
   canonicalisation depth (6); anything larger is declared uncomparable.
2. **Runaway allocation inside the callee.** Capping the *canonical form* does
   not help: the allocation happens **before the function returns**, so Python
   never raises and the process is simply killed — exit 120, **empty log**, even
   buffered output lost. *Defence:* `RLIMIT_AS` before any probing, so the
   allocation raises `MemoryError` inside the harness's own `try`.
3. **The callee closes your file descriptors.** One function in this corpus
   closes fd 1. The collection loop finished and then every `print` raised
   `OSError: [Errno 9] Bad file descriptor` — the work was done and the report
   died on the way out. *Defence:* `os.dup(1)` into a private descriptor
   **before** probing, and write all output through it.

4. **Truncated report from a descriptor the dup did not cover.** With the report
   going through a private dup, output still stopped after the header once
   probing began. Probed code reaches the parent's descriptors in ways one dup
   does not protect, and the only symptom is a report that ends early.
5. **Failure only under buffering — i.e. only in CI.** With fd 1 closed, the
   *interpreter* still flushes `sys.stdout` at shutdown, so the process exits
   non-zero **after the work is correctly done and printed**. Run by hand with
   `-u` it is green; run inside a gate that captures output it is red, with
   identical output. That is the shape of a bug that gets blamed on CI.

**All five have the same root and one answer: never run foreign code in the
process that has to report the result.** Probing now happens in a **child
process** that writes results as JSON and leaves via `os._exit(0)`, skipping
interpreter shutdown entirely — the file is already flushed and fsynced, so
there is no buffer left to fail on a dead descriptor.

That is precisely the decision the claim executor reached in R2, for gates,
for the same reason, with the justification written down at the time
(`fork`/`execve`, never in-process, so a crashing gate cannot corrupt compiler
state). This rung ignored its own line's conclusion and patched around four
symptoms before arriving back at it.

Hazard 2 is the nastiest to diagnose: it produces no output at all. Hazard 5 is
the nastiest to trust: it produces *correct* output and a wrong exit status.

---

## 4. What this is NOT

- **Not a discovery.** Zero new corroborations, again.
- **Not full coverage.** 4 behaviour classes out of a corpus with many more
  computations; only self-contained functions are reachable (§2).
- **Not a proof of equivalence.** Behavioural identity over a finite probe grid
  identifies *candidates*; confirming one is a separate job (R7/R8 did it
  properly for `cds`/`cd_sigma`).
- **Not a compiler change.** The R6–R11 arc is Python-only throughout.

---

## 5. What the arc now says

R10 built the discovery procedure and validated it by rediscovery. R11 widened
it as far as isolation permits and confirmed the negative. Together:

> **This corpus contains exactly one latent corroboration, and it was already
> found. Everything else it computes, it computes once.**

Depth 1 — one derivation, no internal corroboration — is not an anomaly here. It
is the corpus's normal state, now measured rather than assumed, across every
behaviour class the probe can reach.

---

## 6. Reproduce

```bash
python3 -u scripts/research/self_falsifying_compilation_line_r11_contract.py
# expect: 35 probeable functions, 4 behaviour classes, 1 pre-existing
#         corroboration (0 new),
#         SELF_FALSIFYING_R11_VERDICT WIDER_PROBE__NO_NEW_PREEXISTING_CORROBORATION
```

Run it with `-u`. The harness protects its own stdout (§3.3), but unbuffered
output makes a crash inside a probed function diagnosable rather than silent.

---

## 7. AI disclosure

Harness, gate and spec drafted under human direction (2026-07-26). All figures
are machine-computed and re-runnable. The five hazards are recorded because
each was hit, not anticipated — including the one this line had already solved
elsewhere and solved again the hard way. No clinical content. GAIDeT-ICMJE 2025.
