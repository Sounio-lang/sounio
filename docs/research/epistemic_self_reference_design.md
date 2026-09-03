<!-- docs:meta
topic_id: repo.docs.research.epistemic-self-reference-design
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.epistemic-self-reference-design
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Design: genuine epistemic self-reference in the bootstrap compiler

**Status:** design spec (2026-06-03). Goal chosen by the owner: make §6's
self-application headline **true** rather than retract it — give the
bootstrapping compiler `lean_single.sio` a *genuinely uncertain* decision,
annotate it with `Knowledge<T>`, so the self-compile certifies real uncertainty
and the `gen2==gen3` fixed-point becomes epistemically meaningful.

Context: the audit (`docs/audit/EPISTEMIC_SELF_APPLICATION_AUDIT_2026-06-03.md`)
showed `lean_single.sio` is deterministic — `EXPR_CONF = 1000` everywhere, so
"100% certain / 0 guarded" is vacuous. The genuine uncertainty in the codebase
lives in the *modular* compiler's optimizer (`ir/egraph.sio` etc.), which does
not bootstrap. To make *self-reference* real, the uncertainty must live **in the
bundle that bootstraps**.

## The honesty constraint (non-negotiable)

The added uncertainty must be **genuine**, not a number sprinkled on a
deterministic operation. Lexing, parsing, type-checking, and codegen are exact —
annotating them would fabricate uncertainty (the same overclaim the audit
caught). A decision is legitimately epistemic only if the compiler **lacks the
information to be certain** and must *estimate*.

## The genuine uncertainty: static branch-likelihood

A self-hosted compiler with **no profile data** that wants to lay out basic
blocks (place the likely successor as fall-through) must *guess* which branch is
hotter. Standard static heuristics (Ball–Larus; "loops are usually taken",
"branches to `panic`/error paths are usually not taken") are genuinely
uncertain — they are *estimates*, wrong a measurable fraction of the time. This
is real epistemic uncertainty the compiler actually has.

Elegantly, the compiler that *implements* `Knowledge<T>` can **dogfood** it for
its own heuristic — and `lean_single.sio` already type-checks `Knowledge<T>`,
`measure`, and the `with Epistemic` effect, so it can compile such code in its
own source.

## ⚠️ Reachability finding (2026-06-03): the consumer does not exist

A grep of `lean_single.sio` for any uncertain-decision site — block layout,
fall-through/branch ordering, jump threading, register spill, inlining heuristic
— comes back **empty**. The compiler is a pure deterministic single-pass emitter
(emits blocks in source order; `local_bss_spill_bytes` is a fixed `524288`; the
"inline" hits are C-string emission and Option-representation, not optimization).

**Consequence (scope doubles):** there is no existing decision to annotate. To do
genuine self-reference one must **first add a real optimization pass** (a basic-
block layout pass that chooses fall-through using the static branch-likelihood
heuristic) to the bootstrap compiler, *then* annotate its estimate. This is
defensible — block layout is a real, useful optimization that genuinely has no
profile data, so the heuristic is honestly uncertain (not a decision invented
solely to be uncertain) — but it is a **substantial codegen feature on the
bootstrap path**, not a small annotation. The plan below therefore has a Step 0.

## Concrete plan (minimal, reversible)

**Step 0 (the newly-revealed prerequisite): add a layout decision to consume the
estimate.** A block-ordering pass over the emitter's basic blocks that, for each
conditional, places the estimated-likely successor as fall-through. Layout-only
(see neutrality below): it changes block *order*, never program meaning.

1. **A genuinely-uncertain estimator in `lean_single.sio`'s own source:**
   ```
   fn branch_likelihood(is_loop_backedge: bool, leads_to_panic: bool)
       -> Knowledge<i64> with Epistemic {
     // static heuristic, honestly uncertain — no profile data:
     //   loop back-edge  → ~90% taken  (conf 900)
     //   path to panic   → ~10% taken  (conf 100, i.e. likely-not)
     //   otherwise       → 50/50       (conf 500)
     if is_loop_backedge { measure(90, uncertainty: 10) }
     else if leads_to_panic { measure(10, uncertainty: 10) }
     else { measure(50, uncertainty: 25) }
   }
   ```
   These confidences are **below `GATE_THRESHOLD = 950` by construction** — the
   estimate is honestly a guess.

2. **A consumer that acts on it** (block-layout fall-through choice). Reading the
   point estimate requires `.value` → forces `with Epistemic` on the consuming
   codegen function and routes the confidence through the epistemic pass.

3. **What the self-compile then shows (non-vacuous):**
   - the epistemic pass over `lean_single.sio`'s *own* HIR now computes
     `EXPR_CONF < 950` at the branch-layout sites;
   - those sites emit the real `66 90` marker → `gates[direct=N guarded=M>0]`
     when the compiler compiles itself — the first non-zero guarded count on the
     bootstrap;
   - the fixed-point `gen2==gen3` now witnesses **epistemic stability**: the
     compiler's confidence in its own heuristic is identical across generations
     (the original §6.5 claim, now with actual content).

## Verification (what "done" means)

- `bin/souc` (committed binary `6374e52f`) compiles the edited `lean_single.sio`
  → new binary; `gates[…guarded=M]` with **M > 0** (non-vacuous self-uncertainty).
- Re-bootstrap through the **global build lock** (`scripts/dev/souc-build-lock.sh`,
  per CLAUDE.md §4): `gen_k == gen_{k+1}` bit-identical → fixed-point preserved
  *with* the epistemic annotations live.
- The branch-layout heuristic is *functionally* sound (compiler still correct:
  full self-host gate green) — the Knowledge annotation must not change emitted
  semantics, only add the (zero-cost) marker + confidence tracking.

## Risk + discipline (this touches the bootstrap)

- **Brick risk.** Edits to `lean_single.sio` can break self-compilation. Mitigate:
  keep the patch **small and reversible**; build against the **committed** binary
  (the working-tree `souc` `9d4ef541` miscompiles `cd_mul` — must not be used);
  verify `souc check` before any full self-compile.
- **CPU / pod stability.** Full self-compile MUST go through the build lock
  (§4 — pod evicted under CPU saturation before). One build at a time.
- **Scope honesty.** This is multi-session. The first milestone is *the estimator
  + consumer compile and the self-compile shows guarded>0*; the fixed-point
  re-verification is the second; the paper §6 rewrite (now truthful) is the third.
- **Functional neutrality.** The heuristic must be layout-only (fall-through
  choice) so a wrong guess never changes program meaning — only block order.
  Otherwise we'd trade an honesty problem for a correctness one.

## ⚠️ Step 1 finding (2026-06-03): confidence ≠ variance — the estimator must be low-*confidence*

Step 1 added `branch_likelihood(...) -> Knowledge<i64>` (via `measure(v, uncertainty: u)`)
to `lean_single.sio` after `local_bss_spill_bytes` (≈ line 938) and ran
`souc check` (committed binary `6374e52f`): it **type-checks cleanly** — 1396
functions gate-passed, no errors — confirming the compiler can carry epistemic
`Knowledge<i64>`/`measure` code in its own source. **But the census showed
`min=1000 mean=1000` with the estimator present**: `measure(v, uncertainty: u)`
sets the GUM **variance**, not the **confidence** that drives `EXPR_GATE`
(`CONF < 950`). A high-variance measurement still has confidence 1000 → never
gates (consistent with all 8 earlier programs showing `guarded=0`).

**Design correction:** branch-likelihood is a *guess the compiler does not fully
trust* — that is a **confidence** (~70%), not a variance. The estimator must
inject **confidence < 950** via the compiler's confidence mechanism — the
`with Epistemic(N)` floor (`FN_EPISTEMIC_MIN`, parsed at ~24507 with `N ≤ 1000`)
or the `asserted`-named-function path (→ 970), serially composed below 950 —
*not* via `measure`'s `uncertainty:`. Verify the exact low-confidence injector
before re-applying (probe a `with Epistemic(700)` function's census for
`guarded>0`). The WIP `measure`-based estimator was reverted (type-checks but
ineffective and not self-compile-verified).

## ⚠️⚠️ Step-1 probe finding (2026-06-03): the gate only fires on ERRORS, not genuine uncertainty — a wall

The owner-requested probe (`with Epistemic(700)` census for `guarded>0`)
returned **`guarded=0`**, as did `asserted`-named functions and serial
composition. Reading the current confidence model (`lean_single.sio`) explains
why, and it is structural, not incidental:

- Legitimate confidence values are `measured → 990` and `asserted → 970`
  (FN_EFF_CONF), and **both are ≥ `GATE_THRESHOLD = 950` by design** — the
  compiler's own comment states it: `// call_conf = 990*1000/1000 = 990 ≥ 950. ✓`.
- `with Epistemic(N)` is an **obligation** (the body must reach confidence `N`),
  not a confidence *injector*; it does not lower a call's confidence to `N`.
- The **only** way `EXPR_CONF` goes below 950 is the explicit `EXPR_CONF = 0`
  sites — the **error / unresolved-identifier** cases (the "BRONZE" tier).

**Therefore the `66 90` gate marker can fire only on errors, never on
legitimately-uncertain code.** A guarded site means "unresolved/erroneous," not
"genuinely uncertain heuristic." This corroborates the audit (8 + 3 programs,
all `guarded=0`) and reveals the real obstacle: **"real self-reference with a
genuine guarded>0" is not an annotation task — it requires redesigning the
confidence calibration** so genuine uncertainty (a branch-likelihood *guess*)
can land below 950, which the current model deliberately prevents (everything
real is pinned ≥ 970). That is a substantial change to the epistemic pass
itself, a third scope expansion on top of "add an optimization to host it."

**Honest options now (owner's call):** (a) redesign the confidence calibration
so genuine uncertainty gates (large, changes the pass semantics + invalidates
the existing "0 guarded" framing); (b) accept that the gate is an *error/low-
assurance* marker, not an *uncertainty* marker, and reframe §6 around that
honestly; (c) the optimizer-module DEMO instead; (d) stop — the audit + this
probe are the deliverable, and the self-reference headline may simply not be
reachable without rebuilding the confidence model.

## ⚠️ CORRECTION (2026-06-03): the prior "gate fires only on errors" finding was WRONG

Re-probing with the correct form (a `let x = measure(v, uncertainty: u)` *binding*
with HIGH relative uncertainty) overturns the previous section. The confidence
model **already derives gating confidence from uncertainty**:
`m_conf = 1000 − eps_scaled/1000`, `eps_scaled = infer_measure_eps_scaled =
(unc/value)·1e6` (relative uncertainty, ppm; `lean_single.sio:2947`,
~21598). So `measure(10.0, uncertainty: 8.0)` (80% rel) → `eps_scaled=800000` →
`m_conf = 200`, and the census confirms **`min=0`** (genuine sub-950 confidence
from a real high-uncertainty measurement — *not* an error). My earlier probes
missed this because (a) the bare-`measure()` *return* form doesn't hit this
binding path, and (b) the 8 sampled programs had *low* relative uncertainty.

**The real gap is narrower and more tractable:** despite `min=0`, the census
still shows **`guarded=0`** — the `66 90` marker does not fire. The low
confidence is computed but **does not propagate to the consuming call site's
gate** (`consume(lo.value)` stays ≥950; `mean=998`). So the obstacle is **not**
the calibration (sound) and **not** a "≥970 floor" (false) — it is the
**confidence → call-site-gate propagation** (`EXPR_GATE[call_tok]` is not driven
by a low-confidence argument). The §6.2 rule `CONF[call] = min(CONF[args])·
CONF[body]` is the intended behavior; it is not reaching the gate.

**Re-aimed "redesign":** the targeted fix is to complete the
argument-confidence → call-confidence → `EXPR_GATE` propagation so a call
consuming a genuinely-uncertain value gates (`guarded>0`). That is a localized
pass fix, not a calibration rewrite — and it is the real prerequisite for
self-reference (and for the branch-likelihood estimator to gate). VERIFY the
exact propagation site before editing; re-probe the corrected case for
`guarded>0` after.

## ✅ Propagation site LOCATED (2026-06-03): a deliberate dual-channel split

The gap is a **two-channel design**, confirmed by the source comments
("Gen 13: dual-channel measurement confidence (parallel to EXPR_CONF)";
"ESCOPE measurement quality (parallel to ESCOPE_CONF)"):

- `escope_add(name, ty, conf)` (`lean_single.sio:20948`) sets
  `ESCOPE_CONF[slot] = conf` (20952) — the **GATE channel**.
- The `let x = measure(...)` binding calls `escope_add(nh, bind_ty, bind_conf)`
  (21680) with `bind_conf` ≈ 1000, and separately stores the measure's *low*
  confidence `bind_meas_conf` (e.g. 200) into **`ESCOPE_MEAS[slot]`** (21683) —
  a **parallel MEASUREMENT-QUALITY channel**.
- Call-argument propagation (21785) and the gate read **`ESCOPE_CONF`**, never
  `ESCOPE_MEAS`. So measurement uncertainty lowers only the parallel channel;
  the gate channel stays 1000 → `guarded=0`.

Empirical confirmation: `let lo = measure(10.0, uncertainty: 8.0)` then
`consume_k(lo)` → `min=0` (measurement-quality channel) but `guarded=0` (gate
channel ≥ 950). Disambiguated with the bare-Knowledge-arg probe (also 0), ruling
out the `.value` walk and pinning it to the binding's channel split.

### The fix (small, principled — connect the channels)

At the binding (≈21680), fold the measurement confidence into the gate
confidence: `escope_add(nh, bind_ty, min_nonzero(bind_conf, bind_meas_conf))`
(use `bind_meas_conf` when it is > 0 and < `bind_conf`). Then `ESCOPE_CONF[lo]`
carries the genuine low confidence, downstream uses propagate it (21785), and a
call consuming a 20%-confident value **gates** (`guarded>0`). This is the honest
redesign: the gate channel *should* reflect that a value you barely trust is not
"confidently usable" — it does **not** lower the threshold or fabricate
uncertainty. (Mirror the same fold anywhere `ESCOPE_MEAS`/`bind_meas_conf` is set
but `ESCOPE_CONF` isn't, and re-pin the §6 "0 guarded" framing — clinical
high-uncertainty measurements will now legitimately gate, which is correct.)

### Then self-reference becomes reachable

With the channels connected, the branch-likelihood estimator — expressed as a
low-confidence `measure` (e.g. a ~70%-reliable guess) and consumed by the
layout decision — produces a genuine `guarded>0` on the bootstrap self-compile.
The remaining order: (i) make this channel-fold fix + verify `consume_k(lo)`
gates on a probe; (ii) Step 0's layout host + estimator; (iii) self-compile via
build lock + re-pin fixed-point.

## Advisor caveats to honor during implementation

- **Fixed-point hash will change, and that's fine.** Block reorder ⇒ different
  jump offsets ⇒ different bytes. "Fixed-point preserved" means
  `gen_k == gen_{k+1}` for the *new* code — **re-pin from scratch**; do not expect
  the documented `54327028`.
- **Heuristic must be a pure function of the IR.** If `is_loop_backedge` /
  `leads_to_panic` read anything that varies run-to-run, the layout (and bytes)
  won't be stable and the fixed-point won't close.
- **Success = `guarded > 0` AND re-bootstrap closure — a *small* number is still
  success.** The pass may propagate sub-950 confidence only to the immediate
  consumer (e.g. `guarded=3`), leaving the rest at 1000. That is genuine
  non-vacuous self-uncertainty and is the win. **Do not inflate the annotation
  surface to manufacture a bigger count** — one honest guarded site beats fifty
  contrived ones.

## First-session execution order (turnkey; staged to de-risk the bootstrap)

Do these in order; each is a safe checkpoint. Use the **committed** binary
(`md5 artifacts/self-hosted/souc-self-hosted-x86_64 == 6374e52f…`; if the
working tree shows `9d4ef541…` it's the broken `cd_mul` swap — extract HEAD's
binary to `/tmp` and use that). Every full self-compile goes through
`scripts/dev/souc-build-lock.sh`.

0. **Locate the block/jump emission site.** ✅ DONE (2026-06-03). Host =
   **`compile_stmt()` at `lean_single.sio:18655`, the plain-`if` codegen ~19763**
   (after the large `if let`/pattern block). x86 emission structure:
   `em(0x0f); em(0x84)` (`je` → else, back-patched) → then-body → `em(0xe9)`
   (`jmp` end, back-patched) → patch je→else → else-body → patch jmp→end. The
   fall-through is implicit: the **then-body** falls through the `je`; the else is
   jumped-to. The layout choice = **invert the `jcc` (`0x84`↔`0x85`) and swap the
   then/else emission order** when the estimate says the else-branch is hotter.
   Notes: (a) the `if` codegen is split by ARCH — `compile_stmt_a64()` (~26881,
   the if dispatch at 26964) is the **separate aarch64 backend**; the x86
   self-host fixed-point only needs the x86 site, so **Step 1 is one site, not
   two**. (b) `compile_and`/`compile_or` (17058/17092) are the jcc-encoding
   template. (c) The site is entangled with if-let patterns, channel sets
   (`EXPR_CH_SET`), linear-use tracking, and value-yielding — the Step-1 edit must
   be a *local* fall-through swap that leaves all of that untouched.
1. **Plumbing with a NEUTRAL heuristic first (zero-risk).** Add the layout
   decision point but make the heuristic a no-op (never reorder). `souc check`,
   then self-compile via the lock → confirm **bit-identical** to current HEAD
   binary. This proves the decision-site plumbing is safe before any behavior
   change.
2. **Make the heuristic real (still no annotation).** Static branch-likelihood
   as a *pure function of the IR* (loop back-edge → likely; path to
   `panic`/error → unlikely; else 50/50); reorder fall-through accordingly.
   Self-compile; expect **different bytes**; re-pin `gen_k == gen_{k+1}` →
   functional fixed-point with the optimization live. Full self-host gate green.
3. **Annotate epistemically.** Lift the estimate to
   `branch_likelihood(...) -> Knowledge<i64> with Epistemic`; consume via
   `.value`. Self-compile; read `gates[direct=N guarded=M]` → **expect M > 0**
   (first non-vacuous guarded count on the bootstrap).
4. **Re-pin the epistemic fixed-point.** Re-bootstrap; `gen_k == gen_{k+1}` with
   annotations live → *epistemic* stability with real content. Record the new
   hash.
5. **Rewrite §6 truthfully** (now backed): self-application is real; the census
   is non-vacuous; re-pin the convergence/fixed-point numbers.

Stop after step 1 if context tightens — it's a clean, reversible checkpoint.

## ⚠️ Step 1b finding (2026-06-03): low confidence is visible, but `guarded` is not a general low-confidence counter

Follow-up probes on the current `modular/native-v2-e2e-gate` binary show the
low-confidence injector story needs one more correction before editing
`lean_single.sio`:

- `with Epistemic(700)` on a callee does **not** make the callee's return/call
  confidence 700. It is an `EpistemicComplete` body-confidence requirement
  (`FN_EPISTEMIC_MIN`), while plain `with Epistemic` is just the effect tag.
- `measure(10.0, uncertainty: 5.0)` does create real low-confidence expression
  sites: the probe census reported `epistemic_main: 30 expr, 27 certain, 3
  uncertain`, `tier_dist ... BRONZE=2`, `min=0`, and `mean=998`.
- But `gates[guarded]` still stayed **0** even when the low-confidence
  `k.value` fed a normal function call (`consume(k.value)`). The current x86
  marker is emitted only when `EXPR_GATE[call_tok] == 1` at the call emission
  site, and these probes do not make that call token guarded.

**Design correction:** the self-reference success gate cannot be stated as
"any low-confidence expression yields `guarded>0`." There are two honest paths:

1. Treat `epistemic_main`/`tier_dist`/`econf min` as the first non-vacuous
   self-uncertainty evidence, then separately decide whether a codegen marker
   should be generalized beyond call tokens.
2. If the headline specifically requires `gates[guarded]>0`, first add a real
   consumer whose call token is low-confidence, or extend marker emission to the
   actual branch-layout decision site instead of assuming `.value` propagation
   reaches call-token gating.

Do **not** implement the bootstrap annotation until this consumer/marker
boundary is chosen explicitly. Low confidence is real; the old `guarded>0`
acceptance condition is too narrow for the current implementation.

## Why this is the honest headline, not theater

The compiler genuinely cannot know which branch is hotter without a profile;
it estimates; the estimate is wrong sometimes; that uncertainty is now a
first-class `Knowledge<i64>` in the compiler's own source, certified and gated
by the same epistemic pass it applies to user code. "The compiler applies its
epistemic type system to its own source" becomes a mechanical fact with real
content — and the fixed-point proves the self-confidence is stable.

---

## Forensic finding 7 (2026-06-03): the dual-channel fold fix is a NO-OP — REJECTED

The proposed one-line fix — fold the measurement-quality channel into the gate
channel at the binding `escope_add` site

```sounio
let gate_conf = if bind_meas_conf > 0 && bind_meas_conf < bind_conf { bind_meas_conf } else { bind_conf }
escope_add(nh, bind_ty, gate_conf)
```

was implemented, self-compiled through the build lock (committed binary
`6374e52f` as stage 0), and **empirically disproven**. It is now reverted.

### What was measured

Building the fixed compiler `B1 = B0(source+fold)` and compiling programs with
it produced, at first sight, an exciting result:

| program | `B0` (committed) | `B1` (fold) |
|---|---|---|
| `lean_single.sio` (the compiler itself) | `guarded=0` | `guarded=443` |
| `consume_k(measure(10, unc:8))` (80% rel) | `guarded=0` | `guarded=1` |
| `consume_k(measure(10, unc:0.05))` (0.5% rel) | `guarded=0` | `guarded=0` |

and a "graded sweep" (`unc` 0.05→`guarded=0`, ≥1.0→`guarded=1`), plus a stable
bootstrap fixed point `B2==B3` (md5 `33a8ddaf119c`). Read naively this says
"the fold makes genuine uncertainty gate, selectively, and even makes the
compiler gate 443 of its own sites." **All of that is an artifact.**

### The decisive instrument

A diagnostic counter `FOLD_FIRED`, incremented only inside the fold's true
branch and printed beside `gates[...]`, settles it:

```
B1_instr compiling lean_single.sio:  direct=17161 guarded=443 fold_fired=0
B1_instr compiling the high-unc probe: direct=23  guarded=1   fold_fired=0
```

**`fold_fired=0` in every case.** The fold's true branch never executed, so it
provably never lowered a single confidence — yet `guarded` moved from 0 to 443
(compiler) and 0 to 1 (probe). The guarded deltas are therefore **codegen
artifacts**: inserting the `let gate_conf` local into `lean_single.sio`
perturbed its known span/layout-sensitive codegen (see memory
`project_modular_span_sensitive_crash`) and emitted spurious `66 90` NOP
markers. The fixed point holding only proves the *artifact* is stable — a
stable miscompile is still a miscompile.

### Root cause of the no-op

For a `measure()` binding the inference path assigns **the same value** to both
channels: `rhs_conf = m_conf` and `bind_meas_conf = m_conf` (lean_single.sio
~21606-21607). Hence `bind_meas_conf < bind_conf` is structurally **never
true** for measured values, and `min(bind_conf, bind_meas_conf) ≡ bind_conf`.
The two channels were never actually divergent at the binding site for the case
the fix targeted; the fold had nothing to fold. The fix can only fire for a
binding whose RHS is a call to a function named *exactly* `measured`/`asserted`/
`constant` whose `FN_EFF_CONF` exceeds the tier constant — a case that occurs
neither in the compiler source (zero such bindings, confirmed by grep) nor in
any test probe.

### Corrected understanding of the real gap

The genuine bug is **upstream of the binding channel**, not at it. `B0` stores a
low `bind_conf` (e.g. 200 for an 80%-relative measurement) into `ESCOPE_CONF`
already, yet still reports `guarded=0` on the consuming call — i.e. the stored
low confidence does **not** propagate to the call-token gate / marker-emission
site. The §6 audit's "gate never observed firing" stands. Closing it requires
work at the propagation / marker-emission boundary (which call token a
low-confidence `ESCOPE_CONF` entry should mark), **not** a binding-site fold.

### Status

- Fix **reverted**; working tree clean; committed binary `6374e52f` untouched.
- §6 self-application headline remains **overstated / not implemented** per the
  2026-06-03 audit; this finding closes the "dual-channel fold" candidate fix as
  a dead end.
- Method note: the result was only caught because the `FOLD_FIRED` instrument
  separated "the logic fired" from "the count changed." A guarded-count delta is
  **not** evidence that an epistemic mechanism fired; on layout-sensitive codegen
  it can be pure noise. Future epistemic-pass changes must instrument the
  decision branch, not trust the aggregate count.
