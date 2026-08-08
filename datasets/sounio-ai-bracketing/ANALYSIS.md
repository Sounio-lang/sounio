# SBMP v0 — first measurement (2026-08-07)

Target: grok-4.3 (temperature 0), 50 items × 2 induced bracketings.
Judge: glm-5.2 (independent provider), content-level matching.
Raw: `results/grok-4.3_run1.jsonl`; judged: `results/grok-4.3_run1.judged.jsonl`.

## Numbers

- wording-level flip rate (naive): 0.66 — overcounts (hedge variants)
- **semantic flip rate (judged): 0.10** (5/50)
- **directional clean rate (each induction yields its own reading): 0.02** (1/50)
- by gold: left 0.125 / both 0.071 / right 0.10 (semantic flips)

## Reading

The induced bracketing barely moves the model's interpretation, and when it
moves, it does not move *directionally* (toward the induced reading). The
dominant behaviour is a **recency default**: under both inductions the model
tends to the (u₂ u₃)-grouped reading. Under the strong non-associativity
hypothesis (bracketing steers interpretation), the associator should be large
and directional; measured: small and default-driven.

**Status: the strong hypothesis is falsified for this model under framing
induction; the bracket-blindness diagnosis stands as the result.** Caveats,
all declared: (i) framing is the weakest induction — hierarchical
summarisation may move more; (ii) gold labels are machine-authored, human
validation pending; (iii) one model, one run; (iv) judge agreement
unmeasured. None of these rescue the positive claim; they bound its scope.

## Next

- stronger induction (subdialogue summarisation) as induction #2
- multi-model battery (the same harness takes any OpenAI-compatible endpoint)
- human validation of the 50 gold labels
- judge-agreement check (second judge on a sample)

## Preregistered battery predictions (2026-08-07, before any non-grok run)

Reference point (grok-4.3): semantic flip 0.10, directional 0.02, recency
default. Predictions for the battery (llama-3.3-70b via Groq, glm-5.2 via
Z.AI, deepseek-v4-pro if balance allows):

- B1: every model's judged semantic flip rate ≤ 0.25.
- B2: every model's directional clean rate ≤ 0.10.
- B3: a recency default (right-reading rate under LEFT induction > 0.5 among
  decidable items) appears in at least half of the probed models.
- B4: no model achieves directional rate ≥ 0.30 (i.e., none is
  bracket-sensitive under framing induction).

Same scoring rule as before: judged by glm-5.2 (independent of each target,
and itself a target — when glm-5.2 is the target, a second judge (grok-4.3)
scores it, and the two judges' agreement is reported).

## Battery results — 4 models, judged (2026-08-08)

| model | semantic flip | directional clean | recency under left-induction |
|---|---|---|---|
| grok-4.3 | 0.10 | 0.02 | 0.72 |
| grok-4.5 | 0.16 | 0.08 | 0.74 |
| glm-5.2 | 0.46 | 0.04 | 0.45 |
| glm-4.7 | 0.30 | 0.02 | 0.68 |

Judges: glm-5.2 judged the grok runs; grok-4.3 judged glm-5.2; grok-4.5
judged glm-4.7 (no family self-judging anywhere).

**Verdict on the preregistered predictions:**
- **B1 (all flips ≤ 0.25): FALSIFIED** — glm-5.2 (0.46) and glm-4.7 (0.30).
- **B2 (all directional ≤ 0.10): CONFIRMED** — 0.02 / 0.08 / 0.04 / 0.02.
- **B3 (recency default in ≥ half the models): CONFIRMED** — 3 of 4 above
  0.5 (0.72, 0.74, 0.68).
- **B4 (no model directional ≥ 0.30): CONFIRMED** — max 0.08.

**Family profiles.** grok: framing-insensitive (flip ≤ 0.16) with a strong
recency default; grok-4.5 improves directional compliance 4× over 4.3 — the
family is moving toward bracket sensitivity but is not there. glm:
framing-sensitive (flip 0.30–0.46) but undirected; the framing shakes the
interpretation without steering it to the induced reading — concentrated on
`gold=both` items for glm-5.2 (0.71 flip rate on genuinely ambiguous items).
Universal: **no model lands directionally on the induced bracketing** (max
0.08/50 items). Turn-level bracketing is, for these four frontier models,
either invisible (grok) or a source of undirected variance (glm). The
associator as a steering signal is absent in both regimes.

**Infrastructure notes:** deepseek key invalid (401, key revoked — not
balance); OpenRouter has no credits; api.groq.com unresolvable from this
host. Battery ran with 4 models via xAI + Z.AI. Two SIGKILLs hit long tasks
on this host; rescore is now incremental with resume.

**Judge-agreement caveat:** grok-4.3 and glm-5.2 judging the same responses
may disagree; agreement was not measured (declared). The glm-5.2-as-judge
and grok-as-judge tables were computed with family-crossed judges precisely
to bound this.

## Preregistered: kimi-k3 in the dock (2026-08-08, before any Kimi answer)

Protocol exception, declared: the kimi-k3 run is produced by fresh subagent
instances of the same model that authors this repository line (one clean
context per category, identical harness prompts, no temperature control,
single pass — flagged as "in-session, subagent-fresh" and NOT comparable in
kind to the API runs; it is comparable in content).

- **K1:** kimi-k3 directional clean rate > 0.08 (beats the battery's best,
  grok-4.5).
- **K2:** kimi-k3 semantic flip rate ≤ 0.46 (below the battery's worst,
  glm-5.2).
- **K3 (the plot twist):** if directional ≥ 0.30, B4 breaks and the
  bracket-sensitivity hypothesis resurfaces with a champion model — reported
  with the same prominence either way, per protocol.

## kimi-k3 verdict (2026-08-08, judge grok-4.5)

| model | semantic flip | directional | recency |
|---|---|---|---|
| kimi-k3 | 0.82 | 0.06 | 0.20 |

Preregistration verdict: **K1 FALSIFIED** (0.06 < 0.08), **K2 FALSIFIED**
(0.82 ≫ 0.46), **K3 FALSIFIED** (directional nowhere near 0.30). The K3
profile is the most extreme in the battery and in the wrong direction for
the positive hypothesis: maximally framing-reactive (0.82 semantic flips)
but neither directed at the induced reading nor anchored on recency (0.20,
the battery's lowest) — the model generates a third reading of its own.
Loudest undirected associator measured; sensitivity without direction is
variance, not bracket-sensitivity. Declared protocol caveats: in-session
subagent answers (up to 15 words, no temperature control, contexts fresh
per category) are more verbose than the API runs, which may inflate judged
semantic difference; judge was grok-4.5 after glm-5.2 hit HTTP 429
(rate-limited after ~700 judge calls; grok-4.5 has no family conflict with
kimi-k3).

**Battery-wide standing (5 models, 3 families):** directional clean rate
≤ 0.08 everywhere. The associator-as-steering is absent in all measured
regimes; what varies across families is the amount of undirected variance
(0.10 → 0.82) and the recency anchor (0.20 → 0.74).
