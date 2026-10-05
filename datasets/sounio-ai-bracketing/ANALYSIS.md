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

## Preregistered: local/self-hosted wing (2026-08-08, before any local probe)

Endpoints discovered on-cluster (k8s): llm-router service
(`10.96.188.231:4000`, LiteLLM router, OpenAI-compatible, no key needed
from inside the cluster). Alive local backends verified by direct call:
qwen2.5-14b, qwen2.5-coder-32b, spark-qwq-32b (QwQ-32B reasoning),
spark-phi4-reasoning (Phi-4 14B). Dead/slow backends noted: r1-distill-70b,
qwen3.6-35b, hermes-4, baichuan-m2-med, spark-glm-air, spark-deepseek-16b,
hunyuan-7b, olmo3-think (connection errors or >90s cold start).

Predictions (reference: frontier battery directional ≤ 0.08 everywhere):
- **L1:** every local's directional clean rate ≤ 0.10.
- **L2:** at least one local has semantic flip ≥ 0.25 (smaller models are
  more framing-reactive).
- **L3:** no local achieves directional ≥ 0.30 (B4 extends to self-hosted).

## Local wing results (2026-08-09, judge glm-5.2)

| model | params | semantic flip | directional | recency |
|---|---|---|---|---|
| qwen2.5-14b | 14B | 0.16 | 0.00 | 0.53 |
| qwen2.5-coder-32b | 32B | 0.16 | 0.04 | 0.50 |
| spark-deepseek-coder-16b | 16B | 0.38 | 0.00 | 0.53 |
| spark-qwq-32b | 32B reasoning | DNF | — | — |
| spark-phi4-reasoning | 14B reasoning | DNF | — | — |

DNF = per-call generation exceeded the 600 s budget on the shared ollama
backend (reasoning models); declared, not retuned.

**Verdict on the preregistered L-predictions:**
- **L1 (all locals directional ≤ 0.10): CONFIRMED** (0.00 / 0.04 / 0.00).
- **L2 (≥ one local flip ≥ 0.25): CONFIRMED** (dscoder-16b 0.38).
- **L3 (no local directional ≥ 0.30): CONFIRMED** (max 0.04).

## The 8-model map

| model | class | flip | directional | recency |
|---|---|---|---|---|
| grok-4.3 | frontier | 0.10 | 0.02 | 0.72 |
| qwen2.5-14b | local | 0.16 | 0.00 | 0.53 |
| grok-4.5 | frontier | 0.16 | 0.08 | 0.74 |
| qwen2.5-coder-32b | local | 0.16 | 0.04 | 0.50 |
| glm-4.7 | frontier | 0.30 | 0.02 | 0.68 |
| dscoder-16b | local | 0.38 | 0.00 | 0.53 |
| glm-5.2 | frontier | 0.46 | 0.04 | 0.45 |
| kimi-k3 | frontier (in-session) | 0.82 | 0.06 | 0.20 |

Across 8 models, 3+ families, 2 deployment classes: **directional clean
rate ≤ 0.08 everywhere** — the strongest, most replicated finding of the
SBMP v0. Framing reactivity varies by an order of magnitude (0.10–0.82) and
is a family/deployment trait, not steering. Recency anchors everything
except kimi-k3. Local models sit inside the frontier envelope on every
metric — no local champion, no local outlier worse than the worst frontier
model either.

## Preregistered: induction #2 (hierarchical summarisation) — 2026-08-09

Design: the pair is composed into a literal one-sentence summary by the
target model itself, then the probe is answered with that summary in
context (left: summary of (u1,u2) + u3; right: u1 + summary of (u2,u3)).
μ is operationalised as the summarisation step. Declared mechanics test
BEFORE preregistration: 2 items (R-001, R-002) on grok-4.5 to verify the
codepath — R-002 came out clean-flip on the naive matcher; disclosed so the
preregistration below is honest about what was already seen (2/50 items,
one model, naive metric only).

- **S1:** every model's judged directional rate under summary induction
  exceeds its framing-induction rate, but no model exceeds 0.20.
- **S2:** semantic flip rates rise vs framing for every model (the summary
  injects the bracketing into the context explicitly).
- **S3 (the rescue test):** no model reaches directional ≥ 0.30 even under
  summary induction. If S3 fails, the positive hypothesis gains its first
  champion and the battery's null needs re-reading.

## Induction #2 (hierarchical summarisation) — full results (2026-08-09)

| model | framing flip/dir/recency | summary flip/dir/recency |
|---|---|---|
| grok-4.3 | 0.10 / 0.02 / 0.72 | 0.16 / 0.08 / 0.58 |
| grok-4.5 | 0.16 / 0.08 / 0.74 | 0.20 / 0.12 / 0.54 |
| glm-5.2 | 0.46 / 0.04 / 0.45 | 0.60 / 0.00 / 0.62 |
| glm-4.7 | 0.30 / 0.02 / 0.68 | 0.24 / 0.08 / 0.62 |

**Preregistration verdict:**
- **S1 (directional rises for every model, ≤ 0.20): FALSIFIED as universal** —
  3/4 rose (grok-4.3 0.02→0.08, grok-4.5 0.08→0.12, glm-4.7 0.02→0.08) but
  glm-5.2 fell to zero (0.04→0.00). The rise is a property of anchored
  models, not of the induction.
- **S2 (flip rises for every model): FALSIFIED as universal** — 3/4 rose
  (0.10→0.16, 0.16→0.20, 0.46→0.60) but glm-4.7 fell (0.30→0.24).
- **S3 (no model directional ≥ 0.30): CONFIRMED** — battery-wide maximum
  under literal composition is 0.12 (grok-4.5).

**Synthesis.** Literal composition helps anchored models: modest directional
gains (≤ 0.12) and a weakened recency anchor (0.72–0.74 → 0.54–0.58 for the
groks). On the reactive outlier (glm-5.2) it amplifies volume without
direction (flip 0.60, directional 0.00) and even re-anchors it on recency
(0.45 → 0.62). glm-4.7 sits behaviourally closer to the groks than to its
own family's flagship. Two inductions, nine model-runs, one robust null:
**no measured model is bracket-steerable**; the associator exists as
variance everywhere, as steering nowhere. The one lever that produced any
direction at all is explicit composition — which is precisely the operation
a non-associative semantic architecture would need to supply natively.

## Preregistered: does the octonion functor's grouping magnitude track
## semantic ambiguity? (2026-08-09, before any embedding is computed)

Pipeline: sentence-transformers all-MiniLM-L6-v2 (384-dim) for the 150 SBMP
turns → fixed deterministic projection to the 16-dim trajectory payload →
generated .sio driver → Madaros-native associator magnitude per item. The
embedding/projection is frozen BEFORE any magnitude is looked at.

- **C1:** items with gold_human="both" have higher median grouping
  magnitude than items with gold_human ∈ {left, right} (Mann–Whitney
  one-sided, reported with the p-value; no threshold promised in advance).
- **C2 (calibration):** the same pipeline with the canonical hash/Halton
  embedding (semantics-free) concentrates all items in a narrow band; C1's
  separation, if any, must exceed that band to count.

If the semantic embedding cannot be obtained (offline fallback), only C2
runs and C1 is declared UNTESTED, not failed.

## C1/C2 verdict — octonion grouping magnitude vs semantic ambiguity (2026-08-09)

Pipeline: all-MiniLM-L6-v2 embeddings (384-dim, normalised) → frozen
projection (seed 17) to 16-dim payloads → Sounio-native associator
(grouping_magnitude_sq) per item, run in 5 chunked Madaros drivers.

**C1: NOT CONFIRMED.** Median grouping magnitude: gold=both 6.12e-05,
gold∈{left,right} 5.56e-05 — right direction, far too weak
(Mann–Whitney one-sided p = 0.248, two-sided p = 0.496). With MiniLM-class
embeddings, the functor's grouping magnitude does not separate
human-ambiguous from committed items. The associator is exact and real;
its semantic informativeness at this embedding quality is unproven.

**C2 (calibration band): FALSIFIED as "narrow".** The canonical
hash/Halton embedder (unit S⁷) produces magnitudes spanning 0.0–3.22
across the 50 items — a WIDE null band, setting a hard normalisation bar
for any future semantic-separation claim. R-009's exact zero is fully
explained and is itself a double finding: (i) `embed_into` reduces hashes
mod 1024 (`embed_table_rows`), so u1/u3 collided to the same row (549) —
the canonical embedder resolves at most 1024 distinct utterances, a
load-bearing limitation for the O-CSSM line (expected ≈ 11 collisions
over 150 turns); (ii) the resulting [a,b,a] = 0 is a native confirmation
of the octonion flexibility identity, appearing in production data.

**Instrumentation incidents (all resolved, all documented):**
1. The single 561-line driver (300+ vars in main) printed 2.5e17 for
   A-005; the same item in isolated/chunked drivers gives 0.00025269.
   Repro preserved as a suspected Madaros large-main miscompilation —
   reported to the compiler lane (see commit message and coord bus).
2. A python port of oct_mul matched the basis-element ground truth
   (‖[e1,e2,e4]‖²=4) but diverges from the canonical oct_mul on generic
   vectors; basis-level ground truth is necessary but not sufficient.
   All reported magnitudes are Sounio-native.
3. sentence-transformers all-MiniLM-L6-v2 installed into the repo .venv
   for this measurement (declared; not a project dependency).
