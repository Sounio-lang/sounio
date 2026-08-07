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
