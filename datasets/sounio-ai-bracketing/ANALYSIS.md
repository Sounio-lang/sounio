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
