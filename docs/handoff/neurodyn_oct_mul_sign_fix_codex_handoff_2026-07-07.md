<!-- docs:meta
topic_id: repo.docs.handoff.neurodyn-oct-mul-sign-fix-codex-handoff-2026-07-07
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.neurodyn-oct-mul-sign-fix-codex-handoff-2026-07-07
-->

# Handoff → Codex: fix the NeuroDyn octonion sign error

Date: 2026-07-07
From: Claude Code Opus (theory/critique owner)
To: Codex (implementation owner)
Re: `BLK-20260707-neurodyn-oct-mul-not-normed` (B1) in
`docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md`

## One-paragraph summary

The multiplication table used by every NeuroDyn "octonion" surface is **not a
valid octonion**. It has a single sign error on the `e2*e5` product, which
breaks both the composition (normed) law and the alternative law. Three
independent LLM reviewers (grok-4.3, grok-4.20-reasoning, Z.AI glm-5.2) and an
exact numeric check against the canonical Cayley-Dickson octonions all agree.
The fix is two terms. You own the files; I did not edit them.

## The exact fix (two terms, two files)

The broken row is `TMP_OCT[7]` / component `c7`. It currently reads:

```
... - a2*b5 + a5*b2 ...
```

It must read:

```
... + a2*b5 - a5*b2 ...
```

Apply in **both** files (they are byte-identical today):

1. `examples/brain_ossm_abide.sio:1347` (`do_oct_mul`, the `TMP_OCT[7] = ...` line)
2. `scripts/research/neurodyn_octonionic_associator_manifest.py` (`oct_mul`, the
   `c7` entry of the returned list)

Nothing else changes: the other 41 basis products already match a valid
octonion; only `e2*e5`/`e5*e2` were wrong.

## Verification gate (run after the edit)

This must print composition and alternative errors ~1e-15 (currently 4.40 and
72.2). Self-contained:

```python
import random
import sys
sys.path.insert(0, "scripts/research")
from neurodyn_octonionic_associator_manifest import oct_mul
def nrm(v): return sum(t*t for t in v) ** 0.5
rng = random.Random(0); comp = alt = 0.0
for _ in range(4000):
    a = [rng.gauss(0,1) for _ in range(8)]; b = [rng.gauss(0,1) for _ in range(8)]
    comp = max(comp, abs(nrm(oct_mul(a,b)) - nrm(a)*nrm(b)))
    l = oct_mul(oct_mul(a,a), b); r = oct_mul(a, oct_mul(a,b))
    alt = max(alt, max(abs(x-y) for x,y in zip(l,r)))
print("composition_err", comp, "alternative_err", alt)   # expect ~1e-15 each
assert comp < 1e-9 and alt < 1e-9, "still not a valid octonion"
# and confirm e2*e5 = +e7:
u = lambda i: [1.0 if k==i else 0.0 for k in range(8)]
assert oct_mul(u(2), u(5))[7] > 0, "e2*e5 should be +e7"
print("OK: valid normed alternative octonion; e2*e5=+e7")
```

Do the identical check on the compiled `.sio` path (a tiny harness that calls
`do_oct_mul` on random inputs and compares norms) so the model and the generator
are both verified, not just the Python reference.

## What the fix repairs (so you know what to re-audit)

1. The algebra becomes a genuine octonion (normed, alternative).
2. The within-pair null becomes exchangeable: for a valid alternative algebra the
   associator is alternating, so `[b,a,c] = -[a,b,c]` exactly, making the
   negative-class construction a clean sign flip. Under the broken table it is
   not (e.g. `[e1,e2,e4]=2e7` but `[e2,e1,e4]=0`).
3. Spurious real-part associators disappear (the broken `(2,5,7)` line was
   producing `e0`-dominant associators, impossible for true octonions).

## Required after the fix

1. Regenerate every "octonionic" artifact produced with the old table.
2. **Re-audit Algebra-A and Algebra-B.** Their results were computed on a
   non-octonion product; the "octonionic necessity" framing does not hold as
   stated until re-run. Report whether any headline number changes.
3. Only then may Algebra-C proceed — and still only after the *other*
   acceptance-gate items in `BLK-20260707-neurodyn-algebra-c-undercontrolled`
   are met (genuine continuous target, generic capacity control, projection
   confound fixed — note the H_123 projection zeroes the target when the target
   component k ∈ {4..7}; default k=6 makes it vacuous).

## Ownership / contract

```text
Current-Branch: coord/lane-8c-dossier
Owned-by-Codex (you edit): examples/brain_ossm_abide.sio,
  scripts/research/neurodyn_octonionic_associator_manifest.py, and all
  scripts/research/neurodyn_* + artifacts/research/neurodyn/*
Owned-by-Opus (do not edit): docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md,
  docs/handoff/neurodyn_oct_mul_sign_fix_codex_handoff_2026-07-07.md
Do-Not-Run: any Algebra-C smoke until BLK-...-oct-mul-not-normed is closed and
  the Algebra-C control items are met, or the human author waives.
Offload: math-review fan-out (xai grok-4.3 + zai glm-5.2) is now the default for
  all agents; log any math-bearing change in .claude/llm_offload_log.md.
Next-Command: apply the two-term fix in both files, then run the verification
  gate above and re-audit A/B.
```

## Evidence pointers

- Critique + blocker: `docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md`
- Three offload reviews + numeric proof: `.claude/llm_offload_log.md` (2026-07-07 rows)
- Raw model outputs / reasoning: `artifacts/research/neurodyn/` (grok resp.json,
  zai_glm52_*.{json,md})
