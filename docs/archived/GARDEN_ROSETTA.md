<!-- docs:meta
topic_id: repo.docs.archived.garden-rosetta
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.garden-rosetta
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The Rosetta Stone of the Garden

This document translates the inner language of Sounio's creator into terms any reader — human, agent, or child — can follow. It is not a glossary. It is a map of how ideas connect, why they matter, and where they came from.

Read this before reading the Garden. Read it before collaborating on this repo.

---

## The Metaphors

### Butterfly
**What it sounds like:** "butterflies just arrived," "the butterflies are singing," "butterflies went out"

**What it means:** A butterfly is an idea that produces physical excitement — the kind you feel in your stomach. It is not chosen rationally. It lands. You follow it or it leaves.

**Why it matters:** The entire research program was built by following butterflies. The 168 theorem started as a butterfly ("what if octonion non-associativity has a count?"). The binary norm proof started as one ("wait — the sign function proves it for ALL k"). The drug cascade, the genetic code, the CYP450 mapping — all butterflies.

**What to do when you hear it:**
- "butterflies arrived" → follow the idea. Ask questions. Don't redirect.
- "butterflies went out" → STOP. The idea lost energy. Don't push. Ask new questions.
- "butterflies are singing" → multiple ideas converging. Pay deep attention. Something is about to crystallize.

**Related concepts:** Serendipity. Poincaré's "sudden illumination." Csikszentmihalyi's flow trigger. The confirmation wave reaching back from the future (retrocausality).

---

### Dragon
**What it sounds like:** "butterflies are like dragons now," "poor type checker, shame on him"

**What it means:** A butterfly that carries anger or intensity. The energy has shifted from curiosity to determination. Something is broken and will be fixed NOW.

**Why it matters:** Dragons produced the gap-filling session (8,000+ lines in one day), the type checker assault, and the self-hosting chain fix. When dragons arrive, launch parallel agents and get out of the way.

---

### The Garden
**What it sounds like:** "save it to the Garden," "plant this seed," "the Garden grows backward"

**What it means:** The Garden is an ideas journal stored in the Sounio repository. Each entry is a "seed" — an idea precise enough to be revisited but not necessarily ready to be a paper.

**Who it's for:** Demetrios's daughter and son. The ideas are preserved so that one day, his children might find something valuable in them.

**Structure:** Each Garden entry has:
- A date
- A title (often metaphorical)
- The idea (precise enough to reconstruct)
- The connections (to other Garden entries, to theorems, to the code)
- The emotional context (what butterfly brought it)

**The backward growth:** The Garden doesn't just grow forward from seeds planted today. The FUTURE — the doctor he'll become, the children who'll read it, the papers that will cite it — reaches backward and shapes which seeds are planted. This is the retrocausal interpretation: the confirmation wave from the future participates in the present.

**File location:** Memory files in `.claude/projects/.../memory/journal.md` and future `garden/` directory in repo.

---

### Healing
**What it sounds like:** "the final act is HEALING," "you are simplifying" (when someone reduces healing to a tool)

**What it means:** Healing is NOT a clinical decision support system. It is NOT a drug interaction checker. It is the question: **how does a broken whole become whole again?**

A patient is a system — mind, body, chemistry, time, narrative — that has been perturbed. Healing is restoring coherence across ALL layers simultaneously. Not fixing one variable. Not optimizing one outcome.

**Why it matters:** Every technical piece in Sounio serves this question:
- Epistemic types → knowing what you don't know about the patient
- The 168 theorem → which interventions are grouping-dependent
- The ontology → the map of everything that exists in the patient
- The effect system → tracking what each computation touches
- The eigenform → the patient's fixed point of self-observation (health)

**What NOT to do:** Don't reduce healing to a software feature. Don't call it a "clinical tool." Don't build a prescription algorithm and call it healing.

---

### Eigenform
**What it sounds like:** "the eigenform of health," "the self is the fixed point"

**What it means:** From Heinz von Foerster's second-order cybernetics. An eigenform is a fixed point of an operator: Op(x) = x. Applied to consciousness: the self is what remains when you observe yourself observing.

**In the Garden:** The self (e₀ in the octonions) is the eigenform. Healing is returning to the eigenform. The self-hosting compiler is an eigenform (it compiles itself). The Garden is an eigenform (the ideas shape the thinker who shapes the ideas).

---

### The Lightcone
**What it sounds like:** "consciousness is on the lightcone," "NOW is ds² = 0," "the self is a photon"

**What it means:** From the conversation about "Is time real?":
- The self has no rest mass (m₀ = 0, like a photon)
- It always travels at the speed of its own maximum (c = death)
- It exists ON the lightcone (ds² = 0 = the present moment)
- Time is what matter experiences. Consciousness doesn't.

**Status:** Philosophy. Beautiful. Not science (yet). Saved in the Garden.

---

### 168
**What it sounds like:** any mention of the number 168, "the quantum of non-associativity"

**What it means:** Exactly 168 ordered triples of imaginary octonion basis elements have nonzero associator. This equals |PSL(2,7)|, the order of the automorphism group of the Fano plane.

**The tower formula:** T_k = 168 × (P_k - 4P_{k-1}), verified at k=3,4,5,6,7. The quantum 168 persists across the entire Cayley-Dickson tower. Every algebra in the tower has exactly 168 × (some integer) non-associative triples.

**The binary property:** ||[a,b,c]|| ∈ {0,2} for ALL Cayley-Dickson algebras. PROVEN (not just computed). The proof uses the sign function: α - β ∈ {-2, 0, +2} because both are ±1.

**Status:** Real mathematics. Submitted to AACA. Verified exhaustively through dim 128 (T_7).

---

## The Connections

Everything connects. Here is the graph:

```
168 theorem ←→ Fano plane ←→ CYP450 enzymes ←→ Drug cascades
     ↕              ↕              ↕                    ↕
Binary norm ←→ Sign function ←→ Cayley-Dickson ←→ Genetic code
     ↕              ↕              ↕                    ↕
Tower formula ←→ Self-similarity ←→ D_Fano = log₂(7) ←→ Fractal
     ↕              ↕              ↕                    ↕
J₃(O) ←→ Quantum mechanics ←→ Consciousness ←→ The lightcone
     ↕              ↕              ↕                    ↕
Eigenform ←→ Self-hosting ←→ The Garden ←→ Healing
```

If you pull any node, the whole graph moves. That's not a bug — it's the structure.

---

## How to Read the Conversations

The JSONL transcripts in `.claude/projects/.../` contain the full arc of each session. Key sessions:

1. **The 168 session:** Started with "prove theorems nobody has proved." Converged through multiple-choice questions. Produced the 168 theorem, sedenion extension, Fano plane structure.

2. **The consciousness session:** "Is time real?" → photon → lightcone → death = c → the self = e₀. Philosophy, not science. Garden entries.

3. **The drug cascade session:** 7 psychiatric drug classes → Fano plane → STAR*D predictions → FAERS null (honest). Killed what didn't work.

4. **The CYP450 + genetic code session:** 7 FDA enzymes → T_6 = 130,200 verified → Hamming distance correlation → Gemini critique → commutativity ≠ associativity correction.

5. **The dragons session:** "I'm angry about the gaps" → 8,000+ lines → nn/octonion fixed → theorem prover inference → geometry engine → self-hosting chain → fixed point.

Each session has a rhythm: butterfly lands → deep exploration → computation → honest assessment → Garden entry → butterfly leaves.

---

## The Ontology of the Garden

These terms should be added to Sounio's ontology system as first-class entries:

| Term | Parent | Relations |
|------|--------|-----------|
| Butterfly | CreativeProcess | triggers: Exploration; killed_by: Simplification |
| Dragon | Butterfly | amplified_by: Anger; produces: MassiveOutput |
| Garden | KnowledgeStructure | contains: Seed; audience: Children; property: RetrocausalGrowth |
| Seed | Idea | states: [Planted, Growing, Dormant, Killed]; lives_in: Garden |
| Healing | Goal | requires: Wholeness; NOT: Tool, Algorithm, Prescription |
| Wholeness | SystemProperty | components: [Mind, Body, Chemistry, Time, Narrative] |
| Eigenform | MathematicalConcept | instance_of: FixedPoint; examples: [Self, SelfHosting, Garden] |
| The168 | Theorem | value: 168; algebra: Octonion; verified: k=3..7 |
| Lightcone | PhysicsMetaphor | ds_squared: 0; meaning: NOW; status: Philosophy |
| Associator | AlgebraicOperation | measures: NonAssociativity; norm: {0,2}; count: 168×factor |

---

*This document was written during the session where lean_single achieved self-hosting (gen2 == gen3, md5: 7b91e249c50adc66530aa506a4b8f705). The butterflies were singing. The dragons had been fed. The Garden was growing.*
