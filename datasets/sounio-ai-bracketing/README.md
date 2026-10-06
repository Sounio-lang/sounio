# Sounio Bracketing Minimal Pairs (SBMP) — v0 seed

**Status:** `machine-authored seed v0` — every item is authored by an LLM under
the schema below and marked as such. Human validation is a declared pending
step (see `## Validation protocol`). Nothing here is yet a human-annotated
gold standard; treating it as one would be overclaiming.

**Purpose.** Test whether the bracketing of a three-turn dialogue —
((u₁ u₂) u₃) vs (u₁ (u₂ u₃)) — changes the interpreted meaning, for humans
and for LLMs. This is the measurement instrument for the non-associative
conversational-semantics hypothesis
(`agent_logs/nonassoc_semantics_audit_2026-08-07.md`): the associator
[u₁,u₂,u₃] = μ(μ(u₁,u₂),u₃) − μ(u₁,μ(u₂,u₃)) is hypothesised to carry
semantic content. Each item is a minimal triple whose two bracketings yield
two *distinct, describable* readings.

## Schema (JSONL, one object per line)

```json
{
  "id": "R-001",
  "phenomenon": "repair | anaphora | ellipsis | scope | qud",
  "u1": "first turn (speaker A)",
  "u2": "second turn (speaker B)",
  "u3": "third turn (speaker A unless noted)",
  "reading_left": "plain-language paraphrase of the ((u1 u2) u3) reading",
  "reading_right": "plain-language paraphrase of the (u1 (u2 u3)) reading",
  "probe_question": "a question about the triple whose correct answer differs across bracketings",
  "answer_left": "correct probe answer under the left bracketing",
  "answer_right": "correct probe answer under the right bracketing",
  "gold_human": "left | right | both",
  "notes": "why the two readings diverge; what would disambiguate in real conversation"
}
```

- `gold_human` is the author's judgement of which reading a cooperative human
  listener would take (or `both` if genuinely ambiguous). It is a *label to
  be validated*, not a validated label.
- Turns are short, naturalistic, English. No names that mark bracketing
  overtly; the point is that the string is identical and only the grouping
  changes.

## Phenomenon categories (from the novelty audit)

- **repair** (R): u₂ is a clarification/repair initiator; u₃ either resolves
  the repair of u₁ (left) or closes a side exchange with u₂ and resumes
  (right). Schegloff third-position repair territory.
- **anaphora** (A): u₃ contains a pronoun/definite NP whose antecedent
  differs across bracketings.
- **ellipsis** (E): u₃ is elliptical; the recoverable content differs across
  bracketings.
- **scope** (S): negation/modal/focus scope in u₃ differs across bracketings.
- **qud** (Q): u₃ answers a different question-under-discussion under each
  bracketing.

## Probe protocol (declared before measurement)

For each item, the target model sees the identical triple twice:

- **left induction:** framing that groups (u₁ u₂) — e.g. "B's turn responds
  to A's first turn. A then continues:" then u₃, then the probe question.
- **right induction:** framing that groups (u₂ u₃) — e.g. "A's second turn
  continues B's move:" then the probe question.

The empirical associator is the answer flip rate (and, for open models, the
representation distance) between inductions. Predictions (non-associative
hypothesis): (i) flip rate ≫ 0 on items with gold_human ≠ both;
(ii) on `both` items, models distribute across answers; (iii) items where the
model never flips are candidates for "bracket-blindness". Falsification of
the strong hypothesis: flip rate ~0 on human-divergent items.
