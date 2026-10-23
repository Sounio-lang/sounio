# Multi-ontology EL+ offload log — round 13 (2026-08-05)

Lane: `kimi-cli2/elplus-scale-multi-20260805`. Canonical log
`.claude/llm_offload_log.md` is outside this lane's write set (and was
claim-contended in earlier rounds — same pattern as rounds 11/12, which
logged to `agent_logs/`).

## Review 1 — math-review of the round-13 claims

- Command: `bin/llm-offload -t math-review -p xai -i
  artifacts/ontology-frontiers/multi-ontology/MATH_REVIEW_INPUT.md`
- Provider: xai (Grok grok-4.5), 2026-08-05 ~11:55 UTC.
- Input: `artifacts/ontology-frontiers/multi-ontology/MATH_REVIEW_INPUT.md`
  (5 claims: cone partition + atomic decomposition; grouped conflict
  counter + full-GO recomputation; role-edge deficit framing; profile
  theorem coverage for CL/UBERON; NEPW word-generalized masks).

### Outcome: 4 claims [OK], 2 corrections

1. **[OK] Claim 1** (cone partition ⇒ atomic-edge decomposition
   298,203+23,943+73,793 = 395,939): arithmetic exact.
2. **[WRONG→fixed] Claim 2 typo**: my prose wrote intra-cone conflicts
   29,770,678; correct is 29,770,768 (21,144,668+8,621,578+4,522).
   Fixed in RESULTS.md §3. The grouped-counter formula itself: [OK]
   "faithful".
3. **[OK] Claim 3** (role-edge deficit 503,092 from 3,603 measured
   cross-cone restrictions; framed as measured, not identity).
4. **[WRONG→reframed] Claim 4**: "superclass-side restrictions cannot
   change atom-level statistics" is FALSE in the 8-rule system —
   transitivity through an existential node (`A ⊑ ∃r.F`, `∃r.F ⊑ B` ⇒
   `A ⊑ B`) yields atom-atom subsumption. Minimal correction adopted:
   the profile theorem REQUIRES 0 such axioms; CL/UBERON have 1 each
   (the same `∃RO:0000053.PATO:0010006 ⊑ CL:0000000`), which is provably
   inert under the namespace-only extraction (PATO filler never
   interned; CL:0000000 out-of-namespace for UBERON), so the reported
   numbers remain exact for the extracted TBox, with completeness vs
   full OWL semantics documented as a limitation. Fixed in RESULTS.md
   §4 item 4 and §8, and in the gen_multi_data.py docstring. The
   equivalentClass restrictions (2 CL / 15 UBERON) were probed to be
   anonymous/nested (no named onProperty+someValuesFrom), hence never
   extractable as simple existentials regardless.
5. **[OK] Claim 5** (NEPW word-index math preserves pm/epm semantics;
   NEPW=26 covers UBERON's 822 endpoints).

No commit trailer needed (no commit made by this lane; changes left
uncommitted for review per task instructions).
