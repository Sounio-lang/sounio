# Non-Associative Conversational Semantics — novelty audit, 7-agent swarm, 2026-08-07

Hypothesis under audit: semantic composition in conversation is
non-associative — the associator [u1,u2,u3] = μ(μ(u1,u2),u3) − μ(u1,μ(u2,u3))
carries semantic content (which bracketing was meant). Verdict below.

## VERDICT: the hypothesis SURVIVES on all seven fronts — the cell is empty

Nobody, in any indexed literature, has:
1. defined the associator as a semantically interpreted operator over
   dialogue turns;
2. built controlled minimal pairs of discourse bracketing with divergent
   readings and measured humans or models on them;
3. connected non-associative algebras (octonions/Malcev/alternative) to
   dialogue or language semantics (zero hits, multiple independent searches);
4. tested bracket-sensitivity of LLMs at the turn level (the discourse
   analogue of constituent tests — Cao/Kitaev/Klein EMNLP 2020 — never done
   for dialogue);
5. given any non-associative identity (alternativity, flexibility, Moufang)
   a linguistic reading.

## What each front found

- **Lambek NL/CNL/LG/ACG**: 60 years of non-associativity kept strictly on
  the syntax/type side; Curry-Howard always lands in associative λ-calculus.
  Retoré–Salvati 2010 (NL encodable in ACG) means non-associativity is
  always *simulable as data* in an associative metalanguage — the paper must
  argue simulability ≠ algebraic identity of the semantics.
- **DRT/DPL/update semantics**: associativity is a trivial theorem
  (function/relation composition); bracketing is invisible. SDRT/RST:
  bracketing exists as representational input (attachment graph), never as
  the output of an algebra; nobody asks whether the update algebra is
  alternative. Note: "semantics is not associative" as a phrase is taken —
  Di Lavore et al. 2025 — must be cited and distinguished (theirs is not
  dialogue-turn composition).
- **Quantum cognition**: order effects (commutator) established
  (Busemeyer–Bruza); always binary, never triples; nobody imported the
  non-associative sequential product of effects (Gudder–Greechie) into
  cognition.
- **Hypercomplex NNs**: quaternion/octonion nets exist (signal processing);
  gains attributed to parameter efficiency, non-associativity tolerated but
  never measured; PHM (Zhang 2021) may learn non-associative rules
  implicitly — the reply is an explicit, measurable associator.
- **Psycholinguistics**: garden-path (Frazier) is sentential; third-position
  repair (Schegloff) documented but never experimentally controlled by
  bracketing; no grouping-sensitivity benchmark exists (Winograd variants
  test world knowledge; STAC treats attachment ambiguity as annotator noise).
- **LLMs**: causal transformers are fold-left — structurally unable to
  represent μ(u1,μ(u2,u3)); "attention is associative" is imprecise as
  stated (softmax attention is not a binary operator) — the experiment must
  define μ operationally (hierarchical encoder with two bracketings).

## The three battles the paper must fight explicitly

1. "This is rebranded NL calculus" — no: in NL the semantics is associative;
   here the non-associativity is of the semantic composition itself, with
   syntactic order fixed.
2. "SDRT already knows bracketing matters" — yes as external pragmatic
   parameter; the novelty is internalizing it as failure of associativity
   with the associator as the studied object.
3. "PHM/tree-RNNs already do this implicitly" — implicit, unmeasured, and
   without identities; the contribution is explicit quantity + testable laws.

## The publishable shape (all seven agents converge)

Contribution = formalization + measurement, NOT the conceptual claim.

- Formalization: μ over dialogue-turn meanings with the associator as the
  studied object; alternativity ([a,b,c] = −[b,a,c]) as a testable
  structural prediction — swapping the first two turns flips the sign of
  the grouping effect. Canonical hand-worked case: u1 assertion, u2
  clarification request, u3 answer — the two bracketings differ in common
  ground and QUD stack (anchorable in STAC/KoS).
- Experiment: 200–500 dialogue triples with controlled ambiguous bracketing
  (repair, side sequences), human-annotated; induce both bracketings in
  LLMs (subdialogue summarization / hierarchical encoder); measure (i) the
  empirical associator norm, (ii) its correlation with human judged
  meaning difference, (iii) model accuracy on the human reading under each
  induced bracketing. Falsification: associator ~0 or uncorrelated → strong
  hypothesis dies, but "current LLMs are bracket-blind at turn level" is
  still a publishable architectural diagnosis.

## Mandatory citations (from the swarm)

Lambek 1961; Kandulski 1988; Aarts & Trautwein 1995; Moot & Retoré 2012;
de Groote & Lamarche 2002 (CNL); Retoré & Salvati 2010 (ACG encoding);
Došen 1988/89 (groupoid models); Asher & Lascarides 2003/2013 (SDRT);
Kamp (DRT); Roberts 2012 (QUD); Ginzburg 2012 (KoS); Schegloff et al. 1977
(repair); Busemeyer & Bruza 2012 (quantum cognition); Di Lavore et al. 2025
(title collision — distinguish); Sedlár 2019 (substructural, distinguish);
Parcollet et al. (quaternion RNNs); Zhang et al. 2021 (PHM); Wu et al. 2020
(octonion nets); Popa 2016 / Kuroe & Iima 2016 (octonion NNs); Socher 2013
(RNTN); Tai 2015 (Tree-LSTM); Shen 2019 (Ordered Neurons); Drozdov 2019
(DIORA); Serban 2016 (HRED); Davis & van Schijndel 2020 (CoNLL, closest
neighbor); Huber & Carenini 2022; Koto/Lau/Baldwin 2021; Liu et al. 2024
(lost-in-the-middle); Hewitt & Manning 2019 (structural probes);
Dao 2022 (FlashAttention) / Katharopoulos 2020 (linear attention) for the
associative-aggregation architectural contrast.

## Verification gaps

- IEEE/linguistics venues (Dialogue & Discourse, SIGDIAL 2025–2026) were
  quota-limited; one more pass before submission.
- Di Lavore et al. 2025 exact scope to confirm before writing related work.
- de Marrais fringe octonion-language work: zero-divisor graphs, not
  associators — distinguish in one line.
