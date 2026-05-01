<!-- docs:meta
topic_id: repo.docs.research.octonion-conversation-literature-review-framework
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.octonion-conversation-literature-review-framework
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Octonion Conversational State Model — Literature Review Framework

Snapshot date: 2026-04-23

## Purpose

This document defines the literature-review framework for the conversational extension of the O-SSM line.

The goal is not to prove novelty by intuition. The goal is to separate:
- what already exists in adjacent literatures
- what is strongly adjacent but not identical
- what appears genuinely new and should be framed as a provisional novelty claim

This review is specifically for the idea of an **octonion or sedenion conversational state model** that combines:
- temporally structured conversation dynamics
- affective or emotional state representation
- non-associative composition
- optional physiological inputs such as HRV
- possible links to psychopathology-sensitive state trajectories

## Working thesis

The core idea is not “conversation is quantum.”

The core idea is:
- conversation has structured temporality
- meaning can remain coupled across nonlocal turns
- ordinary associative vector composition may be too weak to represent that structure
- non-associative algebra may provide a mathematically grounded way to encode path dependence, cross-time coupling, and controlled instability

The literature review should therefore test five claims separately:

1. Do current models already represent emotion or affect as latent directions in LLMs?
2. Do conversational emotion models already track temporality and multi-party state well?
3. Do hypercomplex neural networks already provide sequence-model precedents?
4. Has anyone already combined octonions with conversation or dialogue?
5. Is there serious evidence that physiological signals like HRV can anchor psychopathology-sensitive state modeling?

## Provisional novelty map

Based on an initial pass, the following look **established or adjacent**, not novel:
- internal emotion-related latent directions in LLMs
- controllable emotion steering in LLMs
- emotion recognition in conversation with speaker-state tracking
- quaternion recurrent models for sequential signals
- octonion neural networks for vision, time series, and control
- HRV as a candidate biomarker or moderator in several psychopathology domains
- multimodal dialogue systems that incorporate emotion recognition and, in some cases, physiological inputs

The following look **plausibly novel**, pending deeper review:
- an octonion **conversational state-space model** rather than a generic octonion network
- use of the **associator** as a dialogue-level telemetry or coherence signal
- use of **Fano-line subalgebras** as structured conversational modes
- use of **sedenion zero-divisor proximity** as a controlled forgetting or instability signal
- an integrated model linking **conversation + affect + non-associativity + physiological markers + psychopathology-sensitive dynamics**

These should still be treated as provisional until the review is complete.

## Search axes

### Axis 1 — Emotion representations inside LLMs

Question:
Do current LLMs already contain steerable or interpretable affective latent directions, and how local or persistent are they?

Seed sources:
- Anthropic, “Emotion concepts and their function in a large language model” (2026)
- Dong et al., “From Rational Answers to Emotional Resonance” / emotion vectors (arXiv:2502.04075, 2025)
- Abdurahman et al., “Large Language Models are Highly Aligned with Human Ratings of Emotional Stimuli” (2025)

Why this matters:
- This tells us whether your idea starts from an already-real phenomenon inside language models.
- Anthropic’s result is especially important because it suggests emotion representations are **functional** and behavior-shaping, but also largely **local** rather than stable persistent states.

What to extract:
- whether emotion directions are persistent or local
- whether they are causal or merely correlational
- whether emotion steering preserves semantic content
- whether existing work already proposes temporally persistent affect-state carriers

Suggested search strings:
- `"emotion concepts" "large language model" Anthropic`
- `"emotion vectors" LLM controllable emotion generation`
- `LLM emotional alignment latent representation affect steering`

### Axis 2 — Emotion in conversation

Question:
How do current conversation models track emotional state across turns, speakers, and context windows?

Seed sources:
- Majumder et al., “DialogueRNN” (arXiv:1811.00405, 2018)
- DialogueGCN (2019)
- DialogueEIN / related ERC work
- 2025 systematic review on AI-based multimodal dialogue systems with emotion recognition

Why this matters:
- This is the main adjacent precedent for “conversation + emotion + temporality.”
- Most of this literature models emotional evolution with RNNs, graph networks, or attention, but not with non-associative algebra.

What to extract:
- how speaker state is modeled
- whether emotions are local labels or latent trajectories
- whether models support dialogue generation or only recognition
- whether physiology is integrated or only text/audio/vision

Suggested search strings:
- `"emotion recognition in conversation" speaker state model`
- `"emotional dialogue generation" context emotion flow`
- `multimodal dialogue emotion physiological review`

### Axis 3 — Hypercomplex sequence models

Question:
What is already known about quaternion and octonion models for sequential or affective signals?

Seed sources:
- Parcollet et al., “Quaternion Recurrent Neural Networks” (arXiv:1806.04418, 2018)
- Parcollet et al., bidirectional quaternion LSTM (2018)
- Berrouiguet et al., “Learning Speech Emotion Representations in the Quaternion Domain” (arXiv:2204.02385, 2022)
- Wu et al., “Deep Octonion Networks” (arXiv:1903.08478, 2019)
- octonion-based echo-state / speech-emotion work

Why this matters:
- This is the strongest direct precedent family for “hypercomplex representations help with structured correlated inputs.”
- If quaternion recurrent and speech-emotion models already work, then your thesis is not coming from nowhere.
- The key distinction is that quaternion models remain associative; your octonion line adds path dependence.

What to extract:
- what advantage hypercomplex structure gave
- whether the gains came from compactness, inductive bias, or better coupling
- whether any work uses octonions for dialogue or temporally persistent affective state
- whether any paper formalizes non-associativity as a feature rather than a nuisance

Suggested search strings:
- `quaternion recurrent neural network sequence modeling`
- `speech emotion quaternion neural network`
- `octonion neural network time series`
- `octonion dialogue emotion neural network`

### Axis 4 — Octonion dynamical systems and stability

Question:
What mathematical control, stability, or delayed-dynamics results already exist for octonion-valued systems?

Seed sources:
- Wang and Liu, “Global μ-stability and finite-time control of octonion-valued neural networks with unbounded delays” (arXiv:2003.11330, 2020)
- IEEE work on metacognitive octonion-valued neural networks and time series

Why this matters:
- This is where we look for rigorous tools, not application demos.
- Even if those papers decompose octonion systems into real-valued subproblems, they may still provide reusable stability language.

What to extract:
- Lyapunov-style conditions
- delay handling
- decomposition tricks
- whether non-associativity is preserved or merely bypassed during analysis

Suggested search strings:
- `octonion-valued neural networks stability delay`
- `octonion time series neural networks`
- `octonion dynamical systems control`

### Axis 5 — State-space models for language and short dialogue

Question:
What is already known about state-space models in language-like or short-context settings, and where do they fail?

Seed sources:
- S4 / Mamba / H3 and their language-modeling follow-ons
- dialogue and conversational sequence modeling work using RNNs or SSM-like recurrence

Why this matters:
- Your idea is not merely “use octonions.”
- It is “use a state-space recurrence whose algebraic composition preserves conversational temporality better than an associative baseline.”

What to extract:
- whether SSMs have been pushed into conversational settings
- where associative scan is an advantage
- where associative compression may discard structure relevant to affective dialogue

Suggested search strings:
- `state space model dialogue language`
- `SSM conversation modeling emotion`
- `structured state space model short dialogue`

### Axis 6 — Physiology, HRV, and psychopathology

Question:
Can physiological signals such as HRV support an affective conversational state model in a clinically serious way?

Seed sources:
- Review: “The Predictive Potential of Heart Rate Variability for Depression” (2024)
- Review: “HRV as a biobehavioral marker of diverse psychopathologies” (2021)
- JMIR Mental Health 2025 longitudinal smartwatch-derived digital phenotypes in psychotic spectrum disorders
- PTSD / emotion-regulation and HRV work

Why this matters:
- This is the bridge from “affective dialogue” to “psychopathology-sensitive dynamics.”
- The literature is promising but heterogeneous, which means we should use it as a grounding signal, not a magic proxy.

What to extract:
- which HRV metrics are most defensible
- whether findings are transdiagnostic or symptom-cluster specific
- whether the evidence is predictive, concurrent, or merely correlational
- what temporal resolution is clinically meaningful

Suggested search strings:
- `"heart rate variability" depression review`
- `"heart rate variability" psychopathology transdiagnostic review`
- `smartwatch digital phenotyping psychosis HRV longitudinal`
- `emotion regulation HRV clinical trial`

### Axis 7 — Multimodal affective dialogue with physiological inputs

Question:
How close is existing work to “conversation + emotion + physiology” already?

Seed sources:
- 2025 multimodal dialogue systematic review
- recent emotional chatbot / multimodal affective agent papers

Why this matters:
- This axis is where novelty can disappear if we are not careful.
- Many systems already combine text, audio, facial signals, or biosignals with dialogue.
- The likely novelty is not multimodality by itself, but the **non-associative state geometry**.

What to extract:
- whether physiology is only used as an input feature
- whether dialogue state is truly dynamical or just classifier-conditioned
- whether any model links biosignals to interpretable latent trajectories

Suggested search strings:
- `multimodal emotional dialogue physiological signals`
- `affective dialogue system biosignals`
- `chatbot emotion physiology wearable`

## Inclusion criteria

Include papers if they meet at least one of these:
- directly study emotion or affect in LLMs
- directly study emotion recognition or generation in conversation
- use quaternion or octonion networks for sequential, recurrent, or affective tasks
- study octonion-valued dynamics, delays, or stability
- use physiological markers, especially HRV, in emotion regulation or psychopathology modeling
- provide systematic reviews that map the surrounding field

## Exclusion criteria

Exclude or demote papers if they are:
- generic sentiment-analysis papers without temporal conversational state
- hypercomplex papers limited to image classification unless they provide reusable representation arguments
- “quantum cognition” or “entanglement” papers that are metaphorical but not mathematically instantiated
- consumer-wellness HRV pieces without validated psychopathology or regulatory grounding
- chatbot papers that use “emotion” only as prompt style rather than a modeled latent or measured signal

## Claim taxonomy

Use this taxonomy while reviewing papers and while writing future briefs.

### A. Established

Use only when the literature clearly supports it.

Examples:
- LLMs can exhibit interpretable emotion-related internal directions.
- Conversation emotion models often maintain speaker-specific or context-specific state.
- Quaternion recurrent models exist for sequential signals.
- Octonion neural network literature exists.
- HRV has meaningful but heterogeneous links to emotion regulation and psychopathology.

### B. Adjacent precedent

Use when the literature is close but not identical.

Examples:
- octonion models for time series or speech emotion
- multimodal dialogue systems with physiology
- digital phenotyping for psychosis using smartwatch data

### C. Speculative extension

Use when the literature does not yet establish the exact claim.

Examples:
- octonion conversational state-space modeling
- associator-based hallucination telemetry
- Fano-line personality modes
- sedenion zero-divisor forgetting gates
- psychopathology-sensitive dialogue geometry derived from hypercomplex temporal coupling

## Review matrix template

For each paper, record:

| Field | Notes |
|---|---|
| Citation | Title, authors, year, URL |
| Axis | 1-7 |
| Domain | LLM / dialogue / hypercomplex / physiology / psychiatry |
| Representation | real / complex / quaternion / octonion / other |
| Temporal model | feedforward / RNN / graph / SSM / hybrid |
| Conversation-specific | yes / no |
| Emotion-specific | yes / no |
| Physiology-specific | yes / no |
| Psychopathology-specific | yes / no |
| Strongest useful takeaway | one sentence |
| Threat to novelty | low / medium / high |
| Relevance to Tapestry | low / medium / high |

## Immediate conclusions from the first pass

1. The strongest starting point for your idea is **not** “nobody has done emotion in AI.”
   It is:
   - emotion is already latent and steerable in LLMs
   - conversation emotion models already track contextual state
   - hypercomplex recurrent models already help with structured correlations
   - octonion systems already exist outside conversation

2. The likely novelty is the **combination**:
   - non-associative recurrence
   - conversational temporality
   - affective state geometry
   - optional physiological anchoring
   - psychopathology-sensitive trajectories

3. Anthropic’s 2026 result is important because it cuts both ways:
   - good news: emotion-like latent structure is real and behaviorally relevant
   - caution: those vectors are reported as mostly local, not stable long-horizon emotional selves

4. HRV is promising but should not be overclaimed.
   The literature supports it as a useful candidate signal, not a universal or diagnosis-complete biomarker.

## Recommended review order

1. Anthropic emotion-concept paper
2. controllable emotion vector papers
3. ERC / emotional dialogue papers
4. quaternion recurrent and speech-emotion papers
5. octonion network and dynamical-systems papers
6. HRV and digital-phenotyping reviews
7. only then write a formal novelty statement

## Suggested multi-model workflow

Use the models differently.

- **Claude**
  Use for literature skepticism, theorem pressure, and foundation discipline.

- **Codex**
  Use for synthesis into repo-grounded architecture, experiment plans, and implementation consequences.

- **Kimi**
  Use for ideation, variant generation, and broad exploratory scans, but verify claims carefully before they shape the thesis.

## Deliverables for the next pass

1. A 20-40 paper annotated bibliography using the matrix above.
2. A novelty table with three columns:
   - already established
   - adjacent but not identical
   - likely new
3. A claim-safe abstract for the conversational project.
4. A benchmark proposal that only uses claims surviving the review.

## Seed bibliography

- Anthropic. “Emotion concepts and their function in a large language model.” 2026.
  https://www.anthropic.com/research/emotion-concepts-function
- Dong et al. “From Rational Answers to Emotional Resonance: The Role of Controllable Emotion Generation in Language Models.” arXiv:2502.04075, 2025.
  https://arxiv.org/abs/2502.04075
- Majumder et al. “DialogueRNN: An Attentive RNN for Emotion Detection in Conversations.” arXiv:1811.00405, 2018.
  https://arxiv.org/abs/1811.00405
- “A Systematic Review on Artificial Intelligence-Based Multimodal Dialogue Systems Capable of Emotion Recognition.” 2025.
  https://www.mdpi.com/2414-4088/9/3/28
- Parcollet et al. “Quaternion Recurrent Neural Networks.” arXiv:1806.04418, 2018.
  https://arxiv.org/abs/1806.04418
- Wu et al. “Deep Octonion Networks.” arXiv:1903.08478, 2019.
  https://arxiv.org/abs/1903.08478
- Berrouiguet et al. “Learning Speech Emotion Representations in the Quaternion Domain.” arXiv:2204.02385, 2022.
  https://arxiv.org/abs/2204.02385
- Wang and Liu. “Global μ-stability and finite-time control of octonion-valued neural networks with unbounded delays.” arXiv:2003.11330, 2020.
  https://arxiv.org/abs/2003.11330
- Vicentini et al. “Smartwatch-Derived Digital Phenotypes Relate to Psychopathology Dimensions in Patients With Psychotic Spectrum Disorders.” JMIR Mental Health, 2025.
  https://mental.jmir.org/2025/1/e75774
- “The Predictive Potential of Heart Rate Variability for Depression.” Neuroscience, 2024.
  https://www.sciencedirect.com/science/article/pii/S030645222400126X
- Heiss et al. “Heart rate variability as a biobehavioral marker of diverse psychopathologies: A review and argument for an ideal range.” Neuroscience & Biobehavioral Reviews, 2021.
  https://doi.org/10.1016/j.neubiorev.2020.12.004
