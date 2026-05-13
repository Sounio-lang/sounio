<!-- docs:meta
topic_id: repo.docs.research.octonion-conversation-annotated-bibliography
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.octonion-conversation-annotated-bibliography
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Octonion Conversational State Model — Annotated Bibliography v1

Snapshot date: 2026-04-23

This bibliography is the first operational pass for the conversational extension of the O-SSM program.

It is organized around the review axes defined in:
- [octonion_conversation_literature_review_framework.md](/workspace/sounio/docs/research/octonion_conversation_literature_review_framework.md:1)

The point is not completeness. The point is to build a **claim-safe map** of the surrounding field.

Each entry records:
- **Axis** — which review axis it informs
- **Contribution** — what it establishes
- **Threat to novelty** — how much it weakens a future novelty claim for Tapestry
- **Relevance to Tapestry** — how directly it matters to the octonion conversational agenda

Threat scale:
- **Low** — useful background, but not close to the core claim
- **Medium** — adjacent precedent that narrows the novelty space
- **High** — close enough that future claims must be phrased carefully

---

## A. Emotion Representations Inside LLMs

### 1. Anthropic (2026)
**Citation:** “Emotion concepts and their function in a large language model.”  
**Link:** https://www.anthropic.com/research/emotion-concepts-function

- **Axis:** 1
- **Contribution:** Strong evidence that emotion-related internal directions in an LLM can be measured, manipulated, and linked to downstream behavior. Also suggests these representations are often functional and local rather than equivalent to stable, human-like selves.
- **Threat to novelty:** High
- **Relevance to Tapestry:** Very high
- **Why it matters:** This is the clearest reason not to claim that “emotion geometry in language models” is itself novel. It does, however, support your premise that affect-like internal structure is a real computational object.

### 2. Dong et al. (2025)
**Citation:** “From Rational Answers to Emotional Resonance: The Role of Controllable Emotion Generation in Language Models.”  
**Link:** https://arxiv.org/abs/2502.04075

- **Axis:** 1
- **Contribution:** Proposes controllable emotion generation using emotion vectors across multiple LLMs without heavy retraining.
- **Threat to novelty:** High
- **Relevance to Tapestry:** High
- **Why it matters:** This is a direct adjacent precedent for steerable affective geometry. It does not, by itself, address temporally persistent conversational state or non-associative recurrence.

### 3. Abdurahman et al. (2025)
**Citation:** “Large Language Models are Highly Aligned with Human Ratings of Emotional Stimuli.”  
**Link:** https://arxiv.org/abs/2508.14214

- **Axis:** 1
- **Contribution:** Shows that current LLMs can align well with human emotion ratings across multiple modalities and scales.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Supports the idea that LLMs can model emotional structure plausibly, but this is about alignment with ratings, not a temporal emotional state model.

### 4. “Do LLMs ‘Feel’?” (2025)
**Citation:** “Do LLMs ‘Feel’? Emotion Circuits Discovery and Control.”  
**Link:** https://arxiv.org/abs/2510.11328

- **Axis:** 1
- **Contribution:** Pushes beyond vector steering toward circuit-level localization and control of emotional expression in LLMs.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Important for interpretability framing. If emotion circuits are real and controllable, Tapestry should avoid overstating any claim that affective internal structure is absent from current models.

### 5. Scientific Reports (2025)
**Citation:** “Correspondence of high dimensional emotion structures elicited from video clips between humans and multimodal LLMs.”  
**Link:** https://www.nature.com/articles/s41598-025-14961-6

- **Axis:** 1
- **Contribution:** Shows multimodal LLMs can approximate rich human emotion structure, not only coarse labels.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Helps justify using richer emotional state spaces than simple valence/arousal vectors, but does not imply a hypercomplex or temporally recurrent formulation.

---

## B. Emotion in Conversation

### 6. Majumder et al. (2018)
**Citation:** “DialogueRNN: An Attentive RNN for Emotion Detection in Conversations.”  
**Link:** https://arxiv.org/abs/1811.00405

- **Axis:** 2
- **Contribution:** Classic speaker-aware recurrent model for emotion recognition in conversation.
- **Threat to novelty:** High
- **Relevance to Tapestry:** Very high
- **Why it matters:** This is a foundational adjacent precedent for tracking emotional state through dialogue context. It does not use non-associative algebra, but it directly occupies the “conversation + emotion + temporality” neighborhood.

### 7. Ghosal et al. (2019)
**Citation:** “DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation.”  
**Link:** https://arxiv.org/abs/1908.11540

- **Axis:** 2
- **Contribution:** Models inter-utterance and inter-speaker dependencies with graph structure rather than simple recurrence.
- **Threat to novelty:** High
- **Relevance to Tapestry:** High
- **Why it matters:** Important because it shows the field already moved beyond flat sequence models for emotional dialogue. Tapestry must therefore emphasize what non-associative state geometry adds beyond graph relational modeling.

### 8. Hu et al. (2021)
**Citation:** “DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations.”  
**Link:** https://arxiv.org/abs/2106.01978

- **Axis:** 2
- **Contribution:** Uses contextual reasoning over conversations for ERC, reinforcing that dialogue emotion is a stateful reasoning task, not a local classification task.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** High
- **Why it matters:** Helps define the baseline problem. Tapestry is not inventing emotional context modeling; it is proposing a different mathematical substrate for it.

### 9. Deep emotion recognition survey (2024)
**Citation:** “Deep emotion recognition in textual conversations: a survey.”  
**Link:** https://link.springer.com/article/10.1007/s10462-024-11010-y

- **Axis:** 2
- **Contribution:** Survey of textual ERC methods, including context modeling, emotion dynamics, and common architectures.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** High
- **Why it matters:** Useful to map the standard baselines and avoid rediscovering solved framing language.

### 10. Context-based emotion recognition survey (2024)
**Citation:** “Context-based emotion recognition: A survey.”  
**Link:** https://www.sciencedirect.com/science/article/pii/S0925231224018447

- **Axis:** 2
- **Contribution:** Focuses on contextual modeling in emotion recognition across modalities.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Reinforces that temporality and context are central in affect modeling. Tapestry’s novelty therefore depends on how it treats these mathematically, not on merely acknowledging them.

### 11. MDPI review (2025)
**Citation:** “A Systematic Review on Artificial Intelligence-Based Multimodal Dialogue Systems Capable of Emotion Recognition.”  
**Link:** https://www.mdpi.com/2414-4088/9/3/28

- **Axis:** 2, 7
- **Contribution:** Systematic review of multimodal dialogue systems with emotion recognition.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** High
- **Why it matters:** This is a direct warning against broad novelty claims around “emotion-aware dialogue systems.” Tapestry must focus on the non-associative state model, not on multimodal emotion dialogue in general.

### 12. Jiang (2025)
**Citation:** “Towards Human-Like Dialogue Systems: Integrating Multimodal Emotion Recognition and Non-Verbal Cue Generation.”  
**Link:** https://aclanthology.org/2025.yrrsds-1.6/

- **Axis:** 2, 7
- **Contribution:** Position paper advocating emotionally aware multimodal dialogue systems.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** Medium
- **Why it matters:** Useful for motivation and design context, but it is not close to the hypercomplex or state-space core.

### 13. EmoBot (2023)
**Citation:** “EmoBot: Artificial emotion generation through an emotional chatbot during general-purpose conversations.”  
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S138904172300102X

- **Axis:** 2, 7
- **Contribution:** Builds a chatbot that detects user emotion and maintains its own emotional state during general-purpose conversations.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** High
- **Why it matters:** This is close enough that Tapestry should not claim “first emotional chatbot with state.” The differentiator must be the algebraic-temporal geometry and the state-space formulation.

---

## C. Hypercomplex Sequence Models

### 14. Parcollet et al. (2018)
**Citation:** “Quaternion Recurrent Neural Networks.”  
**Link:** https://arxiv.org/abs/1806.04418

- **Axis:** 3
- **Contribution:** Introduces quaternion recurrent and quaternion LSTM models for sequence tasks, leveraging structured channel dependencies.
- **Threat to novelty:** High
- **Relevance to Tapestry:** Very high
- **Why it matters:** This is one of the strongest adjacent precedents. It shows that hypercomplex recurrent models for sequences are already real. Tapestry’s differentiator is the move from associative quaternions to non-associative octonions.

### 15. Parcollet et al. (2018)
**Citation:** “Bidirectional Quaternion Long-Short Term Memory Recurrent Neural Networks for Speech Recognition.”  
**Link:** https://arxiv.org/abs/1811.02566

- **Axis:** 3
- **Contribution:** Extends quaternion recurrence to bidirectional sequence processing.
- **Threat to novelty:** High
- **Relevance to Tapestry:** Very high
- **Why it matters:** This is especially important because your bidirectional conversational idea is a key part of the pitch. Bidirectional hypercomplex sequence modeling is not novel by itself; the novelty would have to come from the octonion/non-associative structure.

### 16. Qiu et al. (2020)
**Citation:** “Quaternion Neural Networks for Multi-channel Distant Speech Recognition.”  
**Link:** https://arxiv.org/abs/2005.08566

- **Axis:** 3
- **Contribution:** Reinforces the advantage of quaternion structure for multichannel correlated sequence inputs.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Helpful precedent for the claim that hypercomplex structures can encode coupled modalities or channels more naturally than real-valued baselines.

### 17. Berrouiguet et al. (2022)
**Citation:** “Learning Speech Emotion Representations in the Quaternion Domain.”  
**Link:** https://arxiv.org/abs/2204.02385

- **Axis:** 3
- **Contribution:** Connects quaternion representations directly to speech emotion modeling.
- **Threat to novelty:** High
- **Relevance to Tapestry:** Very high
- **Why it matters:** This is one of the closest adjacent works to your affective-state idea. It significantly narrows the novelty space for any claim like “hypercomplex models for affect are new.”

### 18. Wu et al. (2019)
**Citation:** “Deep Octonion Networks.”  
**Link:** https://arxiv.org/abs/1903.08478

- **Axis:** 3
- **Contribution:** Establishes the main deep-learning building blocks for octonion-valued networks.
- **Threat to novelty:** High
- **Relevance to Tapestry:** Very high
- **Why it matters:** This removes any novelty claim around “using octonions in deep learning” by itself. What remains potentially new is the conversational, temporally structured, non-associative state-space use.

### 19. Quaternion-based HAR (2025)
**Citation:** “Advancing human activity recognition with quaternion-based recurrent neural networks.”  
**Link:** https://www.tandfonline.com/doi/full/10.1080/00051144.2025.2480419

- **Axis:** 3
- **Contribution:** Recent application of quaternion recurrent modeling to wearable-sensor time series.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Supports the plausibility of using hypercomplex recurrence on wearable-derived signals, which matters if HealthKit or biosignals become part of the long-term architecture.

---

## D. Octonion Dynamics, Stability, and Control

### 20. Wang and Liu (2020)
**Citation:** “Global μ-stability and finite-time control of octonion-valued neural networks with unbounded delays.”  
**Link:** https://arxiv.org/abs/2003.11330

- **Axis:** 4
- **Contribution:** Provides a stability/control analysis framework for octonion-valued neural systems with delays.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** High
- **Why it matters:** Important mathematical precedent. Even if the analysis decomposes the octonion system into real-valued components, it offers language and tools for future stability arguments in Tapestry.

### 21. Global exponential stability of OVNNs (2018)
**Citation:** “Global exponential stability of octonion-valued neural networks with leakage delay and mixed delays.”  
**Link:** https://www.sciencedirect.com/science/article/pii/S0893608018301606

- **Axis:** 4
- **Contribution:** Earlier stability result for delayed octonion-valued neural networks.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** High
- **Why it matters:** Reinforces that there is already a mathematical control/stability literature around octonion neural systems.

### 22. Multistability analysis of OVNNs
**Citation:** “Multistability analysis of octonion-valued neural networks with time-varying delays.”  
**Link:** https://jglobal.jst.go.jp/en/public/202202248099776793

- **Axis:** 4
- **Contribution:** Suggests a broader literature on multistability and delayed octonion systems.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** Medium
- **Why it matters:** Worth deeper follow-up if Tapestry needs a rigorous account of regime-switching or multiple stable conversational modes.

---

## E. State-Space Models for Sequence Modeling

### 23. Gu et al. (2021)
**Citation:** “Efficiently Modeling Long Sequences with Structured State Spaces.”  
**Link:** https://arxiv.org/abs/2111.00396

- **Axis:** 5
- **Contribution:** S4 establishes the modern structured-state-space baseline for efficient long-sequence modeling.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** Very high
- **Why it matters:** This is not a threat to the hypercomplex idea, but it is the main baseline literature for why associative state-space scans matter computationally.

### 24. Gu and Dao (2023/2024)
**Citation:** “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.”  
**Link:** https://arxiv.org/abs/2312.00752

- **Axis:** 5
- **Contribution:** Makes selective state-space models competitive with transformers and emphasizes efficient content-aware sequence modeling.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** Very high
- **Why it matters:** Critical contrast case. Mamba shows what you gain from associative/selective scan efficiency. Tapestry’s claim is that non-associativity may recover something structurally richer for conversation at the cost of that efficiency.

### 25. Smith et al. (2022)
**Citation:** “Simplified State Space Layers for Sequence Modeling.”  
**Link:** https://arxiv.org/abs/2208.04933

- **Axis:** 5
- **Contribution:** S5 and related developments broaden the SSM design space beyond the original S4.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** Medium
- **Why it matters:** Helps make sure Tapestry is compared against modern SSM families, not a strawman diagonal recurrence alone.

---

## F. Physiology, HRV, and Psychopathology

### 26. Heiss et al. (2021)
**Citation:** “Heart rate variability as a biobehavioral marker of diverse psychopathologies: A review and argument for an ‘ideal range’.”  
**Link:** https://www.sciencedirect.com/science/article/pii/S0149763420306795

- **Axis:** 6
- **Contribution:** Broad transdiagnostic review arguing HRV is relevant across multiple psychopathology domains.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** Very high
- **Why it matters:** Strong support for using HRV as a candidate grounding signal, but also a warning that HRV is broad and transdiagnostic rather than diagnosis-specific magic.

### 27. Galin and Keren (2024)
**Citation:** “The Predictive Potential of Heart Rate Variability for Depression.”  
**Link:** https://www.sciencedirect.com/science/article/pii/S030645222400126X

- **Axis:** 6
- **Contribution:** Review arguing HRV changes may precede the onset of depression and should be tested as predictive markers.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** High
- **Why it matters:** Important if depression-sensitive conversational dynamics are part of the downstream research program.

### 28. Frontiers smartwatch study (2024)
**Citation:** “Association between heart rate variability metrics from a smartwatch and self-reported depression and anxiety symptoms: a four-week longitudinal study.”  
**Link:** https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2024.1371946/full

- **Axis:** 6
- **Contribution:** Shows consumer wearable HRV can relate to depression and anxiety symptoms longitudinally.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** High
- **Why it matters:** Useful for the Apple Health / wearable integration angle. Supports feasibility, not a strong mechanistic claim.

### 29. Garyfalli et al. (2025)
**Citation:** “Smartwatch-Derived Digital Phenotypes Relate to Psychopathology Dimensions in Patients With Psychotic Spectrum Disorders: Longitudinal Observational Study.”  
**Link:** https://mental.jmir.org/2025/1/e75774

- **Axis:** 6
- **Contribution:** Links wearable-derived physiological and behavioral signals to symptom dimensions in psychotic spectrum disorders.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Very high
- **Why it matters:** Strongly relevant to any proposal to model psychopathology-sensitive temporal trajectories. It supports the clinical plausibility of dimension-sensitive passive sensing.

### 30. HRV in psychiatric disorders review (2023)
**Citation:** “Heart Rate Variability in Psychiatric Disorders: A Systematic Review.”  
**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10596135/

- **Axis:** 6
- **Contribution:** Summarizes associations between HRV and multiple psychiatric conditions, while emphasizing inconsistency in some domains.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** High
- **Why it matters:** Good cautionary source. Supports using HRV as one signal among many, not as a standalone psychiatric truth variable.

### 31. Digital phenotyping monitoring review (2024)
**Citation:** “Digital Phenotyping for Monitoring Mental Disorders: Systematic Review.”  
**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10753422/

- **Axis:** 6
- **Contribution:** Reviews passive sensing for mental disorder monitoring, including heart rate and wearable data.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** High
- **Why it matters:** Important when moving from pure theory toward clinically serious monitoring or assistive applications.

### 32. Depression digital phenotyping review (2025)
**Citation:** “Distinguishing Common Digital Phenotyping and Self-Report Parameters for Monitoring and Predicting Depression: Scoping Review.”  
**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12954677/

- **Axis:** 6
- **Contribution:** Surveys common sensing and self-report variables used in depression-related digital phenotyping, including wearable signals.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** Medium
- **Why it matters:** Useful for designing a claim-safe physiological feature set if the project grows toward depression-sensitive conversational modeling.

---

## G. Emotion Recognition from Physiological Signals and Wearables

### 33. Saganowski et al. (2020)
**Citation:** “Emotion Recognition Using Wearables: A Systematic Literature Review.”  
**Link:** https://arxiv.org/abs/1912.10528

- **Axis:** 6, 7
- **Contribution:** Reviews wearable-based emotion recognition, including physiological sensor quality, device constraints, and study design.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** High
- **Why it matters:** Good baseline for the biofeedback side of the project. It shows wearable affect inference is already an active area.

### 34. Cardio-based ERS review (2023)
**Citation:** “A systematic review of emotion recognition using cardio-based signals.”  
**Link:** https://www.sciencedirect.com/science/article/pii/S2405959523001157

- **Axis:** 6, 7
- **Contribution:** Reviews ECG/PPG-based emotion recognition, including wearable implementations.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** High
- **Why it matters:** Especially relevant if HRV-like measures are used to modulate the conversational state.

### 35. Physiological signal survey (2022)
**Citation:** “A Survey on Physiological Signal Based Emotion Recognition.”  
**Link:** https://arxiv.org/abs/2205.10466

- **Axis:** 6, 7
- **Contribution:** Broad survey on physiological emotion recognition, including preprocessing, multimodal fusion, and inter-subject variance.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Helps avoid naive assumptions about physiology-based emotion inference. Inter-subject heterogeneity is a major issue.

### 36. FEEL (2026)
**Citation:** “FEEL: Quantifying Heterogeneity in Physiological Signals for Generalizable Emotion Recognition.”  
**Link:** https://arxiv.org/abs/2604.05926

- **Axis:** 6, 7
- **Contribution:** Recent work explicitly targeting heterogeneity and transferability in physiological emotion recognition.
- **Threat to novelty:** Low
- **Relevance to Tapestry:** Medium
- **Why it matters:** Useful if you later want biofeedback components that generalize across devices and subjects.

---

## H. Multimodal Emotion Recognition and Affective Agents

### 37. Zhang et al. (2023)
**Citation:** “A Survey of Deep Learning-Based Multimodal Emotion Recognition: Speech, Text, and Face.”  
**Link:** https://www.mdpi.com/1099-4300/25/10/1440

- **Axis:** 7
- **Contribution:** Broad survey of multimodal emotion recognition systems.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Good background for multimodal fusion, but not close to the hypercomplex conversational-state claim.

### 38. Scientific Reports (2025)
**Citation:** “Multi-modal emotion recognition in conversation based on prompt learning with text-audio fusion features.”  
**Link:** https://www.nature.com/articles/s41598-025-89758-8

- **Axis:** 7
- **Contribution:** Shows continuing movement toward stronger multimodal ERC systems.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Another reminder that multimodal emotion-in-conversation is already active. The novelty has to be in the state geometry and temporal algebra.

### 39. Emotional AI chatbot with ontology (2025)
**Citation:** “An Emotional AI Chatbot Using an Ontology and a Novel Audiovisual Emotion Transformer for Improving Nonverbal Communication.”  
**Link:** https://www.mdpi.com/2079-9292/14/21/4304

- **Axis:** 7
- **Contribution:** Builds an emotional chatbot with ontology and multimodal emotion perception.
- **Threat to novelty:** Medium
- **Relevance to Tapestry:** Medium
- **Why it matters:** Important adjacent precedent for “emotion-aware conversational systems,” though still not close to octonion state-space modeling.

---

## Provisional takeaways

### 1. What is clearly **not** novel anymore

- emotion-like internal structure in LLMs
- controllable emotion steering in LLMs
- emotion recognition in conversation
- multimodal emotion-aware dialogue systems
- hypercomplex recurrent sequence models in general
- octonion neural networks in general
- HRV and wearable signals as affect- or psychopathology-relevant features

### 2. What still looks **plausibly novel**

- an **octonion conversational state-space model** specifically
- using **non-associativity** as the central mechanism for conversational temporality
- using the **associator** as a dialogue-telemetry or coherence signal
- using **Fano-line subalgebras** as structured conversational modes
- using **sedenion zero-divisor proximity** as a controlled forgetting geometry
- integrating all of the above with physiology and psychopathology-sensitive trajectories in one model family

### 3. What claim language looks safest right now

Safer:
- “We are not aware of prior work combining octonion non-associative state-space recurrence with conversational affective modeling.”
- “The novelty appears to lie in the integrated architecture, not in any single component alone.”

Less safe:
- “This is the first emotional dialogue system.”
- “This is the first model to use emotion geometry.”
- “This is the first AI system to connect physiology and conversation.”

### 4. Highest-priority next readings

If time is limited, start with these ten:

1. Anthropic 2026 emotion concepts
2. Dong et al. 2025 emotion vectors
3. DialogueRNN
4. DialogueGCN
5. 2025 multimodal dialogue systems review
6. Quaternion Recurrent Neural Networks
7. Learning Speech Emotion Representations in the Quaternion Domain
8. Deep Octonion Networks
9. Wang and Liu 2020 octonion stability/control
10. Garyfalli et al. 2025 smartwatch digital phenotypes

### 5. Recommended claim strategy for Tapestry

The most defensible framing is:

> Tapestry is not novel because it uses emotion, conversation, octonions, or biosignals separately.  
> Tapestry may be novel because it proposes a **non-associative conversational state-space architecture** in which affective and physiological signals are embedded into a temporally structured hypercomplex state whose composition law is itself path-dependent.

That is a serious, testable claim.
