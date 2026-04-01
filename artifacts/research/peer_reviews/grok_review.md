# Review of PAPER A: "Non-Associative State Space Models: Octonion Dynamics for Path-Dependent Sequence Modeling"

### 1. Summary (3-4 sentences)
The paper proposes O-SSM, a state space model using 8D octonion hidden states, leveraging non-associativity to introduce path-dependent dynamics that purportedly enhance sequence modeling beyond associative SSMs like S4 or Mamba. It verifies numerous algebraic properties of octonions (e.g., 168 associators, Moufang identities, G₂ automorphisms) and claims superior performance on 9/11 toy benchmarks against a diagonal baseline, with "kill shots" on sorting and Morse decoding. The work positions O-SSM at the "Cayley-Dickson boundary" for maximal non-commutativity and non-associativity while preserving norms, but acknowledges issues like incompatible parallel scan and weak scaling.

### 2. Strengths (bulleted, specific)
- Novel mathematical foundation: Rigorous verification of octonion properties (e.g., exact 168/343 associators via 3 methods, zero-error triality/D₄ symmetry) grounds the non-associativity claim solidly—impressive algebra, potentially publishable in a math journal.
- Honest negatives: Explicitly lists failures (e.g., H-SSM better optimization, diagonal wins gating tasks, synthetic connectome only), which builds credibility.
- Some relative benchmark wins: 3.18x on Morse decoding and 1.99x on sorting are intriguing for path-dependence, with ablation on α-sweep showing 6.66x order sensitivity.

### 3. Weaknesses (bulleted, specific, actionable)
- Toy benchmarks with abysmal absolute performance: sMNIST at 24% (real MNIST pixels? Baseline 10% is meaningless—state-of-the-art SSMs hit 90%+ on sequential MNIST; report full baselines like Mamba/S4). Action: Run on Long Range Arena or standard long-context tasks (e.g., Path-X, RULER) at realistic scales (1M+ params, not 160).
- Non-associativity kills SSM core advantage: Parallel scan fundamentally broken—no efficient training/inference at scale. Action: Provide serial-scan complexity analysis vs. RWKV/Griffin; this is a dealbreaker for "sequence modeling."
- Math overkill unrelated to ML impact: 31 algebra experiments (Fano planes, E₈ dims, Malcev algebras) dominate; ML reader skims to weak results. Action: Move to appendix or separate math paper—this is algebraic fanfic, not NeurIPS material.
- Suspicious "wins": Sorting 69.5% undefined (accuracy? What input?); connectome "80% curvature" on synthetic ASD data, null on real ABIDE-I—cherry-picked hype. Action: Define metrics precisely, release code/data, compare to SOTA (not diagonal strawman).
- Microscopic scale: 8D states, ~160 params—irrelevant to real models (Mamba is millions). Optimization hacks (momentum fixes divergence) scream brittleness. Action: Scale to 1M params on real tasks or admit it's a curiosity.

### 4. Questions for Authors (numbered)
1. Why no comparison to actual SSMs (S4, Mamba, RetNet) or RNNs (RWKV, Hyena)—diagonal baseline is useless?
2. How do you address parallel scan incompatibility for long sequences (e.g., 1M tokens)—serial only?
3. sMNIST 24%: Is this pixel-by-pixel classification? Report perplexity/loss, not just "%" (what %?)
4. Connectome: Define "curvature density" mathematically; why synthetic only when real ABIDE-I is null?
5. Catalan branching "exactly 2 distinct results" at n=4—what's the sequence task, and why not associative quaternions/H-SSM throughout?

### 5. Missing References
- S4/Mamba papers (Gu et al. 2021/2023) for SSM baselines.
- RWKV (Peng et al. 2023) for path-dependent RNNs without scan issues.
- Octonion ML prior: e.g., "Octonions for Sequence Processing" (if exists) or quaternion RNNs (Parcollet et al. 2018).
- Long-context benchmarks: LRA (Tay et al. 2021), Path-X (Moller et al. 2024).

### 6. Scores
- Soundness (1-4): 2 (Math solid; ML claims weak/underscaled)
- Presentation (1-4): 2 (Dense math walls; vague benchmarks)
- Contribution (1-4): 1 (Algebra neat; zero ML advance)
- Overall (1-10): 3
- Confidence (1-5): 4

### 7. Decision
Reject

---

# Review of PAPER B: "E-KAN: Analytical Uncertainty Propagation in Kolmogorov-Arnold Networks via GUM"

### 1. Summary (3-4 sentences)
The paper introduces E-KAN, a KAN variant with piecewise-linear hat-basis activations that enable exact first-order GUM (Guide to Uncertainty in Measurement) propagation for analytical uncertainty quantification, claiming 20x speed over ensembles with 90-100% coverage on UCI datasets. It validates against Monte Carlo on PBPK pharmacokinetics (σ-ratio 0.986) and ablates factors like noise/knots, while disclosing failures on OOD/heteroscedasticity/Friedman-1 interactions. The key insight is KAN's linearity per segment allowing closed-form σ_y propagation, unlike smooth MLPs.

### 2. Strengths (bulleted, specific)
- Practical UQ advance: Analytical GUM at 20x ensemble speed is compelling for low-data settings; PBPK validation (94.85% MC coverage match) shows real-world utility.
- Rigorous ablations: Knots/width/depth/noise sweeps quantify GUM robustness (e.g., saturates at K=5), with second-order extension improving +1.7% on interactions.
- Honest failures: Explicit OOD/heteroscedastic/adversarial breakdowns (e.g., 0% OOD coverage) and Friedman-1 (10% expected) avoid hype.
- Architecture insight: Piecewise-linearity enabling ∂f/∂x = constant slope per segment is clean math for UQ.

### 3. Weaknesses (bulleted, specific, actionable)
- Tiny datasets, no ML relevance: UCI (Concrete/Wine) overkill for NeurIPS—SOTA models are 99%+ accurate with MC-dropout/BBB better calibrated. 100% vs 0% ensemble on Wine screams tiny N=5 ensembles or miscalibrated baselines. Action: Benchmark on large-scale (ImageNet, GLUE) or time-series (ETTh/Weather) with SOTA UQ (Deep Ensembles, SWA-GP).
- GUM is old hat: JCGM 2008 standard; "exact first-order" ignores higher-order limits (even your extension is marginal +1.7%). Action: Compare to SWAG/SGLD (full posterior) or conformal prediction (distribution-free).
- Suspicious coverage: 100% on Wine/Energy vs. ensemble 0%/76%—ensembles usually 95% calibrated; likely small test sets or poor ensemble setup. Action: Report calibration curves, effective sample size, code for reproduction.
- KAN hype without payoff: KANs underperform MLPs on scale; UQ niche doesn't justify. Failures (OOD blind) are GUM limits, not E-KAN strengths. Action: Show predictive accuracy gains or deploy on high-stakes (e.g., medical imaging).
- No theory guarantees: "20x speed" handwavy; heteroscedastic "ratio 1.007" but assumes homoscedastic—what's the bias? Action: Prove coverage under knot assumptions.

### 4. Questions for Authors (numbered)
1. Ensemble details: N=5-10 how trained? Why 0% Wine coverage—MC-dropout/SWA baselines?
2. Scale: Results on UCI only—coverage/σ-ratio on ImageNet-1k or LLM fine-tuning?
3. Friedman-1 10%: Exact failure mode (which interaction?); why not adaptive knots?
4. PBPK: Public code/data? How does it compare to GP/Surrogate UQ standards?
5. Second-order: Full Hessian propagation cost? Scalable to width>10?

### 5. Missing References
- KAN original (Liu et al. 2024, arXiv).
- SOTA UQ: Deep Ensembles (Lakshminarayanan 2017), SWAG (Maddox 2019), Conformal Prediction (Shafer 2008).
- Piecewise-linear UQ: Prior splines/BART (e.g., Hahn 2017).
- Large-scale UQ: "Uncertainty Baselines" (Chung et al. 2021).

### 6. Scores
- Soundness (1-4): 3 (Solid small-scale; unproven at scale)
- Presentation (1-4): 3 (Clear tables; vague baselines)
- Contribution (1-4): 2 (Niche UQ tweak)
- Overall (1-10): 5
- Confidence (1-5): 4

### 7. Decision
Borderline
