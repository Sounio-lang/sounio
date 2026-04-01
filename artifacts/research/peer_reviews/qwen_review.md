# NeurIPS 2026 Peer Review

## PAPER A: "Non-Associative State Space Models: Octonion Dynamics for Path-Dependent Sequence Modeling"

### 1. Summary
This paper introduces O-SSM, a novel state space model using octonion-valued hidden states that explicitly exploits non-associativity as a feature for path-dependent sequence modeling. It combines mathematical rigor with empirical validation, establishing properties like norm preservation, automorphism groups, and associator counts. The model shows superior performance on sequence modeling tasks like sorting, Morse decoding, and bracket matching compared to diagonal baselines.

### 2. Strengths
- **Mathematical rigor**: Multiple independent verifications of key properties (168-element automorphism group, Moufang identities, triality)
- **Theoretical novelty**: First practical application of octonions in ML with formal guarantees about norm preservation and path dependence
- **Empirical validation**: Extensive benchmarking across 11 tasks with clear kill shots (3.18x advantage on Morse decoding)
- **Boundary analysis**: Demonstrates fundamental limitations (parallel scan incompatibility) and Cayley-Dickson boundary
- **Curvature analysis**: Connectome experiments (synthetic) suggest potential neuroscience applications

### 3. Weaknesses
- **Practical significance unclear**: Performance gains come at cost of 66% more parameters (160 vs 96); no ablation on parameter scaling
- **Synthetic evaluation limitations**: Connectome results use synthetic data only; real neuroimaging validation missing
- **Optimization challenges**: H-SSM outperforms O-SSM on raw training loss; momentum Adam comparison shows optimization complexity
- **Accessibility issues**: Dense mathematical presentation obscures key insights for ML audience
- **Unclear path dependence mechanism**: The role of 168 associators in specific task improvements remains hand-wavy

### 4. Questions for Authors
1. How does the 168-element automorphism group specifically contribute to sequence modeling capabilities?
2. Can you demonstrate capacity-controlled experiments to isolate gains from non-associativity vs parameter count?
3. What specific failure modes occur in real connectome experiments with ABIDE-I data?
4. How does O-SSM's memory footprint compare to associative models given 8×8 full A matrices?
5. Can you provide visualizations of the path-dependent dynamics in Fano plane representation?

### 5. Missing References
- Octonion neural networks (e.g., Parcollet 2019)
- Non-associative algebra applications in physics (G₂ holonomy)
- Composition algebras in signal processing
- Path-dependent PDEs in mathematical finance

### 6. Scores
- Soundness: 4 (Mathematical claims verified, but ML claims lack depth)
- Presentation: 2 (Excessive mathematical jargon obscures ML contributions)
- Contribution: 3 (Niche but theoretically significant)
- Overall: 6.5
- Confidence: 4 (High in math, moderate in ML implications)

### 7. Decision
Weak Accept. Would benefit from significant reformatting for ML audience and more discussion of practical constraints.

---

## PAPER B: "E-KAN: Analytical Uncertainty Propagation in Kolmogorov-Arnold Networks via GUM"

### 1. Summary
This work proposes E-KAN, a Kolmogorov-Arnold Network variant enabling exact first-order GUM uncertainty propagation through piecewise-linear activations. It demonstrates 20x speedup over ensemble methods on UCI benchmarks and PBPK modeling while maintaining rigorous coverage guarantees. The paper honestly documents limitations in OOD, heteroscedastic, and interaction-heavy scenarios.

### 2. Strengths
- **Practical utility**: Implements international standard (JCGM 100) for measurement uncertainty in ML
- **Speed vs accuracy**: 20x efficiency gain over ensembles with comparable coverage (90-100%)
- **Theoretical elegance**: Leverages KAN architecture's piecewise-linear nature for analytical propagation
- **Rigorous validation**: Multiple failure modes systematically characterized
- **Second-order extension**: Principled approach to handling interaction terms through Hessian analysis

### 3. Weaknesses
- **Fundamental limitation**: First-order Taylor inherently fails for interaction terms (Friedman-1 validation)
- **Expressiveness concerns**: Hat-basis activations may limit representation power vs smooth activations
- **Narrow applicability**: Homoscedastic noise assumption limits real-world use cases
- **Coverage overestimation**: Claims of 99.8% coverage despite known failure modes
- **Missing comparisons**: No Bayesian uncertainty baselines or heteroscedastic extensions

### 4. Questions for Authors
1. How does model performance degrade when moving from synthetic to real heteroscedastic noise?
2. Can hat-basis knots be adapted during training to improve interaction handling?
3. What's the computational overhead of second-order propagation in practice?
4. Why does width ≥5 guarantee 100% coverage while depth shows saturation?
5. How does E-KAN's uncertainty calibration compare to Bayesian methods?

### 5. Missing References
- Bayesian neural network uncertainty quantification (Gal 2016)
- Heteroscedastic regression techniques
- Taylor propagation in traditional ML (e.g., random forests)
- Piecewise-linear network expressiveness (Arora et al 2018)

### 6. Scores
- Soundness: 4 (Methodologically rigorous with systematic validation)
- Presentation: 3 (Clear but could better motivate GUM relevance)
- Contribution: 3 (Practically significant for regulated domains)
- Overall: 7.0
- Confidence: 5 (High confidence in assessment)

### 7. Decision
Accept. Provides concrete value to safety-critical applications despite theoretical limitations. Second-order extension shows promising research direction.

---

## Final Notes
- Paper A is borderline due to accessibility issues but represents novel math-ML intersection
- Paper B offers concrete practical advancement in uncertainty quantification
- Both papers should consider splitting mathematical content into appendices for better flow
- Paper A needs clearer path from algebra properties to ML gains
- Paper B should better calibrate claims about "coverage" vs actual uncertainty quality
