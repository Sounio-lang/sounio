## REVIEW FOR PAPER A

### 1. Summary
The paper proposes O-SSM, a novel state space model that uses octonion-valued hidden states to exploit non-associativity as a feature for path-dependent sequence modeling. Key claims: 1) The 168 non-zero associators in octonion multiplication create intrinsic path-dependence, 2) The composition algebra property provides stable norm preservation, 3) O-SSM outperforms diagonal SSMs on 9/11 synthetic sequence tasks, particularly on tasks requiring order sensitivity like sorting and Morse decoding.

### 2. Strengths
- **Genuine mathematical novelty:** Leveraging octonion non-associativity for path-dependence is a creative and theoretically deep idea not seen in prior SSM literature.
- **Extensive mathematical verification:** The 31 mathematical experiments (associator count, Moufang identities, G₂ automorphisms, etc.) provide rigorous grounding and build confidence in the implementation.
- **Compelling "kill shot" results:** The 1.99x advantage on Sorting and 3.18x on Morse Decoding are striking and directly support the core claim about order sensitivity.
- **Honest reporting of negatives:** Including results where O-SSM loses (Adding Problem, H-SSM on raw loss) and limitations (parallel scan incompatibility) strengthens credibility.

### 3. Weaknesses (BLUNT)
- **Extremely narrow experimental scope:** All 11 benchmarks are small-scale, synthetic algorithmic tasks. There is **zero evidence** that this approach scales to real-world, high-dimensional data (language, audio, video) or provides benefits over transformers or modern SSMs (Mamba-2, Griffin) on practical problems.
- **The connection to "sequence modeling" is tenuous:** The paper demonstrates O-SSM is good at synthetic tasks that explicitly test order dependence, but fails to argue why this translates to general sequence modeling. The "Connectome" result is on synthetic data and the real-data attempt gave null—this is a major red flag.
- **Computational practicality is ignored:** The admission that parallel scan is "fundamentally incompatible" is a death knell for modern efficient training of long sequences. No analysis of runtime, memory, or scaling compared to standard SSMs is provided.
- **The architectural description is superficial:** How is the octonion state actually integrated? Is the A matrix learned in the octonion algebra? How are inputs and outputs projected? The ~160 parameters suggest a toy model, not a scalable architecture.
- **Theoretical contributions are self-contained but isolated:** While the math is interesting, it reads like a pure algebra exercise. The paper does not convincingly bridge the gap between abstract octonion properties and tangible advances in machine learning.

### 4. Questions for Authors
1. Can you provide any preliminary results on a standard, non-synthetic long-range arena benchmark (e.g., Long Range Arena, PG-19, a real-world time series dataset) to demonstrate practical utility?
2. The parallel scan incompatibility seems catastrophic for training efficiency. Do you have any proposed algorithmic or architectural modifications to recover sub-quadratic training, or is O-SSM fundamentally limited to short sequences?
3. The hidden state is 8-dimensional. For modeling complex phenomena, how do you propose to scale capacity? Simply stacking independent octonion SSMs, or using a higher-dimensional algebra (e.g., bioctonions)? What are the trade-offs?
4. The "optimization cost" is dismissed as not fundamental, but H-SSM (quaternion) has a significantly lower training loss. Doesn't this suggest the non-associative dynamics create a much more difficult optimization landscape, which will be exacerbated at scale?

### 5. Missing References
- **Modern SSM literature:** No comparison to recent efficient SSMs like Mamba-2, Griffin, or Based, which are the relevant baselines for sequence modeling.
- **Alternative approaches to path-dependence:** Prior work on higher-order RNNs, nonlinear state spaces, or memory-augmented networks that aim to capture complex temporal dependencies.
- **Applications of non-associative algebras in ML:** While rare, there is some work on quaternions and octonions in neural networks (e.g., for 3D rotation, Clifford algebras). These should be cited and differentiated from.

### 6. Scores
- **Soundness (3/4):** The mathematical derivations and synthetic experiments appear technically sound, but the empirical scope is too narrow to fully support the broader claims about sequence modeling.
- **Presentation (3/4):** The paper is densely written and assumes high familiarity with abstract algebra. The connection between mathematical results and machine learning contributions is not clearly motivated for a general ML audience.
- **Contribution (2/4):** The theoretical idea is novel, but its practical significance for the field is highly questionable given the lack of real-world benchmarks and fundamental efficiency limitations.
- **Overall (4/10):** Borderline reject. The idea is fascinating but presented as a mathematical curiosity without a convincing path to impact in machine learning.
- **Confidence (4/5):** Fairly confident in this assessment. The weaknesses are fundamental and not easily addressable in a rebuttal.

### 7. Decision
**Weak Reject**

**Justification:** The core idea is intellectually novel and the paper is executed with mathematical rigor. However, as a *NeurIPS paper*, it fails to demonstrate why this advance matters for machine learning. The experiments are confined to toy tasks, the proposed model is incompatible with modern efficient training paradigms, and no compelling real-world application is shown. This work might be better suited for a mathematics or theoretical computer science venue. For NeurIPS, the contribution is too isolated and the practical relevance too underdeveloped.

---

## REVIEW FOR PAPER B

### 1. Summary
The paper proposes E-KAN, a Kolmogorov-Arnold Network with piecewise-linear (hat-basis) activations that enable exact first-order uncertainty propagation using the GUM (Guide to the Expression of Uncertainty in Measurement) standard. Key claims: 1) The piecewise-linear structure allows for analytical computation of output uncertainty, 2) This method is 20x faster than ensemble methods and provides well-calibrated uncertainty on UCI datasets, 3) The approach fails predictably in known edge cases (OOD, interactions), which are clearly documented.

### 2. Strengths
- **Clear, practical problem and solution:** Uncertainty quantification is a critical need in ML for scientific applications. Leveraging KAN's unique architecture for efficient analytical propagation is a smart and well-motivated idea.
- **Rigorous evaluation against a standard:** Using GUM (JCGM 100:2008) as the foundation and measuring coverage against Monte Carlo is methodologically sound and appeals to scientific rigor.
- **Honest and thorough failure analysis:** The paper proactively identifies and quantifies limitations (OOD: 0%, interactions: 10%), which builds tremendous credibility. The "expected" failure on Friedman-1 is a strength, not a weakness.
- **Strong, clear results:** The 90-100% coverage on UCI datasets vs. ensemble's frequent failure is a compelling demonstration of superiority for in-distribution, homoscedastic uncertainty.

### 3. Weaknesses (BLUNT)
- **The innovation is narrow and architecturaly dependent:** The entire method hinges on using *piecewise-linear* activations in a KAN. This is a very specific design choice that limits the model's expressive power (no smooth nonlinearities) and ties the contribution to a single, still-niche architecture (KANs).
- **The comparison is unfair/insufficient:** Beating a *5-10 model ensemble* is not a high bar. The paper should compare against state-of-the-art UQ methods for neural networks: Deep Ensembles (with more models), MC Dropout, Laplace Approximation, SNGP, or evidential deep learning. The claim of "20x speed" is meaningless without comparing to these efficient baselines.
- **No demonstration on meaningful ML tasks:** UCI datasets are low-dimensional tabular benchmarks. There is no evidence E-KAN works on images, text, or any high-dimensional data where uncertainty quantification is actually challenging and valuable. The PBPK example is better but still relatively small-scale.
- **The "second-order extension" is trivial and offers minimal gain:** A +1.3% improvement is negligible, and the explanation (cross-Hessian terms only) reveals the fundamental limitation of the piecewise-linear approach for capturing curvature.
- **The paper feels like a minor extension of KANs:** It applies a known standard (GUM) to a recently proposed architecture (KAN). The core insight (piecewise-linear → analytical derivatives) is simple. The real contribution is the empirical evaluation, which is strong but limited in scope.

### 4. Questions for Authors
1. Why compare only to a small ensemble and not to standard UQ baselines like MC Dropout, Deep Ensembles (50+ models), or Laplace Approximation? The speed/accuracy trade-off should be benchmarked against these.
2. The piecewise-linear activation seems like a major compromise on expressivity. Have you experimented with a hybrid approach (e.g., smooth activations with piecewise-linear approximations for UQ propagation)? What is the performance drop on prediction accuracy versus standard KANs or MLPs?
3. The method fails completely on OOD data. For real-world scientific use, this is a critical flaw. Do you have any plans or ideas to combine the efficient GUM propagation with an OOD detection mechanism?
4. The paper focuses on epistemic uncertainty (from input noise). Have you considered or tested the method on aleatoric uncertainty (noise in the data itself)? The heteroscedastic failure suggests not.

### 5. Missing References
- **Modern neural network UQ literature:** Papers on SNGP, Laplace Approximation, SWAG, Bayesian neural networks, and evidential deep learning are conspicuously absent.
- **Prior work on uncertainty propagation in neural networks:** There is existing literature on analytical uncertainty propagation (e.g., "Uncertainty Quantification in Neural Networks" by Gast et al., 2018). This should be discussed.
- **KAN variants and critiques:** Since the paper builds on KANs, it should cite follow-up work and discussions about KANs' practical performance and limitations.

### 6. Scores
- **Soundness (4/4):** The methodology is correct, the experiments are well-designed, and the claims are strongly supported by the presented data. The honest discussion of failures adds to the soundness.
- **Presentation (4/4):** Exceptionally clear, well-structured, and honest. The tables and failure modes are presented effectively.
- **Contribution (3/4):** The contribution is solid but incremental. It provides a valuable, rigorously-tested tool for a specific niche (UQ in piecewise-linear KANs) but does not fundamentally advance the broader field of uncertainty in deep learning.
- **Overall (6/10):** Weak Accept. It is a competent, well-executed paper that solves a clear problem in a specific domain. It meets the bar for NeurIPS but is not a standout.
- **Confidence (5/5):** Very confident. The evaluation is thorough and the limitations are self-admitted.

### 7. Decision
**Weak Accept**

**Justification:** This is a well-crafted, honest, and practically useful paper. It identifies a clear problem (efficient uncertainty quantification) in an emerging architecture (KANs) and provides a rigorous solution with a comprehensive evaluation of both strengths and weaknesses. While the contribution is not revolutionary and the scope is somewhat narrow, the paper's clarity, rigor, and reproducibility make it a positive addition to NeurIPS. It serves as a valuable template for how to rigorously evaluate a UQ method. The main criticism is the lack of comparison to modern UQ baselines, which should be addressed in the final version.
