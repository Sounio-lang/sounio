# NeurIPS 2026 Peer Review

---

## PAPER A: "Non-Associative State Space Models: Octonion Dynamics for Path-Dependent Sequence Modeling"

### 1. Summary
The paper proposes O-SSM, a State Space Model (SSM) utilizing octonion hidden states to leverage non-associative dynamics. By moving beyond the associative constraint required by parallel scans, the authors argue that the resulting path-dependence and non-zero associators enhance performance on order-sensitive tasks. The work explores the mathematical properties of octonion algebras (Moufang identities, G₂ automorphisms) and evaluates the model on sequence benchmarks against diagonal SSM baselines.

### 2. Strengths
*   **Mathematical Depth:** The exploration of composition algebras and the "Cayley-Dickson boundary" is theoretically sophisticated. The verification of the 168-automorphism group and Moufang identities shows a high level of algebraic rigor.
*   **Novelty:** While Clifford/Quaternion neural networks exist, moving into non-associative algebras for SSMs is a genuinely novel direction that challenges the "associativity for efficiency" dogma of current SSM research.
*   **Task-specific Gains:** The "kill shot" results on Sorting and Morse Decoding (1.99x and 3.18x improvements) suggest that non-associativity may indeed capture specific structural dependencies that diagonal SSMs miss.
*   **Honesty:** The authors are refreshingly transparent about the failure on the Adding Problem and the superiority of Quaternions (H-SSM) in optimization.

### 3. Weaknesses
*   **Computational Impracticality:** By breaking associativity, the authors lose the $O(L \log L)$ parallel scan. For a NeurIPS audience, a model that scales linearly $O(L)$ in time but cannot be parallelized is a massive step backward for long-sequence modeling (the primary use case for SSMs).
*   **Weak Baselines/Metrics:** The reported accuracies (e.g., 24% on sMNIST, 12% on sCIFAR-10) are abysmal. Modern SSMs (S4, Mamba, Liquid-S4) achieve >90% on sMNIST and >80% on sCIFAR-10. Comparing against a "Diagonal" baseline that only gets 10% suggests neither model is properly tuned or the hidden dimension is far too small.
*   **Parameter Inefficiency:** The O-SSM has ~1.6x the parameters of the diagonal baseline. The performance gains might simply stem from increased capacity and cross-dimensional mixing rather than the "magic" of octonions.
*   **Synthetic Connectome Results:** The inclusion of "ASD vs TD curvature density" on synthetic data feels like "math-washing"—applying complex geometry to a toy simulation to imply biological relevance where none is proven.

### 4. Questions for Authors
1. Why are the baseline accuracies for sMNIST and sCIFAR-10 so low compared to the literature?
2. Can the non-associative dynamics be simulated using an associative algebra in a higher dimension? If so, what is the specific benefit of the octonion constraint?
3. Given the parallel scan is impossible, how do you propose O-SSM scales to the 100k+ sequence lengths where SSMs usually shine?
4. Is the "order sensitivity" truly a result of non-associativity, or simply the result of having a non-diagonal transition matrix $A$?

### 5. Missing References
*   Gu, A., et al. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces" (S4).
*   Poli, M., et al. (2023). "Hyena Hierarchy: Towards Larger Convolutional Language Models" (For non-SSM long-context alternatives).
*   Luo, et al. (2022). "Quasi-topological Neural Networks" (For related higher-order algebraic structures).

### 6. Scores
*   Soundness: 3
*   Presentation: 4
*   Contribution: 2
*   Overall: 4 (Reject)
*   Confidence: 5

### 7. Decision
**Reject.** While the math is beautiful, the empirical results are non-competitive with the state-of-the-art. This paper belongs in a mathematical physics or alternative computing journal rather than a mainstream ML venue until it can show competitive performance on standard benchmarks.

---

## PAPER B: "E-KAN: Analytical Uncertainty Propagation in Kolmogorov-Arnold Networks via GUM"

### 1. Summary
The paper introduces E-KAN, a Kolmogorov-Arnold Network (KAN) variant that uses piecewise-linear "hat" functions to enable analytical uncertainty propagation following the GUM (Guide to the Expression of Uncertainty in Measurement) standard. By exploiting the zero second-derivatives of the basis functions, the authors perform first-order Taylor expansions to estimate output variance at a fraction of the cost of Monte Carlo or Ensemble methods.

### 2. Strengths
*   **Methodological Synergy:** Using KANs specifically for GUM propagation is a clever insight. The piecewise-linear nature of hat-basis functions simplifies the Jacobian-based uncertainty calculation significantly.
*   **Efficiency:** A 20x speedup over ensembles for uncertainty estimation is a significant practical contribution, especially for real-time systems.
*   **Rigorous Validation:** The use of PBPK (pharmacokinetic) modeling and the $\sigma$-ratio comparison against Monte Carlo (0.986) provides strong evidence that the analytical approximation is highly accurate within its local domain.
*   **Honesty about Failures:** The "Honest Failures" section (OOD, Friedman-1) demonstrates scientific integrity and clearly defines the operational envelope of the method.

### 3. Weaknesses
*   **First-Order Limitations:** The failure on the Friedman-1 dataset (10% coverage) highlights a major flaw: GUM propagation ignores the interaction terms ($x_i x_j$) that are central to KAN's multivariate representation.
*   **Homoscedastic Assumption:** The model assumes constant input noise. In most real-world ML applications, noise is heteroscedastic (input-dependent), which the authors admit E-KAN is "blind" to.
*   **Architectural Narrowness:** The method relies entirely on the piecewise-linear "hat" basis. If one uses B-splines (as in the original KAN paper), the second-order terms no longer vanish, and the "E-KAN" advantage disappears or becomes significantly more complex.
*   **Baseline Weakness:** Comparing E-KAN (with GUM) to an MLP (with GUM) is a "strawman" comparison, as MLPs are known to have vanishing/exploding gradients that make first-order GUM propagation notoriously unstable.

### 4. Questions for Authors
1. How does the computational cost scale as the number of "knots" (K) increases, given the Jacobian must be re-evaluated for each segment?
2. Can this be extended to B-splines of degree > 1, or does the loss of the "zero-Hessian" property make it computationally equivalent to standard Delta-method propagation?
3. Why does the Ensemble method perform so poorly (0-11%) on Concrete/Wine datasets compared to GUM? This suggests the ensembles were not properly calibrated (e.g., no temperature scaling or proper prior).

### 5. Missing References
*   Liu, Z., et al. (2024). "KAN: Kolmogorov-Arnold Networks" (The foundational paper).
*   Abdar, M., et al. (2021). "A review of uncertainty quantification in deep learning."
*   Der Kiureghian, A. (2008). "Analysis of structural reliability under uncertainty" (For context on first-order reliability methods/FORM).

### 6. Scores
*   Soundness: 3
*   Presentation: 4
*   Contribution: 3
*   Overall: 6 (Weak Accept)
*   Confidence: 4

### 7. Decision
**Weak Accept.** The paper identifies a specific niche where KANs outperform MLPs (analytical tractability for metrological standards). While the failure on interaction terms is concerning, the speed and accuracy on PBPK models make this a valuable contribution to the "AI for Science" community. It is a solid, focused paper.
