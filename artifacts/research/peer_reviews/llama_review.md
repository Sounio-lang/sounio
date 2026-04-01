## Review of PAPER A: "Non-Associative State Space Models: Octonion Dynamics for Path-Dependent Sequence Modeling"

### 1. Summary (3-4 sentences)
The paper proposes O-SSM, a novel State Space Model using octonion (8-dimensional) hidden states, leveraging non-associativity as a feature for path-dependent sequence modeling. The authors claim that the octonion algebra's properties, such as norm preservation and the presence of 168 non-zero associators, enable more expressive and flexible dynamics. O-SSM is shown to outperform traditional diagonal SSMs on several sequence modeling tasks. The paper provides an extensive mathematical foundation and experimental validation.

### 2. Strengths (bulleted, specific)
* **Novel Mathematical Foundation**: The paper introduces the use of octonions in State Space Models, providing a rigorous mathematical foundation that explores the properties of octonion algebra and its implications for sequence modeling.
* **Extensive Experimental Validation**: The authors conduct a wide range of experiments (31 mathematical verifications and 11 benchmark tasks) to validate the proposed O-SSM, demonstrating its superiority over diagonal SSMs on most tasks.
* **Path-Dependent Dynamics**: The non-associative property of octonions is leveraged to create path-dependent dynamics, which is shown to be beneficial for certain sequence modeling tasks.

### 3. Weaknesses (bulleted, specific, actionable)
* **Computational Complexity**: The paper does not adequately address the computational cost of using full 8x8 matrices versus diagonal matrices. The increased parameter count (~160 vs ~96) and the lack of parallel scan compatibility may hinder practical applications.
* **Task-Specific Performance**: While O-SSM wins on 9 out of 11 tasks, it loses on the Adding Problem and has mixed results on other tasks. The conditions under which O-SSM excels or fails are not clearly analyzed.
* **Lack of Intuition**: The paper is heavy on mathematical formalism but light on intuitive explanations for why O-SSM works better on certain tasks. The significance of the 168 non-zero associators and other mathematical results could be better contextualized.
* **Synthetic Data**: Some results, like the connectome analysis, are on synthetic data, which may not directly translate to real-world performance.

### 4. Questions for Authors (numbered)
1. How does the computational cost of O-SSM compare to diagonal SSMs in practice, and are there any plans to optimize the implementation?
2. Can you provide more insight into why O-SSM performs worse on tasks like the Adding Problem?
3. How do the authors plan to address the lack of parallel scan compatibility in O-SSM?
4. Are there any theoretical or empirical analyses on the sensitivity of O-SSM to hyperparameter tuning compared to other SSMs?

### 5. Missing References
The paper could benefit from referencing other works that explore non-associative algebras in machine learning or sequence modeling, such as earlier attempts to use quaternions or other non-associative structures.

### 6. Scores
- **Soundness (1-4)**: 3 (The claims are generally well-supported, but some experimental results are on synthetic data, and the computational complexity is not fully addressed.)
- **Presentation (1-4)**: 3 (The paper is dense and mathematically rigorous but could be more accessible with additional intuitive explanations.)
- **Contribution (1-4)**: 4 (The introduction of octonion-based SSMs is novel and potentially significant for sequence modeling.)
- **Overall (1-10)**: 7 (The paper is strong in its mathematical contribution and experimental validation but has some weaknesses in practical applicability and clarity.)
- **Confidence (1-5)**: 4 (I am confident in my assessment based on the provided material.)

### 7. Decision
**Weak Accept**: The paper is novel and provides significant mathematical and experimental contributions. However, it requires improvements in addressing computational complexity, providing more intuitive explanations, and clarifying the practical applicability of O-SSM.

## Review of PAPER B: "E-KAN: Analytical Uncertainty Propagation in Kolmogorov-Arnold Networks via GUM"

### 1. Summary (3-4 sentences)
The paper proposes E-KAN, a method for analytical uncertainty propagation in Kolmogorov-Arnold Networks (KANs) using the Guide to the Expression of Uncertainty in Measurement (GUM). E-KAN leverages the piecewise-linear hat-basis edge activations in KANs to enable exact first-order GUM propagation, achieving faster and more accurate uncertainty estimation compared to ensemble methods. The authors validate E-KAN on UCI datasets and a pharmacokinetic modeling task.

### 2. Strengths (bulleted, specific)
* **Novel Application of GUM**: The paper introduces the use of GUM for uncertainty propagation in neural networks, specifically in KANs, which is a novel application.
* **Analytical Uncertainty Propagation**: E-KAN provides an analytical method for uncertainty propagation, which is significantly faster (20x) than ensemble methods.
* **Experimental Validation**: The authors validate E-KAN on multiple UCI datasets and a pharmacokinetic modeling task, showing improved coverage and accuracy compared to ensemble methods.

### 3. Weaknesses (bulleted, specific, actionable)
* **Limited Applicability**: E-KAN is specifically designed for KANs with piecewise-linear hat-basis activations. Its applicability to other network architectures or activation functions is not explored.
* **Failure Modes**: E-KAN fails on tasks with interaction terms (Friedman-1), out-of-distribution (OOD) data, and heteroscedastic noise. These limitations are acknowledged but not addressed.
* **Comparison to Other Methods**: The comparison is primarily with ensemble methods. A comparison with other uncertainty quantification methods (e.g., Bayesian Neural Networks, Monte Carlo Dropout) would provide a more comprehensive evaluation.
* **Depth and Width Ablation**: The ablation studies show that depth and width have limited impact on GUM coverage. More analysis on why this is the case would be beneficial.

### 4. Questions for Authors (numbered)
1. Can E-KAN be extended to other neural network architectures beyond KANs, or are there plans to explore such extensions?
2. How do the authors plan to address the limitations of E-KAN, particularly with respect to interaction terms and OOD data?
3. Would the authors consider comparing E-KAN with other uncertainty quantification methods to further establish its relative strengths and weaknesses?
4. Can you provide more insight into why the depth and width of the network have minimal impact on GUM coverage?

### 5. Missing References
The paper could benefit from referencing other works on uncertainty quantification in neural networks, such as Bayesian Neural Networks, Monte Carlo Dropout, and other ensemble methods, to contextualize the contribution of E-KN.

### 6. Scores
- **Soundness (1-4)**: 3 (The claims are generally well-supported, but the method's limitations and applicability are not fully explored.)
- **Presentation (1-4)**: 3 (The paper is clear but could benefit from additional discussion on the broader applicability and limitations of E-KAN.)
- **Contribution (1-4)**: 3 (The application of GUM to KANs is novel, but the overall impact is somewhat limited by the specificity to KANs and piecewise-linear activations.)
- **Overall (1-10)**: 6 (The paper provides a novel contribution but has limitations in terms of applicability and comparison to other methods.)
- **Confidence (1-5)**: 4 (I am confident in my assessment based on the provided material.)

### 7. Decision
**Borderline**: The paper introduces a novel application of GUM to KANs and demonstrates its effectiveness. However, its limitations and specificity to certain architectures and activation functions need to be more thoroughly discussed. Addressing these concerns could strengthen the paper.
