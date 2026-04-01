## Review of Paper A: "Non-Associative State Space Models: Octonion Dynamics for Path-Dependent Sequence Modeling"

### 1. Summary
This paper proposes a novel state space model, O-SSM, that utilizes octonion hidden states to capture non-associative dynamics in sequence modeling. The authors demonstrate the mathematical foundation of O-SSM, showcasing its ability to preserve norm and exhibit path-dependent behavior. They also present a benchmark suite where O-SSM outperforms diagonal and other baselines in 9 out of 11 tasks.

### 2. Strengths
* The paper introduces a novel and mathematically rigorous approach to sequence modeling, leveraging the properties of octonions.
* The authors provide an extensive mathematical foundation, including 31 experiments that verify the properties of O-SSM.
* The benchmark suite demonstrates the effectiveness of O-SSM in various tasks, including sMNIST, psMNIST, and Copy Task.

### 3. Weaknesses
* The paper assumes a significant amount of background knowledge in mathematics, particularly in the area of octonions and non-associative algebras, which may make it difficult for some readers to follow.
* The authors do not provide a clear explanation of how the octonion dynamics are used to capture path-dependent behavior in sequences, making it hard to understand the intuition behind the model.
* Some results, such as the 3.18x advantage in Morse Decoding, seem unusually high and may require further verification.
* The paper does not provide a clear comparison to other state-of-the-art models in sequence modeling, making it difficult to assess the significance of the results.

### 4. Questions for Authors
1. Can you provide a more detailed explanation of how the octonion dynamics are used to capture path-dependent behavior in sequences?
2. How do you plan to address the issue of parallel scan incompatibility in O-SSM?
3. Can you provide more insights into the optimization process and why momentum significantly improves the performance of O-SSM?
4. How do you plan to extend O-SSM to more complex tasks and larger datasets?

### 5. Missing References
The paper does not cite any prior work on non-associative algebras in machine learning, which may be relevant to the proposed approach.

### 6. Scores
* Soundness: 3
* Presentation: 2
* Contribution: 4
* Overall: 7
* Confidence: 4

### 7. Decision
Weak Accept

---

## Review of Paper B: "E-KAN: Analytical Uncertainty Propagation in Kolmogorov-Arnold Networks via GUM"

### 1. Summary
This paper proposes a novel approach to uncertainty propagation in neural networks, E-KAN, which utilizes the piecewise-linear structure of Kolmogorov-Arnold Networks (KANs) to enable exact first-order GUM propagation. The authors demonstrate the effectiveness of E-KAN on various UCI datasets and a pharmacokinetic modeling task, showing that it outperforms ensemble methods in terms of uncertainty estimation.

### 2. Strengths
* The paper introduces a novel and efficient approach to uncertainty propagation, leveraging the properties of KANs.
* The authors provide a clear and concise explanation of the mathematical foundations of E-KAN.
* The experimental results demonstrate the effectiveness of E-KAN in various tasks, including uncertainty estimation and coverage.

### 3. Weaknesses
* The paper assumes that the uncertainty is homoscedastic, which may not always be the case in real-world applications.
* The authors do not provide a clear explanation of how E-KAN handles out-of-distribution (OOD) data, which is an important aspect of uncertainty estimation.
* Some results, such as the 20x speedup over ensemble methods, may require further verification.
* The paper does not provide a clear comparison to other state-of-the-art methods in uncertainty estimation, making it difficult to assess the significance of the results.

### 4. Questions for Authors
1. Can you provide a more detailed explanation of how E-KAN handles OOD data and heteroscedastic noise?
2. How do you plan to extend E-KAN to more complex tasks and larger datasets?
3. Can you provide more insights into the ablation studies and the effects of different hyperparameters on the performance of E-KAN?
4. How do you plan to address the issue of interaction terms breaking the first-order Taylor approximation?

### 5. Missing References
The paper does not cite any prior work on uncertainty propagation in neural networks using GUM, which may be relevant to the proposed approach.

### 6. Scores
* Soundness: 4
* Presentation: 4
* Contribution: 4
* Overall: 8
* Confidence: 5

### 7. Decision
Accept

Note: The scores and decisions are based on the reviewer's assessment of the papers and may not reflect the final decision of the NeurIPS program committee.
