1. [BLOCKER] The claim that "all implementation checks passed and all 22 negative mutations were rejected" is unsupported because neither the implementation, the mutation harness, nor the test definitions are supplied.
   Location: §Result, paragraph 2.
   Why it matters: A hostile referee cannot distinguish a sound verifier from an incomplete one that simply never emitted a passing mutant.
   Minimal fix: Deposit the exact test suite, mutation operators, and pass/fail logs at the cited commit.

2. [BLOCKER] The statement "aggregate dependence on all six original variables remained positive" is asserted without exhibiting the polynomials or the interval-arithmetic certificates that establish positivity after the depth-8 splits.
   Location: §What survived, bullet 3.
   Why it matters: The subsequent refusal is attributed to orientation ambiguity rather than loss of dependence, yet the dependence itself is not evidenced.
   Minimal fix: Output the six symbolic expressions together with their verified interval bounds at the refusal point.

3. [MAJOR] The report concludes that "the failure is therefore not symbolic-variable loss" while simultaneously reporting `XI=0` and `RHO3=0` split counts; the two observations are consistent only under an unstated assumption about how split selection weights width versus coefficient magnitude.
   Location: §What survived, final paragraph.
   Why it matters: Without the weighting rule or its proof, the inference is circular.
   Minimal fix: State the exact split-selection predicate and prove that zero split count is compatible with strictly positive dependence.

4. [MAJOR] Reproducibility is blocked: the artifact references Git commit `80442bb0d6`, job 9492, and an independent `Fraction`-based verifier, none of which are present or linked.
   Location: §Result, paragraph 2 and §Exhaustive precursor.
   Why it matters: A third party cannot reconstruct the receipt or the refusal class from the supplied text alone.
   Minimal fix: Provide a self-contained tarball or repository snapshot that regenerates the exact receipt.

5. [MINOR] The proposed "witness-local interval-Newton event projection" is described only at the level of three diagnostic cases with no termination argument or soundness statement for the adaptive bisection.
   Location: §Next experiment.
   Why it matters: The text explicitly disclaims a soundness theorem, yet the method is offered as the next step toward certification.
   Minimal fix: Either supply a machine-checked soundness lemma for the local procedure or label it strictly as an unverified heuristic.
