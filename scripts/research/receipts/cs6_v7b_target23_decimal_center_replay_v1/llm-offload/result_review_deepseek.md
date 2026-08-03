1. [BLOCKER] The stated verification claim "331/331 fine determinants were strictly negative" is not established by the supplied artifact, as the determinant values are computed by a non-rigorous RK4 integrator with no truncation-error bound.
   - Location: Result section, "331/331 fine determinants were strictly negative"
   - The work itself admits "Decimal arithmetic plus RK4 supplies no rigorous truncation-error enclosure," so claiming "strictly negative" for the computed values overstates what the computation demonstrates. The values could be positive or zero for the true trajectory.
   - Minimal fix: Replace "strictly negative" with "negative in all computed point estimates" or "negative at all tested center points at the specified numerical precision."

2. [MAJOR] The claim "Both coarse and fine point estimates lay strictly inside both retained CAPD Liouville enclosures on 331/331 leaves" is unfounded as a verification result because the point estimates are not rigorous.
   - Location: Result section, bullet point 3
   - A computed point lying inside an interval does not certify the point's true trajectory or prove the CAPD interval, as the artifact itself acknowledges in "What this does not establish." The word "strictly" implies rigor that the method cannot provide.
   - Minimal fix: Change to "Both coarse and fine computed point estimates lay inside both retained CAPD Liouville enclosures on all 331 leaves at the tested precision."

3. [MAJOR] The "maximum absolute coarse/fine discrepancy was 1.669821881242075889...e-18" is presented without statistical or numerical significance assessment.
   - Location: Result section, bullet point 4
   - The discrepancy is tiny relative to the determinant scale (~3e-11) but the artifact does not state whether this is expected given the step-size difference (2^-9 vs 2^-10) or whether it indicates a hidden bug. The ratio to CAPD width (0.0001646) is reported but not interpreted.
   - Minimal fix: Add a sentence explaining whether this discrepancy is consistent with RK4 convergence theory for the given step sizes and function smoothness.

4. [MAJOR] The "minimum fine-point distance to any CAPD endpoint was 5.070949335831988...e-15" is not meaningful for validation because the CAPD endpoints themselves are not rigorously established as bounds on the true determinant.
   - Location: Result section, bullet point 5
   - If CAPD's enclosures are correct, this margin is the relevant safety factor. But the artifact does not independently verify CAPD's correctness, so the margin is only meaningful assuming CAPD's output is trustworthy—which is the very thing being checked.
   - Minimal fix: State explicitly that this margin is conditional on CAPD's correctness and does not provide independent evidence for it.

5. [MAJOR] The "verifier rejected 14/14 deliberate evidence mutations" is insufficient to establish the verifier's reliability for the actual evidence.
   - Location: Result section, bullet point 8
   - The mutations are not described (what was mutated? how? in what format?), so the reader cannot assess whether the mutations are representative of plausible transcription or implementation errors. A verifier that only catches trivial formatting changes is not tested against substantive scientific errors.
   - Minimal fix: Describe the mutation types (e.g., sign flips, value perturbations, field swaps) and explain why they are representative of the error modes the audit targets.

6. [MAJOR] The "next falsification window" recommendation is internally inconsistent with the stated goal of the current work.
   - Location: Next falsification window section
   - The current work claims to be a "bounded pointwise falsification scout" and its results are presented as evidence against "simple implementation and transcription bugs." But the recommended next step (validated integrator) would test a different question (enclosure-level correctness), not the same question at higher fidelity. The text should clarify that the current work is not a stepping stone to the next, but a complementary check.
   - Minimal fix: Rewrite to state that the current work and the proposed next step are independent checks for different error classes, not sequential refinements.

7. [MINOR] The "failed attempts retained" section is presented as evidence of transparency but omits the actual error messages and the specific fixes.
   - Location: Failed attempts retained section
   - Without knowing what the empty `sys.executable` error was or how the portability repair was implemented, the reader cannot independently assess whether the fix could have introduced new errors.
   - Minimal fix: Include the relevant error tracebacks and the diff of the fix.

8. [MINOR] The "Durable evidence" section claims "pre-execution LLM reviews" are retained, but the artifact does not state what these reviews were or whether they were incorporated.
   - Location: Durable evidence section
   - If the LLM reviews flagged issues that were not addressed, their retention is misleading. If they were addressed, the changes should be documented.
   - Minimal fix: Briefly summarize the LLM review conclusions and any resulting changes.

9. [NIT] The schema identifier `SCHEMA=sounio.cs6.v7b-target23-decimal-center-replay-analysis.v1` is not explained or validated.
   - Location: Durable evidence section
   - The reader cannot verify that the schema is appropriate or that the data conforms to it without a schema definition.
   - Minimal fix: Include the schema definition or a link to it.

10. [NIT] The artifact does not state the exact CAPD enclosures it is comparing against, making it impossible to independently reproduce the "inside enclosure" claims.
    - Location: Result section
    - Without the CAPD interval values, a third party cannot verify the containment claims or assess the margin numbers.
    - Minimal fix: Include the CAPD enclosures in the receipt or reference them by hash.

**Verdict on the specific questions:** The prose correctly limits Decimal RK4 to pointwise falsification (the "What this does not establish" section is honest and precise). However, the result statements overclaim by using "strictly" and "inside" without the rigorous qualifier. The work does not claim global H-PG, V7-B, novelty, or open-problem solution, and correctly identifies the next step as a validated integrator. The audit does not PASS as written due to the overclaiming in the Result section; it would PASS if the language were tightened to match the stated limitations.
