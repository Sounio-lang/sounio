1. [BLOCKER] The supplied text asserts a certified refusal (`PRERECOND_NEXT_RETURN_REFUSED`) and preservation of the exact six-variable split lineage, yet supplies neither the Lean statements, the TM2R formalization, nor any `sorry`-free proof objects that would allow an independent verifier to confirm the classification.
   <location: entire document>
   The claim is therefore an uncheckable assertion rather than a reproducible certificate.
   <minimal fix> Deposit the exact Lean 4 modules, the `seek_upward_return` definition, the 22 negative-mutation tests, and the generated `.olean` or proof-term artifacts.

2. [BLOCKER] The report concludes that the experiment “cannot … support a chaos claim,” but nowhere states the formal predicate that would constitute such a claim or the theorem that would link a completed next-return certificate to it.
   <location: §Consequence>
   Without the statement, the negative result cannot be evaluated for relevance.
   <minimal fix> Exhibit the precise Lean theorem (including any `h-set`, recurrent-graph, or covering-edge predicates) that the missing certificate was intended to discharge.

3. [MAJOR] The witness domain is given with `rho3 ∈ [-1,1]` and zero splits on both `xi` and `rho3`, yet the text asserts “aggregate dependence on all six original variables remained positive.”
   <location: §What survived, table>
   The two assertions are inconsistent unless an unstated projection or marginalization step is applied.
   <minimal fix> Either tighten the domain bounds to reflect the actual splits performed or retract the dependence claim for the unsplit variables.

4. [MAJOR] The next-experiment proposal invokes an “interval-Newton event projection with adaptive time bisection” without specifying the enclosure arithmetic, the handling of the `SECOND_PRIOR_ORIENTATION_UNRESOLVED` tube, or any soundness statement that would guarantee the three-way discriminator terminates with a correct classification.
   <location: §Next experiment>
   A hostile referee will demand a machine-checked guarantee that the local refinement cannot silently accept an invalid return.
   <minimal fix> Supply the Lean formalization of the local Newton operator and its soundness theorem before any further compute is performed.

5. [MINOR] Job identifiers, node name, and wall-clock times are supplied, but neither the exact commit hash of the code base nor the random seed (if any) used by the enumerator is recorded.
   <location: §Result, §Exhaustive precursor>
   Exact reconstruction from the supplied files alone is therefore impossible.
   <minimal fix> Record the git SHA and any nondeterministic parameters in every job log.
