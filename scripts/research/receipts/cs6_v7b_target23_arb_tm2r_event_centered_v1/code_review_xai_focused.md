1. [BLOCKER] receipt_dir hard-codes the wrong prior directory name, so the script cannot locate the expected frozen receipt on disk.
   <location: scripts/research/cs6_v7b_target23_arb_tm2r_event_centered_worker.py:250>
   <why it matters> The entire verification chain rests on `prior_receipt.is_file()` and the subsequent SHA-256 check; a missing file aborts before any mathematical claim is evaluated.
   <minimal fix> Change the literal to `"cs6_v7b_target23_arb_tm2r_event_centered_v1"` (or make it a parameter) and update the expected SHA if the file moved.

2. [BLOCKER] Reproducibility claim fails: the worker imports five modules (`carrier`, `base`, `adaptive`, `chain`, `event`) whose sources are absent from the supplied artifact.
   <location: scripts/research/cs6_v7b_target23_arb_tm2r_event_centered_worker.py:14-18 and throughout>
   <why it matters> A hostile referee or independent verifier cannot reconstruct the analysis from the three files provided; all interval-arithmetic and TM2R invariants live in the missing modules.
   <minimal fix> Either include the full transitive closure of imported sources or ship a single, self-contained snapshot.

3. [MAJOR] The disposition asserts that “the worker now constructs an exact rational interval R” and that the Newton image lies strictly inside it, yet the verifier only checks the summary flags rather than re-executing the interval arithmetic itself.
   <location: scripts/research/cs6_v7b_target23_arb_tm2r_event_centered_verify.py:140-160 and disposition §“Adopted mathematical finding”>
   <why it matters> The soundness theorem (strict inclusion + negative derivative) is accepted on trust of the worker’s JSON; any mis-computation of `newton_domain` or `derivative` is invisible to the verifier.
   <minimal fix> Port the critical interval-Newton predicates into the verifier or embed a machine-checked certificate.

4. [MINOR] `retained_source_monomials` iterates over rows `(0,1,3)` while `variable_weights` defaults to `(0,1,2,3)`; the two functions therefore apply inconsistent row masks to the same preservation invariant.
   <location: scripts/research/cs6_v7b_target23_arb_tm2r_event_centered_worker.py:66 and 48>
   <why it matters> The disposition claims “preservation of all six normalized variables … is a separate positive-weight gate on those three retained rows,” but the code mixes the two masks without comment.
   <minimal fix> Add an explicit comment or factor the row set into a single constant used by both functions.
