# Invalidate stale integer constants on scoped assignment

Assignment previously appended a scoped constant entry. Leaving the block discarded that update and exposed the stale outer initializer. CSV/JSON digit emission then received a false E056: pow remained known as1 after its multiply-by10 loop, so place appeared zero inside a guarded loop.

Both checker spines now forget the nearest visible constant entry before adding the assignment's new scoped value. A shadowing declaration remains independent from its outer declaration. This is conservative cache invalidation; it does not claim a complete constant propagation or alias analysis.

Slurm19240 checks the actual checker kernel built from the production source plus the paged borrow implementation. Four formerly rejected valid programs (loop/block/branch/nested assignment) now pass. Five genuine-zero/shadow cases still reject with E056, and the independent shadow-positive case still passes. The same checker passes all51 ownership verdicts and their original diagnostics, seven wrong-output controls, up to512 bindings and24-level balanced/asymmetric branch controls. Bootstrap compilation has no explicit errors.

The original wide source check19200 using rawbd4ba240 timed out at1200s while reporting borrow E038 in bind_fn_param; it did not qualify the checker source and did not run these cases. It predates the paged borrow overlay. Current source-wide and original multi-module CSV/JSON replay are still required against an integrated rebuilt compiler; this receipt establishes the component behavior only.

Portable reproduction on Slurm:

    BEFORE_RAW_BIN=/absolute/before/madaros bash scripts/ci/wave4_constant_scope_gate.sh /absolute/legacy/seed /absolute/fresh/results

No expected numerical value, test body assertion or corpus deadline was changed.
