# Worker region envelopes: qualified components, integration pending

The root raw `8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932` rejects the unchanged imported direct callback with identity=0, capability=0, closure=119. Its lowered `product_into` replaces the unused region-enter operation with an ordinary integer zero and retains a guarded scalar-cleanup epilogue. The original real loader gate is retained as the integration baseline (Slurm19500); no production opt-in is added here.

The proof delta distinguishes a uniquely defined integer zero from arbitrary scalar provenance. The register cannot be a parameter or float-marked, and every CFG path reaching the cleanup must define it as literal zero. Scalar result provenance and the existing unused-Unit restriction remain required. Ordinary CFG/capability admission uses its unchanged definition mode.

A real frame-token conditional is accepted only for the canonical four-instruction branch/cleanup/label/return envelope. The empty enter/cleanup helpers must have their existing backend identities, flags and arities. The enter is the unique definition, must dominate the guard, and the scalar value is defined at both guard and return. The token has no uses outside this branch and cleanup. Other token branches and arithmetic remain refused. This intentionally restrictive two-use rule has not yet been qualified on the canonical normal kernels with early returns.

| Receipt | Scope | Result |
|---|---|---|
|19529|Unchanged raw8abe direct callback IR dump|Literal-zero token at instruction0, guarded cleanup at19/20, matching label21 and return22|
|19559|Modern checks and clean legacy-native before/after component|Before zero119/native1 and frame116/native1; after native0;18 new mutants refused|
|19555|Native frame-envelope model, actual emitter and C++23 oracle|11 positives,14 invalid words,3 wrong controls; original managed-region controls SIGSEGV under PROT_NONE; cloned helpers inaccessible as entries|
|19555|Mixed serial/cloned native stress, twice for unused Unit variants|Four threads,50000 serial and150000 cloned calls each run; zero value errors and unchanged runtime-context slot|
|19574|Same unmodified ELF/helper exit67 through canonical thunk|Direct token0 returns with PROT_NONE; direct token1 SIGSEGV; no outer guard is used for these calls|
|19538|Unmodified raw8abe emitted zero-token callbacks and canonical ABI thunk|Eight correct zero/plus calls with every writable ELF segment PROT_NONE; real-region control SIGSEGV|
|19563|Full common component gate|OOM at6GiB in modern native-driver check; no verdict for that phase|

The after component also retains15 earlier IR mutations,9 CFG corruptions,3 positive CFG merges/loop, empty-extern namespace refusal, metadata page/layout canaries and retirement/address-reuse controls. The metadata is injected by the component; this does not establish real loader/BFS custody.

The new native-frame fixtures are separate from the older stack fixtures so their Unit140 baseline remains intact. The runtime oracle must call the canonical Sounio callback thunk; calling Sounio entrypoints directly as C functions is invalid for this ABI. Earlier runtime harness attempts19535 (direct C call) and19537 (pedantic rejection of the existing runtime GNU integer extension), and argv-dependent component19548, remain rejected attempts rather than accepted evidence.

The actual preflight through module_frontend/native with literal source paths is tracked separately. Slurm19569 rejected missing legacy cwd imports;19570 rejected an explicit E001 in the private dump before executing its ELF. Corrected helper modern check precedes the next literal producer in19575. Runtime scanners19572/73 rejected an embedded displacement byte as a CALL before executing callbacks;19574 filters executable targets and requires one matching test-rdi/jz helper entry. It retains source origins, source-context binding, dependency arena resets, final-IR reproof and protected-address allowlists. No callback API, pool, normal pipeline, full stdlib corpus or production CLI is certified by these component receipts. Storage is excluded.

Sources and seeds are identified in the receipt manifests. Full source trees, ELF files, before/after guards and disassembly remain under `/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/{worker-region-envelope03,worker-frame-native01,worker-zero-runtime03}` pending durable export.
