# Explicit shared reborrows in the profiler

The two calls to profiler_find_slot in profiler_begin and profiler_end now use `&*p`. Their input remains `p: &!Profiler`, and all other code and signatures are unchanged. Casting that affine mutable reference with `p as &Profiler` consumes it under the existing checker; the later reads/writes then produce E039. The explicit shared reborrow ends after the lookup and leaves the mutable reference available for the existing updates.

The production delta is exactly two changed lines. The original tests/stdlib/profile/test_profiler_e2e.sio is byte-identical to the retained fixture. Its assertions, workload, status checks and 30-second native deadline were preserved. The cast compiler implementation was not changed.

Slurm 19245 (loop regions 1) and 19252 (default loop regions 0) both pass:

- Original profiler: check 1 and compile 1 with E039; no executable emitted.
- Corrected profiler: check 0, compile 0, native 0, exact PROFILE_CONTRACT_OK marker.
- Cast reduction still refuses E039.
- Explicit reborrow and existing implicit coercion reductions check/compile/run successfully.
- Mutation while a shared reborrow is still live refuses E038.

The native run reuses the C++23 platform DSO qualified by the frozen whole-stdlib baseline 19073. No runtime implementation changed. DSO SHA256 is `358c7076be264c069d77da82e79fc0313b27e6ea255c8ab343a9f89ffb0d0a4e`. Its source and baseline submission-script hashes are retained in each receipt.

Both runs use raw compiler `826d9c18efbc8dc878633c355a8a1e76985aa40fa7faa47490e3544ecddb9143`, from the experimental proto03 build with the paged borrow checker. Source, tool and executable guards pass. These are focused profiler receipts; the unrelated original BigFrame native 181/139 and whole-stdlib acceptance remain open.

Reproduce inside Slurm with `bash tests/fixtures/wave4_profiler_reborrow/gate.sh RAW_COMPILER SEALED_PLATFORM_DSO FRESH_RESULTS`. Loop regions default to 0; set SOUNIO_LOOP_REGIONS=1 explicitly for the additional experimental-mode comparison. All source/DSO inputs and execution logs are retained under /scratch/sounio-stdlib-wave3-20261027/services/profiler-reborrow01 and profiler-reborrow02. Durable export19266 passed. The complete sources, ELF files, DSO and receipts are in /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/borrow-profiler-export01/borrow-profiler.tar.gz, SHA2566ee48112220c17f9008b00264e7c58eddf00afd1eee32422c06f8d1f98902edb. Shared archive/Slurm verification is versioned in ../stdlib-wave4-borrow-capacity/custody/.
