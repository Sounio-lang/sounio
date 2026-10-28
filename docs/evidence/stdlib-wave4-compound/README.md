# Native compound coverage for generated stdlib fragments

The full runner now offers --compound-gates. It first preserves each ordinary standalone result, then executes five canonical native/C++ gates for the ten explicitly listed generated fragments. A row can become compound-native only if its standalone result is exactly skip/no-annotation. Existing checking, compilation, runtime failures and explicit ignores cannot be reclassified. New unlisted fragments remain skipped and strict acceptance still rejects them.

Each member requires a fresh successful gate process, STATUS0, a unique completion marker, and its exact original SHA256 in the gate input manifest. The original standalone row is retained as standalone_result. Gate output directories, commands, script hashes and compiler hashes are recorded in compound_producers. Historical replay19392 is unchanged.

Compound mode binds every standalone invocation to a closed generated launcher for the required absolute MADAROS_RAW_BIN ELF. The launcher and raw are hashed before execution and checked again afterward; each standalone producer must report that launcher. Native ELFs are retained by test path before execution. The five gates receive the same raw overrides and must report that exact path/hash in COMPILER.sha256. The historical mode without --compound-gates keeps its existing launcher behavior.

SIGTERM cancellation is consulted between groups; completed and interrupted group receipts are written incrementally. A cancellation, missing member, source mutation, compiler mismatch or incomplete producer cannot produce an accepted aggregate.

Slurm19479 qualified the initial five-gate component and eleven metadata rejection controls. Review found missing cancellation and compiler linkage; these were fixed and Slurm19486 requalified the actual five native/C++ gates with the closed binding. It also refused fourteen malformed/failed receipt controls, absent selection/binding controls, a launcher mutation, and stopped before starting a producer when cancellation was pending. A real standalone XOR test passed through a raw/retention path containing an apostrophe, despite conflicting inherited raw overrides; its ELF was retained. Final full source/tool guards passed. The component report covers ten fragments only, not the full733-file suite. A full invocation of the integrated runner remains part of the next whole replay.

Use inside an allocated worker with the identified raw and SDK:
node scripts/stdlib/run_full_suite.mjs --out-dir ABS_FRESH --jobs 4 --partition file --slow --acceptance --compound-gates
File partition requires a source archive, not a Git worktree.
