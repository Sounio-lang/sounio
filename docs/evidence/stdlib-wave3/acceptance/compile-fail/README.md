# Execute declared compile-fail tests

The canonical runner now recognizes //@ compile-fail plus //@ error-code: E... . Both check and compile must exit exactly1, both must include the named diagnostic, and compilation must leave no output file or symlink. A missing diagnostic, accepted program, crash, timeout, unexpected artifact or conflicting positive annotation is a failure. Complete captured phase logs retain the actual refusal exit codes.

Slurm19181 passes all four operational suites:11new refusal/mutant cases plus the existing capture, emitter and full-suite selection controls.19176 is preserved as a staging failure: the historical emitter JSON fixture was missing; the complete snapshot in19181 includes that exact pinned fixture. The reviewed runner projection hash was updated for this intentional behavior change; its historical result fixture is unchanged.

Slurm19183 exercises real source-built Madaros on the existing double_close(E039) and workspace_private(E259) controls:2pass/0fail/0skip, with check and compile refusals retained and no executable. These files previously had compile-fail annotations but were skipped by this runner. No negative test body or expected error code was changed.

Whole stdlib acceptance remains open. These negative controls establish refusal behavior and do not execute a native worker or authorize its issuance.
