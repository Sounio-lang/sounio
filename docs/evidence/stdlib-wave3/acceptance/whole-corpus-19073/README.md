# Whole corpus baseline19073

Canonical runner inputs were preserved. Slurm19073 ran the730-file test selection with file partitioning,8workers, slow tests enabled and acceptance mode. The raw compiler was af5dd8b4a99263f863a8ef76132abf39a2583ab06e4e86785374f6f9d53ac1d3 with SOUNIO_LOOP_REGIONS=1. This predates the closure isolation correction and the structural borrow environment rewrite.

Result:455pass,54fail,221skip. The221skips are all no-annotation cases, not approved exclusions. The acceptance producer reports not_run because the selection contains skipped cases. It preserves all730 rows. Failure metadata separates19check failures,11compile failures (10exit139,1exit1),16native timeouts and8other native failures. Cause attribution requires focused replay; these categories alone do not establish regressions or fixes.

CLASSIFICATION.json preserves every failure excerpt and every unclassified skipped path. Auxiliary programs, negative controls and executable entrypoints still need explicit applicability review. No annotation, test size, expected value, arena capacity or deadline was changed for this baseline.

Durable export job19101 completed. Archive: /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/stdlib-wide-loop-export01/stdlib-wide-loop.tar.gz

Archive SHA256:0f3db41a7ab74eab231234b3fc57eeb2a03f65971d064d12639955026e6e91c6. The export includes compiler and Node identity, sources, captured outputs and generated native executables. Full stdlib acceptance is open.
