# Bind complete source inputs in every full-suite worker copy

Per-file workers now bind every file under tests, stdlib, scripts and self-hosted when present. This covers auxiliary CSV/JSON inputs and orchestration scripts, as well as Sounio files. The original source archive is also rechecked at completion. Existing strict stdlib symlink handling remains.

Slurm19376 executes the same new operational controls against the old and new runner. The old guard accepts the changed script marker and the test deliberately fails; the new guard refuses changes to both script and CSV fixtures and passes all existing isolation/selection/skip controls. Failed test-syntax attempt19371 is retained separately and did not qualify a guard. These are metadata test doubles, not native/numerical claims.
