<!-- docs:meta
topic_id: repo.docs.getting-started
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.getting-started
-->

# Getting Started with Sounio

This file is not the current walkthrough. Read [`guide/getting-started.md`](guide/getting-started.md).

Measured 2026-10-15 on the committed binary: `bin/souc --version` prints `Madaros v0.80.0`, not `1.0.0-beta.4`. Native `souc run` is a Linux x86-64 ELF. It does not run on macOS. There is no Cranelift JIT in this entry point.

```bash
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
"$SOUC_BIN" --version
"$SOUC_BIN" check examples/hello.sio
```

On Linux x86-64, `"$SOUC_BIN" run examples/hello.sio` prints `Hello, Sounio`.

Do not copy a GPU, checkpoint, or REPL recipe from an older revision of this file. Those commands were not re-run on 2026-10-15. `TOUR.md` §6 (2026-10-04) measured `souc build --backend gpu` as an empty-body PTX skeleton. The root [README.md](../README.md) records scoped Spark evidence for GPU/Pireus and says the generic GPU CLI needs its own status.

## Next

- [guide/getting-started.md](guide/getting-started.md)
- [../README.md](../README.md)
- [../INSTALL.md](../INSTALL.md)
- [../TOUR.md](../TOUR.md)
