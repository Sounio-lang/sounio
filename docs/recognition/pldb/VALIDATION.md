# Package validation receipt

Validated on 2026-11-10 (UTC), in an isolated Linux workspace checkout.

- Sounio evidence commit: `3765b7d024a46a67c7a13675ba1e92857cffb8bd`.
- PLDB source commit: `22cf13b127a581c6d0a25c59f93301a39c99cf7e`.
- Parser packages: `scroll-cli 178.2.3`, `scrollsdk 107.0.1`.
- Entry SHA-256: `4819c0c3f26ca7d51641f65fa3eefbecb138e0f70c6646224d965d5623867a31`.
- Installed dependency lock SHA-256: `42ef728dd3e5f8a0d4449ec494679a9fad972ff1ef79f0140fc365ea179f20ec`.
- Upstream `code/measures.parsers` SHA-256: `d7e09e816395c703df82a83e00f554b37f55cf07df2c96450881b27c1951ec8e`.
- Upstream `code/conceptPage.scroll` SHA-256: `351beb6df6e9eb13ab175fddf89620fb9e60cc02e869c3743b625cca3832d107`.

Command, run from the Sounio worktree:

```sh
node docs/recognition/pldb/validate.cjs /workspace/upstream-lab/pldb-source-20261110
```

Exit **0**, report **PASS**. PLDB's actual parser accepted the entry and rejected
both malformed controls (invalid year and unknown measure). Git source hashes,
text witnesses, field coverage, research-status wording and duplicate checks
passed. `node --check docs/recognition/pldb/validate.cjs` and
`git diff --check` also exited 0. A separate case-insensitive search for
Sounio, Demetrios and demetrios-lang across upstream `concepts/*.scroll` found
no matches at the pinned revision.

The full log and installed lockfile are preserved under
`/workspace/upstream-lab/receipts/SIO-608-recognition-20261110/`.
The dependency install used `--ignore-scripts --no-audit --no-fund`; tracked
upstream PLDB files were unchanged after validation. Installation and schema
parsing were lightweight workspace operations; no compiler build, Sounio
execution, numerical calculation, website build or publication was performed.

This receipt demonstrates preparation and schema acceptance at the named
revisions. It does not demonstrate language semantics, catalogue acceptance,
an external submission, CI completion or an internal merge. Those events need
their own receipts. SIO-610 and the recognition epic cannot be marked Done
from this receipt.
