<!-- docs:meta
topic_id: repo.docs.research.pireus-evidence-location
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-evidence-location
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Where the Pireus evidence corpus lives

The branch `pireus/consolidated` brings the Pireus language surface onto
`main`: stdlib modules, examples, tests and concept docs. It does not bring
the evidence corpus, meaning the bulk receipts, JSON observations, logs and
validation captures that the Pireus continuity harness produced between
2026-08-27 and 2026-09-21. This note says where that corpus is, how big it
is, and proposes a permanent home for it.

## What was left out

| Corpus | Size (measured 2026-10-06) | Where it is today |
|---|---|---|
| `tools/pireus/continuity/validation/**` | 3,728 files, 102.0 MiB of blobs, +343,194 lines. By extension: 1,635 `.json`, 679 `.py`, 144 `.stderr`, 138 `.sha256`, 138 `.receipt`, 136 `.tsv`, 132 `.log`, 122 `.txt` | PR [#2473](https://github.com/Sounio-lang/sounio/pull/2473) (branch `pireus/09-validation-evidence`) and tag `archive/2026-10-05/pireus/09-validation-evidence` |
| `tools/pireus/continuity/validation/external-overhead-v3-preparation-20260910/**` (CI envelope discovery and source coverage), plus `tools/pireus/continuity/ops/audit_journal_source_coverage.py` and its test | 36 files | tag `archive/2026-10-05/codex/pireus-inkling-cycle-20260906` only |
| Pireus CI wrappers (`scripts/ci/pireus_*.sh`, 75 of 76; the typed-XOR GPU gate is included, see below), cluster launchers (`scripts/dev/pireus_*`, 5), the Metal runner (`scripts/gpu-metal-validation/`, 2) and 5 `.github/workflows/pireus-*.yml` | 87 files | PR [#2474](https://github.com/Sounio-lang/sounio/pull/2474) (branch `pireus/10-workflow-wiring`) and tag `archive/2026-10-05/pireus/10-workflow-wiring` |

The CI wrappers are listed here, not carried by `pireus/consolidated`,
because they are not self-contained. Each one pins SHA-256 values of its
inputs and calls the Loom language-authority runtime
(`$GIT_COMMON_DIR/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime`).
That runtime is not in this repository (see
[`LOOM_EXTRACTION_PLAN.md`](LOOM_EXTRACTION_PLAN.md)). The one exception is
`scripts/ci/pireus_typed_xor_gpu_gate.sh`, which reads only in-repo sources.
It is included.

Already on `main` and not touched by this branch: the earlier slice of the
same corpus, about 600 files under `tools/pireus/` (`evidence/`,
`continuity/runtime`, `continuity/ops`, `continuity/reviews`, `continuity/ci`),
the small receipts under `docs/research/receipts/` and `docs/research/evidence/`,
and the six `tools/loom/evidence/pireus-*-acceptance-20260827.txt` files.
Some test `.args` files and the concept docs name these paths, so they have
to stay until the language-side references are rewritten.

## The archive tags are the source of truth

Every remote branch that existed on 2026-10-05 is preserved as a tag:

```bash
git fetch origin 'refs/tags/archive/2026-10-05/*:refs/tags/archive/2026-10-05/*'
git tag -l 'archive/2026-10-05/*' | grep -i pireus      # 43 tags
git ls-tree -r --name-only archive/2026-10-05/pireus/09-validation-evidence \
  -- tools/pireus/continuity/validation | wc -l          # 3728
```

The tags are enough to rebuild the corpus byte for byte after the branches
and PRs are closed. They are not a good place to read it from: nothing
indexes them, and a full clone of `Sounio-lang/sounio` fetches all of them.

## Proposal

Do not merge the evidence into `main`. A 102 MiB JSON/log corpus would grow
every clone and every `git archive`, and ordinary development would never
read it. Move it to one of these two places instead. Option A is the
recommended one.

**A. Release asset (recommended, low effort).** Cut a release named
`pireus-evidence-2026-09`. Attach one tarball made from the tags:

```bash
git archive --format=tar.gz -o pireus-evidence-2026-09.tar.gz \
  --prefix=pireus-evidence/ archive/2026-10-05/pireus/09-validation-evidence \
  -- tools/pireus/continuity/validation
sha256sum pireus-evidence-2026-09.tar.gz > pireus-evidence-2026-09.tar.gz.sha256
```

Attach the 36 inkling-cycle files as a second tarball made from
`archive/2026-10-05/codex/pireus-inkling-cycle-20260906`. Write the SHA-256
of each tarball into this file, so the language-side docs can cite one
immutable digest instead of thousands of paths.

**B. Separate repository `Sounio-lang/pireus-evidence`.** Use this if the
corpus will keep growing, or if the continuity harness (`tools/pireus/continuity`)
moves with it. Build it with `git filter-repo --path tools/pireus/continuity
--path scripts/ci/pireus_ --path .github/workflows/pireus-` on a clone that
has the archive tags checked out as branches. That keeps the history. Creating
the repository needs owner approval, so this branch does not do it.

With either option, PRs #2469, #2473 and #2474 can be closed once
`pireus/consolidated` is reviewed. #2469 (tests) and the language half of
#2474 (concept docs) are superseded by `pireus/consolidated`. #2473 is
superseded by the release asset or repository above.
