# Sounio repository scale

**Do not guess.** Run:

```bash
bash scripts/dev/measure_repo_scale.sh
bash scripts/dev/measure_repo_scale.sh --json artifacts/audit/repo_scale.v1.json
```

## Headline numbers (regenerate before citing)

Measured 2026-10-05 with `bash scripts/dev/measure_repo_scale.sh` on the tree of
`main` at `99d078eb` (plus one commit that touches no `.sio` file). The file and
line counts are sums over `git ls-files '*.sio'`.

| Metric | Measured value |
|---|---:|
| Tracked `.sio` files | 8,777 |
| Tracked `.sio` lines | ~3,367,236 |
| `self-hosted/` lines | ~639,419 (554 files) |
| `stdlib/` lines | ~642,602 (1,680 files) |
| `tests/` lines | ~313,370 (4,696 files) |
| `examples/` lines | ~158,555 (632 files) |
| CI gate scripts (`scripts/ci/*gate*.sh`) | 609 |
| Working tree disk | ~1.0 GB for a fresh worktree; a working checkout with `formal/.lake` and build artifacts is several times larger |

## What people get wrong

1. **`stdlib/` is not the whole language** — it is ~19% of tracked `.sio` lines.
2. **`self-hosted/` alone exceeds many "whole repo" estimates** cited in blog posts.
3. **Test pass badges** (e.g. 814/910) measure harness inventory, not "every module works".
4. **Gate script count ≠ CI coverage** — see `docs/audit/README.md` phase A.4.

## For LLMs

Read `llms.txt` § Repository scale before describing project size or maturity.

Full audit trail: [`docs/audit/README.md`](docs/audit/README.md).
