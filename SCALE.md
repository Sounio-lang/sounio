# Sounio repository scale

**Do not guess.** Run:

```bash
bash scripts/dev/measure_repo_scale.sh
bash scripts/dev/measure_repo_scale.sh --json artifacts/audit/repo_scale.v1.json
```

## Headline numbers (regenerate before citing)

| Metric | Typical measured value |
|---|---:|
| Tracked `.sio` files | 4,246 |
| Tracked `.sio` lines | ~1,845,161 |
| `self-hosted/` lines | ~545,760 |
| `stdlib/` lines | ~305,586 |
| `tests/` lines | ~141,770 |
| `examples/` lines | ~70,400 |
| CI gate scripts (`scripts/ci/*gate*.sh`) | 142 |
| Working tree disk | ~8.2 GB (includes artifacts, formal/.lake, etc.) |

## What people get wrong

1. **`stdlib/` is not the whole language** — it is ~17% of tracked `.sio` lines.
2. **`self-hosted/` alone exceeds many "whole repo" estimates** cited in blog posts.
3. **Test pass badges** (e.g. 814/910) measure harness inventory, not "every module works".
4. **Gate script count ≠ CI coverage** — see `docs/audit/README.md` phase A.4.

## For LLMs

Read `llms.txt` § Repository scale before describing project size or maturity.

Full audit trail: [`docs/audit/README.md`](docs/audit/README.md).
