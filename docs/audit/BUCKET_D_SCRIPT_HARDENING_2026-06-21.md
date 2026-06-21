<!-- docs:meta
topic_id: repo.docs.audit.bucket-d-script-hardening-2026-06-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.bucket-d-script-hardening-2026-06-21
-->

# Bucket D Script Hardening — 2026-06-21

## Scope

This note resolves the primary-checkout reconciliation Bucket D disposition.
Bucket D contains helper scripts found in the dirty protected checkout. They are
not automatically production source.

Current baseline:

- `origin/main` at `45a1e19310ba24005d1b6c83676828945f7fd9ca`
  (`Merge pull request #350 from Sounio-lang/codex/madaros-audit-docs`).
- Archive/source for local-only candidates: protected `/workspace/sounio`.

## Disposition

| Item | Current state | Disposition |
|---|---|---|
| `scripts/coverage_report.sh` | not present on `origin/main`; archived script has a malformed `basename " $f" .mdx` line and no CI contract | do not promote raw; replace only with a tested website coverage command |
| `scripts/validate_mdx.sh` | not present on `origin/main`; archived script mutates dependencies with `npm install --no-save` and uses uninitialized counters under `set -u` | do not promote raw; use existing website quality checks until a no-mutation validator is added |
| `scripts/translate_all_locales.sh` | not present on `origin/main`; archived script assumes `.venv`, a cluster-internal Beagle URL, and writes logs into repo root | do not promote raw; future version must be operator-only and fully parameterized |
| `slurm-jobs/madaros-frame-fix/fetch_stack_fix.sh` | present on `origin/main` and identical to archive | already represented; no replay |
| `slurm-jobs/madaros-frame-fix/submit_stack_fix.sh` | present on `origin/main`; archive copy is stale compared with current diagnostic no-ulimit contract | keep current `origin/main`; no replay |

## Required Gates Before Any Future Script Promotion

Any future PR that promotes one of the three local-only scripts must include:

- a scoped script path under `scripts/dev/`, `scripts/ops/`, or `scripts/website/`
  instead of a vague root-level helper,
- `bash -n` for shell syntax,
- `shellcheck` when available, or a documented reason if unavailable in the
  workspace image,
- a dry-run mode that does not mutate dependencies, write repo-root logs, or
  contact internal services by default,
- a CI or documented operator contract naming exactly when the script is safe
  to run,
- no hardcoded cluster-internal API endpoint unless it is behind an explicit
  environment variable with a safe default.

## Concrete Future Shapes

### Website Coverage

If locale coverage is needed, implement it as a deterministic report command
that reads `website/src/content/**` and prints counts only. It should not write
files and should pass:

```bash
bash -n scripts/website/coverage_report.sh
scripts/website/coverage_report.sh
```

### MDX Validation

If a standalone MDX validator is still useful beyond the website workflow, it
must use checked-in website dependencies. It must not run `npm install`.
Acceptance gate:

```bash
cd website
npm ci
npm run quality
```

A separate shell wrapper may call the website command, but should not become a
second dependency manager.

### Locale Translation

Translation must be operator-only, not CI. A future helper should require:

- `OPENAI_API_BASE`
- `MODEL_NAME`
- `TARGET_LOCALE` or an explicit locale list
- `TRANSLATION_LOG_DIR` outside the repo by default

It must support `--dry-run` and refuse to run when the API base/model is not
explicitly supplied.

### Slurm Frame-Fix Helpers

The Slurm frame-fix helpers are already source-controlled. Changes to them must
be validated against the current foundry/Slurm contract and should not be mixed
with website/locale tooling.

## Result

Bucket D is classified, but no raw script is promoted. This keeps `origin/main`
clean while preserving the exact path to productionize any helper that is still
needed.
