# Website deploy runbook

Production: **https://www.souniolang.org**

## Build surface

| Item | Value |
|------|-------|
| Project root (Vercel) | `website/` |
| Install | `npm ci` |
| Build | `npm run build` |
| Output directory | `dist/` |

**Never commit** `.vercel/output/` or prebuilt Vercel artifacts into git. Vercel builds from source on each deploy.

## Deploy paths

### 1. Vercel Git integration (preferred)

Push to `main` (or merge a PR). The Vercel project linked to this repo runs `npm run build` inside `website/` and publishes `dist/`.

Verify locally before push:

```bash
cd website
npm ci
npm run build
```

### 2. GitHub Actions (`vercel-website.yml`)

Workflow exists but requires repository secret **`VERCEL_TOKEN`**. Without it, the workflow fails — this does **not** block Git-integration deploys.

### 3. Manual CLI (emergency)

```bash
cd website
npm ci
npm run build
npx vercel deploy --prebuilt
```

Requires Vercel CLI auth and project linkage.

## Release tags

Optional website-only tags: `website-v*` (e.g. `website-v2026.05.26`). Tagging is informational; production still tracks `main` via Git integration unless you configure otherwise.

## Pre-deploy checklist

- [ ] `npm run check:quality` green in `website/`
- [ ] No `.vercel/output` staged in git
- [ ] `website/vercel.json` `outputDirectory` is `dist`
- [ ] Node **22.12+** (see `website/package.json` `engines`)

## Rollback

Revert the offending commit on `main` and let Vercel redeploy, or use the Vercel dashboard to promote a previous deployment.

## Common failure: stale static output

If production shows old homepage copy after a source change, check whether `.vercel/output/` was accidentally committed. Remove it from git, add `.vercel/` to `.gitignore`, and redeploy from a clean `npm run build`.
