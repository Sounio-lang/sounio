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

Git-triggered deployments are **off**: the repository-root `vercel.json` and `website/vercel.json` set `git.deploymentEnabled: false`, and both Vercel projects (`sounio`, `sounio-next`) have preview deployments disabled and an ignored build step of `exit 0`. Pushing to `main`, merging a PR or pushing a branch deploys nothing, so Vercel has no deployment to comment on. Pull-request comments themselves are a dashboard setting (Project Settings → Git → Pull Request Comments; `github.silent` in `vercel.json` is deprecated and not used here). This keeps Vercel builds out of the PR loop, where every push used to build the whole repository.

Verify locally before any deploy:

```bash
cd website
npm ci
npm run check:quality
npm run build
```

### 1. GitHub Actions (`vercel-website.yml`, authoritative)

- **Production:** push a `website-v*` tag (e.g. `website-v2026.10.07`) on the commit to publish, or run the workflow by hand with target `production`:
  `gh workflow run vercel-website.yml -f target=production`
- **Preview:** `gh workflow run vercel-website.yml -f target=preview`

The workflow runs `npm run check:quality`, then `vercel build` and `vercel deploy --prebuilt` against the `sounio` project. It needs the repository secret **`VERCEL_TOKEN`** and fails without it.

### 2. Manual CLI (emergency)

```bash
cd website
npm ci
npm run build
npx vercel build --prod
npx vercel deploy --prebuilt --prod
```

Requires Vercel CLI auth and project linkage (`vercel link` to `sounio`).

## Pre-deploy checklist

- [ ] `npm run check:quality` green in `website/`
- [ ] No `.vercel/output` staged in git
- [ ] `website/vercel.json` `outputDirectory` is `dist`
- [ ] Node **22.12+** (see `website/package.json` `engines`)

## Rollback

Reverting a commit on `main` does **not** redeploy. Either promote the previous production deployment (Vercel dashboard → Deployments → Promote, or `npx vercel rollback <deployment-url>`), or check out the last good commit, tag it `website-v<date>-rollback` and push the tag so the workflow redeploys it.

## Common failure: stale static output

If production shows old homepage copy after a source change, check whether `.vercel/output/` was accidentally committed. Remove it from git, add `.vercel/` to `.gitignore`, and redeploy from a clean `npm run build`.
