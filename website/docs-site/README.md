# Sounio Documentation

This is the source for the public Sounio documentation site, built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Quick Start

```bash
pip install mkdocs-material mkdocs-macros-plugin

cd website/docs-site
mkdocs serve    # dev server at http://localhost:8000
mkdocs build    # static HTML -> site/
```

## Content Source

The docs content lives in `docs/` at the repo root. This `docs-site/` directory contains the MkDocs configuration and a thin set of wrapper pages that reference the canonical docs via `mkdocs-macros` includes.

**Do not duplicate content here.** Edit the canonical docs in `docs/`, then update the wrapper pages if navigation changes.

## Deployment

```bash
mkdocs build
# Deploy site/ to your hosting provider
```

For GitHub Pages:

```bash
mkdocs gh-deploy
```

## Structure

```
docs-site/
  mkdocs.yml          # site configuration
  content/            # thin wrapper pages (references docs/)
  hooks/              # status badge rendering hook
  requirements.txt    # Python dependencies
```
