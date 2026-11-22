#!/usr/bin/env bash
# Inspect presence flags only; never accept or print a credential value.
set -euo pipefail

if [[ "${PUBLISH_REQUIRED:-true}" != true ]]; then
  echo mode=none
elif [[ -n "${APP_ID:-}" && "${HAS_APP_KEY:-false}" == true ]]; then
  echo mode=app
elif [[ "${HAS_PAT:-false}" == true ]]; then
  echo mode=pat
else
  echo mode=none
  echo "::error::Madaros refresh publication requires an approved repository-scoped credential: vars.MADAROS_REFRESH_APP_ID + secrets.MADAROS_REFRESH_APP_PRIVATE_KEY, or secrets.MADAROS_REFRESH_PAT (Contents and Pull requests read/write on this repository). Credential creation, installation or permission expansion requires maintainer approval. GITHUB_TOKEN is not a fallback because it does not trigger dependent CI. For build/gate inspection without publication, dispatch with artifact_only=true." >&2
  exit 1
fi
