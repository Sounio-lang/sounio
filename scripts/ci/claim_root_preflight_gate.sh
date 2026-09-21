#!/usr/bin/env bash
# Claim-root preflight. Reads receipts. Does not run any other gate.
# Not wired: putting this in Contracts before measuring receipts exist
# would redden every PR for unprotected public numbers. That is honest,
# and it is a separate decision from landing the instrument.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec python3 scripts/dev/claim_root_preflight.py
