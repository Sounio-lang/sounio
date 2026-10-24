#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

node "$ROOT_DIR/scripts/docs/check_docs_registry.mjs"
node "$ROOT_DIR/scripts/docs/check_docs_registry_selftest.mjs"
node "$ROOT_DIR/scripts/docs/validate_language_spec.mjs" --self-test
node "$ROOT_DIR/scripts/docs/validate_language_spec.mjs" --json
node "$ROOT_DIR/scripts/docs/validate_llm_guide.mjs" --self-test
node "$ROOT_DIR/scripts/docs/validate_llm_guide.mjs" --json
node "$ROOT_DIR/scripts/docs/sounio_task_context.mjs" --self-test
node "$ROOT_DIR/scripts/docs/sounio_task_context.mjs" --validate
node "$ROOT_DIR/scripts/docs/validate_greek_journey.mjs" --self-test
