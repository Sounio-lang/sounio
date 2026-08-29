#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

bash scripts/ci/knowledge_context_phase2_ontology_gate.sh
bash scripts/ci/knowledge_context_unit_gate.sh
bash scripts/ci/knowledge_context_numeric_gate.sh
bash scripts/ci/knowledge_context_composite_gate.sh
bash scripts/ci/knowledge_context_static_value_gate.sh

cat <<'MSG'
Knowledge static proof-context umbrella gate passed.
This covers static ontology, derived-unit dimensional, numeric-refinement, and
composite proof contexts, plus literal Knowledge value initializers and simple
generated value constructors with direct literal, literal-argument, or narrow
const-evaluable numeric-expression evidence, including simple local bindings in
generated constructor bodies and simple linear assignment/compound-assignment
statements before return, plus narrow helper calls with const-evaluable
arguments and helper-local bindings/assignments, and literal-boolean branch
selection plus literal-boolean branch local flow in generated constructors.
The gate also covers simple const-evaluable numeric branch conditions over
literals and literal-backed parameters for branch local flow and direct-return
branch selection, plus narrow helper branch selection and helper-local branch
flow, plus narrow nested helper calls with caller parameter threading. Runtime
checks for dynamic Knowledge<T> fields and dynamic branch conditions remain
future work and are intentionally not claimed by this gate.
MSG
