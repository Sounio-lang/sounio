#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REFRESH_ONTOLOGY=0
WITH_K2_CLASSIFIER=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ci/semantic_knowledge_spine_gate.sh [--refresh-ontology] [--with-k2-classifier]

Runs the focused acceptance surface for the ontology -> units -> Knowledge<T>
spine:

  1. generated ontology manifest and witnesses;
  2. pre-native ontology-bundle directive expansion through the C importer;
  3. compiler-side ontology-bundle directive source scanning;
  4. focused imported .ontocache + frontend Knowledge<T> composition;
  5. unit dimensional-analysis gates;
  6. ontology-linked unit metadata as validation data;
  7. static Knowledge<T> proof-context gates;
  8. dynamic Knowledge<T> runtime-obligation classification;
  9. pre-native dynamic Knowledge<T> runtime guard expansion, including
     checker-obligation drain coverage across return/call/assignment sites and
     supported guard families, plus native compile/run coverage for generated
     guard assertions, opt-in bin/souc launcher integration, and the
     source-level knowledge-runtime-guards directive;
 10. compiler-side raw-source scanning for the knowledge-runtime-guards
     directive as an intent classifier for pre-native expansion;
 11. compiler-side Knowledge<T> runtime guard lowering-plan classification
     joining directive intent, semantic verdicts, runtime obligations, and
     first-obligation payload extraction plus staged backend-guard row counts;
 12. optional K2 check::check bridge classifier when --with-k2-classifier is set.

This is an umbrella gate, not a claim that runtime guard/trap insertion,
ordinary source-file checker-kernel cache hydration, same-Checker
ontology-kernel cache hydration, or the standalone check::check import bridge
are complete.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --refresh-ontology)
      REFRESH_ONTOLOGY=1
      shift
      ;;
    --with-k2-classifier)
      WITH_K2_CLASSIFIER=1
      shift
      ;;
    *)
      echo "error: unsupported argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$REFRESH_ONTOLOGY" -eq 1 ]]; then
  bash scripts/ci/generated_ontology_gate.sh --refresh-only
fi

bash scripts/ci/generated_ontology_manifest_gate.sh
bash scripts/run_sio_test_suite.sh ontology_generated --verbose
bash scripts/run_sio_test_suite.sh ontology_typed_bridge --verbose
bash scripts/ci/ontology_bundle_directive_gate.sh
bash scripts/ci/ontology_bundle_directive_native_scan_gate.sh
bash scripts/ci/ontology_cache_frontend_composition_gate.sh
bash scripts/ci/unit_types_phase1_gate.sh
bash scripts/ci/unit_types_derived_gate.sh
bash scripts/ci/ontology_unit_metadata_gate.sh
bash scripts/ci/knowledge_context_static_gate.sh
bash scripts/ci/knowledge_context_runtime_obligation_gate.sh
bash scripts/ci/knowledge_runtime_guard_expansion_gate.sh
bash scripts/ci/knowledge_runtime_guard_directive_native_scan_gate.sh
bash scripts/ci/knowledge_runtime_guard_lowering_plan_gate.sh
if [[ "$WITH_K2_CLASSIFIER" -eq 1 ]]; then
  bash scripts/ci/k2_check_import_bridge_classifier.sh
fi

cat <<'MSG'
Semantic Knowledge Spine gate passed.
This umbrella proves the current static path from generated ontology stubs
through dimensional unit checking, ontology-linked unit metadata validation,
pre-native ontology-bundle directive expansion, compiler-side raw-source
scanning for ontology-bundle directives with bounded compact side-table
staging and bounded OntologyKernel-like fact materialization plus bounded
native side-table .ontocache write/read rehydration and source-level cache
directive import with checker-shaped semantic verdicts for those staged facts,
including check::mod diagnostic consumption through a check-owned native
side-table cache reader plus focused check::check live-kernel hydration into
Checker.ontology_kernel, an imported check::check sourcecheck collect-boundary
classifier, and a check::mod sourcecheck cache-verdict bridge with positive
and reverse bounded subclass verdicts after ordinary sourcecheck plus a
check::mod same-sourcecheck sidecar that accepts SNOMED subclass and PART
subclass/disjoint verdicts after ordinary sourcecheck plus a focused
check::mod Knowledge<T> proof-context sidecar over imported .ontocache facts
that resolves generic PREFIX_<numeric-id> proof-context names, accepts bounded
SNOMED/PART positive proof contexts, and rejects reverse SNOMED plus disjoint
PART proof contexts through a boot4-style source-path verdict API after
ordinary sourcecheck, with the modular frontend now composing that imported
ontology sidecar with the ordinary value-only check::knowledge_context validator
through a focused .ontocache frontend-composition gate so numeric/unit
proof-context constraints are still checked in files that declare an
ontology-side-table cache directive, and so PART imported-cache subclass plus
disjoint rejection proof contexts, QM imported-cache subclass plus reverse
subclass rejection proof contexts, and generic ALG/ChEBI/GO/HPO/PHYS
slot3-to-slot0 subclass plus reverse-subclass rejection proof contexts are
checked through the same frontend path, while LOINC uses synthetic positive
cache-slot IDs for alphanumeric CURIE tails and checks cache_slot4-to-cache_slot5
subclass plus reverse-subclass rejection through the same frontend path,
plus a
same-Checker ontology-kernel hydration classifier that reaches hydrate_ok and
still rejects the positive kernel verdict plus a same-Checker copy classifier
that reaches copy_ok and rejects the positive same-Checker verdict plus a
same-Checker pointer-query classifier that reaches copy_ok before failing to
produce a query verdict plus a same-Checker precheck classifier that reaches
collect_ok/hydrate_ok before class_count readback remains unsafe plus a
same-Checker shadow-field classifier that reaches hydrate_ok but reads
shadow_count=0/empty shadow state, alongside
static Knowledge<T> proof contexts, checker-visible runtime obligations for
dynamic Knowledge<T> proof-context values and a pre-native executable runtime
guard expansion for numeric lower-bound, upper-bound, and equality comparison
slices, including checker-obligation drain coverage across return,
call-boundary, assignment, conjunctive, upper-bound, equality, unit-suffixed,
and internal-label guard families plus native compile/run coverage for those
generated assertion paths after pre-native expansion, including the opt-in
SOUNIO_KNOWLEDGE_RUNTIME_GUARDS=1 launcher bridge and the source-level
//@ knowledge-runtime-guards directive for compile/run, plus a compiler-side
raw-source scanner that sees that directive before comments are dropped and
classifies it as pre-native expansion intent while explicitly reporting
native_lowering_ready=0, plus a compiler-side runtime guard lowering-plan
surface that joins that intent with ordinary parser/checker semantic acceptance
and checker-visible runtime obligation counts, preserving first-obligation
payload for return/call/assignment sites and unit-suffixed constraints, and
mapping dynamic descriptors to staged backend-guard rows with comparison
opcode, threshold kind, and trap exit code, including one staged row per
checker-visible obligation in conjunctive dynamic proof contexts, while keeping emitted
backend_guard_count=0 until direct native lowering exists. The
standalone K2 check::check import bridge classifier is intentionally optional
because it tracks a separate modular-checker bridge gap. Direct native backend
proof-obligation guard/trap lowering, same-Checker ontology-kernel or
shadow-field cache hydration, and ordinary source-file checker-kernel cache
hydration remain future work.
MSG
