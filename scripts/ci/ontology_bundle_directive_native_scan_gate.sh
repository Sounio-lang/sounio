#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_ontology_bundle_directive_scan_probe.sio"
CHECK_MOD_CACHE_PROBE="self-hosted/compiler/k2_check_mod_ontology_cache_directive_probe.sio"
LIVE_KERNEL_CACHE_PROBE="self-hosted/compiler/k2_check_import_ontology_cache_kernel_probe.sio"
SOURCECHECK_CACHE_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_import_ontology_cache_sourcecheck_classifier_probe.sio"
CHECK_MOD_SOURCECHECK_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_ontology_cache_sourcecheck_classifier_probe.sio"
CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_ontology_cache_sourcecheck_reject_classifier_probe.sio"
CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_same_checker_ontology_cache_kernel_classifier_probe.sio"
CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_same_checker_ontology_cache_kernel_copy_classifier_probe.sio"
CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_same_checker_ontology_cache_kernel_ptr_query_classifier_probe.sio"
CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_same_checker_ontology_cache_kernel_precheck_classifier_probe.sio"
CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe.sio"
CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_PROBE="self-hosted/compiler/k2_check_mod_knowledge_context_ontocache_sidecar_probe.sio"
CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_same_checker_ontology_cache_shadow_classifier_probe.sio"
CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_PROBE="self-hosted/compiler/k2_check_mod_same_checker_ontology_cache_shadow_reject_classifier_probe.sio"
POSITIVE_SRC="tests/frontend/ontology_bundle_directive_snomed.sio"
PART_SRC="$TMP_DIR/ontology-bundle-directive-part.sio"
PART_LOG="$TMP_DIR/ontology-bundle-directive-part.log"
PART_CHECK_MOD_LOG="$TMP_DIR/ontology-bundle-directive-part-check-mod.log"
PART_LIVE_KERNEL_LOG="$TMP_DIR/ontology-bundle-directive-part-live-kernel.log"
PART_CACHE="$TMP_DIR/ontology-bundle-directive-part.native.ontocache"
PART_CACHE_SRC="$TMP_DIR/ontology-bundle-directive-part-cache-consumer.sio"
PART_KNOWLEDGE_POSITIVE_SRC="$TMP_DIR/ontology-bundle-directive-part-knowledge-positive.sio"
PART_KNOWLEDGE_DISJOINT_REJECT_SRC="$TMP_DIR/ontology-bundle-directive-part-knowledge-disjoint-reject.sio"
LOINC_SRC="$TMP_DIR/ontology-bundle-directive-loinc.sio"
LOINC_LOG="$TMP_DIR/ontology-bundle-directive-loinc.log"
LOINC_CACHE="$TMP_DIR/ontology-bundle-directive-loinc.native.ontocache"
LOINC_CACHE_SRC="$TMP_DIR/ontology-bundle-directive-loinc-cache-consumer.sio"
LOG="$TMP_DIR/ontology-bundle-directive-native-scan.log"
CHECK_MOD_LOG="$TMP_DIR/ontology-bundle-directive-check-mod.log"
LIVE_KERNEL_LOG="$TMP_DIR/ontology-bundle-directive-live-kernel.log"
SOURCECHECK_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-sourcecheck-classifier.log"
CHECK_MOD_SOURCECHECK_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-sourcecheck-classifier.log"
CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-sourcecheck-reject-classifier.log"
CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-same-checker-kernel-classifier.log"
CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-same-checker-kernel-copy-classifier.log"
CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-same-checker-kernel-ptr-query-classifier.log"
CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-same-checker-kernel-precheck-classifier.log"
CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-same-checker-sidecar-classifier.log"
PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-part-check-mod-same-checker-sidecar-classifier.log"
CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-knowledge-context-sidecar.log"
CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-knowledge-context-sidecar-reject.log"
PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG="$TMP_DIR/ontology-bundle-directive-part-check-mod-knowledge-context-sidecar.log"
PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG="$TMP_DIR/ontology-bundle-directive-part-check-mod-knowledge-context-sidecar-reject.log"
CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-same-checker-shadow-classifier.log"
CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG="$TMP_DIR/ontology-bundle-directive-check-mod-same-checker-shadow-reject-classifier.log"
SNOMED_CACHE="$TMP_DIR/ontology-bundle-directive-snomed.native.ontocache"
SNOMED_CACHE_SRC="$TMP_DIR/ontology-bundle-directive-snomed-cache-consumer.sio"
SNOMED_KNOWLEDGE_POSITIVE_SRC="$TMP_DIR/ontology-bundle-directive-snomed-knowledge-positive.sio"
SNOMED_KNOWLEDGE_REJECT_SRC="$TMP_DIR/ontology-bundle-directive-snomed-knowledge-reject.sio"
SNOMED_FRONTEND_COMPOSITE_SRC="$TMP_DIR/ontology-bundle-directive-snomed-frontend-composite.sio"
SNOMED_FRONTEND_COMPOSITE_REJECT_SRC="$TMP_DIR/ontology-bundle-directive-snomed-frontend-composite-reject.sio"
SNOMED_FRONTEND_UNIT_COMPOSITE_SRC="$TMP_DIR/ontology-bundle-directive-snomed-frontend-unit-composite.sio"
SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC="$TMP_DIR/ontology-bundle-directive-snomed-frontend-unit-composite-reject.sio"
FRONTEND="$TMP_DIR/souc-lean-frontend-ontocache-sidecar"
FRONTEND_COMPOSITE_LOG="$TMP_DIR/ontology-bundle-directive-frontend-composite.log"
FRONTEND_COMPOSITE_REJECT_LOG="$TMP_DIR/ontology-bundle-directive-frontend-composite-reject.log"
FRONTEND_UNIT_COMPOSITE_LOG="$TMP_DIR/ontology-bundle-directive-frontend-unit-composite.log"
FRONTEND_UNIT_COMPOSITE_REJECT_LOG="$TMP_DIR/ontology-bundle-directive-frontend-unit-composite-reject.log"
MALFORMED_SRC="$TMP_DIR/ontology-bundle-directive-malformed.sio"
MALFORMED_LOG="$TMP_DIR/ontology-bundle-directive-malformed.log"
INVALID_BUNDLE="$TMP_DIR/not-a-dontology.bin"
INVALID_BUNDLE_SRC="$TMP_DIR/ontology-bundle-directive-invalid-bundle.sio"
INVALID_BUNDLE_LOG="$TMP_DIR/ontology-bundle-directive-invalid-bundle.log"
INVALID_PAYLOAD="$TMP_DIR/invalid-payload.dontology"
INVALID_PAYLOAD_SRC="$TMP_DIR/ontology-bundle-directive-invalid-payload.sio"
INVALID_PAYLOAD_LOG="$TMP_DIR/ontology-bundle-directive-invalid-payload.log"

printf '%s\n' '//@ ontology-bundle: "stdlib/data/data/ontology/bundles/snomed.dontology' >"$MALFORMED_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$MALFORMED_SRC"
printf '%s\n' '//@ ontology-bundle: "stdlib/data/data/ontology/bundles/part.dontology"' >"$PART_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$PART_SRC"
printf '%s\n' '//@ ontology-bundle: "stdlib/data/data/ontology/bundles/loinc.dontology"' >"$LOINC_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$LOINC_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$SNOMED_CACHE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_CACHE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$PART_CACHE" >"$PART_CACHE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$PART_CACHE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$LOINC_CACHE" >"$LOINC_CACHE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$LOINC_CACHE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$SNOMED_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$SNOMED_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$SNOMED_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'struct PatientSnomedOntocachePositive { dx: CACHEGEN_73211009 }' >>"$SNOMED_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'fn accept_snomed_ontocache_positive(p: Knowledge<PatientSnomedOntocachePositive where { dx subclass_of CACHEGEN_138875005 }>) -> i32 { 1 }' >>"$SNOMED_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_KNOWLEDGE_POSITIVE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$SNOMED_KNOWLEDGE_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$SNOMED_KNOWLEDGE_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$SNOMED_KNOWLEDGE_REJECT_SRC"
printf '%s\n' 'struct PatientSnomedOntocacheReject { dx: CACHEGEN_138875005 }' >>"$SNOMED_KNOWLEDGE_REJECT_SRC"
printf '%s\n' 'fn reject_snomed_ontocache_reverse(p: Knowledge<PatientSnomedOntocacheReject where { dx subclass_of CACHEGEN_73211009 }>) -> i32 { 1 }' >>"$SNOMED_KNOWLEDGE_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_KNOWLEDGE_REJECT_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$SNOMED_FRONTEND_COMPOSITE_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$SNOMED_FRONTEND_COMPOSITE_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$SNOMED_FRONTEND_COMPOSITE_SRC"
printf '%s\n' 'struct PatientSnomedFrontendComposite { dx: CACHEGEN_73211009, age: i64 }' >>"$SNOMED_FRONTEND_COMPOSITE_SRC"
printf '%s\n' 'fn accept_snomed_frontend_composite(p: Knowledge<PatientSnomedFrontendComposite where { dx subclass_of CACHEGEN_138875005, age >= 18 }>) -> i32 { 1 }' >>"$SNOMED_FRONTEND_COMPOSITE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_FRONTEND_COMPOSITE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$SNOMED_FRONTEND_COMPOSITE_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$SNOMED_FRONTEND_COMPOSITE_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$SNOMED_FRONTEND_COMPOSITE_REJECT_SRC"
printf '%s\n' 'struct PatientSnomedFrontendCompositeReject { dx: CACHEGEN_73211009, age: bool }' >>"$SNOMED_FRONTEND_COMPOSITE_REJECT_SRC"
printf '%s\n' 'fn reject_snomed_frontend_composite(p: Knowledge<PatientSnomedFrontendCompositeReject where { dx subclass_of CACHEGEN_138875005, age >= 18 }>) -> i32 { 1 }' >>"$SNOMED_FRONTEND_COMPOSITE_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_FRONTEND_COMPOSITE_REJECT_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$SNOMED_FRONTEND_UNIT_COMPOSITE_SRC"
printf '%s\n' 'unit acceleration = m / s / s;' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_SRC"
printf '%s\n' 'struct PatientSnomedFrontendUnitComposite { dx: CACHEGEN_73211009, accel: acceleration }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_SRC"
printf '%s\n' 'fn accept_snomed_frontend_unit_composite(p: Knowledge<PatientSnomedFrontendUnitComposite where { dx subclass_of CACHEGEN_138875005, accel >= 1.0 <acceleration> }>) -> i32 { 1 }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC"
printf '%s\n' 'unit velocity = m / s;' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC"
printf '%s\n' 'unit acceleration = m / s / s;' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC"
printf '%s\n' 'struct PatientSnomedFrontendUnitCompositeReject { dx: CACHEGEN_73211009, accel: velocity }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC"
printf '%s\n' 'fn reject_snomed_frontend_unit_composite(p: Knowledge<PatientSnomedFrontendUnitCompositeReject where { dx subclass_of CACHEGEN_138875005, accel >= 1.0 <acceleration> }>) -> i32 { 1 }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$PART_CACHE" >"$PART_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'struct cache_slot3 { code: i64 }' >>"$PART_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'struct cache_slot1 { code: i64 }' >>"$PART_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'struct PatientPartOntocachePositive { dx: cache_slot3 }' >>"$PART_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'fn accept_part_ontocache_positive(p: Knowledge<PatientPartOntocachePositive where { dx subclass_of cache_slot1 }>) -> i32 { 1 }' >>"$PART_KNOWLEDGE_POSITIVE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$PART_KNOWLEDGE_POSITIVE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$PART_CACHE" >"$PART_KNOWLEDGE_DISJOINT_REJECT_SRC"
printf '%s\n' 'struct cache_slot2 { code: i64 }' >>"$PART_KNOWLEDGE_DISJOINT_REJECT_SRC"
printf '%s\n' 'struct cache_slot4 { code: i64 }' >>"$PART_KNOWLEDGE_DISJOINT_REJECT_SRC"
printf '%s\n' 'struct PatientPartOntocacheDisjointReject { dx: cache_slot2 }' >>"$PART_KNOWLEDGE_DISJOINT_REJECT_SRC"
printf '%s\n' 'fn reject_part_ontocache_disjoint(p: Knowledge<PatientPartOntocacheDisjointReject where { dx subclass_of cache_slot4 }>) -> i32 { 1 }' >>"$PART_KNOWLEDGE_DISJOINT_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$PART_KNOWLEDGE_DISJOINT_REJECT_SRC"
printf '%s\n' 'not a DONT envelope' >"$INVALID_BUNDLE"
printf '//@ ontology-bundle: "%s"\n' "$INVALID_BUNDLE" >"$INVALID_BUNDLE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$INVALID_BUNDLE_SRC"
printf 'DONT\001\000\000\000\002\000\000\000{}' >"$INVALID_PAYLOAD"
printf '//@ ontology-bundle: "%s"\n' "$INVALID_PAYLOAD" >"$INVALID_PAYLOAD_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$INVALID_PAYLOAD_SRC"

bin/souc check "$CHECK_MOD_CACHE_PROBE"
bin/souc check "$LIVE_KERNEL_CACHE_PROBE"
bin/souc check "$SOURCECHECK_CACHE_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_SOURCECHECK_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_PROBE"
bin/souc check "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_PROBE"
bin/souc check "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_PROBE"
bin/souc self-hosted/compiler/lean_frontend.sio "$FRONTEND"

if bin/souc run "$PROBE" -- "$POSITIVE_SRC" "$SNOMED_CACHE" "$SNOMED_CACHE_SRC" >"$LOG" 2>&1 &&
   grep -q 'ontology_bundle_directive_found=1' "$LOG" &&
   grep -q 'count=1' "$LOG" &&
   grep -q 'malformed=0' "$LOG" &&
   grep -q 'path=stdlib/data/data/ontology/bundles/snomed.dontology' "$LOG" &&
   grep -q 'bundle_exists=1' "$LOG" &&
   grep -Eq 'bundle_size=[1-9][0-9]*' "$LOG" &&
   grep -q 'dont_magic_valid=1' "$LOG" &&
   grep -q 'dont_version=1' "$LOG" &&
   grep -Eq 'dont_payload_size=[1-9][0-9]*' "$LOG" &&
   grep -q 'dont_payload_size_matches=1' "$LOG" &&
   grep -q 'payload_has_ontology=1' "$LOG" &&
   grep -q 'payload_has_terms=1' "$LOG" &&
   grep -q 'payload_has_disjoints=1' "$LOG" &&
   grep -q 'payload_term_count=112' "$LOG" &&
   grep -q 'payload_disjoint_count=0' "$LOG" &&
   grep -q 'payload_ontology=SNOMED' "$LOG" &&
   grep -q 'first_term_present=1' "$LOG" &&
   grep -q 'first_term_curie=SNOMED:138875005' "$LOG" &&
   grep -q 'first_term_numeric_id=138875005' "$LOG" &&
   grep -q 'first_parent_edge_present=1' "$LOG" &&
   grep -q 'first_parent_child_curie=SNOMED:404684003' "$LOG" &&
   grep -q 'first_parent_parent_curie=SNOMED:138875005' "$LOG" &&
   grep -q 'first_parent_child_numeric_id=404684003' "$LOG" &&
   grep -q 'first_parent_parent_numeric_id=138875005' "$LOG" &&
   grep -q 'first_disjoint_pair_present=0' "$LOG" &&
   grep -q 'mini_table_valid=1' "$LOG" &&
   grep -q 'mini_table_ontology=SNOMED' "$LOG" &&
   grep -q 'mini_table_class_count=112' "$LOG" &&
   grep -q 'mini_table_class0=138875005' "$LOG" &&
   grep -q 'mini_table_class1=404684003' "$LOG" &&
   grep -q 'mini_table_class2=64572001' "$LOG" &&
   grep -q 'mini_table_class3=73211009' "$LOG" &&
   grep -q 'mini_table_parent_edge_count=111' "$LOG" &&
   grep -q 'mini_table_parent_child0=404684003' "$LOG" &&
   grep -q 'mini_table_parent_parent0=138875005' "$LOG" &&
   grep -q 'mini_table_parent_child1=64572001' "$LOG" &&
   grep -q 'mini_table_parent_parent1=404684003' "$LOG" &&
   grep -q 'mini_table_disjoint_pair_count=0' "$LOG" &&
   grep -q 'mini_table_disjoint_left0=-1' "$LOG" &&
   grep -q 'mini_table_disjoint_right0=-1' "$LOG" &&
   grep -q 'mini_table_truncated=1' "$LOG" &&
   grep -q 'mini_query_first_edge_direct=1' "$LOG" &&
   grep -q 'mini_query_snomed_73211009_subclass_138875005=1' "$LOG" &&
   grep -q 'mini_query_part_4_subclass_2=0' "$LOG" &&
   grep -q 'mini_query_first_disjoint=0' "$LOG" &&
   grep -q 'mini_query_first_disjoint_symmetric=0' "$LOG" &&
   grep -q 'kernel_stage_valid=1' "$LOG" &&
   grep -q 'kernel_slot_count=4' "$LOG" &&
   grep -q 'kernel_slot0=138875005' "$LOG" &&
   grep -q 'kernel_slot1=404684003' "$LOG" &&
   grep -q 'kernel_slot2=64572001' "$LOG" &&
   grep -q 'kernel_slot3=73211009' "$LOG" &&
   grep -q 'kernel_subclass_axiom_count=3' "$LOG" &&
   grep -q 'kernel_subclass_subject0=1' "$LOG" &&
   grep -q 'kernel_subclass_object0=0' "$LOG" &&
   grep -q 'kernel_subclass_subject1=2' "$LOG" &&
   grep -q 'kernel_subclass_object1=1' "$LOG" &&
   grep -q 'kernel_subclass_subject2=3' "$LOG" &&
   grep -q 'kernel_subclass_object2=2' "$LOG" &&
   grep -q 'kernel_disjoint_axiom_count=0' "$LOG" &&
   grep -q 'kernel_query_first_edge_direct=1' "$LOG" &&
   grep -q 'kernel_query_snomed_73211009_subclass_138875005=1' "$LOG" &&
   grep -q 'kernel_query_part_4_subclass_2=0' "$LOG" &&
   grep -q 'kernel_query_first_disjoint=0' "$LOG" &&
   grep -q 'kernel_query_first_disjoint_symmetric=0' "$LOG" &&
   grep -q 'kernel_materialized_valid=1' "$LOG" &&
   grep -q 'kernel_materialized_reflexive_count=4' "$LOG" &&
   grep -q 'kernel_materialized_subclass_fact_count=10' "$LOG" &&
   grep -q 'kernel_materialized_disjoint_fact_count=0' "$LOG" &&
   grep -q 'kernel_materialized_query_first_edge=1' "$LOG" &&
   grep -q 'kernel_materialized_query_snomed_73211009_subclass_138875005=1' "$LOG" &&
   grep -q 'kernel_materialized_query_part_4_subclass_2=0' "$LOG" &&
   grep -q 'kernel_materialized_query_first_disjoint=0' "$LOG" &&
   grep -q 'kernel_materialized_query_first_disjoint_symmetric=0' "$LOG" &&
   grep -q 'imported_cache_valid=1' "$LOG" &&
   grep -q 'imported_cache_ontology=SNOMED' "$LOG" &&
   grep -q 'imported_cache_slot_count=4' "$LOG" &&
   grep -q 'imported_cache_slot0=138875005' "$LOG" &&
   grep -q 'imported_cache_slot1=404684003' "$LOG" &&
   grep -q 'imported_cache_slot2=64572001' "$LOG" &&
   grep -q 'imported_cache_slot3=73211009' "$LOG" &&
   grep -q 'imported_cache_subclass_axiom_count=3' "$LOG" &&
   grep -q 'imported_cache_disjoint_axiom_count=0' "$LOG" &&
   grep -q 'imported_cache_reflexive_count=4' "$LOG" &&
   grep -q 'imported_cache_subclass_fact_count=10' "$LOG" &&
   grep -q 'imported_cache_disjoint_fact_count=0' "$LOG" &&
   grep -q 'imported_cache_query_snomed_73211009_subclass_138875005=1' "$LOG" &&
   grep -q 'imported_cache_query_part_4_subclass_2=0' "$LOG" &&
   grep -q 'imported_cache_query_part_3_disjoint_12=0' "$LOG" &&
   grep -q 'cache_directive_import_valid=1' "$LOG" &&
   grep -q 'cache_directive_import_count=1' "$LOG" &&
   grep -q 'cache_directive_import_ontology=SNOMED' "$LOG" &&
   grep -q 'cache_directive_import_source_bundle=stdlib/data/data/ontology/bundles/snomed.dontology' "$LOG" &&
   grep -q 'cache_directive_import_slot_count=4' "$LOG" &&
   grep -q 'cache_directive_import_subclass_fact_count=10' "$LOG" &&
   grep -q 'cache_directive_import_disjoint_fact_count=0' "$LOG" &&
   grep -q 'cache_directive_query_snomed_73211009_subclass_138875005=1' "$LOG" &&
   grep -q 'cache_directive_query_part_4_subclass_2=0' "$LOG" &&
   grep -q 'cache_directive_query_part_3_disjoint_12=0' "$LOG" &&
   grep -q 'cache_directive_verdict_snomed_73211009_subclass_138875005=0' "$LOG" &&
   grep -q 'cache_directive_verdict_snomed_138875005_subclass_73211009=152' "$LOG" &&
   grep -q 'cache_directive_verdict_part_4_subclass_2=152' "$LOG" &&
   grep -q 'cache_directive_verdict_part_3_disjoint_12=152' "$LOG" &&
   grep -q 'payload_shape_valid=1' "$LOG"; then
  printf 'PASS  compiler-side ontology-bundle directive load plan materialized bounded SNOMED kernel facts\n'
else
  printf 'FAIL  compiler-side ontology-bundle directive load plan did not materialize expected bounded SNOMED kernel facts\n' >&2
  cat "$LOG" >&2
  exit 1
fi

if grep -q 'native_side_table_cache_written=1' "$LOG" &&
   grep -q 'format=bounded-kernel-facts' "$SNOMED_CACHE" &&
   grep -q 'source_bundle=stdlib/data/data/ontology/bundles/snomed.dontology' "$SNOMED_CACHE" &&
   grep -q 'ontology=SNOMED' "$SNOMED_CACHE" &&
   grep -q 'kernel_slot_count=4' "$SNOMED_CACHE" &&
   grep -q 'kernel_slots=138875005,404684003,64572001,73211009' "$SNOMED_CACHE" &&
   grep -q 'kernel_subclass_axioms=1>0,2>1,3>2' "$SNOMED_CACHE" &&
   grep -q 'kernel_materialized_subclass_fact_count=10' "$SNOMED_CACHE" &&
   grep -q 'authority=compiler-side-bounded-side-table-cache-not-live-checker-kernel' "$SNOMED_CACHE"; then
  printf 'PASS  compiler-side ontology-bundle directive wrote bounded SNOMED native .ontocache\n'
else
  printf 'FAIL  compiler-side ontology-bundle directive did not write expected bounded SNOMED native .ontocache\n' >&2
  cat "$LOG" >&2
  if [[ -f "$SNOMED_CACHE" ]]; then
    cat "$SNOMED_CACHE" >&2
  fi
  exit 1
fi
printf 'PASS  compiler-side ontology-bundle directive reimported bounded SNOMED native .ontocache\n'
printf 'PASS  compiler-side ontology side-table cache directive imported bounded SNOMED native .ontocache\n'

if bin/souc run "$CHECK_MOD_CACHE_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_LOG" 2>&1 &&
   grep -q 'check_mod_cache_verdict_snomed_73211009_subclass_138875005=0' "$CHECK_MOD_LOG" &&
   grep -q 'check_mod_cache_verdict_snomed_138875005_subclass_73211009=152' "$CHECK_MOD_LOG" &&
   grep -q 'check_mod_cache_verdict_part_4_subclass_2=152' "$CHECK_MOD_LOG" &&
   grep -q 'check_mod_cache_verdict_part_3_disjoint_12=152' "$CHECK_MOD_LOG" &&
   grep -q 'k2_check_mod_ontology_cache_directive_probe=ok' "$CHECK_MOD_LOG"; then
  printf 'PASS  check::mod consumed bounded SNOMED native .ontocache semantic verdicts\n'
else
  printf 'FAIL  check::mod did not consume expected bounded SNOMED native .ontocache semantic verdicts\n' >&2
  cat "$CHECK_MOD_LOG" >&2
  exit 1
fi

if bin/souc run "$LIVE_KERNEL_CACHE_PROBE" -- "$SNOMED_CACHE_SRC" >"$LIVE_KERNEL_LOG" 2>&1 &&
   grep -q 'live_kernel_cache_verdict_snomed_73211009_subclass_138875005=0' "$LIVE_KERNEL_LOG" &&
   grep -q 'live_kernel_cache_verdict_snomed_138875005_subclass_73211009=152' "$LIVE_KERNEL_LOG" &&
   grep -q 'live_kernel_cache_verdict_part_4_subclass_2=152' "$LIVE_KERNEL_LOG" &&
   grep -q 'live_kernel_cache_verdict_part_3_disjoint_12=152' "$LIVE_KERNEL_LOG" &&
   grep -q 'k2_check_import_ontology_cache_kernel_probe=ok' "$LIVE_KERNEL_LOG"; then
  printf 'PASS  check::check hydrated live Checker.ontology_kernel from bounded SNOMED native .ontocache\n'
else
  printf 'FAIL  check::check did not hydrate expected live Checker.ontology_kernel from bounded SNOMED native .ontocache\n' >&2
  cat "$LIVE_KERNEL_LOG" >&2
  exit 1
fi

if bin/souc run "$SOURCECHECK_CACHE_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$SOURCECHECK_CLASSIFIER_LOG" 2>&1; then
  printf 'FAIL  imported sourcecheck cache hydrator unexpectedly completed; expected current imported collect classifier failure\n' >&2
  cat "$SOURCECHECK_CLASSIFIER_LOG" >&2
  exit 1
fi
if grep -q 'k2_check_import_ontology_cache_sourcecheck_classifier_probe=parse_ok' "$SOURCECHECK_CLASSIFIER_LOG" &&
   grep -q 'k2_check_import_ontology_cache_sourcecheck_classifier_probe=sourcecheck_enter' "$SOURCECHECK_CLASSIFIER_LOG" &&
   grep -q 'checker_check_items_with_ontology_cache_subclass_verdict=idx_ok' "$SOURCECHECK_CLASSIFIER_LOG" &&
   ! grep -q 'checker_check_items_with_ontology_cache_subclass_verdict=collect_ok' "$SOURCECHECK_CLASSIFIER_LOG"; then
  printf 'PASS  imported sourcecheck cache hydrator remains blocked at check::check collect_items after cache index resolution\n'
else
  printf 'FAIL  imported sourcecheck cache hydrator failed in an unexpected classifier shape\n' >&2
  cat "$SOURCECHECK_CLASSIFIER_LOG" >&2
  exit 1
fi

if ! bin/souc run "$CHECK_MOD_SOURCECHECK_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SOURCECHECK_CLASSIFIER_LOG" 2>&1; then
  printf 'FAIL  check::mod sourcecheck cache hydrator did not complete positive/reverse semantic verdict probe\n' >&2
  cat "$CHECK_MOD_SOURCECHECK_CLASSIFIER_LOG" >&2
  exit 1
fi
if grep -q 'k2_check_mod_ontology_cache_sourcecheck_classifier_probe=parse_ok' "$CHECK_MOD_SOURCECHECK_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_ontology_cache_sourcecheck_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SOURCECHECK_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_ontology_cache_sourcecheck_classifier_probe=positive_verdict=0' "$CHECK_MOD_SOURCECHECK_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_ontology_cache_sourcecheck_classifier_probe=ok' "$CHECK_MOD_SOURCECHECK_CLASSIFIER_LOG"; then
  printf 'PASS  check::mod sourcecheck cache hydrator accepted positive semantic verdict after ordinary sourcecheck\n'
else
  printf 'FAIL  check::mod sourcecheck cache hydrator failed in an unexpected classifier shape\n' >&2
  cat "$CHECK_MOD_SOURCECHECK_CLASSIFIER_LOG" >&2
  exit 1
fi

if ! bin/souc run "$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_LOG" 2>&1; then
  printf 'FAIL  check::mod sourcecheck cache reject classifier did not complete reverse semantic verdict probe\n' >&2
  cat "$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_LOG" >&2
  exit 1
fi
if grep -q 'k2_check_mod_ontology_cache_sourcecheck_reject_classifier_probe=parse_ok' "$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_ontology_cache_sourcecheck_reject_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_ontology_cache_sourcecheck_reject_classifier_probe=reverse_verdict=152' "$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_ontology_cache_sourcecheck_reject_classifier_probe=ok' "$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_LOG"; then
  printf 'PASS  check::mod sourcecheck cache hydrator rejected reverse semantic verdict after ordinary sourcecheck\n'
else
  printf 'FAIL  check::mod sourcecheck cache reject classifier failed in an unexpected classifier shape\n' >&2
  cat "$CHECK_MOD_SOURCECHECK_REJECT_CLASSIFIER_LOG" >&2
  exit 1
fi

if bin/souc run "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" 2>&1; then
  if grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_classifier_probe=ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker ontology-kernel cache hydration accepted positive semantic verdict\n'
  else
    printf 'FAIL  check::mod same-Checker ontology-kernel cache classifier completed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" >&2
    exit 1
  fi
else
  if grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" &&
     grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v1=idx_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v1=collect_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v1=check_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v1=hydrate_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v1=query_reject' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" &&
     grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_classifier_probe=verdict=152' "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker ontology-kernel cache hydration remains classified after hydrate_ok with rejected positive verdict\n'
  else
    printf 'FAIL  check::mod same-Checker ontology-kernel cache classifier failed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_KERNEL_CLASSIFIER_LOG" >&2
    exit 1
  fi
fi

if bin/souc run "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" 2>&1; then
  if grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_copy_classifier_probe=ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker ontology-kernel copy hydration accepted positive semantic verdict\n'
  else
    printf 'FAIL  check::mod same-Checker ontology-kernel copy classifier completed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" >&2
    exit 1
  fi
else
  if grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_copy_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_copy_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=idx_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=collect_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=check_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=local_hydrate_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     ! grep -q 'check_mod_same_checker_cache_kernel_v2=copy_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker ontology-kernel copy hydration remains classified before copy_ok after local_hydrate_ok\n'
  elif grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_copy_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_copy_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=idx_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=collect_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=check_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=local_hydrate_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=copy_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v2=query_reject' "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker ontology-kernel copy hydration remains classified after copy_ok with rejected positive verdict\n'
  else
    printf 'FAIL  check::mod same-Checker ontology-kernel copy classifier failed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_KERNEL_COPY_CLASSIFIER_LOG" >&2
    exit 1
  fi
fi

if bin/souc run "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" 2>&1; then
  if grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_ptr_query_classifier_probe=ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker ontology-kernel pointer-query classifier accepted positive semantic verdict\n'
  else
    printf 'FAIL  check::mod same-Checker ontology-kernel pointer-query classifier completed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" >&2
    exit 1
  fi
else
  if grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_ptr_query_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_ptr_query_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=idx_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=collect_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=check_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=local_hydrate_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=copy_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=query_reject' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker ontology-kernel pointer-query classifier remains classified after copy_ok with rejected positive verdict\n'
  elif grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_ptr_query_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_ptr_query_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=idx_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=collect_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=check_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=local_hydrate_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_v3=copy_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     ! grep -q 'check_mod_same_checker_cache_kernel_v3=query_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" &&
     ! grep -q 'check_mod_same_checker_cache_kernel_v3=query_reject' "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker ontology-kernel pointer-query classifier remains classified after copy_ok before query verdict\n'
  else
    printf 'FAIL  check::mod same-Checker ontology-kernel pointer-query classifier failed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_KERNEL_PTR_QUERY_CLASSIFIER_LOG" >&2
    exit 1
  fi
fi

if bin/souc run "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG" 2>&1; then
  if grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_precheck_classifier_probe=ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker precheck ontology-kernel hydration accepted positive semantic verdict before ordinary check\n'
  else
    printf 'FAIL  check::mod same-Checker precheck ontology-kernel classifier completed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG" >&2
    exit 1
  fi
else
  if grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_precheck_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG" &&
     grep -q 'k2_check_mod_same_checker_ontology_cache_kernel_precheck_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_precheck=idx_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_precheck=collect_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_precheck=hydrate_ok' "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG" &&
     grep -q 'check_mod_same_checker_cache_kernel_precheck=class_count=' "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker precheck ontology-kernel hydration remains classified at collected-checker class_count read\n'
  else
    printf 'FAIL  check::mod same-Checker precheck ontology-kernel classifier failed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_KERNEL_PRECHECK_CLASSIFIER_LOG" >&2
    exit 1
  fi
fi

if bin/souc run "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" 2>&1 &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=snomed_positive=0' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=snomed_reverse=152' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=part_positive=152' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=part_disjoint=152' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar=collect_ok' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar=check_ok' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar=sidecar_hydrate_ok' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar=sidecar_count=4' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=ok' "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG"; then
  printf 'PASS  check::mod same-sourcecheck sidecar accepted bounded SNOMED cache verdict after ordinary sourcecheck\n'
else
  printf 'FAIL  check::mod same-sourcecheck sidecar failed SNOMED cache verdict probe\n' >&2
  cat "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" >&2
  exit 1
fi

if bin/souc run "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_PROBE" -- "$SNOMED_KNOWLEDGE_POSITIVE_SRC" >"$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG" 2>&1 &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=sourcecheck_enter' "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG" &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=verdict=0' "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG" &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=ok' "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG"; then
  printf 'PASS  check::mod sidecar accepted SNOMED Knowledge<T> proof-context after ordinary sourcecheck\n'
else
  printf 'FAIL  check::mod sidecar did not accept SNOMED Knowledge<T> proof-context\n' >&2
  cat "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG" >&2
  exit 1
fi

if bin/souc run "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_PROBE" -- "$SNOMED_KNOWLEDGE_REJECT_SRC" >"$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" 2>&1; then
  printf 'FAIL  check::mod sidecar unexpectedly accepted reverse SNOMED Knowledge<T> proof-context\n' >&2
  cat "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" >&2
  exit 1
fi
if grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=sourcecheck_enter' "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=verdict=1' "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=rejected' "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG"; then
  printf 'PASS  check::mod sidecar rejected reverse SNOMED Knowledge<T> proof-context after ordinary sourcecheck\n'
else
  printf 'FAIL  check::mod sidecar reverse SNOMED Knowledge<T> rejection had unexpected shape\n' >&2
  cat "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$SNOMED_FRONTEND_COMPOSITE_SRC" >"$FRONTEND_COMPOSITE_LOG" 2>&1; then
  printf 'PASS  modular frontend accepted imported-cache Knowledge<T> ontology plus numeric proof-context\n'
else
  printf 'FAIL  modular frontend rejected imported-cache Knowledge<T> ontology plus numeric proof-context\n' >&2
  cat "$FRONTEND_COMPOSITE_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$SNOMED_FRONTEND_COMPOSITE_REJECT_SRC" >"$FRONTEND_COMPOSITE_REJECT_LOG" 2>&1; then
  printf 'FAIL  modular frontend unexpectedly accepted imported-cache Knowledge<T> with invalid numeric field\n' >&2
  cat "$FRONTEND_COMPOSITE_REJECT_LOG" >&2
  exit 1
fi
printf 'PASS  modular frontend rejected imported-cache Knowledge<T> when numeric proof-context fails\n'

if "$FRONTEND" --check "$SNOMED_FRONTEND_UNIT_COMPOSITE_SRC" >"$FRONTEND_UNIT_COMPOSITE_LOG" 2>&1; then
  printf 'PASS  modular frontend accepted imported-cache Knowledge<T> ontology plus unit proof-context\n'
else
  printf 'FAIL  modular frontend rejected imported-cache Knowledge<T> ontology plus unit proof-context\n' >&2
  cat "$FRONTEND_UNIT_COMPOSITE_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$SNOMED_FRONTEND_UNIT_COMPOSITE_REJECT_SRC" >"$FRONTEND_UNIT_COMPOSITE_REJECT_LOG" 2>&1; then
  printf 'FAIL  modular frontend unexpectedly accepted imported-cache Knowledge<T> with invalid unit field\n' >&2
  cat "$FRONTEND_UNIT_COMPOSITE_REJECT_LOG" >&2
  exit 1
fi
printf 'PASS  modular frontend rejected imported-cache Knowledge<T> when unit proof-context fails\n'

if bin/souc run "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" 2>&1; then
  if grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_classifier_probe=ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG"; then
    printf 'PASS  check::mod same-Checker shadow cache accepted positive semantic verdict after ordinary sourcecheck\n'
  else
    printf 'FAIL  check::mod same-Checker shadow cache classifier completed in an unexpected shape\n' >&2
    cat "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" >&2
    exit 1
  fi
elif grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=collect_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=check_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=hydrate_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=shadow_count=0' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=shadow_empty' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_classifier_probe=verdict=152' "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG"; then
  printf 'PASS  check::mod same-Checker shadow cache remains classified after hydrate_ok with empty shadow state\n'
else
  printf 'FAIL  check::mod same-Checker shadow cache classifier failed in an unexpected shape\n' >&2
  cat "$CHECK_MOD_SAME_CHECKER_SHADOW_CLASSIFIER_LOG" >&2
  exit 1
fi

if ! bin/souc run "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_PROBE" -- "$SNOMED_CACHE_SRC" >"$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" 2>&1; then
  printf 'FAIL  check::mod same-Checker shadow cache reject classifier did not complete reverse semantic verdict probe\n' >&2
  cat "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" >&2
  exit 1
fi
if grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_reject_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_reject_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=collect_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=check_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=hydrate_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=query_reject' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_reject_classifier_probe=verdict=152' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG"; then
  printf 'PASS  check::mod same-Checker shadow cache rejected reverse semantic verdict after ordinary sourcecheck\n'
elif grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_reject_classifier_probe=parse_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_reject_classifier_probe=sourcecheck_enter' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=collect_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=check_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=hydrate_ok' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=shadow_count=0' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_shadow_v1=shadow_empty' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_shadow_reject_classifier_probe=verdict=152' "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG"; then
  printf 'PASS  check::mod same-Checker shadow cache rejected reverse semantic verdict through empty-shadow classifier\n'
else
  printf 'FAIL  check::mod same-Checker shadow cache reject classifier failed in an unexpected shape\n' >&2
  cat "$CHECK_MOD_SAME_CHECKER_SHADOW_REJECT_CLASSIFIER_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$PART_SRC" "$PART_CACHE" "$PART_CACHE_SRC" >"$PART_LOG" 2>&1 &&
   grep -q 'ontology_bundle_directive_found=1' "$PART_LOG" &&
   grep -q 'count=1' "$PART_LOG" &&
   grep -q 'malformed=0' "$PART_LOG" &&
   grep -q 'path=stdlib/data/data/ontology/bundles/part.dontology' "$PART_LOG" &&
   grep -q 'bundle_exists=1' "$PART_LOG" &&
   grep -Eq 'bundle_size=[1-9][0-9]*' "$PART_LOG" &&
   grep -q 'dont_magic_valid=1' "$PART_LOG" &&
   grep -q 'dont_version=1' "$PART_LOG" &&
   grep -Eq 'dont_payload_size=[1-9][0-9]*' "$PART_LOG" &&
   grep -q 'dont_payload_size_matches=1' "$PART_LOG" &&
   grep -q 'payload_has_ontology=1' "$PART_LOG" &&
   grep -q 'payload_has_terms=1' "$PART_LOG" &&
   grep -q 'payload_has_disjoints=1' "$PART_LOG" &&
   grep -q 'payload_term_count=112' "$PART_LOG" &&
   grep -q 'payload_disjoint_count=1' "$PART_LOG" &&
   grep -q 'payload_ontology=PART' "$PART_LOG" &&
   grep -q 'first_term_present=1' "$PART_LOG" &&
   grep -q 'first_term_curie=PART:0000001' "$PART_LOG" &&
   grep -q 'first_term_numeric_id=1' "$PART_LOG" &&
   grep -q 'first_parent_edge_present=1' "$PART_LOG" &&
   grep -q 'first_parent_child_curie=PART:0000002' "$PART_LOG" &&
   grep -q 'first_parent_parent_curie=PART:0000001' "$PART_LOG" &&
   grep -q 'first_parent_child_numeric_id=2' "$PART_LOG" &&
   grep -q 'first_parent_parent_numeric_id=1' "$PART_LOG" &&
   grep -q 'first_disjoint_pair_present=1' "$PART_LOG" &&
   grep -q 'first_disjoint_left_curie=PART:0000003' "$PART_LOG" &&
   grep -q 'first_disjoint_right_curie=PART:0000012' "$PART_LOG" &&
   grep -q 'first_disjoint_left_numeric_id=3' "$PART_LOG" &&
   grep -q 'first_disjoint_right_numeric_id=12' "$PART_LOG" &&
   grep -q 'mini_table_valid=1' "$PART_LOG" &&
   grep -q 'mini_table_ontology=PART' "$PART_LOG" &&
   grep -q 'mini_table_class_count=112' "$PART_LOG" &&
   grep -q 'mini_table_class0=1' "$PART_LOG" &&
   grep -q 'mini_table_class1=2' "$PART_LOG" &&
   grep -q 'mini_table_class2=3' "$PART_LOG" &&
   grep -q 'mini_table_class3=4' "$PART_LOG" &&
   grep -q 'mini_table_parent_edge_count=111' "$PART_LOG" &&
   grep -q 'mini_table_parent_child0=2' "$PART_LOG" &&
   grep -q 'mini_table_parent_parent0=1' "$PART_LOG" &&
   grep -q 'mini_table_parent_child1=3' "$PART_LOG" &&
   grep -q 'mini_table_parent_parent1=2' "$PART_LOG" &&
   grep -q 'mini_table_disjoint_pair_count=1' "$PART_LOG" &&
   grep -q 'mini_table_disjoint_left0=3' "$PART_LOG" &&
   grep -q 'mini_table_disjoint_right0=12' "$PART_LOG" &&
   grep -q 'mini_table_truncated=1' "$PART_LOG" &&
   grep -q 'mini_query_first_edge_direct=1' "$PART_LOG" &&
   grep -q 'mini_query_snomed_73211009_subclass_138875005=0' "$PART_LOG" &&
   grep -q 'mini_query_part_4_subclass_2=1' "$PART_LOG" &&
   grep -q 'mini_query_first_disjoint=1' "$PART_LOG" &&
   grep -q 'mini_query_first_disjoint_symmetric=1' "$PART_LOG" &&
   grep -q 'kernel_stage_valid=1' "$PART_LOG" &&
   grep -q 'kernel_slot_count=5' "$PART_LOG" &&
   grep -q 'kernel_slot0=1' "$PART_LOG" &&
   grep -q 'kernel_slot1=2' "$PART_LOG" &&
   grep -q 'kernel_slot2=3' "$PART_LOG" &&
   grep -q 'kernel_slot3=4' "$PART_LOG" &&
   grep -q 'kernel_slot4=12' "$PART_LOG" &&
   grep -q 'kernel_subclass_axiom_count=3' "$PART_LOG" &&
   grep -q 'kernel_subclass_subject0=1' "$PART_LOG" &&
   grep -q 'kernel_subclass_object0=0' "$PART_LOG" &&
   grep -q 'kernel_subclass_subject1=2' "$PART_LOG" &&
   grep -q 'kernel_subclass_object1=1' "$PART_LOG" &&
   grep -q 'kernel_subclass_subject2=3' "$PART_LOG" &&
   grep -q 'kernel_subclass_object2=2' "$PART_LOG" &&
   grep -q 'kernel_disjoint_axiom_count=1' "$PART_LOG" &&
   grep -q 'kernel_disjoint_subject0=2' "$PART_LOG" &&
   grep -q 'kernel_disjoint_object0=4' "$PART_LOG" &&
   grep -q 'kernel_query_first_edge_direct=1' "$PART_LOG" &&
   grep -q 'kernel_query_snomed_73211009_subclass_138875005=0' "$PART_LOG" &&
   grep -q 'kernel_query_part_4_subclass_2=1' "$PART_LOG" &&
   grep -q 'kernel_query_first_disjoint=1' "$PART_LOG" &&
   grep -q 'kernel_query_first_disjoint_symmetric=1' "$PART_LOG" &&
   grep -q 'kernel_materialized_valid=1' "$PART_LOG" &&
   grep -q 'kernel_materialized_reflexive_count=5' "$PART_LOG" &&
   grep -q 'kernel_materialized_subclass_fact_count=11' "$PART_LOG" &&
   grep -q 'kernel_materialized_disjoint_fact_count=2' "$PART_LOG" &&
   grep -q 'kernel_materialized_query_first_edge=1' "$PART_LOG" &&
   grep -q 'kernel_materialized_query_snomed_73211009_subclass_138875005=0' "$PART_LOG" &&
   grep -q 'kernel_materialized_query_part_4_subclass_2=1' "$PART_LOG" &&
   grep -q 'kernel_materialized_query_first_disjoint=1' "$PART_LOG" &&
   grep -q 'kernel_materialized_query_first_disjoint_symmetric=1' "$PART_LOG" &&
   grep -q 'imported_cache_valid=1' "$PART_LOG" &&
   grep -q 'imported_cache_ontology=PART' "$PART_LOG" &&
   grep -q 'imported_cache_slot_count=5' "$PART_LOG" &&
   grep -q 'imported_cache_slot0=1' "$PART_LOG" &&
   grep -q 'imported_cache_slot1=2' "$PART_LOG" &&
   grep -q 'imported_cache_slot2=3' "$PART_LOG" &&
   grep -q 'imported_cache_slot3=4' "$PART_LOG" &&
   grep -q 'imported_cache_slot4=12' "$PART_LOG" &&
   grep -q 'imported_cache_subclass_axiom_count=3' "$PART_LOG" &&
   grep -q 'imported_cache_disjoint_axiom_count=1' "$PART_LOG" &&
   grep -q 'imported_cache_reflexive_count=5' "$PART_LOG" &&
   grep -q 'imported_cache_subclass_fact_count=11' "$PART_LOG" &&
   grep -q 'imported_cache_disjoint_fact_count=2' "$PART_LOG" &&
   grep -q 'imported_cache_query_snomed_73211009_subclass_138875005=0' "$PART_LOG" &&
   grep -q 'imported_cache_query_part_4_subclass_2=1' "$PART_LOG" &&
   grep -q 'imported_cache_query_part_3_disjoint_12=1' "$PART_LOG" &&
   grep -q 'imported_cache_query_part_12_disjoint_3=1' "$PART_LOG" &&
   grep -q 'cache_directive_import_valid=1' "$PART_LOG" &&
   grep -q 'cache_directive_import_count=1' "$PART_LOG" &&
   grep -q 'cache_directive_import_ontology=PART' "$PART_LOG" &&
   grep -q 'cache_directive_import_source_bundle=stdlib/data/data/ontology/bundles/part.dontology' "$PART_LOG" &&
   grep -q 'cache_directive_import_slot_count=5' "$PART_LOG" &&
   grep -q 'cache_directive_import_subclass_fact_count=11' "$PART_LOG" &&
   grep -q 'cache_directive_import_disjoint_fact_count=2' "$PART_LOG" &&
   grep -q 'cache_directive_query_snomed_73211009_subclass_138875005=0' "$PART_LOG" &&
   grep -q 'cache_directive_query_part_4_subclass_2=1' "$PART_LOG" &&
   grep -q 'cache_directive_query_part_3_disjoint_12=1' "$PART_LOG" &&
   grep -q 'cache_directive_query_part_12_disjoint_3=1' "$PART_LOG" &&
   grep -q 'cache_directive_verdict_snomed_73211009_subclass_138875005=152' "$PART_LOG" &&
   grep -q 'cache_directive_verdict_part_4_subclass_2=0' "$PART_LOG" &&
   grep -q 'cache_directive_verdict_part_2_subclass_4=152' "$PART_LOG" &&
   grep -q 'cache_directive_verdict_part_3_disjoint_12=0' "$PART_LOG" &&
   grep -q 'cache_directive_verdict_part_2_disjoint_4=152' "$PART_LOG" &&
   grep -q 'payload_shape_valid=1' "$PART_LOG"; then
  printf 'PASS  compiler-side ontology-bundle directive load plan materialized bounded PART kernel facts with disjoint pair\n'
else
  printf 'FAIL  compiler-side ontology-bundle directive load plan did not materialize expected bounded PART kernel facts with disjoint pair\n' >&2
  cat "$PART_LOG" >&2
  exit 1
fi

if grep -q 'native_side_table_cache_written=1' "$PART_LOG" &&
   grep -q 'format=bounded-kernel-facts' "$PART_CACHE" &&
   grep -q 'source_bundle=stdlib/data/data/ontology/bundles/part.dontology' "$PART_CACHE" &&
   grep -q 'ontology=PART' "$PART_CACHE" &&
   grep -q 'kernel_slot_count=5' "$PART_CACHE" &&
   grep -q 'kernel_slots=1,2,3,4,12' "$PART_CACHE" &&
   grep -q 'kernel_subclass_axioms=1>0,2>1,3>2' "$PART_CACHE" &&
   grep -q 'kernel_disjoint_axioms=2<>4' "$PART_CACHE" &&
   grep -q 'kernel_materialized_subclass_fact_count=11' "$PART_CACHE" &&
   grep -q 'kernel_materialized_disjoint_fact_count=2' "$PART_CACHE" &&
   grep -q 'authority=compiler-side-bounded-side-table-cache-not-live-checker-kernel' "$PART_CACHE"; then
  printf 'PASS  compiler-side ontology-bundle directive wrote bounded PART native .ontocache\n'
else
  printf 'FAIL  compiler-side ontology-bundle directive did not write expected bounded PART native .ontocache\n' >&2
  cat "$PART_LOG" >&2
  if [[ -f "$PART_CACHE" ]]; then
    cat "$PART_CACHE" >&2
  fi
  exit 1
fi
printf 'PASS  compiler-side ontology-bundle directive reimported bounded PART native .ontocache\n'
printf 'PASS  compiler-side ontology side-table cache directive imported bounded PART native .ontocache\n'

bin/souc run "$PROBE" -- "$LOINC_SRC" "$LOINC_CACHE" "$LOINC_CACHE_SRC" >"$LOINC_LOG" 2>&1 || true
if grep -q 'ontology_bundle_directive_found=1' "$LOINC_LOG" &&
   grep -q 'count=1' "$LOINC_LOG" &&
   grep -q 'malformed=0' "$LOINC_LOG" &&
   grep -q 'path=stdlib/data/data/ontology/bundles/loinc.dontology' "$LOINC_LOG" &&
   grep -q 'bundle_exists=1' "$LOINC_LOG" &&
   grep -q 'dont_magic_valid=1' "$LOINC_LOG" &&
   grep -q 'dont_version=1' "$LOINC_LOG" &&
   grep -q 'dont_payload_size_matches=1' "$LOINC_LOG" &&
   grep -q 'payload_ontology=LOINC' "$LOINC_LOG" &&
   grep -q 'payload_term_count=112' "$LOINC_LOG" &&
   grep -q 'payload_disjoint_count=0' "$LOINC_LOG" &&
   grep -q 'first_term_curie=LOINC:LP31435-1' "$LOINC_LOG" &&
   grep -q 'first_term_numeric_id=-1' "$LOINC_LOG" &&
   grep -q 'first_parent_edge_present=1' "$LOINC_LOG" &&
   grep -q 'first_parent_child_curie=LOINC:LP14693-3' "$LOINC_LOG" &&
   grep -q 'first_parent_parent_curie=LOINC:LP15090-4' "$LOINC_LOG" &&
   grep -q 'first_parent_child_numeric_id=-1' "$LOINC_LOG" &&
   grep -q 'first_parent_parent_numeric_id=-1' "$LOINC_LOG" &&
   grep -q 'mini_table_valid=1' "$LOINC_LOG" &&
   grep -q 'mini_table_ontology=LOINC' "$LOINC_LOG" &&
   grep -q 'mini_table_class_count=112' "$LOINC_LOG" &&
   grep -q 'mini_table_class0=900000000' "$LOINC_LOG" &&
   grep -q 'mini_table_class1=900000001' "$LOINC_LOG" &&
   grep -q 'mini_table_class2=900000002' "$LOINC_LOG" &&
   grep -q 'mini_table_class3=900000003' "$LOINC_LOG" &&
   grep -q 'mini_table_parent_edge_count=95' "$LOINC_LOG" &&
   grep -q 'mini_table_parent_child0=900000007' "$LOINC_LOG" &&
   grep -q 'mini_table_parent_parent0=900000006' "$LOINC_LOG" &&
   grep -q 'mini_table_truncated=1' "$LOINC_LOG" &&
   grep -q 'kernel_stage_valid=1' "$LOINC_LOG" &&
   grep -q 'kernel_slot_count=6' "$LOINC_LOG" &&
   grep -q 'kernel_slot0=900000000' "$LOINC_LOG" &&
   grep -q 'kernel_slot1=900000001' "$LOINC_LOG" &&
   grep -q 'kernel_slot2=900000002' "$LOINC_LOG" &&
   grep -q 'kernel_slot3=900000003' "$LOINC_LOG" &&
   grep -q 'kernel_slot4=900000007' "$LOINC_LOG" &&
   grep -q 'kernel_subclass_axiom_count=1' "$LOINC_LOG" &&
   grep -q 'kernel_subclass_subject0=4' "$LOINC_LOG" &&
   grep -q 'kernel_subclass_object0=5' "$LOINC_LOG" &&
   grep -q 'kernel_disjoint_axiom_count=0' "$LOINC_LOG" &&
   grep -q 'kernel_materialized_valid=1' "$LOINC_LOG" &&
   grep -q 'kernel_materialized_reflexive_count=6' "$LOINC_LOG" &&
   grep -q 'kernel_materialized_subclass_fact_count=7' "$LOINC_LOG" &&
   grep -q 'kernel_materialized_disjoint_fact_count=0' "$LOINC_LOG" &&
   grep -q 'imported_cache_valid=1' "$LOINC_LOG" &&
   grep -q 'imported_cache_ontology=LOINC' "$LOINC_LOG" &&
   grep -q 'imported_cache_slot_count=6' "$LOINC_LOG" &&
   grep -q 'imported_cache_slot0=900000000' "$LOINC_LOG" &&
   grep -q 'imported_cache_slot4=900000007' "$LOINC_LOG" &&
   grep -q 'imported_cache_subclass_axiom_count=1' "$LOINC_LOG" &&
   grep -q 'imported_cache_subclass_fact_count=7' "$LOINC_LOG" &&
   grep -q 'cache_directive_import_valid=1' "$LOINC_LOG" &&
   grep -q 'cache_directive_import_ontology=LOINC' "$LOINC_LOG" &&
   grep -q 'cache_directive_import_slot_count=6' "$LOINC_LOG" &&
   grep -q 'cache_directive_import_subclass_fact_count=7' "$LOINC_LOG" &&
   grep -q 'payload_shape_valid=1' "$LOINC_LOG"; then
  printf 'PASS  compiler-side ontology-bundle directive load plan materialized bounded LOINC alphanumeric kernel facts\n'
else
  printf 'FAIL  compiler-side ontology-bundle directive load plan did not materialize expected bounded LOINC alphanumeric kernel facts\n' >&2
  cat "$LOINC_LOG" >&2
  exit 1
fi

if grep -q 'native_side_table_cache_written=1' "$LOINC_LOG" &&
   grep -q 'format=bounded-kernel-facts' "$LOINC_CACHE" &&
   grep -q 'source_bundle=stdlib/data/data/ontology/bundles/loinc.dontology' "$LOINC_CACHE" &&
   grep -q 'ontology=LOINC' "$LOINC_CACHE" &&
   grep -q 'kernel_slot_count=6' "$LOINC_CACHE" &&
   grep -q 'kernel_slots=900000000,900000001,900000002,900000003,900000007,900000006' "$LOINC_CACHE" &&
   grep -q 'kernel_subclass_axioms=4>5' "$LOINC_CACHE" &&
   grep -q 'kernel_materialized_subclass_fact_count=7' "$LOINC_CACHE" &&
   grep -q 'authority=compiler-side-bounded-side-table-cache-not-live-checker-kernel' "$LOINC_CACHE"; then
  printf 'PASS  compiler-side ontology-bundle directive wrote bounded LOINC native .ontocache with synthetic alphanumeric slots\n'
else
  printf 'FAIL  compiler-side ontology-bundle directive did not write expected bounded LOINC native .ontocache\n' >&2
  cat "$LOINC_LOG" >&2
  if [[ -f "$LOINC_CACHE" ]]; then
    cat "$LOINC_CACHE" >&2
  fi
  exit 1
fi
printf 'PASS  compiler-side ontology-bundle directive reimported bounded LOINC native .ontocache\n'
printf 'PASS  compiler-side ontology side-table cache directive imported bounded LOINC native .ontocache\n'

if bin/souc run "$CHECK_MOD_CACHE_PROBE" -- "$PART_CACHE_SRC" >"$PART_CHECK_MOD_LOG" 2>&1 &&
   grep -q 'check_mod_cache_verdict_snomed_73211009_subclass_138875005=152' "$PART_CHECK_MOD_LOG" &&
   grep -q 'check_mod_cache_verdict_part_4_subclass_2=0' "$PART_CHECK_MOD_LOG" &&
   grep -q 'check_mod_cache_verdict_part_2_subclass_4=152' "$PART_CHECK_MOD_LOG" &&
   grep -q 'check_mod_cache_verdict_part_3_disjoint_12=0' "$PART_CHECK_MOD_LOG" &&
   grep -q 'check_mod_cache_verdict_part_2_disjoint_4=152' "$PART_CHECK_MOD_LOG" &&
   grep -q 'k2_check_mod_ontology_cache_directive_probe=ok' "$PART_CHECK_MOD_LOG"; then
  printf 'PASS  check::mod consumed bounded PART native .ontocache semantic verdicts\n'
else
  printf 'FAIL  check::mod did not consume expected bounded PART native .ontocache semantic verdicts\n' >&2
  cat "$PART_CHECK_MOD_LOG" >&2
  exit 1
fi

if bin/souc run "$LIVE_KERNEL_CACHE_PROBE" -- "$PART_CACHE_SRC" >"$PART_LIVE_KERNEL_LOG" 2>&1 &&
   grep -q 'live_kernel_cache_verdict_snomed_73211009_subclass_138875005=152' "$PART_LIVE_KERNEL_LOG" &&
   grep -q 'live_kernel_cache_verdict_part_4_subclass_2=0' "$PART_LIVE_KERNEL_LOG" &&
   grep -q 'live_kernel_cache_verdict_part_2_subclass_4=152' "$PART_LIVE_KERNEL_LOG" &&
   grep -q 'live_kernel_cache_verdict_part_3_disjoint_12=0' "$PART_LIVE_KERNEL_LOG" &&
   grep -q 'live_kernel_cache_verdict_part_2_disjoint_4=152' "$PART_LIVE_KERNEL_LOG" &&
   grep -q 'k2_check_import_ontology_cache_kernel_probe=ok' "$PART_LIVE_KERNEL_LOG"; then
  printf 'PASS  check::check hydrated live Checker.ontology_kernel from bounded PART native .ontocache\n'
else
  printf 'FAIL  check::check did not hydrate expected live Checker.ontology_kernel from bounded PART native .ontocache\n' >&2
  cat "$PART_LIVE_KERNEL_LOG" >&2
  exit 1
fi

if bin/souc run "$CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_PROBE" -- "$PART_CACHE_SRC" >"$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" 2>&1 &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=snomed_positive=152' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=part_positive=0' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=part_reverse=152' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=part_disjoint=0' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=part_disjoint_reverse=0' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar=collect_ok' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar=check_ok' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar=sidecar_hydrate_ok' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar=sidecar_count=5' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'check_mod_same_checker_cache_sidecar_disjoint=sidecar_count=5' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" &&
   grep -q 'k2_check_mod_same_checker_ontology_cache_sidecar_classifier_probe=ok' "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG"; then
  printf 'PASS  check::mod same-sourcecheck sidecar accepted bounded PART subclass/disjoint cache verdicts after ordinary sourcecheck\n'
else
  printf 'FAIL  check::mod same-sourcecheck sidecar failed PART cache verdict probe\n' >&2
  cat "$PART_CHECK_MOD_SAME_CHECKER_SIDECAR_CLASSIFIER_LOG" >&2
  exit 1
fi

if bin/souc run "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_PROBE" -- "$PART_KNOWLEDGE_POSITIVE_SRC" >"$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG" 2>&1 &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=sourcecheck_enter' "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG" &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=verdict=0' "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG" &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=ok' "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG"; then
  printf 'PASS  check::mod sidecar accepted PART Knowledge<T> proof-context after ordinary sourcecheck\n'
else
  printf 'FAIL  check::mod sidecar did not accept PART Knowledge<T> proof-context\n' >&2
  cat "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_LOG" >&2
  exit 1
fi

if bin/souc run "$CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_PROBE" -- "$PART_KNOWLEDGE_DISJOINT_REJECT_SRC" >"$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" 2>&1; then
  printf 'FAIL  check::mod sidecar unexpectedly accepted disjoint PART Knowledge<T> proof-context\n' >&2
  cat "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" >&2
  exit 1
fi
if grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=sourcecheck_enter' "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=verdict=1' "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" &&
   grep -q 'k2_check_mod_knowledge_context_ontocache_sidecar_probe=rejected' "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG"; then
  printf 'PASS  check::mod sidecar rejected disjoint PART Knowledge<T> proof-context after ordinary sourcecheck\n'
else
  printf 'FAIL  check::mod sidecar disjoint PART Knowledge<T> rejection had unexpected shape\n' >&2
  cat "$PART_CHECK_MOD_KNOWLEDGE_CONTEXT_SIDECAR_REJECT_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$MALFORMED_SRC" >"$MALFORMED_LOG" 2>&1; then
  printf 'FAIL  malformed ontology-bundle directive unexpectedly produced a valid load plan\n' >&2
  cat "$MALFORMED_LOG" >&2
  exit 1
fi
if grep -q 'ontology_bundle_directive_found=0' "$MALFORMED_LOG" &&
   grep -q 'malformed=1' "$MALFORMED_LOG" &&
   grep -q 'bundle_exists=0' "$MALFORMED_LOG"; then
  printf 'PASS  compiler-side ontology-bundle directive load plan classified malformed directive\n'
else
  printf 'FAIL  malformed ontology-bundle directive did not produce expected classifier output\n' >&2
  cat "$MALFORMED_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$INVALID_BUNDLE_SRC" >"$INVALID_BUNDLE_LOG" 2>&1; then
  printf 'FAIL  invalid .dontology payload unexpectedly produced a valid load plan\n' >&2
  cat "$INVALID_BUNDLE_LOG" >&2
  exit 1
fi
if grep -q 'ontology_bundle_directive_found=1' "$INVALID_BUNDLE_LOG" &&
   grep -q 'malformed=0' "$INVALID_BUNDLE_LOG" &&
   grep -q 'bundle_exists=1' "$INVALID_BUNDLE_LOG" &&
   grep -q 'dont_magic_valid=0' "$INVALID_BUNDLE_LOG" &&
   grep -q 'dont_payload_size_matches=0' "$INVALID_BUNDLE_LOG"; then
  printf 'PASS  compiler-side ontology-bundle directive load plan rejected invalid DONT envelope\n'
else
  printf 'FAIL  invalid .dontology payload did not produce expected envelope classifier output\n' >&2
  cat "$INVALID_BUNDLE_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$INVALID_PAYLOAD_SRC" >"$INVALID_PAYLOAD_LOG" 2>&1; then
  printf 'FAIL  invalid .dontology payload shape unexpectedly produced a valid load plan\n' >&2
  cat "$INVALID_PAYLOAD_LOG" >&2
  exit 1
fi
if grep -q 'ontology_bundle_directive_found=1' "$INVALID_PAYLOAD_LOG" &&
   grep -q 'malformed=0' "$INVALID_PAYLOAD_LOG" &&
   grep -q 'bundle_exists=1' "$INVALID_PAYLOAD_LOG" &&
   grep -q 'dont_magic_valid=1' "$INVALID_PAYLOAD_LOG" &&
   grep -q 'dont_payload_size_matches=1' "$INVALID_PAYLOAD_LOG" &&
   grep -q 'payload_shape_valid=0' "$INVALID_PAYLOAD_LOG"; then
  printf 'PASS  compiler-side ontology-bundle directive load plan rejected invalid DONT payload shape\n'
else
  printf 'FAIL  invalid .dontology payload shape did not produce expected classifier output\n' >&2
  cat "$INVALID_PAYLOAD_LOG" >&2
  exit 1
fi

cat <<'MSG'
Ontology bundle directive native scan gate passed.
This proves the self-hosted compiler has a Sounio-side source scanner that can
see `//@ ontology-bundle: "..."` in raw source bytes and extract the bundle
path before ordinary lexing drops comments, build a compiler-side load plan,
verify that the referenced .dontology bundle exists, has bytes, and carries a
consistent DONT v1 envelope with the canonical payload fields needed by the
next ingestion stage, including extraction of the ontology name and first term
CURIE plus the first child-parent subclass edge and the first disjoint pair
when present. It also builds a bounded compact side-table prefix with primitive
parallel fields for class IDs, parent edges, and disjoint pairs, with explicit
truncation, and answers direct subclass, bounded subclass-chain, and symmetric
disjoint queries over that staged prefix. It also stages kernel-shaped class
slots plus subclass/disjoint axiom rows and answers equivalent kernel-row
queries, then materializes a bounded OntologyKernel-like subclass/disjoint fact
surface with reflexive subclass facts, transitive subclass closure, and
symmetric disjoint facts without yet mutating Checker.ontology_kernel. It also
writes and reimports a stable bounded native side-table .ontocache artifact for
those staged kernel facts, carrying an explicit authority boundary that the
artifact is not the live checker kernel. A second source file can also declare
`//@ ontology-side-table-cache: "..."` and import that cache through the
compiler-side raw-source scanner. Reimported SNOMED and PART caches answer the
same bounded materialized subclass/disjoint queries as the original load plans,
including through the cache directive path. The same native-cache rung also
handles LOINC alphanumeric CURIE tails by assigning synthetic positive
cache-slot IDs in the bounded artifact, writing and reimporting the first
bounded LOINC parent edge as `4>5`. The cache directive also exposes a
checker-shaped semantic verdict surface: proven subclass/disjoint queries return
0 and rejected subsumption/disjoint queries return 152. The same bounded cache
verdicts are now consumable through a check::mod diagnostic wrapper backed by a
check-owned native side-table cache reader, which makes the imported cache
visible at the checker module boundary. It also proves a focused check::check
hydrator can build a real Checker, populate Checker.ontology_kernel from the
bounded imported cache slots and axioms, materialize the kernel, and answer the
same subclass/disjoint verdicts through ontology_kernel_is_subclass and
ontology_kernel_is_disjoint.
It also classifies the next integration blocker: the candidate sourcecheck
hydrator that parses a source file, resolves the cache, and then tries to run
check::check collect_items still segfaults after cache index resolution and
before collect_ok, matching the known imported full-checker collector boundary.
The check::mod-local sourcecheck path now completes after the ordinary checker
surface and proves positive and reverse bounded subclass verdicts through the
check-owned cache verdict surface without explicit check::check imports.
It also hydrates a stable check::mod sidecar after ordinary sourcecheck:
SNOMED accepts 73211009 -> 138875005 while rejecting the reverse edge, and
PART accepts 4 -> 2 plus symmetric 3/12 disjointness while rejecting reverse
subclass. This is a checker-module sidecar surface, not yet mutation of
Checker.ontology_kernel. That sidecar is also now used as a focused
Knowledge<T> proof-context authority over imported .ontocache facts. The
proof-context resolver now parses generic `PREFIX_<numeric-id>` names rather
than relying on a fixed SNOMED/PART name list, and the path is exposed through
a boot4-style `check_items_verdict_boot4_with_ontocache_sidecar(items,
source_path)` API returning 0 for accepted and 1 for rejected source items. The
SNOMED cache accepts a bounded `dx subclass_of CACHEGEN_138875005` proof
context and rejects the reverse proof context, while the PART cache accepts a
bounded `cache_slot3 subclass_of cache_slot1` proof context and rejects a
disjoint cache_slot2/cache_slot4 proof context after ordinary sourcecheck.
The modular frontend now also composes the imported-cache ontology sidecar with
ordinary value-only proof-context validation: one source accepts SNOMED ontology
plus numeric constraints, another accepts SNOMED ontology plus unit constraints,
and paired reject witnesses fail invalid numeric/unit fields through the same
frontend path.
The fresh same-Checker kernel classifier reaches hydrate_ok after ordinary
sourcecheck, then still rejects the positive subclass verdict from the
pointer-owned ordinary Checker kernel. Hydrating or mutating that same Checker
kernel remains a separate future rung. A second same-Checker copy classifier
can hydrate a local cache-derived checker kernel, but also remains classified
after copy_ok with a rejected positive same-Checker verdict. A third
same-Checker pointer-query classifier also reaches copy_ok, then remains
classified before a query verdict, so pointer-query over the copied ordinary
Checker kernel is still unsafe. A value-threaded same-Checker precheck
classifier reaches collect_ok and hydrate_ok, then remains classified at
collected-checker class_count readback before it can prove the positive kernel
verdict.
It also classifies a direct shadow-field experiment on the ordinary Checker:
after ordinary sourcecheck, the hydrator reaches hydrate_ok, but the pointer
surface still reads shadow_count=0 and returns the empty-shadow reject verdict.
That proves adding small direct fields to Checker is not yet a stable
same-Checker state surface for imported ontology cache facts.
It classifies malformed directives, invalid bundle envelopes, and invalid
payload shapes without producing a valid bundle plan. This is still a
native-loader rung: the scanner does not yet call the C importer through FFI or
wire cache hydration into Checker.ontology_kernel during ordinary source-file
check/compile execution.
MSG
