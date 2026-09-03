#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_ontology_bundle_directive_scan_probe.sio"
FRONTEND_SRC="self-hosted/compiler/lean_frontend.sio"
FRONTEND="$TMP_DIR/souc-lean-frontend-ontology-cache-composition"

SNOMED_BUNDLE_SRC="$TMP_DIR/ontology-cache-composition-snomed-bundle.sio"
SNOMED_CACHE="$TMP_DIR/ontology-cache-composition-snomed.native.ontocache"
SNOMED_CACHE_CONSUMER_SRC="$TMP_DIR/ontology-cache-composition-snomed-cache-consumer.sio"
SNOMED_CACHE_LOG="$TMP_DIR/ontology-cache-composition-snomed-cache.log"
PART_BUNDLE_SRC="$TMP_DIR/ontology-cache-composition-part-bundle.sio"
PART_CACHE="$TMP_DIR/ontology-cache-composition-part.native.ontocache"
PART_CACHE_CONSUMER_SRC="$TMP_DIR/ontology-cache-composition-part-cache-consumer.sio"
PART_CACHE_LOG="$TMP_DIR/ontology-cache-composition-part-cache.log"
QM_BUNDLE_SRC="$TMP_DIR/ontology-cache-composition-qm-bundle.sio"
QM_CACHE="$TMP_DIR/ontology-cache-composition-qm.native.ontocache"
QM_CACHE_CONSUMER_SRC="$TMP_DIR/ontology-cache-composition-qm-cache-consumer.sio"
QM_CACHE_LOG="$TMP_DIR/ontology-cache-composition-qm-cache.log"

NUMERIC_POSITIVE_SRC="$TMP_DIR/ontology-cache-composition-numeric-positive.sio"
NUMERIC_REJECT_SRC="$TMP_DIR/ontology-cache-composition-numeric-reject.sio"
UNIT_POSITIVE_SRC="$TMP_DIR/ontology-cache-composition-unit-positive.sio"
UNIT_REJECT_SRC="$TMP_DIR/ontology-cache-composition-unit-reject.sio"
PART_POSITIVE_SRC="$TMP_DIR/ontology-cache-composition-part-positive.sio"
PART_DISJOINT_REJECT_SRC="$TMP_DIR/ontology-cache-composition-part-disjoint-reject.sio"
QM_POSITIVE_SRC="$TMP_DIR/ontology-cache-composition-qm-positive.sio"
QM_REVERSE_REJECT_SRC="$TMP_DIR/ontology-cache-composition-qm-reverse-reject.sio"

NUMERIC_POSITIVE_LOG="$TMP_DIR/ontology-cache-composition-numeric-positive.log"
NUMERIC_REJECT_LOG="$TMP_DIR/ontology-cache-composition-numeric-reject.log"
UNIT_POSITIVE_LOG="$TMP_DIR/ontology-cache-composition-unit-positive.log"
UNIT_REJECT_LOG="$TMP_DIR/ontology-cache-composition-unit-reject.log"
PART_POSITIVE_LOG="$TMP_DIR/ontology-cache-composition-part-positive.log"
PART_DISJOINT_REJECT_LOG="$TMP_DIR/ontology-cache-composition-part-disjoint-reject.log"
QM_POSITIVE_LOG="$TMP_DIR/ontology-cache-composition-qm-positive.log"
QM_REVERSE_REJECT_LOG="$TMP_DIR/ontology-cache-composition-qm-reverse-reject.log"

printf '%s\n' '//@ ontology-bundle: "stdlib/data/data/ontology/bundles/snomed.dontology"' >"$SNOMED_BUNDLE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_BUNDLE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$SNOMED_CACHE_CONSUMER_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$SNOMED_CACHE_CONSUMER_SRC"
printf '%s\n' '//@ ontology-bundle: "stdlib/data/data/ontology/bundles/part.dontology"' >"$PART_BUNDLE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$PART_BUNDLE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$PART_CACHE" >"$PART_CACHE_CONSUMER_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$PART_CACHE_CONSUMER_SRC"
printf '%s\n' '//@ ontology-bundle: "stdlib/data/data/ontology/bundles/qm.dontology"' >"$QM_BUNDLE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$QM_BUNDLE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$QM_CACHE" >"$QM_CACHE_CONSUMER_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$QM_CACHE_CONSUMER_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$NUMERIC_POSITIVE_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$NUMERIC_POSITIVE_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$NUMERIC_POSITIVE_SRC"
printf '%s\n' 'struct PatientOntocacheNumericPositive { dx: CACHEGEN_73211009, age: i64 }' >>"$NUMERIC_POSITIVE_SRC"
printf '%s\n' 'fn accept_numeric(p: Knowledge<PatientOntocacheNumericPositive where { dx subclass_of CACHEGEN_138875005, age >= 18 }>) -> i32 { 1 }' >>"$NUMERIC_POSITIVE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$NUMERIC_POSITIVE_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$NUMERIC_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$NUMERIC_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$NUMERIC_REJECT_SRC"
printf '%s\n' 'struct PatientOntocacheNumericReject { dx: CACHEGEN_73211009, age: bool }' >>"$NUMERIC_REJECT_SRC"
printf '%s\n' 'fn reject_numeric(p: Knowledge<PatientOntocacheNumericReject where { dx subclass_of CACHEGEN_138875005, age >= 18 }>) -> i32 { 1 }' >>"$NUMERIC_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$NUMERIC_REJECT_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$UNIT_POSITIVE_SRC"
printf '%s\n' 'unit acceleration = m / s / s;' >>"$UNIT_POSITIVE_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$UNIT_POSITIVE_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$UNIT_POSITIVE_SRC"
printf '%s\n' 'struct PatientOntocacheUnitPositive { dx: CACHEGEN_73211009, accel: acceleration }' >>"$UNIT_POSITIVE_SRC"
printf '%s\n' 'fn accept_unit(p: Knowledge<PatientOntocacheUnitPositive where { dx subclass_of CACHEGEN_138875005, accel >= 1.0 <acceleration> }>) -> i32 { 1 }' >>"$UNIT_POSITIVE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$UNIT_POSITIVE_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$SNOMED_CACHE" >"$UNIT_REJECT_SRC"
printf '%s\n' 'unit velocity = m / s;' >>"$UNIT_REJECT_SRC"
printf '%s\n' 'unit acceleration = m / s / s;' >>"$UNIT_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_73211009 { code: i64 }' >>"$UNIT_REJECT_SRC"
printf '%s\n' 'struct CACHEGEN_138875005 { code: i64 }' >>"$UNIT_REJECT_SRC"
printf '%s\n' 'struct PatientOntocacheUnitReject { dx: CACHEGEN_73211009, accel: velocity }' >>"$UNIT_REJECT_SRC"
printf '%s\n' 'fn reject_unit(p: Knowledge<PatientOntocacheUnitReject where { dx subclass_of CACHEGEN_138875005, accel >= 1.0 <acceleration> }>) -> i32 { 1 }' >>"$UNIT_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$UNIT_REJECT_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$PART_CACHE" >"$PART_POSITIVE_SRC"
printf '%s\n' 'struct cache_slot3 { code: i64 }' >>"$PART_POSITIVE_SRC"
printf '%s\n' 'struct cache_slot1 { code: i64 }' >>"$PART_POSITIVE_SRC"
printf '%s\n' 'struct PatientPartOntocachePositive { dx: cache_slot3 }' >>"$PART_POSITIVE_SRC"
printf '%s\n' 'fn accept_part(p: Knowledge<PatientPartOntocachePositive where { dx subclass_of cache_slot1 }>) -> i32 { 1 }' >>"$PART_POSITIVE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$PART_POSITIVE_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$PART_CACHE" >"$PART_DISJOINT_REJECT_SRC"
printf '%s\n' 'struct cache_slot2 { code: i64 }' >>"$PART_DISJOINT_REJECT_SRC"
printf '%s\n' 'struct cache_slot4 { code: i64 }' >>"$PART_DISJOINT_REJECT_SRC"
printf '%s\n' 'struct PatientPartOntocacheDisjointReject { dx: cache_slot2 }' >>"$PART_DISJOINT_REJECT_SRC"
printf '%s\n' 'fn reject_part_disjoint(p: Knowledge<PatientPartOntocacheDisjointReject where { dx subclass_of cache_slot4 }>) -> i32 { 1 }' >>"$PART_DISJOINT_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$PART_DISJOINT_REJECT_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$QM_CACHE" >"$QM_POSITIVE_SRC"
printf '%s\n' 'struct cache_slot3 { code: i64 }' >>"$QM_POSITIVE_SRC"
printf '%s\n' 'struct cache_slot0 { code: i64 }' >>"$QM_POSITIVE_SRC"
printf '%s\n' 'struct PatientQmOntocachePositive { dx: cache_slot3 }' >>"$QM_POSITIVE_SRC"
printf '%s\n' 'fn accept_qm(p: Knowledge<PatientQmOntocachePositive where { dx subclass_of cache_slot0 }>) -> i32 { 1 }' >>"$QM_POSITIVE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$QM_POSITIVE_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$QM_CACHE" >"$QM_REVERSE_REJECT_SRC"
printf '%s\n' 'struct cache_slot0 { code: i64 }' >>"$QM_REVERSE_REJECT_SRC"
printf '%s\n' 'struct cache_slot3 { code: i64 }' >>"$QM_REVERSE_REJECT_SRC"
printf '%s\n' 'struct PatientQmOntocacheReverseReject { dx: cache_slot0 }' >>"$QM_REVERSE_REJECT_SRC"
printf '%s\n' 'fn reject_qm_reverse(p: Knowledge<PatientQmOntocacheReverseReject where { dx subclass_of cache_slot3 }>) -> i32 { 1 }' >>"$QM_REVERSE_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$QM_REVERSE_REJECT_SRC"

bin/souc check "$PROBE"
bin/souc "$FRONTEND_SRC" "$FRONTEND"

if bin/souc run "$PROBE" -- "$SNOMED_BUNDLE_SRC" "$SNOMED_CACHE" "$SNOMED_CACHE_CONSUMER_SRC" >"$SNOMED_CACHE_LOG" 2>&1 &&
   grep -q 'native_side_table_cache_written=1' "$SNOMED_CACHE_LOG" &&
   grep -q 'imported_cache_valid=1' "$SNOMED_CACHE_LOG" &&
   grep -q 'cache_directive_import_valid=1' "$SNOMED_CACHE_LOG" &&
   grep -q 'cache_directive_verdict_snomed_73211009_subclass_138875005=0' "$SNOMED_CACHE_LOG" &&
   grep -q 'cache_directive_verdict_snomed_138875005_subclass_73211009=152' "$SNOMED_CACHE_LOG"; then
  printf 'PASS  generated and reimported bounded SNOMED .ontocache for frontend composition\n'
else
  printf 'FAIL  could not generate/reimport bounded SNOMED .ontocache for frontend composition\n' >&2
  cat "$SNOMED_CACHE_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$PART_BUNDLE_SRC" "$PART_CACHE" "$PART_CACHE_CONSUMER_SRC" >"$PART_CACHE_LOG" 2>&1 &&
   grep -q 'native_side_table_cache_written=1' "$PART_CACHE_LOG" &&
   grep -q 'imported_cache_valid=1' "$PART_CACHE_LOG" &&
   grep -q 'cache_directive_import_valid=1' "$PART_CACHE_LOG" &&
   grep -q 'cache_directive_verdict_part_4_subclass_2=0' "$PART_CACHE_LOG" &&
   grep -q 'cache_directive_verdict_part_2_subclass_4=152' "$PART_CACHE_LOG" &&
   grep -q 'cache_directive_verdict_part_3_disjoint_12=0' "$PART_CACHE_LOG"; then
  printf 'PASS  generated and reimported bounded PART .ontocache for frontend composition\n'
else
  printf 'FAIL  could not generate/reimport bounded PART .ontocache for frontend composition\n' >&2
  cat "$PART_CACHE_LOG" >&2
  exit 1
fi

bin/souc run "$PROBE" -- "$QM_BUNDLE_SRC" "$QM_CACHE" "$QM_CACHE_CONSUMER_SRC" >"$QM_CACHE_LOG" 2>&1 || true
if grep -q 'native_side_table_cache_written=1' "$QM_CACHE_LOG" &&
   grep -q 'imported_cache_valid=1' "$QM_CACHE_LOG" &&
   grep -q 'imported_cache_ontology=QM' "$QM_CACHE_LOG" &&
   grep -q 'imported_cache_slot_count=4' "$QM_CACHE_LOG" &&
   grep -q 'cache_directive_import_valid=1' "$QM_CACHE_LOG"; then
  printf 'PASS  generated and reimported bounded QM .ontocache for frontend composition via generic cache evidence\n'
else
  printf 'FAIL  could not generate/reimport bounded QM .ontocache for frontend composition\n' >&2
  cat "$QM_CACHE_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$NUMERIC_POSITIVE_SRC" >"$NUMERIC_POSITIVE_LOG" 2>&1; then
  printf 'PASS  modular frontend accepted imported-cache ontology plus numeric Knowledge<T> proof-context\n'
else
  printf 'FAIL  modular frontend rejected imported-cache ontology plus numeric Knowledge<T> proof-context\n' >&2
  cat "$NUMERIC_POSITIVE_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$NUMERIC_REJECT_SRC" >"$NUMERIC_REJECT_LOG" 2>&1; then
  printf 'FAIL  modular frontend unexpectedly accepted invalid numeric Knowledge<T> proof-context\n' >&2
  cat "$NUMERIC_REJECT_LOG" >&2
  exit 1
fi
printf 'PASS  modular frontend rejected imported-cache ontology when numeric Knowledge<T> proof-context fails\n'

if "$FRONTEND" --check "$UNIT_POSITIVE_SRC" >"$UNIT_POSITIVE_LOG" 2>&1; then
  printf 'PASS  modular frontend accepted imported-cache ontology plus unit Knowledge<T> proof-context\n'
else
  printf 'FAIL  modular frontend rejected imported-cache ontology plus unit Knowledge<T> proof-context\n' >&2
  cat "$UNIT_POSITIVE_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$UNIT_REJECT_SRC" >"$UNIT_REJECT_LOG" 2>&1; then
  printf 'FAIL  modular frontend unexpectedly accepted invalid unit Knowledge<T> proof-context\n' >&2
  cat "$UNIT_REJECT_LOG" >&2
  exit 1
fi
printf 'PASS  modular frontend rejected imported-cache ontology when unit Knowledge<T> proof-context fails\n'

if "$FRONTEND" --check "$PART_POSITIVE_SRC" >"$PART_POSITIVE_LOG" 2>&1; then
  printf 'PASS  modular frontend accepted PART imported-cache subclass Knowledge<T> proof-context\n'
else
  printf 'FAIL  modular frontend rejected PART imported-cache subclass Knowledge<T> proof-context\n' >&2
  cat "$PART_POSITIVE_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$PART_DISJOINT_REJECT_SRC" >"$PART_DISJOINT_REJECT_LOG" 2>&1; then
  printf 'FAIL  modular frontend unexpectedly accepted PART imported-cache disjoint Knowledge<T> proof-context\n' >&2
  cat "$PART_DISJOINT_REJECT_LOG" >&2
  exit 1
fi
printf 'PASS  modular frontend rejected PART imported-cache disjoint Knowledge<T> proof-context\n'

if "$FRONTEND" --check "$QM_POSITIVE_SRC" >"$QM_POSITIVE_LOG" 2>&1; then
  printf 'PASS  modular frontend accepted QM imported-cache subclass Knowledge<T> proof-context\n'
else
  printf 'FAIL  modular frontend rejected QM imported-cache subclass Knowledge<T> proof-context\n' >&2
  cat "$QM_POSITIVE_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$QM_REVERSE_REJECT_SRC" >"$QM_REVERSE_REJECT_LOG" 2>&1; then
  printf 'FAIL  modular frontend unexpectedly accepted QM imported-cache reverse subclass Knowledge<T> proof-context\n' >&2
  cat "$QM_REVERSE_REJECT_LOG" >&2
  exit 1
fi
printf 'PASS  modular frontend rejected QM imported-cache reverse subclass Knowledge<T> proof-context\n'

GENERIC_BUNDLE_FRONTEND_PAIRS=(
  "ALG:alg:ALG"
  "ChEBI:chebi:CHEBI"
  "GO:go:GO"
  "HPO:hpo:HPO"
  "PHYS:phys:PHYS"
)

for bundle_spec in "${GENERIC_BUNDLE_FRONTEND_PAIRS[@]}"; do
  ontology_name="${bundle_spec%%:*}"
  bundle_rest="${bundle_spec#*:}"
  bundle_slug="${bundle_rest%%:*}"
  bundle_label="${bundle_rest#*:}"

  bundle_src="$TMP_DIR/ontology-cache-composition-${bundle_slug}-bundle.sio"
  bundle_cache="$TMP_DIR/ontology-cache-composition-${bundle_slug}.native.ontocache"
  bundle_consumer_src="$TMP_DIR/ontology-cache-composition-${bundle_slug}-cache-consumer.sio"
  bundle_cache_log="$TMP_DIR/ontology-cache-composition-${bundle_slug}-cache.log"
  bundle_positive_src="$TMP_DIR/ontology-cache-composition-${bundle_slug}-generic-positive.sio"
  bundle_reverse_reject_src="$TMP_DIR/ontology-cache-composition-${bundle_slug}-generic-reverse-reject.sio"
  bundle_positive_log="$TMP_DIR/ontology-cache-composition-${bundle_slug}-generic-positive.log"
  bundle_reverse_reject_log="$TMP_DIR/ontology-cache-composition-${bundle_slug}-generic-reverse-reject.log"

  printf '//@ ontology-bundle: "stdlib/data/data/ontology/bundles/%s.dontology"\n' "$bundle_slug" >"$bundle_src"
  printf '%s\n' 'fn main() -> i64 { 0 }' >>"$bundle_src"
  printf '//@ ontology-side-table-cache: "%s"\n' "$bundle_cache" >"$bundle_consumer_src"
  printf '%s\n' 'fn main() -> i64 { 0 }' >>"$bundle_consumer_src"

  bin/souc run "$PROBE" -- "$bundle_src" "$bundle_cache" "$bundle_consumer_src" >"$bundle_cache_log" 2>&1 || true
  if grep -q 'native_side_table_cache_written=1' "$bundle_cache_log" &&
     grep -q 'imported_cache_valid=1' "$bundle_cache_log" &&
     grep -q "imported_cache_ontology=${ontology_name}" "$bundle_cache_log" &&
     grep -q 'imported_cache_slot_count=4' "$bundle_cache_log" &&
     grep -q 'cache_directive_import_valid=1' "$bundle_cache_log"; then
    printf 'PASS  generated and reimported bounded %s .ontocache for generic frontend composition\n' "$bundle_label"
  else
    printf 'FAIL  could not generate/reimport bounded %s .ontocache for generic frontend composition\n' "$bundle_label" >&2
    cat "$bundle_cache_log" >&2
    exit 1
  fi

  printf '//@ ontology-side-table-cache: "%s"\n' "$bundle_cache" >"$bundle_positive_src"
  printf '%s\n' 'struct cache_slot3 { code: i64 }' >>"$bundle_positive_src"
  printf '%s\n' 'struct cache_slot0 { code: i64 }' >>"$bundle_positive_src"
  printf '%s\n' 'struct GenericOntocachePositive { dx: cache_slot3 }' >>"$bundle_positive_src"
  printf '%s\n' 'fn accept_generic(p: Knowledge<GenericOntocachePositive where { dx subclass_of cache_slot0 }>) -> i32 { 1 }' >>"$bundle_positive_src"
  printf '%s\n' 'fn main() -> i64 { 0 }' >>"$bundle_positive_src"

  printf '//@ ontology-side-table-cache: "%s"\n' "$bundle_cache" >"$bundle_reverse_reject_src"
  printf '%s\n' 'struct cache_slot0 { code: i64 }' >>"$bundle_reverse_reject_src"
  printf '%s\n' 'struct cache_slot3 { code: i64 }' >>"$bundle_reverse_reject_src"
  printf '%s\n' 'struct GenericOntocacheReverseReject { dx: cache_slot0 }' >>"$bundle_reverse_reject_src"
  printf '%s\n' 'fn reject_generic_reverse(p: Knowledge<GenericOntocacheReverseReject where { dx subclass_of cache_slot3 }>) -> i32 { 1 }' >>"$bundle_reverse_reject_src"
  printf '%s\n' 'fn main() -> i64 { 0 }' >>"$bundle_reverse_reject_src"

  if "$FRONTEND" --check "$bundle_positive_src" >"$bundle_positive_log" 2>&1; then
    printf 'PASS  modular frontend accepted %s generic imported-cache subclass Knowledge<T> proof-context\n' "$bundle_label"
  else
    printf 'FAIL  modular frontend rejected %s generic imported-cache subclass Knowledge<T> proof-context\n' "$bundle_label" >&2
    cat "$bundle_positive_log" >&2
    exit 1
  fi

  if "$FRONTEND" --check "$bundle_reverse_reject_src" >"$bundle_reverse_reject_log" 2>&1; then
    printf 'FAIL  modular frontend unexpectedly accepted %s generic imported-cache reverse subclass Knowledge<T> proof-context\n' "$bundle_label" >&2
    cat "$bundle_reverse_reject_log" >&2
    exit 1
  fi
  printf 'PASS  modular frontend rejected %s generic imported-cache reverse subclass Knowledge<T> proof-context\n' "$bundle_label"
done

LOINC_BUNDLE_SRC="$TMP_DIR/ontology-cache-composition-loinc-bundle.sio"
LOINC_CACHE="$TMP_DIR/ontology-cache-composition-loinc.native.ontocache"
LOINC_CACHE_CONSUMER_SRC="$TMP_DIR/ontology-cache-composition-loinc-cache-consumer.sio"
LOINC_CACHE_LOG="$TMP_DIR/ontology-cache-composition-loinc-cache.log"
LOINC_POSITIVE_SRC="$TMP_DIR/ontology-cache-composition-loinc-positive.sio"
LOINC_REVERSE_REJECT_SRC="$TMP_DIR/ontology-cache-composition-loinc-reverse-reject.sio"
LOINC_POSITIVE_LOG="$TMP_DIR/ontology-cache-composition-loinc-positive.log"
LOINC_REVERSE_REJECT_LOG="$TMP_DIR/ontology-cache-composition-loinc-reverse-reject.log"

printf '%s\n' '//@ ontology-bundle: "stdlib/data/data/ontology/bundles/loinc.dontology"' >"$LOINC_BUNDLE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$LOINC_BUNDLE_SRC"
printf '//@ ontology-side-table-cache: "%s"\n' "$LOINC_CACHE" >"$LOINC_CACHE_CONSUMER_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$LOINC_CACHE_CONSUMER_SRC"

bin/souc run "$PROBE" -- "$LOINC_BUNDLE_SRC" "$LOINC_CACHE" "$LOINC_CACHE_CONSUMER_SRC" >"$LOINC_CACHE_LOG" 2>&1 || true
if grep -q 'native_side_table_cache_written=1' "$LOINC_CACHE_LOG" &&
   grep -q 'imported_cache_valid=1' "$LOINC_CACHE_LOG" &&
   grep -q 'imported_cache_ontology=LOINC' "$LOINC_CACHE_LOG" &&
   grep -q 'imported_cache_slot_count=6' "$LOINC_CACHE_LOG" &&
   grep -q 'cache_directive_import_valid=1' "$LOINC_CACHE_LOG"; then
  printf 'PASS  generated and reimported bounded LOINC .ontocache for alphanumeric frontend composition\n'
else
  printf 'FAIL  could not generate/reimport bounded LOINC .ontocache for alphanumeric frontend composition\n' >&2
  cat "$LOINC_CACHE_LOG" >&2
  exit 1
fi

printf '//@ ontology-side-table-cache: "%s"\n' "$LOINC_CACHE" >"$LOINC_POSITIVE_SRC"
printf '%s\n' 'struct cache_slot4 { code: i64 }' >>"$LOINC_POSITIVE_SRC"
printf '%s\n' 'struct cache_slot5 { code: i64 }' >>"$LOINC_POSITIVE_SRC"
printf '%s\n' 'struct LoincOntocachePositive { dx: cache_slot4 }' >>"$LOINC_POSITIVE_SRC"
printf '%s\n' 'fn accept_loinc(p: Knowledge<LoincOntocachePositive where { dx subclass_of cache_slot5 }>) -> i32 { 1 }' >>"$LOINC_POSITIVE_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$LOINC_POSITIVE_SRC"

printf '//@ ontology-side-table-cache: "%s"\n' "$LOINC_CACHE" >"$LOINC_REVERSE_REJECT_SRC"
printf '%s\n' 'struct cache_slot5 { code: i64 }' >>"$LOINC_REVERSE_REJECT_SRC"
printf '%s\n' 'struct cache_slot4 { code: i64 }' >>"$LOINC_REVERSE_REJECT_SRC"
printf '%s\n' 'struct LoincOntocacheReverseReject { dx: cache_slot5 }' >>"$LOINC_REVERSE_REJECT_SRC"
printf '%s\n' 'fn reject_loinc_reverse(p: Knowledge<LoincOntocacheReverseReject where { dx subclass_of cache_slot4 }>) -> i32 { 1 }' >>"$LOINC_REVERSE_REJECT_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$LOINC_REVERSE_REJECT_SRC"

if "$FRONTEND" --check "$LOINC_POSITIVE_SRC" >"$LOINC_POSITIVE_LOG" 2>&1; then
  printf 'PASS  modular frontend accepted LOINC alphanumeric imported-cache subclass Knowledge<T> proof-context\n'
else
  printf 'FAIL  modular frontend rejected LOINC alphanumeric imported-cache subclass Knowledge<T> proof-context\n' >&2
  cat "$LOINC_POSITIVE_LOG" >&2
  exit 1
fi

if "$FRONTEND" --check "$LOINC_REVERSE_REJECT_SRC" >"$LOINC_REVERSE_REJECT_LOG" 2>&1; then
  printf 'FAIL  modular frontend unexpectedly accepted LOINC alphanumeric imported-cache reverse subclass Knowledge<T> proof-context\n' >&2
  cat "$LOINC_REVERSE_REJECT_LOG" >&2
  exit 1
fi
printf 'PASS  modular frontend rejected LOINC alphanumeric imported-cache reverse subclass Knowledge<T> proof-context\n'

cat <<'MSG'
Ontology cache frontend composition gate passed.
This proves bounded SNOMED, PART, QM, ALG, ChEBI, GO, HPO, PHYS, and LOINC .ontocache
artifacts generated by Sounio can be consumed by the modular frontend through
`//@ ontology-side-table-cache: "..."`, where the imported-cache ontology
sidecar composes with ordinary value-only check::knowledge_context validation.
The gate accepts SNOMED ontology+numeric and SNOMED ontology+unit Knowledge<T>
proof contexts, rejects paired invalid numeric/unit fields, accepts a PART
imported-cache subclass proof context, and rejects a PART disjoint
proof-context. It also accepts a QM imported-cache subclass proof context and
rejects the reverse subclass proof context through the same frontend path. It
also accepts generic slot3-to-slot0 subclass proof contexts and rejects the
reverse proof context for ALG, ChEBI, GO, HPO, and PHYS. LOINC uses synthetic
positive cache-slot IDs for alphanumeric CURIE tails and proves the first
bounded parent edge through cache_slot4-to-cache_slot5 plus reverse rejection.
This is still a sidecar composition proof, not mutation of Checker.ontology_kernel
during ordinary source-file check/compile execution.
MSG
