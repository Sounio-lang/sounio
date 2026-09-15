#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_knowledge_unit_constraint_probe.sio"
POSITIVE="tests/frontend/knowledge_static_unit_constraint_positive.sio"
NEGATIVE="tests/frontend/knowledge_static_unit_constraint_reject.sio"
INTERNAL_LABEL_POSITIVE="tests/frontend/knowledge_static_unit_internal_label_positive.sio"
INTERNAL_LABEL_NEGATIVE="tests/frontend/knowledge_static_unit_internal_label_reject.sio"

PROBE_ELF="$TMP_DIR/knowledge-unit-probe.elf"
# `bin/souc run PROBE -- FILE` forwards nothing after the `--`: bin/madaros
# documents `run` as taking a source and nothing else. The probe therefore never
# received the file it was asked to analyse and fell back to its own default,
# which is the positive fixture. Every check below was reading that same file --
# so the positive passed vacuously, and the negative could not fail for its own
# reason. Build the probe once and hand it the path directly.
if ! bin/madaros build "$PROBE" -o "$PROBE_ELF" >"$TMP_DIR/probe-build.log" 2>&1; then
  printf 'FAIL  could not build %s\n' "$PROBE" >&2
  cat "$TMP_DIR/probe-build.log" >&2
  exit 1
fi
chmod +x "$PROBE_ELF"

POSITIVE_LOG="$TMP_DIR/knowledge-unit-positive.log"
if "$PROBE_ELF" "$POSITIVE" >"$POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=0' "$POSITIVE_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=0' "$POSITIVE_LOG"; then
  printf 'PASS  %s accepted matching Knowledge<T> unit constraint\n' "$POSITIVE"
else
  printf 'FAIL  %s did not accept the matching Knowledge<T> unit constraint\n' "$POSITIVE" >&2
  cat "$POSITIVE_LOG" >&2
  exit 1
fi

NEGATIVE_LOG="$TMP_DIR/knowledge-unit-negative.log"
if "$PROBE_ELF" "$NEGATIVE" >"$NEGATIVE_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=1' "$NEGATIVE_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=1' "$NEGATIVE_LOG"; then
  printf 'PASS  %s rejected incompatible Knowledge<T> unit constraint\n' "$NEGATIVE"
else
  printf 'FAIL  %s did not reject the incompatible Knowledge<T> unit constraint\n' "$NEGATIVE" >&2
  cat "$NEGATIVE_LOG" >&2
  exit 1
fi

INTERNAL_LABEL_POSITIVE_LOG="$TMP_DIR/knowledge-unit-internal-label-positive.log"
if "$PROBE_ELF" "$INTERNAL_LABEL_POSITIVE" >"$INTERNAL_LABEL_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=0' "$INTERNAL_LABEL_POSITIVE_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=0' "$INTERNAL_LABEL_POSITIVE_LOG"; then
  printf 'PASS  %s accepted internal validation-data unit labels in Knowledge<T> constraints\n' "$INTERNAL_LABEL_POSITIVE"
else
  printf 'FAIL  %s did not accept internal validation-data unit labels in Knowledge<T> constraints\n' "$INTERNAL_LABEL_POSITIVE" >&2
  cat "$INTERNAL_LABEL_POSITIVE_LOG" >&2
  exit 1
fi

INTERNAL_LABEL_NEGATIVE_LOG="$TMP_DIR/knowledge-unit-internal-label-negative.log"
if "$PROBE_ELF" "$INTERNAL_LABEL_NEGATIVE" >"$INTERNAL_LABEL_NEGATIVE_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=1' "$INTERNAL_LABEL_NEGATIVE_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=1' "$INTERNAL_LABEL_NEGATIVE_LOG"; then
  printf 'PASS  %s rejected incompatible internal validation-data unit label in Knowledge<T> constraint\n' "$INTERNAL_LABEL_NEGATIVE"
else
  printf 'FAIL  %s did not reject incompatible internal validation-data unit label in Knowledge<T> constraint\n' "$INTERNAL_LABEL_NEGATIVE" >&2
  cat "$INTERNAL_LABEL_NEGATIVE_LOG" >&2
  exit 1
fi

cat <<'MSG'
Knowledge unit constraint gate passed.
This proves that `Knowledge<T where { field >= value <unit> }>` retains its
unit payload and that the ordinary check::knowledge_context semantic bridge
rejects incompatible dimensions over derived units. It also proves the modular
UnitRegistry mirrors internal validation-data labels such as mg_dL, mmol_L,
U_L, and mm_h for dimensional checking only; this is not clinical, conversion,
UCUM, LOINC, ChEBI, dosing, or regulatory-exchange authority. The standalone
check::check import bridge remains covered by its own classifier, not by this
gate.
MSG
