#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

TMP="${TMPDIR:-/tmp}/sounio-m6-closure-v2"
rm -rf "$TMP"
mkdir -p "$TMP"

BOOT4="./artifacts/bootstrap/boot4.elf"

echo "[m6] gen1"
"$BOOT4" bootstrap/boot4.sio "$TMP/gen1.elf"
chmod +x "$TMP/gen1.elf"

echo "[m6] gen2"
"$TMP/gen1.elf" bootstrap/boot4.sio "$TMP/gen2.elf"
chmod +x "$TMP/gen2.elf"

echo "[m6] gen3"
"$TMP/gen2.elf" bootstrap/boot4.sio "$TMP/gen3.elf"
chmod +x "$TMP/gen3.elf"

cmp "$TMP/gen2.elf" "$TMP/gen3.elf"
echo "[m6] fixed-point PASS"

B="$TMP/gen2.elf"
for probe in m4_flat_direct m4_flat_transitive m4_namespaced_direct m4_namespaced_transitive; do
  echo "[m6] probe $probe"
  "$B" "scripts/ci/${probe}_probe.sio" "$TMP/${probe}.elf" >"$TMP/${probe}.build.log" 2>&1
  chmod +x "$TMP/${probe}.elf"
  timeout 30 "$TMP/${probe}.elf" >"$TMP/${probe}.run.log" 2>&1
done

echo "[m6] telemetry probe"
"$B" scripts/ci/m4_flat_transitive_probe.sio "$TMP/v2test.elf" >"$TMP/v2test.build.log" 2>&1
grep -q "CLOSURE_BUNDLE_V2" "$TMP/v2test.build.log"
grep -q "CLOSURE_BUNDLE_V1" "$TMP/v2test.build.log"
test "$(grep -c '^CLOSURE_BUNDLE_V2_MODULE ' "$TMP/v2test.build.log")" -ge 2

echo "[m6] ontology_min_input"
"$B" scripts/ci/ontology_min_input.sio "$TMP/ontology_min_input.elf" >"$TMP/ontology_min_input.build.log" 2>&1
chmod +x "$TMP/ontology_min_input.elf"

echo "[m6] ontology_witness diagnostics"
set +e
"$B" scripts/ci/ontology_witness_program_probe.sio "$TMP/ontology_witness_program_probe.elf" >"$TMP/ontology_witness_program_probe.build.log" 2>&1
OWP_RC=$?
set -e
grep -q "CLOSURE_BUNDLE_V2" "$TMP/ontology_witness_program_probe.build.log"
echo "[m6] ontology_witness_program_probe rc=$OWP_RC"

echo "[m6] PASS"
