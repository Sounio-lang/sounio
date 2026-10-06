#!/bin/bash
# scripts/tour.sh -- run every claim in TOUR.md and print what happened.
#
#   bash scripts/tour.sh          # fast set, about 1 minute
#   bash scripts/tour.sh --full   # adds the slow runs, about 6 minutes
#   bash scripts/tour.sh --lean   # also builds the Lean proofs (needs elan)
#
# Each line is PASS or FAIL against the sentinel the program itself prints.
# Nothing is summarised that was not run. Linux x86-64 only (the compiler
# emits static x86-64 ELF).
set -u
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
FULL=0; LEAN=0
for a in "$@"; do
  case "$a" in --full) FULL=1 ;; --lean) LEAN=1 ;; esac
done

pass=0; fail=0
row() {  # row <label> <engine: mad|lean> <expected sentinel> <file> [timeout]
  local label=$1 eng=$2 want=$3 file=$4 to=${5:-300} out rc t0 t1
  if [ ! -f "$file" ]; then printf '  %-4s %-46s %s\n' "SKIP" "$label" "(not on this branch: $file)"; return; fi
  t0=$(date +%s)
  if [ "$eng" = lean ]; then out=$(SOUNIO_SOUC_ENGINE=lean_single timeout "$to" bin/souc run "$file" 2>&1)
  else out=$(timeout "$to" bin/souc run "$file" 2>&1); fi
  t1=$(date +%s)
  if printf '%s\n' "$out" | grep -qF -- "$want"; then
    printf '  PASS %-46s %4ss  %s\n' "$label" "$((t1 - t0))" "$want"; pass=$((pass + 1))
  else
    printf '  FAIL %-46s %4ss  expected "%s"\n' "$label" "$((t1 - t0))" "$want"; fail=$((fail + 1))
  fi
}

echo "== 1. The epistemic core =="
row "GUM uncertainty propagation"            mad  "Sucesso! Incerteza propagada" demo_incerteza.sio
row "units: m/s derived and checked"         lean "Velocidade calculada" demo_unidades.sio
# The guard fires only after the source-to-source expansion (issue #2753).
if bash scripts/ontology/expand_knowledge_runtime_guards.sh demo_portas_rejeicao.sio /tmp/tour_guard.sio >/dev/null 2>&1; then
  if timeout 120 bin/souc run /tmp/tour_guard.sio >/tmp/tour_guard.out 2>&1; then
    printf '  FAIL %-46s        guard did not fire (rc=0)\n' "Knowledge<T where ..> guard rejects age 13"; fail=$((fail + 1))
  elif grep -q "Este print" /tmp/tour_guard.out; then
    printf '  FAIL %-46s        guarded line ran\n' "Knowledge<T where ..> guard rejects age 13"; fail=$((fail + 1))
  else
    printf '  PASS %-46s        rc!=0, guarded line never ran\n' "Knowledge<T where ..> guard rejects age 13"; pass=$((pass + 1))
  fi
fi

echo "== 2. Knowledge graphs: the verified EL+ reasoner, executed =="
row "EL+ role-aware closure (SNOMED TBox)"   mad  "ALL PASS" examples/ontology_elplus_closure_demo.sio
row "SNOMED fragment: Pericarditis < ∃fs.Heart" mad "ALL PASS" examples/ontology/biomedical/snomed_elplus_demo.sio
row "PROV-DM provenance as EL+ role axioms"  mad  "ALL PASS" examples/epistemic/prov_elplus_demo.sio
row "VIM3 metrological traceability chains"  mad  "ALL PASS" examples/epistemic/traceability_elplus_demo.sio
row "drug-drug interaction (smoke demo)"     mad  "ALL PASS" examples/clinical/ddi_elplus_demo.sio
row "pharmacogenomics (smoke demo)"          mad  "ALL PASS" examples/clinical/pgx_elplus_demo.sio
row "pipeline: closure + repair + query"     mad  "ALL PASS" examples/ontology_pipeline_demo.sio

echo "== 3. Simulation: hydrogen chemistry =="
row "MH7 cascade ceiling, nothing fitted"    mad  "MH7_COUPLED_CEILING_OK" demos/hydrogen/mh7_coupled_ceiling.sio
row "van 't Hoff extrapolation gate"         mad  "VANTHOFF_GATE_OK" demos/hydrogen/vanthoff_gate.sio
row "methanation log K gate"                 mad  "METHANATION_LOGK_GATE_OK" demos/hydrogen/methanation_logk_gate.sio
if [ "$FULL" = 1 ]; then
  row "UHS H2-brine-calcite network + p-boxes" lean "UHS_BRINE_CALCITE_OK" demos/hydrogen/uhs_brine_calcite.sio 400
  row "GRI-Mech 3.0 H2 ignition with UQ"      mad  "DEMO DONE" examples/chemistry/h2_ignition_uq_demo.sio 400
fi

if [ "$FULL" = 1 ]; then
  echo "== 4. Self-hosting: rebuild the compiler and check the fixed point =="
  t0=$(date +%s)
  if make build >/tmp/tour_build.log 2>&1 && grep -q "FIXED POINT OK" /tmp/tour_build.log; then
    printf '  PASS %-46s %4ss  %s\n' "gen2 == gen3, bit-identical" "$(( $(date +%s) - t0 ))" "$(grep -o 'FIXED POINT OK ([0-9a-f]*)' /tmp/tour_build.log)"
    pass=$((pass + 1))
  else
    printf '  FAIL %-46s\n' "gen2 == gen3, bit-identical"; fail=$((fail + 1))
  fi
  rm -f gen1.elf gen2.elf gen3.elf
fi

if [ "$LEAN" = 1 ]; then
  echo "== 5. Lean 4: the proofs behind sections 2 and 3 =="
  if command -v lake >/dev/null; then
    t0=$(date +%s)
    if (cd formal && lake build OntologyELPlusClosureComplete) >/tmp/tour_lean1.log 2>&1 \
       && ! grep -q "declaration uses 'sorry'" /tmp/tour_lean1.log; then
      printf '  PASS %-46s %4ss\n' "EL+ closure soundness+completeness, no sorry" "$(( $(date +%s) - t0 ))"; pass=$((pass + 1))
    else printf '  FAIL %-46s\n' "EL+ closure (formal/)"; fail=$((fail + 1)); fi
    for f in SounioHydrogenPbox SounioHydrogenReceipt SounioHydrogenVanthoff; do
      if (cd formal/lean4 && lake env lean "$f.lean") >/tmp/tour_lean2.log 2>&1 \
         && ! grep -q "declaration uses 'sorry'" /tmp/tour_lean2.log; then
        printf '  PASS %-46s\n' "$f, no sorry"; pass=$((pass + 1))
      else printf '  FAIL %-46s\n' "$f"; fail=$((fail + 1)); fi
    done
  else
    echo "  SKIP (no lake on PATH; install elan, then re-run with --lean)"
  fi
fi

echo
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
