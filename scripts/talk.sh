#!/usr/bin/env bash
# scripts/talk.sh — the NCSR Demokritos run-of-show as one command.
#
#   bash scripts/talk.sh            run every live demo in order (target < 6 min)
#   bash scripts/talk.sh --fast     run demos under 20 s; for the rest, verify the
#                                   committed golden output (target < 90 s)
#   bash scripts/talk.sh --record   run everything, including golden-only demos,
#                                   and rewrite demos/demokritos/golden/
#   bash scripts/talk.sh --list     print the run-of-show and exit
#
# Every demo pins its engine and is judged by a sentinel its own output must
# contain; one PASS/FAIL line per demo with wall time; exit non-zero on any
# failure. A demo whose file is missing FAILS — nothing here is skipped quietly.
# See demos/demokritos/README.md for what each step shows and what it does not.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 9
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
GOLDEN_DIR="$ROOT_DIR/demos/demokritos/golden"
FAST_LIMIT_S=20

mode=full
only=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) mode=fast ;;
    --record) mode=record ;;
    --list) mode=list ;;
    --only) only="${2:?--only needs a step id}"; shift ;;
    *) echo "usage: $0 [--fast|--record|--list] [--only <step-id>]" >&2; exit 2 ;;
  esac
  shift
done
# --only <id>: run (or with --record, re-record) a single step, e.g. after
# changing one demo; everything else is skipped silently.
selected() { [[ -z "$only" || "$1" == "$only" ]]; }

bash scripts/lib/materialize_madaros_prebuilt.sh >/dev/null 2>&1 || {
  echo "talk: cannot materialize bin/madaros-linux-x86_64" >&2; exit 9; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-talk.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

pass=0; fail=0; total_ms=0
failed=()

now_ms() { date +%s%3N; }

# Program output only: Madaros prints its compile log up to "   Output: <elf>".
program_output() {
  if grep -q '^   Output: ' "$1"; then sed '1,/^   Output: /d' "$1"; else cat "$1"; fi
}

report() { # id title status ms detail
  local id="$1" title="$2" st="$3" ms="$4" detail="${5:-}"
  printf '%-4s %-58s %-6s %6.1fs  %s\n' "$id" "$title" "$st" "$(awk "BEGIN{print $ms/1000}")" "$detail"
  total_ms=$((total_ms + ms))
  if [[ "$st" == FAIL ]]; then fail=$((fail + 1)); failed+=("$id"); else pass=$((pass + 1)); fi
}

# souc_run <engine> <file> <out>  -> rc
souc_run() {
  SOUNIO_SOUC_ENGINE="$1" ./bin/souc run "$2" >"$3" 2>&1
}

golden_check() { # id title sentinel
  local id="$1" title="$2" sentinel="$3" g="$GOLDEN_DIR/$1.out"
  local t0; t0=$(now_ms)
  if [[ ! -f "$g" ]]; then report "$id" "$title" FAIL $(( $(now_ms) - t0 )) "golden missing: ${g#$ROOT_DIR/} (run --record)"; return; fi
  if ! grep -qF -- "$sentinel" "$g"; then report "$id" "$title" FAIL $(( $(now_ms) - t0 )) "golden lacks sentinel '$sentinel'"; return; fi
  report "$id" "$title" GOLDEN $(( $(now_ms) - t0 )) "$(sed -n 's/^# recorded: //p' "$g" | head -1)"
}

record_golden() { # id engine wall_ms outfile
  mkdir -p "$GOLDEN_DIR"
  { echo "# recorded: $(date -u +%F) at $(git rev-parse --short HEAD 2>/dev/null || echo unknown) on $2, wall $(awk "BEGIN{printf \"%.1f\", $3/1000}") s"
    echo "# regenerate: bash scripts/talk.sh --record"
    cat "$4"; } > "$GOLDEN_DIR/$1.out"
}

# demo <id> <title> <engine> <budget_s> <golden_only 0|1> <file> <sentinel>
# Runs a program and requires the sentinel in its program output.
demo() {
  selected "$1" || return 0
  local id="$1" title="$2" eng="$3" budget="$4" gonly="$5" file="$6" sentinel="$7"
  [[ "$mode" == list ]] && { printf '%-4s %-58s %-11s ~%ss%s\n' "$id" "$title" "$eng" "$budget" "$([[ $gonly == 1 ]] && echo ' (golden)')"; return; }
  if [[ "$mode" == fast && "$budget" -gt "$FAST_LIMIT_S" ]] || [[ "$mode" != record && "$gonly" == 1 ]]; then
    golden_check "$id" "$title" "$sentinel"; return
  fi
  [[ -f "$file" ]] || { report "$id" "$title" FAIL 0 "missing $file"; return; }
  local raw="$WORK/$id.raw" out="$WORK/$id.out" t0 rc ms
  t0=$(now_ms); souc_run "$eng" "$file" "$raw"; rc=$?; ms=$(( $(now_ms) - t0 ))
  program_output "$raw" > "$out"
  if [[ $rc -eq 0 ]] && grep -qF -- "$sentinel" "$out"; then
    if [[ "$mode" == record ]]; then
      record_golden "$id" "$eng" "$ms" "$out"
    elif [[ -f "$GOLDEN_DIR/$id.out" ]] && ! diff -q <(sed '1,2d' "$GOLDEN_DIR/$id.out") "$out" >/dev/null; then
      # Every demo is seeded: a printed number that moved is a regression
      # until someone re-records the golden on purpose.
      report "$id" "$title" FAIL "$ms" "$eng output differs from golden:"
      diff <(sed '1,2d' "$GOLDEN_DIR/$id.out") "$out" | head -n 6 | sed 's/^/       | /'
      return
    fi
    report "$id" "$title" PASS "$ms" "$eng"
  else
    report "$id" "$title" FAIL "$ms" "$eng rc=$rc, sentinel '$sentinel' $(grep -qF -- "$sentinel" "$out" && echo found || echo absent)"
    tail -n 5 "$raw" | sed 's/^/       | /'
  fi
}

# refuse <id> <title> <engine> <file> <pattern>  — must FAIL to compile with pattern.
refuse() {
  selected "$1" || return 0
  local id="$1" title="$2" eng="$3" file="$4" pat="$5"
  [[ "$mode" == list ]] && { printf '%-4s %-58s %-11s compile-fail\n' "$id" "$title" "$eng"; return; }
  [[ -f "$file" ]] || { report "$id" "$title" FAIL 0 "missing $file"; return; }
  local raw="$WORK/$id.raw" t0 rc ms
  t0=$(now_ms); SOUNIO_SOUC_ENGINE="$eng" ./bin/souc check "$file" >"$raw" 2>&1; rc=$?; ms=$(( $(now_ms) - t0 ))
  if [[ $rc -ne 0 ]] && grep -qE -- "$pat" "$raw"; then
    report "$id" "$title" PASS "$ms" "$eng refused: $(grep -oE -m1 -- "$pat[^=]{0,40}" "$raw")"
  else
    report "$id" "$title" FAIL "$ms" "$eng rc=$rc, expected refusal matching '$pat'"
  fi
}

# --- the C++23 cross-check of the MH7 ceiling ---------------------------------
mh7_crosscheck() {
  selected 3x || return 0
  local id=3x title="MH7 ceiling: independent C++23 cross-check"
  [[ "$mode" == list ]] && { printf '%-4s %-58s %-11s ~3s\n' "$id" "$title" "g++ -std=c++23"; return; }
  local t0 ms; t0=$(now_ms)
  if ! command -v g++ >/dev/null; then report "$id" "$title" FAIL 0 "g++ not found"; return; fi
  if ! g++ -std=c++23 -O2 -o "$WORK/mh7x" demos/hydrogen/tools/mh7_ceiling_crosscheck.cpp 2>"$WORK/mh7x.err"; then
    report "$id" "$title" FAIL $(( $(now_ms) - t0 )) "C++ build failed"; return; fi
  "$WORK/mh7x" > "$WORK/mh7x.out"
  souc_run madaros demos/hydrogen/mh7_coupled_ceiling.sio "$WORK/mh7.raw"
  program_output "$WORK/mh7.raw" > "$WORK/mh7.out"
  ms=$(( $(now_ms) - t0 ))
  # Sounio prints the ceiling to 2 decimals, C++ to 4: agreement is judged at
  # Sounio's printed precision (|diff| <= 0.005 bar), the most the output can show.
  local verdict
  verdict=$(awk '
    FNR==NR { if ($1=="case" && $3=="ceiling") cpp[$2]=$4; next }
    /^   [0-9]+ +[0-9.]+ +[0-9.]+/ { s[$1]=$2 }
    END { n=0; worst=0; for (k in cpp) { if (!(k in s)) { print "MISSING case " k; exit } d=cpp[k]-s[k]; if (d<0) d=-d; if (d>worst) worst=d; n++ }
          if (n!=6) { print "CASES " n; exit } printf "%s %.4f", (worst<=0.0050001 ? "OK" : "DIFF"), worst }' \
    "$WORK/mh7x.out" "$WORK/mh7.out")
  if [[ "$verdict" == OK* ]] && grep -q 'infeasible couplings 0' "$WORK/mh7x.out"; then
    report "$id" "$title" PASS "$ms" "6/6 ceilings agree at printed precision (max |diff| ${verdict#OK } bar)"
  else
    report "$id" "$title" FAIL "$ms" "$verdict"
  fi
}

# --- PBPK28 parity gate --------------------------------------------------------
pbpk_parity() {
  selected 9 || return 0
  local id=9 title="PBPK28: Sounio vs Node, five drugs, RMSE < 1 %" budget=90
  [[ "$mode" == list ]] && { printf '%-4s %-58s %-11s ~%ss\n' "$id" "$title" "gate" "$budget"; return; }
  if [[ "$mode" == fast ]]; then golden_check "$id" "$title" "_PASS"; return; fi
  local t0 rc ms n; t0=$(now_ms)
  # The gate rewrites two committed CSVs (P1.9); keep the tree clean.
  bash scripts/ci/dissertation_pbpk28_parity_gate.sh > "$WORK/pbpk.out" 2>&1; rc=$?
  git checkout -q -- benchmarks/pbpk/qss_residual.csv benchmarks/pbpk/model_form_uc.csv 2>/dev/null
  ms=$(( $(now_ms) - t0 )); n=$(grep -c '_PASS' "$WORK/pbpk.out")
  if [[ $rc -eq 0 && $n -ge 22 ]]; then
    [[ "$mode" == record ]] && record_golden "$id" gate "$ms" "$WORK/pbpk.out"
    report "$id" "$title" PASS "$ms" "$n _PASS verdicts"
  else
    report "$id" "$title" FAIL "$ms" "rc=$rc, $n _PASS verdicts (need 22)"
  fi
}

# --- compiler fixed point ------------------------------------------------------
fixed_point() {
  selected 11 || return 0
  local id=11 title="Self-hosted compiler reaches its fixed point (make build)" budget=40
  [[ "$mode" == list ]] && { printf '%-4s %-58s %-11s ~%ss\n' "$id" "$title" "make" "$budget"; return; }
  if [[ "$mode" == fast ]]; then golden_check "$id" "$title" "FIXED POINT OK"; return; fi
  local t0 rc ms; t0=$(now_ms)
  make build > "$WORK/fp.out" 2>&1; rc=$?; ms=$(( $(now_ms) - t0 ))
  if [[ $rc -eq 0 ]] && grep -q 'FIXED POINT OK' "$WORK/fp.out"; then
    [[ "$mode" == record ]] && record_golden "$id" make "$ms" <(grep -E 'FIXED POINT|gen[0-9]|sha' "$WORK/fp.out")
    report "$id" "$title" PASS "$ms" "$(grep -o 'FIXED POINT OK[^)]*)\?' "$WORK/fp.out" | head -1)"
  else
    report "$id" "$title" FAIL "$ms" "rc=$rc"
  fi
}

[[ "$mode" != list ]] && printf 'Sounio — NCSR Demokritos run-of-show (%s mode) at %s\n\n' "$mode" "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# 1. Uncertainty propagates (ISO GUM, first order)
demo 1  "Uncertainty propagates through arithmetic (GUM)" madaros 5 0 \
  demo_incerteza.sio "Área        = 50.000000 ± 0.707107"
# 2. No-cloning as a type error, with its positive control
demo   2a "No-cloning: a linear qubit used once compiles and runs" madaros 5 0 \
  demos/quantum/no_cloning_single_use.sio "1"
refuse 2b "No-cloning: using the same qubit twice is refused" madaros \
  demos/quantum/no_cloning_double_use.sio "E039"
refuse 2c "No-deleting: a qubit never consumed is refused" madaros \
  demos/quantum/no_cloning_dropped.sio "E040"
# 3. The centrepiece and its independent cross-check
demo 3  "MH7 seven-stage compressor: thermodynamic ceiling" madaros 5 0 \
  demos/hydrogen/mh7_coupled_ceiling.sio "MH7_COUPLED_CEILING_OK"
mh7_crosscheck
# 4. Batch tolerance per stage, on both engines
demo 4a "MHHC batch margins per stage (Madaros)" madaros 10 0 \
  examples/hydrogen/mhhc_batch_margins.sio "MHHC_BATCH_MARGINS_OK"
demo 4b "MHHC batch margins per stage (lean_single)" lean_single 10 0 \
  examples/hydrogen/mhhc_batch_margins.sio "MHHC_BATCH_MARGINS_OK"
# 5. Probability boxes through a chain
demo 5a "Trieres chain: p-boxes through a process chain" madaros 12 0 \
  demos/hydrogen/trieres_chain.sio "TRIERES_CHAIN_OK"
demo 5b "Valley chain: epistemic p-boxes" madaros 15 0 \
  demos/hydrogen/valley_chain_epistemic.sio "VALLEY_CHAIN_OK"
# 6. Underground hydrogen storage geochemistry (lean_single only today, P0.4)
demo 6  "UHS brine-calcite equilibrium (lean_single; Madaros: arena full)" lean_single 60 0 \
  demos/hydrogen/uhs_brine_calcite.sio "UHS_BRINE_CALCITE_OK"
# 7. H2 ignition with uncertainty quantification
# No in-program assertion: its Cantera parity lives in benchmarks/chemistry; here
# the full run must reproduce the recorded golden byte for byte.
# Madaros ~220 s; lean_single ~1700 s (measured 2026-10-06): shown from golden.
demo 7  "H2 ignition delay with uncertainty quantification" madaros 220 1 \
  examples/chemistry/h2_ignition_uq_demo.sio "DEMO DONE"
# 8. Ontologies: EL+ reasoning at scale
demo 8a "SNOMED-style EL+ classification" madaros 5 0 \
  examples/ontology/biomedical/snomed_elplus_demo.sio "ALL PASS"
demo 8b "Traceability through EL+ provenance" madaros 5 0 \
  examples/epistemic/traceability_elplus_demo.sio "ALL PASS"
demo 8c "Full Gene Ontology EL+ closure (38,245 classes)" madaros 10 0 \
  artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio "ALL PASS"
demo 8d "OAEI Anatomy alignment repair (mouse <-> human)" madaros 30 0 \
  artifacts/ontology-frontiers/real-data/real_repair_driver.sio "ALL PASS"
demo 8e "Cell Ontology EL+ closure" madaros 330 1 \
  artifacts/ontology-frontiers/multi-ontology/obo_elplus_driver.sio "ALL PASS"
demo 8f "UBERON EL+ closure" madaros 70 1 \
  artifacts/ontology-frontiers/multi-ontology/uberon_open_elplus_driver.sio "ALL PASS"
demo 8g "Gene Ontology roots EL+ closure" madaros 160 1 \
  artifacts/ontology-frontiers/multi-ontology/go_roots_elplus_driver.sio "ALL PASS"
# 9. PBPK parity against an independent implementation
pbpk_parity
# 10. Quantum: two-qubit H2 VQE with an exact-diagonalisation oracle (P0.3)
demo 10 "Two-qubit H2 VQE vs exact diagonalisation" madaros 15 0 \
  demos/quantum/h2_vqe_2q.sio "H2_VQE_2Q_OK"
# 11. The compiler compiles itself to a fixed point
fixed_point

[[ "$mode" == list ]] && exit 0
printf '\n%d passed, %d failed, %.1f s total' "$pass" "$fail" "$(awk "BEGIN{print $total_ms/1000}")"
if [[ $fail -gt 0 ]]; then printf '  — FAILED: %s\n' "${failed[*]}"; exit 1; fi
printf '\n'
