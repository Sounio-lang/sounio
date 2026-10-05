#!/usr/bin/env bash
# Type-check refinement/Div/Panic witnesses on both engines.
# Intended to run on a Slurm node (no /workspace). Set ROOT to the extracted bundle.
set -u
ROOT="${1:-.}"
WIT="$ROOT/docs/audit/repro/refinement_div_discharge"
MAD="$ROOT/bin/madaros"
LS="$ROOT/bin/souc-lean-single-x86_64"
OUT="${2:-/tmp/refinement_div_discharge.tsv}"

ulimit -s 524288 2>/dev/null || true

codes_from() {
  local log="$1"
  local codes=""
  if grep -q 'error\[E035' "$log"; then codes="${codes}|E035"; fi
  if grep -q 'error\[E042' "$log"; then codes="${codes}|E042"; fi
  if grep -q 'error\[E056' "$log"; then codes="${codes}|E056"; fi
  if grep -q 'error\[E001' "$log"; then codes="${codes}|E001"; fi
  if grep -q 'warning\[W040' "$log" || grep -q 'W040' "$log"; then codes="${codes}|W040"; fi
  if grep -q 'error\[E008' "$log"; then codes="${codes}|E008"; fi
  if grep -qi 'division requires' "$log"; then codes="${codes}|E031msg"; fi
  if grep -qi 'not declared in function signature' "$log"; then codes="${codes}|E035msg"; fi
  if grep -qi 'refinement type violation' "$log"; then codes="${codes}|E042msg"; fi
  if [[ -z "$codes" ]]; then
    if grep -qi 'error\[' "$log"; then
      codes="|OTHER:$(grep -o 'error\[E[0-9]*' "$log" | head -1)"
    fi
  fi
  printf '%s' "${codes#|}"
}

echo -e "cell\tengine\texit\tcodes\tfirst_line" > "$OUT"

for src in "$WIT"/*.sio; do
  cell="$(basename "$src" .sio)"
  mad_log="/tmp/${cell}.madaros.log"
  ls_log="/tmp/${cell}.lean_single.log"
  "$MAD" check "$src" >"$mad_log" 2>&1
  mad_rc=$?
  mad_codes="$(codes_from "$mad_log")"
  mad_first="$(tr '\n' ' ' <"$mad_log" | sed 's/Madaros v0.80.0[^c]*compiler[^c]*//' | head -c 280)"
  echo -e "${cell}\tmadaros\t${mad_rc}\t${mad_codes}\t${mad_first}" >> "$OUT"

  ls_tmp="/tmp/${cell}.lean.elf"
  "$LS" "$src" "$ls_tmp" >"$ls_log" 2>&1
  ls_rc=$?
  rm -f "$ls_tmp"
  ls_codes="$(codes_from "$ls_log")"
  ls_first="$(tr '\n' ' ' <"$ls_log" | head -c 200)"
  echo -e "${cell}\tlean_single\t${ls_rc}\t${ls_codes}\t${ls_first}" >> "$OUT"
done

echo "__TSV_BEGIN__"
cat "$OUT"
echo "__TSV_END__"
echo "host=$(hostname) date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
