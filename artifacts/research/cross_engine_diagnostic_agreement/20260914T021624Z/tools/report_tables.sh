#!/usr/bin/env bash
# Usage: report_tables.sh AGREEMENT_TSV CLASS_TSV OUT_DIR
# Writes: OUT_DIR/tables.md (markdown), OUT_DIR/summary.txt, OUT_DIR/metrics.json, OUT_DIR/classified.tsv
A="$1"; C="$2"; OUTD="$3"; mkdir -p "$OUTD"
n=$(($(wc -l < "$A") - 1))
{
  echo "### Verdicts over $n files"; echo; echo "| Verdict | Files |"; echo "|---|---:|"
  awk -F'\t' 'NR>1{c[$4]++} END{for(k in c) printf "| %s | %d |\n", k, c[k]}' "$A" | sort -t'|' -k3 -rn
  echo; echo "### Timeouts by exit codes (lean_rc / madaros_rc; 124 = timed out)"; echo; echo "| lean_rc | madaros_rc | Files |"; echo "|---:|---:|---:|"
  awk -F'\t' '$4=="TIMEOUT"{c[$2"|"$3]++} END{for(k in c){split(k,p,"|"); printf "| %s | %s | %d |\n", p[1], p[2], c[k]}}' "$A" | sort -t'|' -k4 -rn
  echo; echo "### Disagreements by class"; echo; echo "| Class | LEAN_ONLY | MADAROS_ONLY | Total |"; echo "|---|---:|---:|---:|"
  awk -F'\t' '{t[$4]++; if($2=="LEAN_ONLY") l[$4]++; else m[$4]++} END{for(k in t) printf "| %s | %d | %d | %d |\n", k, l[k], m[k], t[k]}' "$C" | sort -t'|' -k5 -rn
  echo; echo "### Groups"; echo; echo "| Verdict | Class | Group | Files | Example |"; echo "|---|---|---|---:|---|"
  awk -F'\t' '{k=$2"\t"$4"\t"$5; c[k]++; if(!(k in ex)) ex[k]=$1} END{for(k in c){split(k,p,"\t"); printf "| %s | %s | %s | %d | `%s` |\n", p[1], p[2], p[3], c[k], ex[k]}}' "$C" | sort -t'|' -k5 -rn
} > "$OUTD/tables.md"
{
  echo "files=$n"
  awk -F'\t' 'NR>1{c[$4]++} END{for(k in c) printf "verdict.%s=%d\n", k, c[k]}' "$A" | sort
  awk -F'\t' '{c[$4]++} END{for(k in c) printf "class.%s=%d\n", k, c[k]}' "$C" | sort
} > "$OUTD/summary.txt"
{
  printf '{\n  "files": %d,\n  "verdicts": {' "$n"
  awk -F'\t' 'NR>1{c[$4]++} END{s=""; for(k in c){printf "%s\"%s\": %d", s, k, c[k]; s=", "}}' "$A"
  printf '},\n  "classes": {'
  awk -F'\t' '{c[$4]++} END{s=""; for(k in c){printf "%s\"%s\": %d", s, k, c[k]; s=", "}}' "$C"
  printf '}\n}\n'
} > "$OUTD/metrics.json"
cut -f1,2,4,5 "$C" > "$OUTD/classified.tsv"
