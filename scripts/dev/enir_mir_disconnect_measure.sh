#!/usr/bin/env bash
# ENIR/MIR/MLI disconnect cost measurement — Slurm only.
# Writes refutation criteria FIRST, then measures.
set -euo pipefail

ROOT="${SOUNIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${DENOM_OUT:-$ROOT/docs/audit/enir_mir_disconnect}"
mkdir -p "$OUT"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
ulimit -s 1048576 2>/dev/null || true

SHA="${DENOM_SHA:-unknown}"
HOST="$(hostname)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# ---------------------------------------------------------------------------
# REFUTATION CRITERIA (written BEFORE any measurement number is trusted)
# ---------------------------------------------------------------------------
cat > "$OUT/REFUTATION_CRITERIA.txt" <<'RC'
R1 RECOVERABILITY of enir/ as a live pipeline layer
   Falsify "enir is recoverable by rewiring only" if ANY of:
   (a) majority of enir/*.sio fail `souc check` with structural errors (not
       missing-import-from-unbuilt-deps), OR
   (b) zero external `use enir::` AND no historical external caller ever, AND
       no driver entrypoint that can be compiled standalone, OR
   (c) EMIR/ENIR capacities cannot express a real main.sio-scale function
       without redesign (cap is fixture-scale AND no multi-function path).
   If (a) and (c) both hold → NOT recoverable without rewrite (cost = rewrite).
   If only (b) → disconnected but source may still check → rewiring cost TBD.

R2 RECOVERABILITY of mli/ as machine IR for production
   Falsify "mli is the next native IR" if:
   (a) mli does not consume production ir:: (no ir_to_mli bridge), OR
   (b) MLI_MAX_INSTRS per block is below the smallest live machine_ir body
       without a documented multi-block strategy that already works, OR
   (c) native/machine_ir.sio already owns the production path and mli has
       zero external importers and no call from compiler/main.sio.

R3 CONCURRENT vs SEQUENTIAL designs (enir MIR vs mli vs machine_ir)
   Decide by consume→produce edges, not names.
   Falsify "same layer, two names" if consume/produce sets are disjoint
   (different input IR families AND different output consumers).
   Falsify "sequential pipeline stages" if neither feeds the other and both
   claim the same slot (e.g. both claim "after ir, before native").

R4 "Was enir ever executable?"
   Falsify "never ran" if git history or gate scripts show driver.sio compiled
   and invoked (emit/verify/run). Falsify "production executable" if the only
   callers are shadow gates outside the default compiler path.

R5 INSTRUMENT validity
   Import counts using `use <dir>::` must match positive control:
   parser external ≥ 80, ir external ≥ 40 (founder band). If instrument shows
   enir external > 0 on this tree, re-check path filter (internal vs external).
   Bad control: grep use|import|from|mod must NOT be used (hits "model").

R6 AGGREGATE FORBIDDEN
   souc check must be per-file rc. A single green/red aggregate is RULER FAIL.
RC

# ---------------------------------------------------------------------------
# Positive controls for instruments
# ---------------------------------------------------------------------------
SOUC=""
for cand in "$ROOT/bin/souc" "$ROOT/bin/madaros" \
            "${MADAROS_RAW_BIN:-}" "$ROOT/bin/madaros-linux-x86_64" \
            "$ROOT/artifacts/self-hosted/madaros"; do
  if [[ -n "$cand" && -x "$cand" ]]; then SOUC="$cand"; break; fi
done
# Prefer wrapper if present
if [[ -x "$ROOT/bin/souc" ]]; then SOUC="$ROOT/bin/souc"; fi

{
  echo "sha=$SHA"
  echo "host=$HOST"
  echo "ts_utc=$TS"
  echo "souc=$SOUC"
  echo "root=$ROOT"
} > "$OUT/MEASUREMENT_RECEIPT.txt"

# Import instrument + positive control
imp_tsv="$OUT/import_external.tsv"
echo -e "dir\texternal\tinternal\ttotal\tpositive_control_note" > "$imp_tsv"
for d in parser check ir native wasm hlir gpu enir mli llvm vm effects; do
  mapfile -t files < <(grep -RIl --include="*.sio" "use ${d}::" self-hosted 2>/dev/null || true)
  ext=0; int=0
  for f in "${files[@]:-}"; do
    [[ -z "${f:-}" ]] && continue
    case "$f" in
      self-hosted/${d}/*) int=$((int+1)) ;;
      *) ext=$((ext+1)) ;;
    esac
  done
  note=""
  case "$d" in
    parser) note="POS_CTRL expected external~93" ;;
    ir) note="POS_CTRL expected external~48" ;;
    enir|mli) note="expect external=0 if disconnected" ;;
  esac
  echo -e "${d}\t${ext}\t${int}\t$((ext+int))\t${note}" >> "$imp_tsv"
done

parser_ext=$(awk -F'\t' '$1=="parser"{print $2}' "$imp_tsv")
ir_ext=$(awk -F'\t' '$1=="ir"{print $2}' "$imp_tsv")
echo "import_pos_ctrl parser_external=$parser_ext ir_external=$ir_ext" >> "$OUT/MEASUREMENT_RECEIPT.txt"
if [[ "${parser_ext:-0}" -lt 80 || "${ir_ext:-0}" -lt 40 ]]; then
  echo "RULER_SUSPECT import_instrument parser_ext=$parser_ext ir_ext=$ir_ext (below founder band)" >> "$OUT/MEASUREMENT_RECEIPT.txt"
else
  echo "import_instrument_OK parser_ext=$parser_ext ir_ext=$ir_ext" >> "$OUT/MEASUREMENT_RECEIPT.txt"
fi

# Bad instrument trap (must be large / not used)
bad=$(grep -RIl --include="*.sio" -E "use |import |from |mod " self-hosted 2>/dev/null | wc -l || echo 0)
echo "bad_instrument_use_import_from_mod_file_hits=$bad (do not use)" >> "$OUT/MEASUREMENT_RECEIPT.txt"

# External importer lists for enir/mli
{
  echo "=== external use enir:: (expect empty) ==="
  grep -Rn --include="*.sio" "use enir::" self-hosted 2>/dev/null | grep -v 'self-hosted/enir/' || echo "(none)"
  echo "=== external use mli:: (expect empty) ==="
  grep -Rn --include="*.sio" "use mli::" self-hosted 2>/dev/null | grep -v 'self-hosted/mli/' || echo "(none)"
  echo "=== machine_ir consumers (expect non-empty) ==="
  grep -Rn --include="*.sio" "use native::machine_ir" self-hosted 2>/dev/null || true
} > "$OUT/importer_detail.txt"

# Caps
{
  echo "=== capacity constants ==="
  grep -Rn -E "EMIR_MAX_INSTRS|EMIR_MAX_VALUES|MLI_MAX_INSTRS|MLI_MAX_BLOCKS|MIR_MAX_INSTRS|IR_MAX_INSTRS|HLIR_MAX_INSTRS" \
    self-hosted/enir self-hosted/mli self-hosted/ir/ir.sio self-hosted/native/machine_ir.sio self-hosted/hlir/ir.sio 2>/dev/null || true
  echo "=== EMIR one-block comment ==="
  sed -n '1,15p' self-hosted/enir/mir.sio
  echo "=== MLI pool geometry ==="
  sed -n '40,55p' self-hosted/mli/ir.sio
  sed -n '410,420p' self-hosted/mli/ir.sio
} > "$OUT/caps_detail.txt"

# Bytes
{
  for d in enir mli hlir effects; do
    if [[ -d "self-hosted/$d" ]]; then
      n=$(find "self-hosted/$d" -name '*.sio' | wc -l)
      b=$(find "self-hosted/$d" -name '*.sio' -print0 | xargs -0 cat | wc -c)
      echo "$d files=$n bytes=$b"
    fi
  done
} >> "$OUT/MEASUREMENT_RECEIPT.txt"

# Consume/produce edges
{
  echo "ENIR_CONSUMES: only enir::* (self) — no use ir::, no use native::"
  grep -Rn "^use " self-hosted/enir/*.sio | sed 's/^/  /'
  echo "ENIR_PRODUCES: EnirModule, EnirMirModule, EnirCfgMirModule, EnirJoinMirModule; interpreter results; canonical text"
  echo "ENIR_ENTRY: enir/driver.sio CLI verbs emit|verify|roundtrip|lower|run|mir*"
  echo
  echo "MLI_CONSUMES:"
  grep -Rn "^use " self-hosted/mli/*.sio | sed 's/^/  /'
  echo "MLI_PRODUCES: MliFunction; legalize_x86 output; gate runners self-test"
  echo
  echo "MACHINE_IR_CONSUMES: ir + parser types; used by native codegen + compiler drivers"
  grep -n "^use " self-hosted/native/machine_ir.sio | head -20
} > "$OUT/consume_produce.txt"

# ---------------------------------------------------------------------------
# Per-file souc check (NO aggregate verdict as sole answer)
# ---------------------------------------------------------------------------
check_tsv="$OUT/souc_check_per_file.tsv"
echo -e "path\trc\tbytes\tseconds\tfirst_error" > "$check_tsv"

if [[ -z "$SOUC" || ! -x "$SOUC" ]]; then
  echo "RULER_FAIL missing_souc" >> "$OUT/MEASUREMENT_RECEIPT.txt"
  echo "NO_SOUC" > "$OUT/souc_check_status.txt"
else
  # Positive control: a file known to be on the live pipeline
  pos="self-hosted/parser/ast.sio"
  if [[ ! -f "$pos" ]]; then pos="self-hosted/ir/ir.sio"; fi
  t0=$(date +%s%N)
  set +e
  "$SOUC" check "$pos" >"$OUT/pos_ctrl_check.log" 2>&1
  pos_rc=$?
  set -e
  t1=$(date +%s%N)
  pos_s=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)/1e9}')
  echo "pos_ctrl_check file=$pos rc=$pos_rc seconds=$pos_s" >> "$OUT/MEASUREMENT_RECEIPT.txt"
  if [[ "$pos_rc" -ne 0 ]]; then
    echo "RULER_SUSPECT pos_ctrl_check_failed — do not trust target zeros" >> "$OUT/MEASUREMENT_RECEIPT.txt"
  else
    echo "pos_ctrl_check_OK" >> "$OUT/MEASUREMENT_RECEIPT.txt"
  fi

  while IFS= read -r -d '' f; do
    b=$(wc -c <"$f")
    t0=$(date +%s%N)
    set +e
    "$SOUC" check "$f" >"$OUT/check_$(echo "$f" | tr '/' '_').log" 2>&1
    rc=$?
    set -e
    t1=$(date +%s%N)
    sec=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)/1e9}')
    err=$(grep -oE 'error\[[^]]+\]|error:.*' "$OUT/check_$(echo "$f" | tr '/' '_').log" 2>/dev/null | head -1 | tr '\t' ' ' | cut -c1-120 || true)
    echo -e "${f}\t${rc}\t${b}\t${sec}\t${err}" >> "$check_tsv"
    echo "checked $f rc=$rc"
  done < <(find self-hosted/enir self-hosted/mli -name '*.sio' -print0 | sort -z)
fi

# Cap comparison row
{
  echo -e "symbol\tvalue\tscope_comment"
  echo -e "IR_MAX_INSTRS\t16384\tproduction ir:: IrFunction arena (#1649)"
  echo -e "HLIR_MAX_INSTRS\t16384\thlir (disconnected external=2)"
  emir=$(grep -n "pub let EMIR_MAX_INSTRS" self-hosted/enir/mir.sio | head -1)
  echo -e "EMIR_MAX_INSTRS\t128\t${emir} ; E3A one-block qd128 only (mir.sio:2)"
  mli=$(grep -n "pub let MLI_MAX_INSTRS" self-hosted/mli/ir.sio | head -1)
  echo -e "MLI_MAX_INSTRS\t32\t${mli} ; per-block body slots; MLI_BLOCK_STRIDE=33"
  mirn=$(grep -n "MIR_MAX_INSTRS" self-hosted/native/machine_ir.sio | head -1 || true)
  echo -e "native_MIR_MAX\tsee_file\t${mirn}"
} > "$OUT/caps_table.tsv"

# Shortest link to ir/
{
  echo "SHORTEST_LINK_ANALYSIS"
  echo "enir -> ir: NO use ir:: in enir/*.sio. ENIR is a parallel shadow IR"
  echo "  (EISA/qd128 epistemic numeric). source_lower lowers EISA sources to ENIR,"
  echo "  not production ir::. Shortest link today: NONE (rewire would invent the edge)."
  echo "mli -> ir: YES — self-hosted/mli/ir_to_mli.sio:52 'use ir::ir::*'"
  echo "  Function: IR arena columns -> MliFunction (S2a/S2b). This IS the shortest"
  echo "  existing link from production ir/ into mli/."
  echo "mli -> native: legalize_x86.sio exists but no external importer from native/"
  echo "  or compiler/main. Production path uses native/machine_ir.sio instead."
  echo "machine_ir -> ir: YES — live production (compiler/main, codegen, module_loader)."
} > "$OUT/shortest_link.txt"

# Summary counts from check tsv
python3 - <<'PY' "$OUT"
import sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
p = out / "souc_check_per_file.tsv"
if p.exists():
    for line in p.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append(parts)
enir = [r for r in rows if r[0].startswith("self-hosted/enir/")]
mli = [r for r in rows if r[0].startswith("self-hosted/mli/")]
def stats(rs, name):
    oks = sum(1 for r in rs if r[1] == "0")
    print(f"{name}_files={len(rs)} check_rc0={oks} check_nonzero={len(rs)-oks}")
stats(enir, "enir")
stats(mli, "mli")
# write compact
(out / "check_summary.txt").write_text(
    f"enir_files={len(enir)} enir_rc0={sum(1 for r in enir if r[1]=='0')} enir_fail={sum(1 for r in enir if r[1]!='0')}\n"
    f"mli_files={len(mli)} mli_rc0={sum(1 for r in mli if r[1]=='0')} mli_fail={sum(1 for r in mli if r[1]!='0')}\n"
)
print(open(out / "check_summary.txt").read())
PY

echo "MEASURE_DONE out=$OUT" | tee -a "$OUT/MEASUREMENT_RECEIPT.txt"
