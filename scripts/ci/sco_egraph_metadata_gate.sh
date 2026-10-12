#!/usr/bin/env bash
# Execute compiler-IR metadata regressions, not scientific non-vacuity.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ci/sco_stack_contract.sh"
MADAROS="${SCO_MADAROS:-$ROOT/artifacts/self-hosted/madaros-sco}"
OUT="${SCO_OUTPUT_DIR:-$ROOT/artifacts/sco}/egraph-metadata"
[[ -x "$MADAROS" ]] || { echo 'FAIL: missing source-built compiler'; exit 1; }
source_digest() {
 find self-hosted stdlib -type f -name '*.sio' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}
receipt="$MADAROS.sco-source.sha256"
[[ -f "$receipt" ]] || { echo 'FAIL: missing source/ELF-bound SCO receipt'; exit 1; }
grep -qx "source_sha256=$(source_digest)" "$receipt" || { echo 'FAIL: source mismatch'; exit 1; }
grep -qx "elf_sha256=$(sha256sum "$MADAROS" | cut -d' ' -f1)" "$receipt" || { echo 'FAIL: compiler mismatch'; exit 1; }
# Import compiler IR from a private probe directory without mutating source tree.
# This compiler-library fixture belongs to its dedicated executed CI gate; it
# is outside the ordinary language-test corpus and uses no skip annotation.
mkdir -p "$OUT"
probe=$(mktemp -d "$OUT/probe.XXXXXX")
ln -s "$ROOT/self-hosted/ir" "$probe/ir"
cp scripts/ci/fixtures/sco_epistemic_metadata.sio "$probe/main.sio"
printf 'FIRST_VALID_LARGE_UNC_PASS\nMISSING_UNC_REFUSED_PASS\nREASSOC_UNC_PROPAGATED_PASS\nSUM_OVERFLOW_REFUSED_PASS\nEXACT_REASSOCIATED_SHAPE_UNC_PASS\nFSUB_REASSOCIATION_REFUSED_PASS\nNONADDITIVE_RSS_REFUSED_PASS\n' > "$OUT/expected.stdout"
printf 'mode\tcompile_rc\trun_rc\tstdout_cmp_rc\n' > "$OUT/status.tsv"
for mode in default optimized; do
 flags=(); [[ "$mode" == default ]] || flags=(-O)
 compile_rc=0
 timeout 180 "$MADAROS" compile "$probe/main.sio" "${flags[@]}" -o "$probe/$mode.elf" > "$OUT/$mode.compile.log" 2>&1 || compile_rc=$?
 run_rc=not_run; cmp_rc=not_run
 if [[ "$compile_rc" == 0 && -s "$probe/$mode.elf" ]]; then
  chmod +x "$probe/$mode.elf"
  run_rc=0
  timeout 30 "$probe/$mode.elf" > "$OUT/$mode.stdout" 2> "$OUT/$mode.stderr" || run_rc=$?
  cmp_rc=0
  cmp "$OUT/expected.stdout" "$OUT/$mode.stdout" > "$OUT/$mode.cmp.log" 2>&1 || cmp_rc=$?
 fi
 printf '%s\t%s\t%s\t%s\n' "$mode" "$compile_rc" "$run_rc" "$cmp_rc" >> "$OUT/status.tsv"
 [[ "$compile_rc" == 0 && "$run_rc" == 0 && "$cmp_rc" == 0 ]] || { echo "FAIL: metadata regression $mode"; exit 1; }
done
echo 'SCO_EGRAPH_METADATA_GATE_OK'
