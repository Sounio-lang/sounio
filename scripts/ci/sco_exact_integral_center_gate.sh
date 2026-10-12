#!/usr/bin/env bash
# Explicit research contract, bounded integer nominal centers; no generic RSS accuracy theorem.
# Default IEEE source order remains selected unless the research option is exactly1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ci/sco_stack_contract.sh"
MADAROS="${SCO_MADAROS:-$ROOT/artifacts/self-hosted/madaros-sco}"
OUT="${SCO_OUTPUT_DIR:-$ROOT/artifacts/sco}/exact-integral-center"
FIX="$ROOT/scripts/ci/fixtures/sco_exact_integral_center"
[[ -x "$MADAROS" ]] || { echo 'FAIL: missing source-built compiler'; exit 1; }
source_digest() {
 find self-hosted stdlib -type f -name '*.sio' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}
source_before="$(source_digest)"
elf_before="$(sha256sum "$MADAROS" | cut -d' ' -f1)"
receipt="$MADAROS.sco-source.sha256"
[[ -f "$receipt" ]] || { echo 'FAIL: missing source/ELF-bound receipt'; exit 1; }
grep -qx "source_sha256=$source_before" "$receipt" || { echo 'FAIL: source mismatch'; exit 1; }
grep -qx "elf_sha256=$elf_before" "$receipt" || { echo 'FAIL: compiler mismatch'; exit 1; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
MADAROS="$(cd "$(dirname "$MADAROS")" && pwd)/$(basename "$MADAROS")"
export TMPDIR="$OUT" SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_TARGET_OVERRIDE SOUNIO_MATRIX_OVERRIDE SOUNIO_PRIVATE_MFI_GUARD_CONTROLS SOUNIO_SCO_ABLATION SOUNIO_RESEARCH_EXACT_INTEGRAL_CENTER
find "$FIX" -type f -print0 | sort -z | xargs -0 sha256sum >"$OUT/fixtures.sha256"
printf 'source_sha256=%s\nelf_sha256=%s\nfixture_manifest_sha256=%s\n' "$source_before" "$elf_before" "$(sha256sum "$OUT/fixtures.sha256" | cut -d' ' -f1)" >"$OUT/receipt.txt"
CXX="${CXX:-g++}"
"$CXX" --version >"$OUT/cpp-compiler.txt"
"$CXX" -std=c++23 -O2 -fno-fast-math -ffp-contract=off "$FIX/center_oracle.cpp" -o "$OUT/center-oracle" >"$OUT/center-oracle-build.log" 2>&1
timeout 30 "$OUT/center-oracle" >"$OUT/center-oracle.stdout"
"$CXX" -std=c++23 -O2 -fno-fast-math -ffp-contract=off "$FIX/decoder_oracle.cpp" -o "$OUT/decoder-oracle" >"$OUT/decoder-oracle-build.log" 2>&1
timeout 30 "$OUT/decoder-oracle" >"$OUT/decoder-oracle.stdout"
probe=$(mktemp -d "$OUT/decoder-probe.XXXXXX")
timeout 30 "$OUT/decoder-oracle" --sounio-fixture >"$probe/main.sio"
cp -a "$ROOT/self-hosted/ir" "$probe/ir"
ln -s "$ROOT/self-hosted/parser" "$probe/parser"
cat "$FIX/decoder_export.sio.txt" >>"$probe/ir/opt_cleanup.sio"
printf 'mode\tcompile_rc\trun_rc\tstdout_cmp_rc\n' >"$OUT/decoder.tsv"
printf 'BOUNDED_DECODER_SOUNIO_22_VECTORS_PASS\n' >"$OUT/decoder.expected"
for mode in default optimized; do
 flags=(); [[ $mode == default ]] || flags=(-O)
 rc=0; rr=not_run; cr=not_run
 timeout 600 env SOUNIO_RESEARCH_EXACT_INTEGRAL_CENTER=0 "$MADAROS" compile "$probe/main.sio" "${flags[@]}" -o "$probe/$mode.elf" >"$OUT/decoder-$mode.compile.log" 2>&1 || rc=$?
 if [[ $rc == 0 && -s "$probe/$mode.elf" ]]; then
  chmod +x "$probe/$mode.elf"; rr=0
  timeout 30 "$probe/$mode.elf" >"$OUT/decoder-$mode.stdout" 2>"$OUT/decoder-$mode.stderr" || rr=$?
  cr=0; cmp "$OUT/decoder.expected" "$OUT/decoder-$mode.stdout" >"$OUT/decoder-$mode.cmp.log" 2>&1 || cr=$?
 fi
 printf '%s\t%s\t%s\t%s\n' "$mode" "$rc" "$rr" "$cr" >>"$OUT/decoder.tsv"
 [[ $rc == 0 && $rr == 0 && $cr == 0 ]] || { echo "FAIL: actual Sounio decoder $mode"; exit 1; }
done
mkdir -p "$OUT/matrix"
printf 'case\tcompile_rc\trun_rc\tapplication\n' >"$OUT/matrix.tsv"
for case in default off enforce ablation ordinary fractional nan alias multiuse two_unc exported pubfn global; do
 flags=(-O); policy=1; ablation=0; expected=0
 [[ $case != default ]] || { flags=(); policy=0; }
 [[ $case != off ]] || policy=0
 [[ $case != ablation && $case != ordinary ]] || ablation=1
 [[ $case != enforce ]] || expected=1
 rc=0; rr=not_run; actual=NOT_APPLIED
 timeout 180 env SOUNIO_RESEARCH_EXACT_INTEGRAL_CENTER="$policy" SOUNIO_SCO_ABLATION="$ablation" SOUNIO_SCO_STRATEGY_TRACE=1 "$MADAROS" compile "$FIX/$case.sio" "${flags[@]}" -o "$OUT/matrix/$case.elf" >"$OUT/matrix/$case.compile.log" 2>&1 || rc=$?
 if [[ $rc == 0 && -s "$OUT/matrix/$case.elf" ]]; then
  chmod +x "$OUT/matrix/$case.elf"; rr=0
  timeout 30 "$OUT/matrix/$case.elf" >"$OUT/matrix/$case.stdout" 2>"$OUT/matrix/$case.stderr" || rr=$?
  if grep -qx 'RESEARCH_EXACT_CENTER applications=1 last_function=exact_integral_center' "$OUT/matrix/$case.compile.log"; then actual=APPLIED; fi
 fi
 printf '%s\t%s\t%s\t%s\n' "$case" "$rc" "$rr" "$actual" >>"$OUT/matrix.tsv"
 [[ $rc == 0 && $rr == 0 ]] || { echo "FAIL: runtime $case"; exit 1; }
 if [[ $expected == 1 ]]; then [[ $actual == APPLIED ]] || exit 1; else [[ $actual == NOT_APPLIED ]] || exit 1; fi
 if [[ $case == default || $case == off || $case == ablation || $case == enforce || $case == exported || $case == pubfn ]]; then
  value=0; [[ $case != enforce ]] || value=1
  printf '%s.000000\n' "$value" >"$OUT/matrix/$case.expected"
  cmp "$OUT/matrix/$case.expected" "$OUT/matrix/$case.stdout" >"$OUT/matrix/$case.cmp.log" 2>&1
 fi
 if [[ $case == exported || $case == pubfn ]]; then grep -qx 'SCO_STRATEGY function=exact_integral_center strategy=7' "$OUT/matrix/$case.compile.log"; fi
done
printf '8\n' > "$OUT/matrix/global.expected"
cmp "$OUT/matrix/global.expected" "$OUT/matrix/global.stdout"
# Same-source existing controls and retained authoritative boundaries.
printf 'case\tcompile_rc\trun_rc\n' > "$OUT/boundary-controls.tsv"
for n in global_scalar_init_still_works global_array_repeat_init retained; do
 input="$ROOT/tests/run-pass/$n.sio"
 [[ $n != retained ]] || input="$FIX/retained.sio"
 cr=0; rr=not_run
 timeout 180 env SOUNIO_RESEARCH_EXACT_INTEGRAL_CENTER=1 SOUNIO_SCO_STRATEGY_TRACE=1 "$MADAROS" compile "$input" -O -o "$OUT/matrix/$n.elf" >"$OUT/matrix/$n.compile.log" 2>&1 || cr=$?
 if [[ $cr == 0 && -s "$OUT/matrix/$n.elf" ]]; then chmod +x "$OUT/matrix/$n.elf"; rr=0; timeout 30 "$OUT/matrix/$n.elf" >"$OUT/matrix/$n.stdout" 2>"$OUT/matrix/$n.stderr" || rr=$?; fi
 printf '%s\t%s\t%s\n' "$n" "$cr" "$rr" >> "$OUT/boundary-controls.tsv"
 [[ $cr == 0 && $rr == 0 ]] || failed=1
 if [[ $n == retained ]]; then
  for pair in 'private_boundary 2' 'public_boundary 7' 'kernel_priority 6'; do read -r fn st <<<"$pair"; grep -qx "SCO_STRATEGY function=$fn strategy=$st" "$OUT/matrix/$n.compile.log" || failed=1; done
  printf '1.000000\n0.000000\n' > "$OUT/matrix/$n.expected"
 elif [[ $n == global_scalar_init_still_works ]]; then printf '42\nSCALAR_OK\n' > "$OUT/matrix/$n.expected"
 else printf '42 42 sum=84\n7 9\nCT_OK\n' > "$OUT/matrix/$n.expected"; fi
 cmp "$OUT/matrix/$n.expected" "$OUT/matrix/$n.stdout" > "$OUT/matrix/$n.cmp.log" 2>&1 || failed=1
done
# Bounded entitlement witness: explicit research center agrees with independent
# exact-integer C++23 reference; strategy ablation restores source-order center0.
grep -qx 'fixed_exact_center=1' "$OUT/center-oracle.stdout"
grep -qx 'fixed_source_left=0' "$OUT/center-oracle.stdout"
grep -qx 'fixed_alternative_right=1' "$OUT/center-oracle.stdout"
printf '1.000000\n' > "$OUT/entitled.expected"
printf '0.000000\n' > "$OUT/ablated.expected"
cmp "$OUT/entitled.expected" "$OUT/matrix/enforce.stdout"
cmp "$OUT/ablated.expected" "$OUT/matrix/ablation.stdout"
if cmp -s "$OUT/matrix/enforce.stdout" "$OUT/matrix/ablation.stdout"; then echo 'FAIL: vacuous exact-center entitlement witness'; exit 1; fi
printf 'explicit_exact_integral_center\tenforce_entitled=1\tablated_entitled=0\n' > "$OUT/nonvacuity.tsv"
[[ ${failed:-0} == 0 ]] || { echo 'FAIL: durable boundary controls'; exit 1; }

# Compare unset policy with explicit OFF in both codegen modes; same current ELF.
for mode in default optimized; do
 flags=(); [[ $mode == default ]] || flags=(-O)
 for policy in unset off; do
  env_args=(-u SOUNIO_RESEARCH_EXACT_INTEGRAL_CENTER)
  [[ $policy == unset ]] || env_args=(SOUNIO_RESEARCH_EXACT_INTEGRAL_CENTER=0)
  timeout 180 env "${env_args[@]}" SOUNIO_SCO_ABLATION=0 "$MADAROS" compile "$FIX/default.sio" "${flags[@]}" -o "$OUT/$mode-$policy.elf" >"$OUT/$mode-$policy.compile.log" 2>&1
  chmod +x "$OUT/$mode-$policy.elf"; timeout 30 "$OUT/$mode-$policy.elf" >"$OUT/$mode-$policy.stdout"
 done
 cmp "$OUT/$mode-unset.elf" "$OUT/$mode-off.elf" >"$OUT/$mode-policy.elf-cmp.log" 2>&1
 cmp "$OUT/$mode-unset.stdout" "$OUT/$mode-off.stdout" >"$OUT/$mode-policy.stdout-cmp.log" 2>&1
 printf '0.000000\n' >"$OUT/off.expected"; cmp "$OUT/off.expected" "$OUT/$mode-off.stdout"
done
# Build a distinct private copy for guard mutations. The real pass/producer stay
# byte-identical; only appended test bridge and private frontend hook differ.
private=$(mktemp -d "$OUT/private-source.XXXXXX")
cp -a "$ROOT/self-hosted" "$ROOT/stdlib" "$ROOT/scripts" "$ROOT/bin" "$private/"
node "$FIX/instrument_private_controls.cjs" "$private" "$FIX/real_controls.sio.txt"
private_digest=$(cd "$private"; source_digest)
printf 'private_source_sha256=%s\n' "$private_digest" >>"$OUT/receipt.txt"
SOUNIO_STDLIB_PATH="$private/stdlib" SOUNIO_BUILD_LOCK="$OUT/private-build.lock" timeout 2400 bash "$private/scripts/ci/build_modular_madaros.sh" "$OUT/private-madaros" >"$OUT/private-build.log" 2>&1
[[ -x "$OUT/private-madaros" ]] || exit 1
[[ $(cd "$private"; source_digest) == "$private_digest" ]] || { echo 'FAIL: private source changed'; exit 1; }
printf 'private_elf_sha256=%s\n' "$(sha256sum "$OUT/private-madaros" | cut -d' ' -f1)" >>"$OUT/receipt.txt"
SOUNIO_STDLIB_PATH="$private/stdlib" timeout 180 env SOUNIO_RESEARCH_EXACT_INTEGRAL_CENTER=1 SOUNIO_PRIVATE_MFI_GUARD_CONTROLS=1 "$OUT/private-madaros" compile "$FIX/enforce.sio" -O -o "$OUT/real-controls.elf" >"$OUT/real-controls.compile.log" 2>&1
chmod +x "$OUT/real-controls.elf"; timeout 30 "$OUT/real-controls.elf" >"$OUT/real-controls.stdout"
grep -qx 'PRIVATE_REAL_OWNED_CONTROLS baseline=1 refusals=6 restored=1 PASS' "$OUT/real-controls.compile.log"
grep -qx 'RESEARCH_EXACT_CENTER applications=1 last_function=exact_integral_center' "$OUT/real-controls.compile.log"
printf '1.000000\n' >"$OUT/real-controls.expected"; cmp "$OUT/real-controls.expected" "$OUT/real-controls.stdout"
[[ $(source_digest) == "$source_before" ]] || { echo 'FAIL: original source changed'; exit 1; }
[[ $(sha256sum "$MADAROS" | cut -d' ' -f1) == "$elf_before" ]] || { echo 'FAIL: original ELF changed'; exit 1; }
sha256sum -c "$OUT/fixtures.sha256" >"$OUT/fixtures-final.log"
echo 'SCO_EXACT_INTEGRAL_CENTER_BOUNDED_GATE_OK'
