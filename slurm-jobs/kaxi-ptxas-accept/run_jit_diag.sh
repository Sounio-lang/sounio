#!/usr/bin/env bash
# Worker-side DIAGNOSTIC for the K-AXI->PTX runtime differential follow-on.
# (Not the gate; run via: RUNNER_SH=.../run_jit_diag.sh bash submit_jit.sh --wait)
#
# Answers: are the rc=1 launches under the acceptance gate's generic invocation
# real defects, or under-provisioning artifacts?
#
#   Config A — reproduce the gate's invocation verbatim (--threads 1
#              --mem-words 64) and bucket every kernel by status/reason.
#   Config B — invoke each kernel with its mode's required buffers/type and
#              ample memory; a kernel that clears in B was an A artifact.
#   Phase 2 — arithmetic differential: feed a known input, compare the printed
#             output to a PTX-derived CPU oracle (proves compute, not just load).
set -uo pipefail

PTX_DIR="${1:-./ptx}"
RESULTS_DIR="${2:-./results}"
RUNNER="${3:-./kaxi_ptx_runner}"
mkdir -p "$RESULTS_DIR"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true

SUMMARY="$RESULTS_DIR/summary.txt"; : > "$SUMMARY"
DETAIL="$RESULTS_DIR/diag.tsv"; printf "config\tfile\trc\tstatus\treason\tcuda_result\n" > "$DETAIL"

# mode (filename prefix before __) -> extra runner flags it needs at runtime
mode_flags() {
  case "$1" in
    default)  echo "" ;;
    epistemic) echo "--epistemic" ;;
    f32)      echo "--type f32" ;;
    f32_2c)   echo "--type f32 --epistemic" ;;
    f32e)     echo "--type f32 --epistemic" ;;
    f32_gum)  echo "--type f32 --gum" ;;
    *)        echo "" ;;
  esac
}

run_config() {  # $1=label  $2..=common extra args; per-file mode flags added in B
  local cfg="$1"; shift
  local common=("$@")
  local pass=0 reject=0 syncfail=0 other=0 total=0
  echo "===== config ${cfg} (common: ${common[*]:-none}) =====" | tee -a "$SUMMARY"
  for f in "$PTX_DIR"/*.ptx; do
    local base mode flags out rc line status reason cures
    base="$(basename "$f")"; mode="${base%%__*}"; total=$((total+1))
    flags=""
    [[ "$cfg" == "B" ]] && flags="$(mode_flags "$mode")"
    out="$(timeout 30 "$RUNNER" "$f" --kernel kaxi_kernel "${common[@]}" $flags 2>&1)"
    rc=$?
    line="$(printf '%s\n' "$out" | grep -m1 'sounio_kaxi_runtime' || true)"
    status="$(printf '%s' "$line" | sed -n 's/.*status=\([^ ]*\).*/\1/p')"
    reason="$(printf '%s' "$line" | sed -n 's/.*reason=\([^ ]*\).*/\1/p')"
    cures="$(printf '%s' "$line"  | sed -n 's/.*cuda_result=\([0-9-]*\).*/\1/p')"
    printf "%s\t%s\t%d\t%s\t%s\t%s\n" "$cfg" "$base" "$rc" "${status:-NA}" "${reason:-NA}" "${cures:-NA}" >> "$DETAIL"
    case "${reason:-NA}" in
      launch_pass) pass=$((pass+1)) ;;
      cuModuleLoadData_rejected|cuModuleGetFunction_rejected) reject=$((reject+1)) ;;
      cuCtxSynchronize_failed|cuLaunchKernel_rejected|cuMemcpyDtoH_failed) syncfail=$((syncfail+1)) ;;
      *) other=$((other+1)) ;;
    esac
  done
  echo "[$cfg] total=$total pass=$pass jit_reject=$reject launch/sync_fail=$syncfail other=$other" | tee -a "$SUMMARY"
}

# ── Phase 1 ────────────────────────────────────────────────────────────────
run_config A --threads 1 --mem-words 64
run_config B --threads 1 --mem-words 4096

# reason histogram for config A (the gate's invocation = the committed 212)
echo "--- config A reason histogram ---" | tee -a "$SUMMARY"
awk -F'\t' '$1=="A"{print $5}' "$DETAIL" | sort | uniq -c | sort -rn | tee -a "$SUMMARY"
# kernels that FAILED in A but PASS in B (= under-provisioning artifacts)
echo "--- A-fail -> B-pass (artifact) vs A-fail & B-fail (real) ---" | tee -a "$SUMMARY"
awk -F'\t' '
  $1=="A"{a[$2]=$5}
  $1=="B"{b[$2]=$5}
  END{
    art=0; real=0;
    for (k in a) if (a[k]!="launch_pass") { if (b[k]=="launch_pass") art++; else { real++; print "  STILL-FAILS:",k,"A="a[k]" B="b[k] } }
    print "A-fail count:", art+real, "| cleared in B (artifact):", art, "| still failing in B (investigate):", real
  }' "$DETAIL" | tee -a "$SUMMARY"

# ── Phase 2: arithmetic differential (PTX-derived oracle, input x=3) ─────────
echo "===== Phase 2 differential (x=3.0) =====" | tee -a "$SUMMARY"
# op -> expected output of in-place self-op on mem[0]=3
diff_one() {  # $1=ptx $2=type(f32|f64) $3=expected
  local ptx="$1" ty="$2" exp="$3" out got
  [[ -f "$PTX_DIR/$ptx" ]] || { echo "  SKIP $ptx (absent)" | tee -a "$SUMMARY"; return; }
  out="$(timeout 30 "$RUNNER" "$PTX_DIR/$ptx" --kernel kaxi_kernel --threads 1 --mem-words 8 \
        --init-mem "3.0" --type "$ty" --print-count 1 2>/dev/null)"
  got="$(printf '%s\n' "$out" | sed -n 's/^MEM: //p' | awk '{print $1}')"
  awk -v g="$got" -v e="$exp" 'BEGIN{d=g-e; if(d<0)d=-d; if(d<=1e-4){print "  PASS"}else{print "  FAIL"}}' \
      | sed "s|$|  $ptx got=$got exp=$exp|" | tee -a "$SUMMARY"
}
# f32 mode kernels
diff_one "f32__source_vec_add_f32.ptx" f32 6
diff_one "f32__source_vec_sub_f32.ptx" f32 0
diff_one "f32__source_vec_mul_f32.ptx" f32 9
diff_one "f32__source_vec_div_f32.ptx" f32 1
diff_one "f32__source_fma_f32.ptx"     f32 12
# f64 kernels live in default mode
diff_one "default__source_vec_add_f64.ptx" f64 6
diff_one "default__source_vec_sub_f64.ptx" f64 0
diff_one "default__source_vec_mul_f64.ptx" f64 9
diff_one "default__source_vec_div_f64.ptx" f64 1
diff_one "default__source_fma_f64.ptx"     f64 12

echo "DIAG_DONE" | tee -a "$SUMMARY"
