#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
NODE="${PIREUS_XEON_NODE:-gpuorangefs-r770-proxmox}"
PARTITION="${PIREUS_XEON_PARTITION:-all}"
RECEIPT="${PIREUS_XEON_RECEIPT:-tools/cluster/evidence/pireus_xeon_avx512_material.receipt}"
MATERIALIZER_SHA="e62c0c639ceece49823240565c4c2bd90fe2a766d98f059a9c59f9965f8c21d2"
SEMANTICS_SHA="100404ef5ea29c6d7fb945bfca3fb2433eb2f88aece42d6f5ef8e6b9067c326e"
HARNESS_SHA="34598aebbaeb5191484392635c21fee066736b74d1c490a6831bf8c816f17fea"
BRIDGE_SHA="df9031018c7d5074c0bfadaa94c47650bfc349b01cd84d7258f94b8b18d18c8f"
LOWERING_SHA="7c7d1ecb1bb6a7e263ca7597c1923c47371c39ab6f2725b78fe77dd68a1df283"

fail() { printf 'pireus-xeon-material-slurm: FAIL: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
[[ "$(sha self-hosted/native/pireus_xor_materializer.sio)" == "$MATERIALIZER_SHA" ]] || fail 'materializer source drifted'
[[ "$(sha tools/pireus/xor_basis4_semantics.values.v1)" == "$SEMANTICS_SHA" ]] || fail 'semantic freeze drifted'
[[ "$(sha tools/pireus/xeon_avx512_xor_material_harness.cpp)" == "$HARNESS_SHA" ]] || fail 'material harness drifted'
[[ "$(sha self-hosted/hlir/native_bridge.sio)" == "$BRIDGE_SHA" ]] || fail 'HLIR bridge drifted'
[[ "$(sha self-hosted/native/lower_ir.sio)" == "$LOWERING_SHA" ]] || fail 'native lowering drifted'
command -v salloc >/dev/null && command -v sbcast >/dev/null && command -v srun >/dev/null || fail 'Slurm transport unavailable'

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-xeon-slurm.XXXXXX")"
trap 'rm -rf "$work"' EXIT
SOUNIO_SOUC_ENGINE=lean_single bin/souc run self-hosted/native/pireus_xor_materializer.sio >"$work/materializer.log"
grep -q '^PIREUS_XEON_MATERIALIZER_PASS$' "$work/materializer.log" || fail 'Sounio materializer failed'
sed -n '/^---BEGIN-ASSEMBLY---$/,/^---END-ASSEMBLY---$/p' "$work/materializer.log" | sed '1d;$d' >"$work/kernel.S"
cc -c "$work/kernel.S" -o "$work/kernel.o"
c++ -std=c++20 -O2 tools/pireus/xeon_avx512_xor_material_harness.cpp "$work/kernel.o" -o "$work/material"
assembly_sha="$(sha "$work/kernel.S")"
binary_sha="$(sha "$work/material")"
toolchain="$(c++ --version | head -1)"

set +e
salloc --partition="$PARTITION" --nodelist="$NODE" --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --exclusive --time=00:05:00 bash -lc "
    sbcast --force '$work/material' /tmp/pireus-xeon-avx512-material
    srun --ntasks=1 --kill-on-bad-exit=1 bash -lc '
      printf \"hardware_node=%s\n\" \"\$(hostname)\"
      lscpu | sed -n \"s/^Model name:[[:space:]]*/hardware_cpu=/p\"
      /tmp/pireus-xeon-avx512-material
    '
  " >"$work/run.log" 2>&1
run_rc=$?
set -e
if [[ $run_rc -ne 0 ]]; then
  cat "$work/run.log" >&2
  fail "Slurm execution failed rc=$run_rc"
fi
grep -q '^basis_pairs=256 component_checks=4096 failures=0$' "$work/run.log" || fail '256-pair material comparison failed'
grep -q '^result=PASS$' "$work/run.log" || fail 'material result did not pass'
grep -q '^hardware_cpu=Intel(R) Xeon(R) 6730P$' "$work/run.log" || fail 'canonical Xeon identity mismatch'
job_id="$(sed -n 's/^salloc: Granted job allocation \([0-9][0-9]*\)$/\1/p' "$work/run.log" | tail -1)"
mkdir -p "$(dirname "$RECEIPT")"
{
  printf 'schema=pireus-xeon-avx512-material-receipt-v1\n'
  printf 'sounio_source_sha256=%s\n' "$MATERIALIZER_SHA"
  printf 'frozen_semantics_sha256=%s\n' "$SEMANTICS_SHA"
  printf 'hlir_bridge_sha256=%s\nnative_lowering_sha256=%s\n' "$BRIDGE_SHA" "$LOWERING_SHA"
  printf 'assembly_sha256=%s\nharness_sha256=%s\nbinary_sha256=%s\n' "$assembly_sha" "$HARNESS_SHA" "$binary_sha"
  printf 'language_producer=Sounio\nlanguage_role=SEMANTIC_AUTHORITY\n'
  printf 'parity_language=C++\nparity_role=MATERIAL_PARITY\n'
  printf 'toolchain=%s\n' "$toolchain"
  sed -n '/^hardware_node=/p;/^hardware_cpu=/p' "$work/run.log"
  printf 'command=salloc+sbcast+srun /tmp/pireus-xeon-avx512-material\n'
  printf 'slurm_job_id=%s\n' "${job_id:-unknown}"
  printf 'basis_pairs=256\ncomponent_checks=4096\nfailures=0\nresult=PASS\n'
} >"$RECEIPT"
cat "$work/run.log"
printf 'PIREUS_XEON_AVX512_MATERIAL_SLURM_PASS receipt=%s sha256=%s\n' "$RECEIPT" "$(sha "$RECEIPT")"
