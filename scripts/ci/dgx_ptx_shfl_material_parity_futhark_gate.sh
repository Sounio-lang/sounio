#!/usr/bin/env bash
# ADR-009 verified_foreign_reference gate: independently re-derives the
# abstract shfl.sync.bfly.b32 XOR-butterfly semantics in Futhark and
# checks internal (algebraic) consistency. This does not execute on the
# DGX Spark GPU -- it validates that Sounio's frozen abstract semantics
# for the butterfly law are self-consistent, independent of the CUDA
# measurement harness in tools/pireus/dgx_ptx_shfl_material_parity.cu.
#
# GPU-vs-Futhark empirical parity: scripts/dev/dgx_ptx_shfl_material_parity_gpu_check.sh
# runs the companion dump probe on a real DGX Spark GPU (best-effort;
# SKIPs if unreachable from this host) and diffs it against this
# oracle's `check` entry. Last verified 2026-09-04 on NVIDIA GB10
# (compute_capability 12.1): 0/256 mismatched cells.

set -euo pipefail
umask 077

fail() {
  printf 'dgx-ptx-shfl-material-parity-futhark: FAIL: %s\n' "$*" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOURCE="$ROOT_DIR/tools/pireus/dgx_ptx_shfl_material_parity.fut"
EXPECTED_FUTHARK_VERSION="0.25.27"

[[ -r "$SOURCE" ]] || fail "oracle source not found: $SOURCE"

command -v futhark >/dev/null 2>&1 || \
  fail "futhark not on PATH; install ${EXPECTED_FUTHARK_VERSION} from https://github.com/diku-dk/futhark/releases"

observed_version="$(futhark -V | sed -n '1s/^Futhark \([0-9.]*\)\.$/\1/p')"
[[ "$observed_version" == "$EXPECTED_FUTHARK_VERSION" ]] || \
  fail "futhark version drift: expected $EXPECTED_FUTHARK_VERSION, observed ${observed_version:-unknown}"

source_sha256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"

work="$(mktemp -d "${TMPDIR:-/tmp}/dgx-ptx-shfl-futhark.XXXXXX")"
trap 'rm -rf "$work"' EXIT

futhark c "$SOURCE" -o "$work/oracle" >/dev/null 2>&1 || \
  fail "futhark compilation failed"

output="$(printf '' | "$work/oracle" -e standalone 2>&1)" || \
  fail "oracle execution failed: $output"

involution_holds="$(printf '%s\n' "$output" | sed -n '1p')"
[[ "$involution_holds" == "true" ]] || \
  fail "involution self-check failed (butterfly law is not its own inverse under the frozen semantics)"

printf 'PIREUS_DGX_PTX_SHFL_MATERIAL_PARITY_FUTHARK_V1\n'
printf 'oracle_class=verified_foreign_reference\n'
printf 'producer_language=Futhark\n'
printf 'producer_role=MATERIAL_PARITY_ABSTRACT_SEMANTICS\n'
printf 'semantic_authority_language=Sounio\n'
printf 'futhark_version=%s\n' "$observed_version"
printf 'oracle_source_sha256=%s\n' "$source_sha256"
printf 'dimension=16\n'
printf 'involution_check=PASS\n'
printf 'gpu_empirical_parity=SEE_scripts_dev_dgx_ptx_shfl_material_parity_gpu_check\n'
printf 'result=PASS\n'
