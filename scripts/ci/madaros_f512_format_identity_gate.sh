#!/usr/bin/env bash
set -euo pipefail

# Source-owned identity gate for the f512 (IEEE 754-2008 binary512) format
# family. Mirrors scripts/ci/madaros_f128_f256_format_identity_gate.sh so the
# V0-A identity contract is asserted per format; the sibling covers f128/f256.
#
# What this gate asserts:
#   * self-hosted/compiler/f512_format_descriptor_probe.sio compiles under the
#     bootstrap seed compiler (bin/souc-lean-single-x86_64).
#   * The probe returns rc=0 and prints the exact `PASS f512_format_descriptor_probe`
#     receipt — anything else (probe missing the PASS line, or a probe-side
#     failure encoded as a non-zero rc) is reported FAIL.
#
# What this gate does NOT assert (explicitly deferred):
#   * Source-level use of f512. The parser/checker treat it as a wide-float
#     reserved keyword today, same shape as f128/f256 in V0-A.
#   * f512 value semantics, arithmetic, normalisation, or RNE rounding. Those
#     are owned by the V0-D ladder (docs/architecture/F128_F256_LADDER.md).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

STRUCTURAL_ONLY=0
if [[ "${1:-}" == "--structural-only" ]]; then
  STRUCTURAL_ONLY=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--structural-only]" >&2
  exit 64
fi

if [[ "$STRUCTURAL_ONLY" -eq 0 && -z "${SOUNIO_F512_COMPILER:-}" ]]; then
  echo "BLOCKED manual gate requires SOUNIO_F512_COMPILER=/path/to/source-fresh-madaros-elf" >&2
  exit 2
fi
COMPILER=""
COMPILER_CLI="$ROOT_DIR/bin/madaros"
if [[ "$STRUCTURAL_ONLY" -eq 0 ]]; then
  COMPILER="$(realpath "$SOUNIO_F512_COMPILER")"
fi
SEED_COMPILER="$(realpath "${SOUNIO_F512_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"
SOURCE_HEAD="$(git rev-parse HEAD)"
COMPILER_SOURCE_SHA="${SOUNIO_F512_COMPILER_SOURCE_SHA:-}"

for binary in "$SEED_COMPILER" ${COMPILER:+"$COMPILER"}; do
  if [[ ! -x "$binary" ]]; then
    echo "FAIL compiler is not executable: $binary" >&2
    exit 2
  fi
  if [[ "$(head -c2 "$binary" 2>/dev/null)" == '#!' ]]; then
    echo "FAIL gate requires a resolved ELF, not a wrapper: $binary" >&2
    exit 2
  fi
done
if [[ "$STRUCTURAL_ONLY" -eq 0 && ! -x "$COMPILER_CLI" ]]; then
  echo "FAIL canonical Madaros CLI is not executable: $binary" >&2
  exit 2
fi
if [[ "$STRUCTURAL_ONLY" -eq 0 && -z "$COMPILER_SOURCE_SHA" ]]; then
  echo "BLOCKED set SOUNIO_F512_COMPILER_SOURCE_SHA to the source SHA used to build the Madaros ELF" >&2
  exit 2
fi
if [[ "$STRUCTURAL_ONLY" -eq 0 && "$COMPILER_SOURCE_SHA" != "$SOURCE_HEAD" ]]; then
  echo "FAIL compiler/source pin mismatch: compiler=$COMPILER_SOURCE_SHA source=$SOURCE_HEAD" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ "$STRUCTURAL_ONLY" -eq 0 ]]; then
  echo "compiler_elf=$COMPILER"
  echo "compiler_sha256=$(sha256sum "$COMPILER" | awk '{print $1}')"
  echo "compiler_source_sha=$COMPILER_SOURCE_SHA"
  echo "compiler_cli=$COMPILER_CLI"
fi
echo "seed_elf=$SEED_COMPILER"
echo "seed_sha256=$(sha256sum "$SEED_COMPILER" | awk '{print $1}')"
echo "source_head=$SOURCE_HEAD"
if [[ "$STRUCTURAL_ONLY" -eq 1 ]]; then
  echo "gate_mode=structural_only"
else
  echo "gate_mode=manual_source_fresh_evidence"
fi

DESCRIPTOR="self-hosted/compiler/f512_format_descriptor_probe.sio"
probe_elf="$TMP_DIR/f512-identity-probe.elf"
probe_build_log="$TMP_DIR/identity-probe-build.log"
probe_run_log="$TMP_DIR/identity-probe-run.log"
if ! "$SEED_COMPILER" "$DESCRIPTOR" "$probe_elf" >"$probe_build_log" 2>&1; then
  echo "FAIL source-owned identity probe did not compile with the bootstrap seed" >&2
  cat "$probe_build_log" >&2
  exit 1
fi
chmod +x "$probe_elf"
if ! "$probe_elf" >"$probe_run_log" 2>&1; then
  echo "FAIL source-owned identity probe returned nonzero" >&2
  cat "$probe_run_log" >&2
  exit 1
fi
if ! grep -Fxq 'PASS f512_format_descriptor_probe' "$probe_run_log"; then
  echo "FAIL source-owned identity probe omitted exact PASS receipt" >&2
  cat "$probe_run_log" >&2
  exit 1
fi
echo "PASS internal identity: TyRawPtr=96 TyF128=97 TyF256=98 TyF512=99 descriptors=exact compatibility=identity-only names=distinct limbs=8"