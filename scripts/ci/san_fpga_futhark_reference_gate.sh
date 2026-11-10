#!/usr/bin/env bash
# CI gate for the Futhark verified_foreign_reference of the U250 SAN
# catastrophe-scan semantics (ADR-021 steps 2-4; ADR-009 section 1).
#
# The reference (scripts/research/san_scan_reference.fut) is authored
# from docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md (T2),
# independently of the HLS kernel and of the Python contract model.
# The Python model remains external_corroboration_only (ADR-021 step 1).
#
# Pinned toolchain (ADR-009 criterion 3): Futhark 0.25.27.
# A version drift fails this gate until the pin is deliberately moved
# here and in artifacts/audit/claim_oracle_inventory.overrides.tsv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC="${REPO_ROOT}/scripts/research/san_scan_reference.fut"
PINNED_FUTHARK="Futhark 0.25.27"

FUTHARK_BIN="${FUTHARK:-futhark}"
if ! command -v "${FUTHARK_BIN}" >/dev/null 2>&1; then
    echo "SAN_FPGA_FUTHARK_REFERENCE_FAIL: futhark not found (${FUTHARK_BIN})"
    exit 1
fi

VERSION="$("${FUTHARK_BIN}" --version | head -n 1)"
if [[ "${VERSION}" != "${PINNED_FUTHARK}"* ]]; then
    echo "SAN_FPGA_FUTHARK_REFERENCE_FAIL: pinned toolchain is ${PINNED_FUTHARK}, found: ${VERSION}"
    exit 1
fi

if [[ ! -f "${SRC}" ]]; then
    echo "SAN_FPGA_FUTHARK_REFERENCE_FAIL: missing ${SRC}"
    exit 1
fi

cd "${REPO_ROOT}"
"${FUTHARK_BIN}" check "${SRC}"
"${FUTHARK_BIN}" test "${SRC}"

echo "SAN_FPGA_FUTHARK_REFERENCE_PASS toolchain=\"${VERSION}\" reference=scripts/research/san_scan_reference.fut"
