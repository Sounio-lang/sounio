#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SOUNIO_DIR:-/home/devsounio/sounio}"
OUT_ROOT="${OUT_ROOT:-$(mktemp -d /tmp/abide-campaign-snapshot.XXXXXX)}"

mkdir -p \
  "${OUT_ROOT}/bin" \
  "${OUT_ROOT}/artifacts/self-hosted" \
  "${OUT_ROOT}/examples" \
  "${OUT_ROOT}/scripts/research"

_copy() { cp -a "$1" "$2"; }
_copy "${SRC_ROOT}/bin/souc" "${OUT_ROOT}/bin/"
_copy "${SRC_ROOT}/bin/souc-native" "${OUT_ROOT}/bin/"
_copy "${SRC_ROOT}/bin/souc-linux-x86_64" "${OUT_ROOT}/bin/"
_copy "${SRC_ROOT}/artifacts/self-hosted/souc-self-hosted-x86_64" "${OUT_ROOT}/artifacts/self-hosted/"
_copy "${SRC_ROOT}/examples/brain_ossm_abide.sio" "${OUT_ROOT}/examples/"
_copy "${SRC_ROOT}/scripts/research/abide_campaign_lib.py" "${OUT_ROOT}/scripts/research/"
_copy "${SRC_ROOT}/scripts/research/build_abide_temporal_manifest.py" "${OUT_ROOT}/scripts/research/"
_copy "${SRC_ROOT}/scripts/research/normalize_abide_manifest.py" "${OUT_ROOT}/scripts/research/"
_copy "${SRC_ROOT}/scripts/research/abide_manifest_quality_gate.py" "${OUT_ROOT}/scripts/research/"
_copy "${SRC_ROOT}/scripts/research/parse_brain_ossm_abide_output.py" "${OUT_ROOT}/scripts/research/"
_copy "${SRC_ROOT}/scripts/research/abide_external_baselines.py" "${OUT_ROOT}/scripts/research/"
_copy "${SRC_ROOT}/scripts/research/aggregate_brain_ossm_campaign.py" "${OUT_ROOT}/scripts/research/"

if [[ -n "${ABIDE_MANIFEST_PATH:-}" && -r "${ABIDE_MANIFEST_PATH}" ]]; then
  rsync -a "${ABIDE_MANIFEST_PATH}" "${OUT_ROOT}/abide_source_manifest.tsv"
fi

if [[ -n "${ABIDE_MANIFEST_PATH:-}" && -r "${ABIDE_MANIFEST_PATH}" ]]; then
  rsync -a "${ABIDE_MANIFEST_PATH}" "${OUT_ROOT}/abide_source_manifest.tsv"
fi

echo "${OUT_ROOT}"
