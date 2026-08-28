#!/usr/bin/env bash
set -euo pipefail

# Submit every source in a Kretikos manifest. This intentionally delegates each
# case to submit-kretikos-source.sh so every source gets an independent Slurm
# runtime verdict and job comment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_SUBMIT="${ROOT_DIR}/slurm-jobs/kretikos/submit-kretikos-source.sh"

usage() {
  cat >&2 <<'EOF'
usage: slurm-jobs/kretikos/submit-kretikos-manifest.sh <manifest.tsv>

Manifest format:
  # label<TAB>source
  vec_add_f32<TAB>real_vec_add.sio

Relative source paths are resolved relative to the manifest directory first,
then relative to the repository root.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MANIFEST="$1"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "manifest not found: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -x "${SOURCE_SUBMIT}" ]]; then
  echo "source submitter is missing or not executable: ${SOURCE_SUBMIT}" >&2
  exit 1
fi

MANIFEST_DIR="$(cd "$(dirname "${MANIFEST}")" && pwd)"
SUMMARY="${SUMMARY:-/tmp/kretikos-manifest-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}.tsv}"
printf 'label\tsource\tstatus\tjob_id\tcomment\tartifact_dir\tartifact_json\n' >"${SUMMARY}"

total=0
failed=0
while IFS=$'\t' read -r label source rest || [[ -n "${label:-}" ]]; do
  [[ -z "${label:-}" ]] && continue
  [[ "${label}" == \#* ]] && continue
  if [[ -z "${source:-}" ]]; then
    source="${label}"
    label="$(basename "${source}" .sio)"
  fi
  case "${source}" in
    /*) src_path="${source}" ;;
    *)
      if [[ -f "${MANIFEST_DIR}/${source}" ]]; then
        src_path="${MANIFEST_DIR}/${source}"
      else
        src_path="${ROOT_DIR}/${source}"
      fi
      ;;
  esac

  echo "=== kretikos manifest case: ${label} ==="
  set +e
  output="$(WAIT_FOR_RESULT=1 "${SOURCE_SUBMIT}" "${src_path}" 2>&1)"
  rc=$?
  set -e
  printf '%s\n' "${output}"
  total=$((total + 1))
  job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job/ { print $4; exit }')"
  comment="$(printf '%s\n' "${output}" | awk '/^Comment=kretikos_source=/ { sub(/^Comment=/, ""); print; exit }')"
  artifact_dir="$(printf '%s\n' "${output}" | awk -F= '/^Artifacts=/ { print $2; exit }')"
  artifact_json="$(printf '%s\n' "${output}" | awk -F= '/^ArtifactJSON=/ { print $2; exit }')"
  if [[ "${rc}" -eq 0 ]]; then
    status="pass"
  else
    status="fail"
    failed=$((failed + 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${label}" "${src_path}" "${status}" "${job_id}" "${comment}" "${artifact_dir}" "${artifact_json}" >>"${SUMMARY}"
done <"${MANIFEST}"

echo "=== kretikos manifest summary ==="
cat "${SUMMARY}"
echo "kretikos_manifest_result total=${total} failed=${failed} summary=${SUMMARY}"
[[ "${failed}" -eq 0 ]]
