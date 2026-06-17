#!/usr/bin/env bash
# Validate a local chi>=6 Foundry/Slurm handoff package without submitting jobs.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <handoff-package-dir>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DIR="$1"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
PROMOTABLE_VALIDATOR="$ROOT/examples/erdos/validate_chi6_promotable_candidate.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
[[ -d "$PACKAGE_DIR" ]] || { echo "error: missing package dir: $PACKAGE_DIR" >&2; exit 2; }
PACKAGE_DIR="$(cd "$PACKAGE_DIR" && pwd)"

HANDOFF="$PACKAGE_DIR/handoff.txt"
SUMS="$PACKAGE_DIR/SHA256SUMS"
[[ -s "$HANDOFF" ]] || { echo "error: missing/empty handoff.txt: $HANDOFF" >&2; exit 2; }
[[ -s "$SUMS" ]] || { echo "error: missing/empty SHA256SUMS: $SUMS" >&2; exit 2; }

declare -A H
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == "Heavy Validation Handoff" ]] && continue
  if [[ "$line" != *": "* ]]; then
    echo "error: malformed handoff line: $line" >&2
    exit 2
  fi
  key="${line%%: *}"
  val="${line#*: }"
  if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "error: invalid handoff key: $key" >&2
    exit 2
  fi
  if [[ -n "${H[$key]+x}" ]]; then
    echo "error: duplicate handoff key: $key" >&2
    exit 2
  fi
  H[$key]="$val"
done < "$HANDOFF"

need_handoff() {
  local key="$1"
  if [[ -z "${H[$key]+x}" || -z "${H[$key]}" ]]; then
    echo "error: missing handoff field: $key" >&2
    exit 2
  fi
}

for key in \
  package_root chi6_candidate_id chi6_manifest_path chi6_manifest_sha256 promotable n m k \
  geometry_proof_type sat_proof_route triangle_sb edge_path edge_sha256 \
  lean_sat_module_path lean_sat_module_sha256 artifact_sha256s_path \
  artifact_sha256s_sha256 trust_boundary
do
  need_handoff "$key"
done

[[ "${H[package_root]}" == "chi6-package" ]] || {
  echo "error: package_root must be chi6-package" >&2
  exit 2
}
[[ "${H[artifact_sha256s_path]}" == "SHA256SUMS" ]] || {
  echo "error: artifact_sha256s_path must be SHA256SUMS" >&2
  exit 2
}
[[ "${H[artifact_sha256s_sha256],,}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "error: artifact_sha256s_sha256 is not a SHA256 digest" >&2
  exit 2
}
actual_sums_sha="$(sha256sum "$SUMS" | awk '{print $1}')"
if [[ "$actual_sums_sha" != "${H[artifact_sha256s_sha256],,}" ]]; then
  echo "error: SHA256SUMS digest mismatch: got $actual_sums_sha expected ${H[artifact_sha256s_sha256],,}" >&2
  exit 1
fi

case "${H[chi6_manifest_path]}" in
  chi6-package/*) ;;
  *)
    echo "error: chi6_manifest_path must live under chi6-package/" >&2
    exit 2
    ;;
esac
case "${H[chi6_manifest_path]}" in
  /*|*..*|*[\ \;\&\|\`\$\(\)\<\>]*)
    echo "error: unsafe chi6_manifest_path: ${H[chi6_manifest_path]}" >&2
    exit 2
    ;;
esac
MANIFEST="$PACKAGE_DIR/${H[chi6_manifest_path]}"
[[ -s "$MANIFEST" ]] || { echo "error: missing packaged manifest: $MANIFEST" >&2; exit 2; }
actual_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
if [[ "$actual_manifest_sha" != "${H[chi6_manifest_sha256],,}" ]]; then
  echo "error: chi6_manifest_sha256 mismatch: got $actual_manifest_sha expected ${H[chi6_manifest_sha256],,}" >&2
  exit 1
fi

(cd "$PACKAGE_DIR" && sha256sum -c SHA256SUMS)
"$VALIDATOR" "$MANIFEST" >/dev/null

declare -A M
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  [[ "$line" == *=* ]] || continue
  key="${line%%=*}"
  val="${line#*=}"
  M[$key]="$val"
done < "$MANIFEST"

need_manifest() {
  local key="$1"
  if [[ -z "${M[$key]+x}" || -z "${M[$key]}" ]]; then
    echo "error: packaged manifest missing field: $key" >&2
    exit 2
  fi
}

manifest_field() {
  local key="$1"
  printf '%s\n' "${M[$key]:-NONE}"
}

compare_field() {
  local manifest_key="$1"
  local handoff_key="${2:-$manifest_key}"
  local manifest_value
  manifest_value="$(manifest_field "$manifest_key")"
  if [[ "${H[$handoff_key]:-}" != "$manifest_value" ]]; then
    echo "error: handoff/manifest mismatch for $manifest_key: ${H[$handoff_key]:-MISSING} != $manifest_value" >&2
    exit 2
  fi
}

for key in candidate_id promotable n m k geometry_proof_type sat_proof_route triangle_sb \
  edge_path edge_sha256 lean_sat_module_path lean_sat_module_sha256
do
  need_manifest "$key"
done

compare_field candidate_id chi6_candidate_id
for key in \
  promotable n m k geometry_proof_type sat_proof_route triangle_sb \
  edge_path edge_sha256 \
  source_meta_path source_meta_sha256 \
  cube_batch_path cube_batch_sha256 \
  cube_refutation_summary_path cube_refutation_summary_sha256 \
  cube_cover_certificate_path cube_cover_certificate_sha256 \
  cube_cover_complement_cnf_path cube_cover_complement_cnf_sha256 \
  cube_cover_complement_lrat_path cube_cover_complement_lrat_sha256 \
  lean_sat_module_path lean_sat_module_sha256 \
  geometry_module_path geometry_module_sha256 \
  geometry_source_path geometry_source_sha256 \
  lean_module lean_build_command producer_command \
  offload_review_raw offload_review_sha256
do
  compare_field "$key"
done

if [[ "${M[promotable]}" == "1" ]]; then
  "$PROMOTABLE_VALIDATOR" "$MANIFEST" >/dev/null
fi

expected_trust_boundary="local package/format/hash validation only; no Slurm execution; no Euclidean chi>=6 claim unless promotable=1 plus Lean/offload gates pass"
if [[ "${H[trust_boundary]}" != "$expected_trust_boundary" ]]; then
  echo "error: trust_boundary mismatch" >&2
  exit 2
fi

echo "chi6_foundry_handoff_package: VALID candidate=${M[candidate_id]} promotable=${M[promotable]}"
