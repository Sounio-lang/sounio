#!/usr/bin/env bash
# Build a local, hash-pinned Foundry/Slurm handoff package for a chi>=6 candidate.
#
# This script does not submit jobs and does not write Slurm/Kubernetes YAML. It
# packages the candidate manifest plus the directly referenced hash-checked
# artifacts into a small replay bundle for a host/control-plane agent.
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: make_chi6_foundry_handoff_package.sh <candidate.manifest> <out-dir>

Creates:
  <out-dir>/chi6-package/      manifest plus directly referenced artifacts
  <out-dir>/SHA256SUMS         package hashes
  <out-dir>/handoff.txt        Foundry/Slurm handoff request
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_IN="$1"
OUT_DIR="$2"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
PROMOTABLE_VALIDATOR="$ROOT/examples/erdos/validate_chi6_promotable_candidate.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
[[ -s "$MANIFEST_IN" ]] || { echo "error: missing/empty manifest: $MANIFEST_IN" >&2; exit 2; }

mkdir -p "$OUT_DIR/chi6-package"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
PKG_DIR="$OUT_DIR/chi6-package"
HANDOFF="$OUT_DIR/handoff.txt"
SUMS="$OUT_DIR/SHA256SUMS"

"$VALIDATOR" "$MANIFEST_IN" > "$OUT_DIR/source_manifest_validator.out"

declare -A FIELDS
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  [[ "$line" == *=* ]] || continue
  key="${line%%=*}"
  val="${line#*=}"
  FIELDS[$key]="$val"
done < "$MANIFEST_IN"

need() {
  local key="$1"
  if [[ -z "${FIELDS[$key]+x}" || -z "${FIELDS[$key]}" ]]; then
    echo "error: manifest missing field: $key" >&2
    exit 2
  fi
}

for key in \
  candidate_id promotable n m k geometry_proof_type sat_proof_route triangle_sb \
  edge_path edge_sha256 lean_sat_module_path lean_sat_module_sha256 \
  geometry_module_path geometry_module_sha256 lean_build_command producer_command
do
  need "$key"
done

MANIFEST_DIR="$(cd "$(dirname "$MANIFEST_IN")" && pwd)"
MANIFEST_BASENAME="$(basename "$MANIFEST_IN")"

resolve_path() {
  local p="$1"
  if [[ "$p" == "NONE" ]]; then
    printf '%s\n' "NONE"
  elif [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$MANIFEST_DIR/$p"
  fi
}

copy_pair() {
  local path_key="$1"
  local hash_key="$2"
  local p="${FIELDS[$path_key]:-NONE}"
  local h="${FIELDS[$hash_key]:-NONE}"
  [[ "$p" == "NONE" && "$h" == "NONE" ]] && return
  [[ "$p" != "NONE" && "$h" != "NONE" ]] || {
    echo "error: $path_key/$hash_key must both be concrete or both be NONE" >&2
    exit 2
  }
  case "$p" in
    /*|*..*|"")
      echo "error: packageable artifact path must be relative and contain no '..': $path_key=$p" >&2
      exit 2
      ;;
  esac
  local src dst actual
  src="$(resolve_path "$p")"
  [[ -s "$src" ]] || { echo "error: missing referenced artifact: $src" >&2; exit 2; }
  if [[ -L "$src" ]]; then
    echo "error: refusing to package symlink artifact: $src" >&2
    exit 2
  fi
  actual="$(sha256sum "$src" | awk '{print $1}')"
  if [[ "$actual" != "${h,,}" ]]; then
    echo "error: artifact hash mismatch for $path_key: got $actual expected ${h,,}" >&2
    exit 1
  fi
  dst="$PKG_DIR/$p"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

for pair in \
  edge_path:edge_sha256 \
  cnf_path:cnf_sha256 \
  drat_or_lrat_path:drat_or_lrat_sha256 \
  lean_sat_module_path:lean_sat_module_sha256 \
  geometry_module_path:geometry_module_sha256 \
  offload_review_raw:offload_review_sha256 \
  source_meta_path:source_meta_sha256 \
  cube_batch_path:cube_batch_sha256 \
  cube_refutation_summary_path:cube_refutation_summary_sha256 \
  cube_cover_certificate_path:cube_cover_certificate_sha256 \
  cube_cover_complement_cnf_path:cube_cover_complement_cnf_sha256 \
  cube_cover_complement_lrat_path:cube_cover_complement_lrat_sha256 \
  geometry_source_path:geometry_source_sha256
do
  copy_pair "${pair%%:*}" "${pair##*:}"
done

cp "$MANIFEST_IN" "$PKG_DIR/$MANIFEST_BASENAME"
MANIFEST_PKG="$PKG_DIR/$MANIFEST_BASENAME"
"$VALIDATOR" "$MANIFEST_PKG" > "$OUT_DIR/package_manifest_validator.out"
if [[ "${FIELDS[promotable]}" == "1" ]]; then
  "$PROMOTABLE_VALIDATOR" "$MANIFEST_PKG" > "$OUT_DIR/package_promotable_validator.out"
fi

(
  cd "$OUT_DIR"
  find chi6-package -type f -print0 | sort -z | xargs -0 sha256sum
) > "$SUMS"

if ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "error: repository root is not a git checkout: $ROOT" >&2
  exit 1
fi

branch="$(git -C "$ROOT" branch --show-current)"
[[ -n "$branch" ]] || branch="detached"
commit="$(git -C "$ROOT" rev-parse --verify HEAD)"
dirty_count="$(git -C "$ROOT" status --porcelain=v1 | wc -l | tr -d ' ')"
if [[ "$dirty_count" == "0" ]]; then
  dirty_state="clean"
else
  dirty_state="dirty (${dirty_count} status entries)"
fi
manifest_sha="$(sha256sum "$MANIFEST_PKG" | awk '{print $1}')"
sums_sha="$(sha256sum "$SUMS" | awk '{print $1}')"
source_surface="workspace pod"
repo_path_seen="$ROOT"
host_repo_root="/home/devsounio/projects/sounio"

get_field() {
  local key="$1"
  printf '%s\n' "${FIELDS[$key]:-NONE}"
}

cat > "$HANDOFF" <<EOF
Heavy Validation Handoff
requested_by: ${USER:-codex}
source_surface: $source_surface
repo_path_seen: $repo_path_seen
branch: $branch
commit: $commit
dirty_state: $dirty_state
gate_requested: chi6 candidate package validation / chi6 SAT+Lean replay
command_or_foundry_target: proposed Foundry target chi6-candidate; if absent, host agent should run host_replay_commands verbatim and return the standard artifact payload
reason: replay hash-pinned chi6 candidate package outside the interactive workspace; no cluster submission was available in this Codex session
gpu_requirement: none for package validation; optional/required only for upstream solver search
slurm_requirement: yes for host/control-plane heavy replay
expected_artifact_root: artifacts/foundry/chi6-candidate-<run-id>/
acceptance_criterion: sha256sum -c passes; validate_chi6_candidate_manifest passes; if promotable=1 validate_chi6_promotable_candidate passes; return first classified blocker otherwise
known_blockers: CHI6-WITNESS-ABSENT unless this package is promotable=1 with Euclidean geometry and Real bridge artifacts
return_payload: summary, artifact root, command used, validator logs, Lean logs, first failing log if any, failure class

package_root: chi6-package
repo_root_seen: $repo_path_seen
host_repo_root: $host_repo_root
chi6_candidate_id: ${FIELDS[candidate_id]}
chi6_manifest_path: chi6-package/$MANIFEST_BASENAME
chi6_manifest_sha256: $manifest_sha
promotable: ${FIELDS[promotable]}
n: ${FIELDS[n]}
m: ${FIELDS[m]}
k: ${FIELDS[k]}
geometry_proof_type: ${FIELDS[geometry_proof_type]}
sat_proof_route: ${FIELDS[sat_proof_route]}
triangle_sb: ${FIELDS[triangle_sb]}
edge_path: $(get_field edge_path)
edge_sha256: $(get_field edge_sha256)
source_meta_path: $(get_field source_meta_path)
source_meta_sha256: $(get_field source_meta_sha256)
cube_batch_path: $(get_field cube_batch_path)
cube_batch_sha256: $(get_field cube_batch_sha256)
cube_refutation_summary_path: $(get_field cube_refutation_summary_path)
cube_refutation_summary_sha256: $(get_field cube_refutation_summary_sha256)
cube_cover_certificate_path: $(get_field cube_cover_certificate_path)
cube_cover_certificate_sha256: $(get_field cube_cover_certificate_sha256)
cube_cover_complement_cnf_path: $(get_field cube_cover_complement_cnf_path)
cube_cover_complement_cnf_sha256: $(get_field cube_cover_complement_cnf_sha256)
cube_cover_complement_lrat_path: $(get_field cube_cover_complement_lrat_path)
cube_cover_complement_lrat_sha256: $(get_field cube_cover_complement_lrat_sha256)
lean_sat_module_path: $(get_field lean_sat_module_path)
lean_sat_module_sha256: $(get_field lean_sat_module_sha256)
geometry_module_path: $(get_field geometry_module_path)
geometry_module_sha256: $(get_field geometry_module_sha256)
geometry_source_path: $(get_field geometry_source_path)
geometry_source_sha256: $(get_field geometry_source_sha256)
lean_module: $(get_field lean_module)
lean_build_command: ${FIELDS[lean_build_command]}
producer_command: ${FIELDS[producer_command]}
offload_review_raw: $(get_field offload_review_raw)
offload_review_sha256: $(get_field offload_review_sha256)

local_preflight_commands: cd $OUT_DIR && sha256sum -c SHA256SUMS && cd $ROOT && examples/erdos/validate_chi6_candidate_manifest.sh $OUT_DIR/chi6-package/$MANIFEST_BASENAME
host_replay_commands: cd $host_repo_root && sha256sum -c $OUT_DIR/SHA256SUMS && examples/erdos/validate_chi6_candidate_manifest.sh $OUT_DIR/chi6-package/$MANIFEST_BASENAME
promotable_replay_command: cd $host_repo_root && examples/erdos/validate_chi6_promotable_candidate.sh $OUT_DIR/chi6-package/$MANIFEST_BASENAME
artifact_sha256s_path: SHA256SUMS
artifact_sha256s_sha256: $sums_sha
trust_boundary: local package/format/hash validation only; no Slurm execution; no Euclidean chi>=6 claim unless promotable=1 plus Lean/offload gates pass
EOF

(cd "$OUT_DIR" && sha256sum -c SHA256SUMS > "$OUT_DIR/sha256sum_check.out")

echo "chi6_foundry_handoff_package: PASS out=$OUT_DIR manifest=$MANIFEST_PKG handoff=$HANDOFF sha256s=$SUMS"
