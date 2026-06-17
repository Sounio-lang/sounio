#!/usr/bin/env bash
# Format-check a chi>=6 candidate manifest without trusting it as mathematics.
#
# This checks field presence, basic graph/header consistency, SHA256s for listed
# artifacts, triangle-precolour metadata, and promotable-candidate Lean term
# declarations. It does not execute manifest commands, verify DRAT/LRAT, or
# prove exact geometry. A manifest is promotable only if this format check passes
# and the separate Lean/SAT-proof/offload gates in CHI6_CANDIDATE_CONTRACT.md
# pass.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <candidate.manifest>" >&2
  exit 2
fi

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }

MANIFEST="$1"
[[ -s "$MANIFEST" ]] || { echo "error: missing/empty manifest: $MANIFEST" >&2; exit 2; }
MANIFEST_DIR="$(cd "$(dirname "$MANIFEST")" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

declare -A FIELDS

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  if [[ "$line" != *=* ]]; then
    echo "error: manifest line lacks '=': $line" >&2
    exit 2
  fi
  key="${line%%=*}"
  val="${line#*=}"
  if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "error: invalid manifest key: $key" >&2
    exit 2
  fi
  if [[ -n "${FIELDS[$key]+x}" ]]; then
    echo "error: duplicate manifest key: $key" >&2
    exit 2
  fi
  FIELDS[$key]="$val"
done < "$MANIFEST"

need() {
  local key="$1"
  if [[ -z "${FIELDS[$key]+x}" || -z "${FIELDS[$key]}" ]]; then
    echo "error: missing required field: $key" >&2
    exit 2
  fi
}

need_promotable_lean_name() {
  local key="$1"
  if [[ -z "${FIELDS[$key]+x}" || -z "${FIELDS[$key]}" || "${FIELDS[$key]}" == "NONE" ]]; then
    echo "error: promotable=1 requires $key" >&2
    exit 2
  fi
  if [[ ! "${FIELDS[$key]}" =~ ^[A-Za-z_][A-Za-z0-9_\']*(\.[A-Za-z_][A-Za-z0-9_\']*)*$ ]]; then
    echo "error: invalid Lean name in $key: ${FIELDS[$key]}" >&2
    exit 2
  fi
}

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

check_hash_pair() {
  local path_key="$1"
  local hash_key="$2"
  local label="$3"
  local p="${FIELDS[$path_key]}"
  local h="${FIELDS[$hash_key]}"

  if [[ "$p" == "NONE" || "$h" == "NONE" ]]; then
    if [[ "$p" != "NONE" || "$h" != "NONE" ]]; then
      echo "error: $label path/hash must both be NONE or both concrete" >&2
      exit 2
    fi
    if [[ "${FIELDS[promotable]}" == "1" ]]; then
      echo "error: promotable=1 requires concrete $label artifact" >&2
      exit 2
    fi
    return
  fi

  h="${h,,}"
  FIELDS[$hash_key]="$h"
  if [[ ! "$h" =~ ^[0-9a-f]{64}$ ]]; then
    echo "error: invalid $label SHA256: $h" >&2
    exit 2
  fi

  local abs
  abs="$(resolve_path "$p")"
  [[ -s "$abs" ]] || { echo "error: missing/empty $label artifact: $abs" >&2; exit 2; }

  local actual
  actual="$(sha256sum "$abs" | awk '{print $1}')"
  if [[ "$actual" != "$h" ]]; then
    echo "error: $label SHA256 mismatch: got $actual expected $h ($abs)" >&2
    exit 1
  fi
}

check_optional_hash_pair() {
  local path_key="$1"
  local hash_key="$2"
  local label="$3"
  if [[ -z "${FIELDS[$path_key]+x}" && -z "${FIELDS[$hash_key]+x}" ]]; then
    return
  fi
  if [[ -z "${FIELDS[$path_key]+x}" || -z "${FIELDS[$hash_key]+x}" ]]; then
    echo "error: optional $label path/hash must be supplied together" >&2
    exit 2
  fi
  check_hash_pair "$path_key" "$hash_key" "$label"
}

for key in \
  candidate_manifest_version promotable candidate_id n m k \
  edge_path edge_sha256 cnf_path cnf_sha256 drat_or_lrat_path drat_or_lrat_sha256 \
  lean_sat_module_path lean_sat_module_sha256 geometry_module_path geometry_module_sha256 \
  geometry_proof_type sat_proof_route triangle_sb generator_commit producer_command lean_build_command \
  offload_review_raw offload_review_sha256
do
  need "$key"
done

[[ "${FIELDS[candidate_manifest_version]}" == "1" ]] || {
  echo "error: expected candidate_manifest_version=1" >&2
  exit 2
}
[[ "${FIELDS[promotable]}" == "0" || "${FIELDS[promotable]}" == "1" ]] || {
  echo "error: promotable must be 0 or 1" >&2
  exit 2
}
[[ "${FIELDS[candidate_id]}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "error: candidate_id must use only letters, digits, '.', '_', or '-'" >&2
  exit 2
}
[[ "${FIELDS[n]}" =~ ^[1-9][0-9]*$ ]] || { echo "error: n must be positive Nat" >&2; exit 2; }
[[ "${FIELDS[m]}" =~ ^[1-9][0-9]*$ ]] || { echo "error: m must be positive Nat" >&2; exit 2; }
[[ "${FIELDS[k]}" == "5" ]] || { echo "error: chi>=6 candidate manifests require k=5" >&2; exit 2; }
case "${FIELDS[geometry_proof_type]}" in
  none|finite_smoke|euclidean) ;;
  *)
    echo "error: geometry_proof_type must be none, finite_smoke, or euclidean" >&2
    exit 2
    ;;
esac
if [[ "${FIELDS[promotable]}" == "1" && "${FIELDS[geometry_proof_type]}" != "euclidean" ]]; then
  echo "error: promotable=1 requires geometry_proof_type=euclidean" >&2
  exit 2
fi
if [[ "${FIELDS[promotable]}" == "1" ]]; then
  if [[ "${FIELDS[geometry_module_path]}" == "NONE" || "${FIELDS[geometry_module_sha256]}" == "NONE" ]]; then
    echo "error: promotable=1 requires concrete Euclidean geometry module path/hash" >&2
    exit 2
  fi
  if [[ ! "${FIELDS[generator_commit]}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "error: promotable=1 requires generator_commit to be a full 40-hex commit SHA" >&2
    exit 2
  fi
  if ! git -C "$ROOT" cat-file -e "${FIELDS[generator_commit]}^{commit}" 2>/dev/null; then
    echo "error: promotable=1 generator_commit is not present in this repo" >&2
    exit 2
  fi
  if ! git -C "$ROOT" merge-base --is-ancestor "${FIELDS[generator_commit]}" HEAD 2>/dev/null; then
    echo "error: promotable=1 generator_commit must be an ancestor of HEAD" >&2
    exit 2
  fi
  if [[ "${FIELDS[producer_command]}" == "NONE" || "${FIELDS[lean_build_command]}" == "NONE" ]]; then
    echo "error: promotable=1 requires concrete producer_command and lean_build_command" >&2
    exit 2
  fi
fi
if [[ "${FIELDS[geometry_proof_type]}" == "none" && "${FIELDS[geometry_module_path]}" != "NONE" ]]; then
  echo "error: geometry_proof_type=none requires geometry_module_path=NONE" >&2
  exit 2
fi
if [[ "${FIELDS[geometry_proof_type]}" != "none" && "${FIELDS[geometry_module_path]}" == "NONE" ]]; then
  echo "error: geometry_proof_type=${FIELDS[geometry_proof_type]} requires concrete geometry_module artifact" >&2
  exit 2
fi

check_hash_pair edge_path edge_sha256 edge
check_hash_pair cnf_path cnf_sha256 cnf
check_hash_pair drat_or_lrat_path drat_or_lrat_sha256 drat_or_lrat
check_hash_pair lean_sat_module_path lean_sat_module_sha256 lean_sat_module
check_hash_pair geometry_module_path geometry_module_sha256 geometry_module
check_hash_pair offload_review_raw offload_review_sha256 offload_review_raw
check_optional_hash_pair cube_batch_path cube_batch_sha256 cube_batch
check_optional_hash_pair cube_refutation_summary_path cube_refutation_summary_sha256 cube_refutation_summary
check_optional_hash_pair cube_cover_certificate_path cube_cover_certificate_sha256 cube_cover_certificate
check_optional_hash_pair cube_cover_complement_cnf_path cube_cover_complement_cnf_sha256 cube_cover_complement_cnf
check_optional_hash_pair cube_cover_complement_lrat_path cube_cover_complement_lrat_sha256 cube_cover_complement_lrat
check_optional_hash_pair source_meta_path source_meta_sha256 source_meta
check_optional_hash_pair geometry_source_path geometry_source_sha256 geometry_source

check_source_meta_semantics() {
  local p="${FIELDS[source_meta_path]:-NONE}"
  [[ "$p" == "NONE" ]] && return
  local abs
  abs="$(resolve_path "$p")"
  if ! python3 - "$abs" "${FIELDS[candidate_id]}" "${FIELDS[n]}" "${FIELDS[m]}" \
      "${FIELDS[k]}" "${FIELDS[edge_sha256]}" <<'PY'
import json
import sys

path, candidate_id, n, m, k, edge_sha = sys.argv[1:]
try:
    with open(path, encoding="ascii") as f:
        meta = json.load(f)
except Exception as exc:
    print(f"error: source_meta is not valid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

def require(cond, msg):
    if not cond:
        print(f"error: source_meta {msg}", file=sys.stderr)
        raise SystemExit(1)

require(meta.get("schema") == "chi6_external_dimacs_edge_package.v1", "schema mismatch")
require(meta.get("candidate_id") == candidate_id, "candidate_id mismatch")
require(meta.get("n") == int(n), "n mismatch")
require(meta.get("m") == int(m), "m mismatch")
require(meta.get("k") == int(k), "k mismatch")
source_sha = meta.get("source_edge_sha256")
packaged_sha = meta.get("packaged_edge_sha256")
require(isinstance(source_sha, str) and len(source_sha) == 64, "source_edge_sha256 malformed")
require(isinstance(packaged_sha, str) and len(packaged_sha) == 64, "packaged_edge_sha256 malformed")
require(source_sha == packaged_sha, "source/packaged edge hash mismatch")
require(packaged_sha == edge_sha, "packaged edge hash does not match manifest edge_sha256")
require(isinstance(meta.get("source_edge_path"), str) and meta["source_edge_path"], "source_edge_path missing")
require(isinstance(meta.get("packaged_edge_path"), str) and meta["packaged_edge_path"], "packaged_edge_path missing")
require(meta.get("provenance_scope") == "edge_packaging_only", "provenance_scope mismatch")
require(
    meta.get("promotion_gate") == "requires_lrat_lean_and_exact_euclidean_geometry",
    "promotion_gate mismatch",
)
split_vertices = meta.get("split_vertices")
require(isinstance(split_vertices, list), "split_vertices must be a list")
require(
    all(isinstance(v, int) and v >= 0 for v in split_vertices),
    "split_vertices must contain non-negative integers",
)
PY
  then
    exit 2
  fi
}

check_nonempty_text_artifact() {
  local path_key="$1"
  local label="$2"
  local p="${FIELDS[$path_key]}"
  [[ "$p" == "NONE" ]] && return
  local abs
  abs="$(resolve_path "$p")"
  if [[ ! -s "$abs" ]] || ! rg -q '\S' "$abs"; then
    echo "error: $label artifact must be non-empty text: $abs" >&2
    exit 2
  fi
}

check_nonempty_text_artifact offload_review_raw offload_review_raw

check_cube_batch_has_rows() {
  local p="${FIELDS[cube_batch_path]:-NONE}"
  [[ "$p" == "NONE" ]] && return
  local abs
  abs="$(resolve_path "$p")"
  if ! rg -q '^[[:space:]]*[^#[:space:]]' "$abs"; then
    echo "error: cube-cover route requires at least one cube row" >&2
    exit 2
  fi
}

sat_route="${FIELDS[sat_proof_route]}"
case "$sat_route" in
  none|plain_lrat|triangle_sb5_lrat|cube_cover_split5|cube_cover_generic) ;;
  *)
    echo "error: sat_proof_route must be none, plain_lrat, triangle_sb5_lrat, cube_cover_split5, or cube_cover_generic" >&2
    exit 2
    ;;
esac
if [[ "${FIELDS[promotable]}" == "1" && "$sat_route" == "none" ]]; then
  echo "error: promotable=1 requires a non-none sat_proof_route" >&2
  exit 2
fi
need_concrete_optional_artifact() {
  local path_key="$1"
  local hash_key="$2"
  local label="$3"
  if [[ -z "${FIELDS[$path_key]+x}" || -z "${FIELDS[$hash_key]+x}" || \
        "${FIELDS[$path_key]}" == "NONE" || "${FIELDS[$hash_key]}" == "NONE" ]]; then
    echo "error: sat_proof_route=$sat_route requires concrete $label artifact" >&2
    exit 2
  fi
}

if [[ "${FIELDS[edge_path]}" == "NONE" ]]; then
  echo "error: candidate manifest requires a concrete edge artifact" >&2
  exit 2
fi

check_no_sorry_admit() {
  local path_key="$1"
  local label="$2"
  local p="${FIELDS[$path_key]}"
  [[ "$p" == "NONE" ]] && return
  local abs
  abs="$(resolve_path "$p")"
  if rg -q '\b(sorry|admit)\b|#exit|#eval|#check' "$abs"; then
    echo "error: $label artifact contains sorry/admit/#exit/#eval/#check: $abs" >&2
    exit 2
  fi
}

check_no_sorry_admit lean_sat_module_path lean_sat_module
check_no_sorry_admit geometry_module_path geometry_module

check_no_untrusted_axiom_surface() {
  local path_key="$1"
  local label="$2"
  local p="${FIELDS[$path_key]}"
  [[ "$p" == "NONE" ]] && return
  local abs
  abs="$(resolve_path "$p")"
  if rg -q '^[[:space:]]*(axiom|constant|opaque)\b' "$abs"; then
    echo "error: promotable $label artifact declares axiom/constant/opaque: $abs" >&2
    exit 2
  fi
  if rg -q '(^|[^A-Za-z0-9_])(unsafe|partial)([^A-Za-z0-9_]|$)|#eval|#check|#exit' "$abs"; then
    echo "error: promotable $label artifact contains unsafe/partial/#eval/#check/#exit: $abs" >&2
    exit 2
  fi
}

lean_decl_basename() {
  local name="$1"
  printf '%s\n' "${name##*.}"
}

module_name_from_formal_lean_path() {
  local abs="$1"
  local formal_root canon
  formal_root="$(readlink -f "$ROOT/formal/lean4")"
  canon="$(readlink -f "$abs")" || {
    echo "error: cannot canonicalize promotable geometry module path: $abs" >&2
    exit 2
  }
  case "$canon" in
    "$formal_root"/*.lean)
      local rel="${canon#$formal_root/}"
      rel="${rel%.lean}"
      printf '%s\n' "${rel//\//.}"
      ;;
    *)
      echo "error: promotable geometry module must live under formal/lean4: $abs" >&2
      exit 2
      ;;
  esac
}

check_lean_decl_present() {
  local path_key="$1"
  local term_key="$2"
  local label="$3"
  local p="${FIELDS[$path_key]}"
  local term="${FIELDS[$term_key]}"
  local abs decl
  abs="$(resolve_path "$p")"
  decl="$(lean_decl_basename "$term")"
  if ! rg -q "^[[:space:]]*(noncomputable[[:space:]]+)?(def|theorem|abbrev)[[:space:]]+$decl([^A-Za-z0-9_']|$)" "$abs"; then
    echo "error: promotable $label term $term is not declared in $(basename "$abs")" >&2
    exit 2
  fi
}

if [[ "${FIELDS[promotable]}" == "1" ]]; then
  need_promotable_lean_name lean_module
  need_promotable_lean_name lean_sat_edges_term
  need_promotable_lean_name lean_point_type
  need_promotable_lean_name lean_unit_term
  need_promotable_lean_name lean_geometry_term
  need_promotable_lean_name lean_edges_sync_term
  need_promotable_lean_name lean_no_five_witness_term
  need_promotable_lean_name lean_final_theorem
  need_promotable_lean_name lean_real_unit_term
  need_promotable_lean_name lean_real_unit_iff_standard
  need_promotable_lean_name lean_real_final_theorem

  geometry_abs="$(resolve_path "${FIELDS[geometry_module_path]}")"
  expected_module="$(module_name_from_formal_lean_path "$geometry_abs")"
  if [[ "${FIELDS[lean_module]}" != "$expected_module" ]]; then
    echo "error: lean_module=${FIELDS[lean_module]} does not match geometry_module_path module $expected_module" >&2
    exit 2
  fi
  if [[ ! "${FIELDS[lean_build_command]}" =~ ^lake[[:space:]]+build[[:space:]][A-Za-z0-9_.-]+([[:space:]][A-Za-z0-9_.-]+)*$ ]]; then
    echo "error: promotable=1 requires lean_build_command to be a simple 'lake build ...' command" >&2
    exit 2
  fi
  case " ${FIELDS[lean_build_command]#lake build } " in
    *" ${FIELDS[lean_module]} "*) ;;
    *)
      echo "error: promotable=1 lean_build_command must include lean_module target ${FIELDS[lean_module]}" >&2
      exit 2
      ;;
  esac
  check_no_untrusted_axiom_surface lean_sat_module_path lean_sat_module
  check_no_untrusted_axiom_surface geometry_module_path geometry_module
  rg -q 'EuclideanNatEdgeExactGeometry' "$geometry_abs" || {
    echo "error: promotable geometry module lacks EuclideanNatEdgeExactGeometry shape" >&2
    exit 2
  }
  rg -q 'ExactFieldLike' "$geometry_abs" || {
    echo "error: promotable geometry module lacks ExactFieldLike scalar-law shape" >&2
    exit 2
  }
  rg -q 'chi_ge_6_euclidean_plugin_contract|noFivePlaneColouringOfColourCNF|noFivePlaneColouringOfSplitVertex5Unsat|noFiveWitnessOfSplitVertex5Unsat|noFivePlaneColouringOfCubeCoverUnsat|noFiveWitnessOfCubeCoverUnsat' "$geometry_abs" || {
    echo "error: promotable geometry module does not expose the Euclidean chi>=6 plug-in contract" >&2
    exit 2
  }
  case "$sat_route" in
    plain_lrat)
      rg -q 'noFiveWitnessOfColourCNFUnsat|noFivePlaneColouringOfColourCNFUnsat' "$geometry_abs" || {
        echo "error: sat_proof_route=plain_lrat requires plain colourCNF witness adapter in geometry module" >&2
        exit 2
      }
      ;;
    triangle_sb5_lrat)
      rg -q 'noFiveWitnessOfColourCNFsb5UnsatTri|noFivePlaneColouringOfColourCNFsb5UnsatTri' "$geometry_abs" || {
        echo "error: sat_proof_route=triangle_sb5_lrat requires SB5 witness adapter in geometry module" >&2
        exit 2
      }
      ;;
    cube_cover_split5)
      rg -q 'noFiveWitnessOfSplitVertex5Unsat|noFivePlaneColouringOfSplitVertex5Unsat' "$geometry_abs" || {
        echo "error: sat_proof_route=cube_cover_split5 requires cube-cover witness adapter in geometry module" >&2
        exit 2
      }
      ;;
    cube_cover_generic)
      rg -q 'noFiveWitnessOfCubeCoverUnsat|noFivePlaneColouringOfCubeCoverUnsat' "$geometry_abs" || {
        echo "error: sat_proof_route=cube_cover_generic requires generic cube-cover witness adapter in geometry module" >&2
        exit 2
      }
      ;;
  esac
  check_lean_decl_present geometry_module_path lean_unit_term unit
  check_lean_decl_present lean_sat_module_path lean_sat_edges_term sat_edges
  check_lean_decl_present geometry_module_path lean_geometry_term geometry
  check_lean_decl_present geometry_module_path lean_edges_sync_term edges_sync
  check_lean_decl_present geometry_module_path lean_no_five_witness_term no_five_witness
  check_lean_decl_present geometry_module_path lean_final_theorem final_theorem
  check_lean_decl_present geometry_module_path lean_real_unit_term real_unit
  check_lean_decl_present geometry_module_path lean_real_unit_iff_standard real_unit_iff_standard
  check_lean_decl_present geometry_module_path lean_real_final_theorem real_final_theorem
fi

edge_abs="$(resolve_path "${FIELDS[edge_path]}")"
read -r header_n header_m < <(awk '$1 == "p" && $2 == "edge" {print $3, $4; exit}' "$edge_abs")
if [[ "$header_n" != "${FIELDS[n]}" || "$header_m" != "${FIELDS[m]}" ]]; then
  echo "error: edge header mismatch: p edge $header_n $header_m, manifest n=${FIELDS[n]} m=${FIELDS[m]}" >&2
  exit 2
fi

edge_lines="$(awk '$1 == "e" {c++} END {print c+0}' "$edge_abs")"
if [[ "$edge_lines" != "${FIELDS[m]}" ]]; then
  echo "error: edge count mismatch: found $edge_lines, manifest m=${FIELDS[m]}" >&2
  exit 2
fi

awk -v n="${FIELDS[n]}" '
  $1 == "e" {
    if ($2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/) exit 10
    if ($2 ~ /^0[0-9]+$/ || $3 ~ /^0[0-9]+$/) exit 10
    if ($2 < 1 || $2 > n || $3 < 1 || $3 > n || $2 == $3) exit 11
  }
' "$edge_abs" || {
  echo "error: edge file has malformed/out-of-range/self-loop edge" >&2
  exit 2
}

awk '
  $1 == "e" {
    u=$2; v=$3
    if (u > v) {t=u; u=v; v=t}
    key=u "," v
    if (seen[key]++) exit 12
  }
' "$edge_abs" || {
  echo "error: edge file has duplicate unordered edge" >&2
  exit 2
}

check_source_meta_semantics

if [[ "${FIELDS[cnf_path]}" != "NONE" ]]; then
  cnf_abs="$(resolve_path "${FIELDS[cnf_path]}")"
  read -r cnf_vars _ < <(awk '$1 == "p" && $2 == "cnf" {print $3, $4; exit}' "$cnf_abs")
  expected_vars=$(( ${FIELDS[n]} * ${FIELDS[k]} ))
  if [[ "$cnf_vars" != "$expected_vars" ]]; then
    echo "error: CNF var count mismatch: got $cnf_vars expected n*k=$expected_vars" >&2
    exit 2
  fi
fi

tri="${FIELDS[triangle_sb]}"
if [[ "$sat_route" == "triangle_sb5_lrat" && "$tri" == "none" ]]; then
  echo "error: sat_proof_route=$sat_route requires triangle_sb metadata" >&2
  exit 2
fi
if [[ "$tri" != "none" ]]; then
  if [[ "$sat_route" == "plain_lrat" || "$sat_route" == "cube_cover_split5" || "$sat_route" == "cube_cover_generic" ]]; then
    echo "error: sat_proof_route=$sat_route requires triangle_sb=none" >&2
    exit 2
  fi
  if [[ ! "$tri" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
    echo "error: triangle_sb must be none or zero-based a,b,c" >&2
    exit 2
  fi
  IFS=, read -r a b c <<< "$tri"
  if [[ "$a" == "$b" || "$a" == "$c" || "$b" == "$c" ]]; then
    echo "error: triangle_sb vertices must be distinct" >&2
    exit 2
  fi
  if (( a >= ${FIELDS[n]} || b >= ${FIELDS[n]} || c >= ${FIELDS[n]} )); then
    echo "error: triangle_sb vertex out of range" >&2
    exit 2
  fi
  for pair in "$a,$b" "$b,$c" "$c,$a"; do
    IFS=, read -r x y <<< "$pair"
    ux=$((x + 1)); uy=$((y + 1))
    if ! awk -v u="$ux" -v v="$uy" '
      $1 == "e" && (($2 == u && $3 == v) || ($2 == v && $3 == u)) {found=1}
      END {exit found ? 0 : 1}
    ' "$edge_abs"; then
      echo "error: unordered triangle_sb edge missing from edge file: $x,$y" >&2
      exit 2
    fi
  done
  if [[ "${FIELDS[lean_sat_module_path]}" != "NONE" ]]; then
    lean_abs="$(resolve_path "${FIELDS[lean_sat_module_path]}")"
    rg -q "colourCNFsb5[[:space:]]+$a[[:space:]]+$b[[:space:]]+$c[[:space:]]+${FIELDS[n]}([^0-9]|$)" "$lean_abs" || {
      echo "error: Lean SAT module does not use matching colourCNFsb5 triangle" >&2
      exit 2
    }
  fi
else
  if [[ "$sat_route" == "cube_cover_split5" || "$sat_route" == "cube_cover_generic" ]]; then
    need_concrete_optional_artifact cube_batch_path cube_batch_sha256 cube_batch
    need_concrete_optional_artifact cube_refutation_summary_path cube_refutation_summary_sha256 cube_refutation_summary
    check_cube_batch_has_rows
    if [[ "$sat_route" == "cube_cover_split5" ]]; then
      need_concrete_optional_artifact cube_cover_certificate_path cube_cover_certificate_sha256 cube_cover_certificate
    fi
  fi
  if [[ "${FIELDS[lean_sat_module_path]}" != "NONE" ]]; then
    lean_abs="$(resolve_path "${FIELDS[lean_sat_module_path]}")"
    if rg -q 'colourCNFsb5|colourCNFsb ' "$lean_abs"; then
      echo "error: triangle_sb=none but Lean SAT module uses triangle-precolour CNF" >&2
      exit 2
    fi
    rg -q "colourCNF ${FIELDS[n]} 5 " "$lean_abs" || {
      echo "error: triangle_sb=none requires plain colourCNF n 5 in Lean SAT module" >&2
      exit 2
    }
    if [[ "$sat_route" == "cube_cover_split5" ]]; then
      rg -q 'colourCNFWithUnit' "$lean_abs" || {
        echo "error: sat_proof_route=cube_cover_split5 requires colourCNFWithUnit in Lean SAT module" >&2
        exit 2
      }
      rg -q 'unsat_of_split_vertex5' "$lean_abs" || {
        echo "error: sat_proof_route=cube_cover_split5 requires unsat_of_split_vertex5 in Lean SAT module" >&2
        exit 2
      }
    fi
    if [[ "$sat_route" == "cube_cover_generic" ]]; then
      rg -q 'colourCNFWithCube' "$lean_abs" || {
        echo "error: sat_proof_route=cube_cover_generic requires colourCNFWithCube in Lean SAT module" >&2
        exit 2
      }
      rg -q 'CubeCover' "$lean_abs" || {
        echo "error: sat_proof_route=cube_cover_generic requires CubeCover in Lean SAT module" >&2
        exit 2
      }
      rg -q 'unsat_of_cube_cover' "$lean_abs" || {
        echo "error: sat_proof_route=cube_cover_generic requires unsat_of_cube_cover in Lean SAT module" >&2
        exit 2
      }
      if rg -q 'cube_cover_of_complement_unsat|cubeCoverComplementCNF' "$lean_abs"; then
        need_concrete_optional_artifact \
          cube_cover_complement_cnf_path cube_cover_complement_cnf_sha256 cube_cover_complement_cnf
        need_concrete_optional_artifact \
          cube_cover_complement_lrat_path cube_cover_complement_lrat_sha256 cube_cover_complement_lrat
      else
        need_concrete_optional_artifact cube_cover_certificate_path cube_cover_certificate_sha256 cube_cover_certificate
      fi
    fi
  elif [[ "$sat_route" == "cube_cover_generic" ]]; then
    need_concrete_optional_artifact cube_cover_certificate_path cube_cover_certificate_sha256 cube_cover_certificate
  fi
fi

if [[ "${FIELDS[promotable]}" == "1" ]]; then
  echo "chi6_manifest: VALID_PROMOTABLE_FORMAT candidate=${FIELDS[candidate_id]}"
else
  echo "chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=${FIELDS[candidate_id]}"
fi
