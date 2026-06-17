#!/usr/bin/env bash
# Promotable-only Lean gate for chi>=6 candidate manifests.
#
# The lightweight manifest validator checks hashes and fail-closed metadata.
# This script adds the kernel-facing check: it imports the candidate module and
# asks Lean to type-check the named Euclidean geometry, no-five witness, and
# final no-five-colouring theorem at the concrete manifest arity.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <candidate.manifest>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
MANIFEST="$1"
MANIFEST_DIR="$(cd "$(dirname "$MANIFEST")" && pwd)"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

"$VALIDATOR" "$MANIFEST"

declare -A FIELDS
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  [[ "$line" == *=* ]] || continue
  key="${line%%=*}"
  val="${line#*=}"
  FIELDS[$key]="$val"
done < "$MANIFEST"

need() {
  local key="$1"
  if [[ -z "${FIELDS[$key]+x}" || -z "${FIELDS[$key]}" || "${FIELDS[$key]}" == "NONE" ]]; then
    echo "error: promotable Lean gate requires $key" >&2
    exit 2
  fi
}

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$MANIFEST_DIR/$p"
  fi
}

[[ "${FIELDS[promotable]:-}" == "1" ]] || {
  echo "error: promotable Lean gate requires promotable=1" >&2
  exit 2
}

for key in \
  n lean_sat_module_path geometry_module_path lean_module lean_sat_edges_term lean_point_type lean_unit_term \
  lean_geometry_term lean_edges_sync_term lean_no_five_witness_term lean_final_theorem \
  lean_real_unit_term lean_real_unit_iff_standard lean_real_final_theorem
do
  need "$key"
done

SAT_ABS="$(resolve_path "${FIELDS[lean_sat_module_path]}")"
GEOM_ABS="$(resolve_path "${FIELDS[geometry_module_path]}")"
[[ -s "$SAT_ABS" ]] || { echo "error: missing Lean SAT module: $SAT_ABS" >&2; exit 2; }
[[ -s "$GEOM_ABS" ]] || { echo "error: missing Lean geometry module: $GEOM_ABS" >&2; exit 2; }

module_name_from_formal_lean_path() {
  local abs="$1"
  local formal_root canon
  formal_root="$(readlink -f "$ROOT/formal/lean4")"
  canon="$(readlink -f "$abs")" || {
    echo "error: cannot canonicalize Lean module path: $abs" >&2
    exit 2
  }
  case "$canon" in
    "$formal_root"/*.lean)
      local rel="${canon#$formal_root/}"
      rel="${rel%.lean}"
      printf '%s\n' "${rel//\//.}"
      ;;
    *)
      echo "error: Lean module must live under formal/lean4: $abs" >&2
      exit 2
      ;;
  esac
}

SAT_MODULE="$(module_name_from_formal_lean_path "$SAT_ABS")"

if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
mkdir -p "$WORK"
VERIFY="$WORK/Chi6PromotableCandidateVerify.lean"
LEANLIB="$WORK/leanlib"
rm -rf "$LEANLIB"
mkdir -p "$LEANLIB"
SAT_OLEAN="$LEANLIB/${SAT_MODULE//.//}.olean"
GEOM_OLEAN="$LEANLIB/${FIELDS[lean_module]//.//}.olean"
mkdir -p "$(dirname "$SAT_OLEAN")" "$(dirname "$GEOM_OLEAN")"
LEAN_PATH_WITH_WORK="$LEANLIB${LEAN_PATH:+:$LEAN_PATH}"

cat > "$VERIFY" <<EOF
import SounioFiniteUnitDistanceWitness
import SounioRootedFieldReal
import SounioMultiquadIndep
import SounioRealPlaneGeometry
import ${SAT_MODULE}
import ${FIELDS[lean_module]}

open UnitDistanceChromatic
open SounioSqrt.RealCauchyField

#check (${FIELDS[lean_geometry_term]} :
  EuclideanNatEdgeExactGeometry ${FIELDS[n]} ${FIELDS[lean_point_type]} ${FIELDS[lean_unit_term]})
#check (${FIELDS[lean_sat_edges_term]} :
  List (Nat × Nat))
#check (${FIELDS[lean_edges_sync_term]} :
  (${FIELDS[lean_geometry_term]}).exact.edges = ${FIELDS[lean_sat_edges_term]})
theorem chi6EdgesSyncByComputation :
    (${FIELDS[lean_geometry_term]}).exact.edges = ${FIELDS[lean_sat_edges_term]} := by
  native_decide
#check (chi6EdgesSyncByComputation :
  (${FIELDS[lean_geometry_term]}).exact.edges = ${FIELDS[lean_sat_edges_term]})
#check (${FIELDS[lean_no_five_witness_term]} :
  NatEdgeUnitDistanceCertificate.NoFiveColourWitness ${FIELDS[n]} ${FIELDS[lean_point_type]} ${FIELDS[lean_unit_term]})
#check (${FIELDS[lean_final_theorem]} :
  Not (Nonempty (PlaneColouring ${FIELDS[lean_point_type]} ${FIELDS[lean_unit_term]} 5)))
#check (${FIELDS[lean_real_unit_term]} :
  Real × Real → Real × Real → Prop)
#check (${FIELDS[lean_real_unit_iff_standard]} :
  ∀ p q : Real × Real,
    ${FIELDS[lean_real_unit_term]} p q ↔ standardRealPlaneDist2 p q = qR (1 : Rat))
#check (${FIELDS[lean_real_final_theorem]} :
  Not (Nonempty (PlaneColouring (Real × Real) ${FIELDS[lean_real_unit_term]} 5)))

theorem chi6StandardRealNoFiveColouring :
    Not (Nonempty (PlaneColouring (Real × Real) standardRealPlaneUnit 5)) := by
  intro h
  rcases h with ⟨pc⟩
  exact ${FIELDS[lean_real_final_theorem]} ⟨⟨pc.1, by
    intro p q hpq
    exact pc.2 p q (by
      show standardRealPlaneUnit p q
      exact ((${FIELDS[lean_real_unit_iff_standard]}) p q).mp hpq)⟩⟩

#check (chi6StandardRealNoFiveColouring :
  Not (Nonempty (PlaneColouring (Real × Real) standardRealPlaneUnit 5)))

#print axioms ${FIELDS[lean_geometry_term]}
#print axioms ${FIELDS[lean_edges_sync_term]}
#print axioms chi6EdgesSyncByComputation
#print axioms ${FIELDS[lean_no_five_witness_term]}
#print axioms ${FIELDS[lean_final_theorem]}
#print axioms ${FIELDS[lean_real_unit_iff_standard]}
#print axioms ${FIELDS[lean_real_final_theorem]}
#print axioms chi6StandardRealNoFiveColouring
EOF

if rg -q '\b(sorry|admit)\b|#exit' "$VERIFY"; then
  echo "error: generated promotable verifier contains sorry/admit/#exit" >&2
  exit 1
fi

if ! (
  cd "$ROOT/formal/lean4"
  "$LAKE" env lean -o "$SAT_OLEAN" "$SAT_ABS" > "$WORK/sat.out" 2> "$WORK/sat.err"
  LEAN_PATH="$LEAN_PATH_WITH_WORK" \
    "$LAKE" env lean -o "$GEOM_OLEAN" "$GEOM_ABS" > "$WORK/geometry.out" 2> "$WORK/geometry.err"
  LEAN_PATH="$LEAN_PATH_WITH_WORK" \
    "$LAKE" env lean "$VERIFY" > "$WORK/verifier.out" 2> "$WORK/verifier.err"
); then
  cat "$WORK/sat.out" "$WORK/sat.err" "$WORK/geometry.out" "$WORK/geometry.err" \
    "$WORK/verifier.out" "$WORK/verifier.err" >&2
  exit 1
fi

if rg -q 'error:' "$WORK/sat.out" "$WORK/sat.err" "$WORK/geometry.out" "$WORK/geometry.err" \
    "$WORK/verifier.out" "$WORK/verifier.err"; then
  cat "$WORK/sat.out" "$WORK/sat.err" "$WORK/geometry.out" "$WORK/geometry.err" \
    "$WORK/verifier.out" "$WORK/verifier.err" >&2
  exit 1
fi
if rg -q 'sorryAx' "$WORK/verifier.out" "$WORK/verifier.err"; then
  cat "$WORK/verifier.out" "$WORK/verifier.err" >&2
  echo "error: promotable verifier reports sorryAx" >&2
  exit 1
fi

AXIOM_LINES="$WORK/axiom-lines.txt"
awk '
  /depends on axioms:/ {
    in_axioms = 1
    sub(/^.*depends on axioms:[[:space:]]*/, "")
    print
    if ($0 ~ /\]/) in_axioms = 0
    next
  }
  in_axioms {
    print
    if ($0 ~ /\]/) in_axioms = 0
  }
' "$WORK/verifier.out" "$WORK/verifier.err" > "$AXIOM_LINES"

UNEXPECTED_AXIOMS="$WORK/unexpected-axioms.txt"
rg -o "[A-Za-z_][A-Za-z0-9_']*(\\.[A-Za-z_][A-Za-z0-9_']*)*" "$AXIOM_LINES" \
  | rg -v '^(propext|Quot\.sound|Classical\.choice|choice|[A-Za-z_][A-Za-z0-9_'\'']*(\.[A-Za-z_][A-Za-z0-9_'\'']*)*\._native\.native_decide\.ax_[A-Za-z0-9_'\'']*)$' \
  | sort -u > "$UNEXPECTED_AXIOMS" || true
if [[ -s "$UNEXPECTED_AXIOMS" ]]; then
  cat "$WORK/verifier.out" "$WORK/verifier.err" >&2
  echo "error: promotable verifier reports unexpected axiom dependencies:" >&2
  cat "$UNEXPECTED_AXIOMS" >&2
  exit 1
fi

echo "chi6_promotable_candidate: PASS candidate=${FIELDS[candidate_id]:-UNKNOWN}"
