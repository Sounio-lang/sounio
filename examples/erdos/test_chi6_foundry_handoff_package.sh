#!/usr/bin/env bash
# Gate the local, non-submitting chi>=6 Foundry/Slurm handoff package.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

EXTERNAL_MAKER="$ROOT/examples/erdos/make_chi6_external_cube_cover_candidate_manifest.sh"
HANDOFF_MAKER="$ROOT/examples/erdos/make_chi6_foundry_handoff_package.sh"
HANDOFF_VALIDATOR="$ROOT/examples/erdos/validate_chi6_foundry_handoff_package.sh"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
mkdir -p "$WORK"

EDGE="$WORK/k6_external.edge"
cat > "$EDGE" <<'EOF'
p edge 6 15
e 1 2
e 1 3
e 1 4
e 1 5
e 1 6
e 2 3
e 2 4
e 2 5
e 2 6
e 3 4
e 3 5
e 3 6
e 4 5
e 4 6
e 5 6
EOF

echo "chi6_foundry_handoff_package_gate: workdir=$WORK"
PKG_SOURCE="$WORK/source"
WORK="$PKG_SOURCE" "$EXTERNAL_MAKER" "$EDGE" k6_foundry_handoff_smoke 0 \
  > "$WORK/external-maker.out"
rg -q '^chi6_external_cube_cover_candidate: PASS$' "$WORK/external-maker.out"

HANDOFF_OUT="$WORK/handoff"
"$HANDOFF_MAKER" "$PKG_SOURCE/candidate.manifest" "$HANDOFF_OUT" \
  > "$WORK/handoff-maker.out"
rg -q '^chi6_foundry_handoff_package: PASS ' "$WORK/handoff-maker.out"

HANDOFF="$HANDOFF_OUT/handoff.txt"
SUMS="$HANDOFF_OUT/SHA256SUMS"
MANIFEST="$HANDOFF_OUT/chi6-package/candidate.manifest"
[[ -s "$HANDOFF" ]] || { echo "error: missing handoff.txt" >&2; exit 1; }
[[ -s "$SUMS" ]] || { echo "error: missing SHA256SUMS" >&2; exit 1; }
[[ -s "$MANIFEST" ]] || { echo "error: missing packaged candidate.manifest" >&2; exit 1; }

rg -q '^Heavy Validation Handoff$' "$HANDOFF"
rg -q '^gate_requested: chi6 candidate package validation / chi6 SAT\+Lean replay$' "$HANDOFF"
rg -q '^package_root: chi6-package$' "$HANDOFF"
rg -q '^repo_root_seen: /workspace/sounio$' "$HANDOFF"
rg -q '^host_repo_root: /home/devsounio/projects/sounio$' "$HANDOFF"
rg -q '^chi6_candidate_id: k6_foundry_handoff_smoke$' "$HANDOFF"
rg -q '^chi6_manifest_path: chi6-package/candidate\.manifest$' "$HANDOFF"
rg -q '^chi6_manifest_sha256: [0-9a-f]{64}$' "$HANDOFF"
rg -q '^sat_proof_route: cube_cover_generic$' "$HANDOFF"
rg -q '^promotable: 0$' "$HANDOFF"
rg -q '^geometry_proof_type: none$' "$HANDOFF"
rg -q '^source_meta_path: package/k6_foundry_handoff_smoke\.meta\.json$' "$HANDOFF"
rg -q '^source_meta_sha256: [0-9a-f]{64}$' "$HANDOFF"
rg -q '^artifact_sha256s_path: SHA256SUMS$' "$HANDOFF"
rg -q '^artifact_sha256s_sha256: [0-9a-f]{64}$' "$HANDOFF"
rg -q '^host_replay_commands: cd /home/devsounio/projects/sounio && sha256sum -c .*/SHA256SUMS && examples/erdos/validate_chi6_candidate_manifest.sh .*/chi6-package/candidate.manifest$' \
  "$HANDOFF"
rg -q '^trust_boundary: local package/format/hash validation only; no Slurm execution; no Euclidean chi>=6 claim unless promotable=1 plus Lean/offload gates pass$' \
  "$HANDOFF"

(cd "$HANDOFF_OUT" && sha256sum -c SHA256SUMS > "$WORK/sha256sum.out")
"$VALIDATOR" "$MANIFEST" > "$WORK/validator.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_foundry_handoff_smoke$' \
  "$WORK/validator.out"
"$HANDOFF_VALIDATOR" "$HANDOFF_OUT" > "$WORK/handoff-validator.out"
rg -q '^chi6_foundry_handoff_package: VALID candidate=k6_foundry_handoff_smoke promotable=0$' \
  "$WORK/handoff-validator.out"

rg -q '^chi6-package/candidate\.manifest$' <(awk '{print $2}' "$SUMS")
rg -q '^chi6-package/package/k6_foundry_handoff_smoke\.edge$' <(awk '{print $2}' "$SUMS")
rg -q '^chi6-package/package/k6_foundry_handoff_smoke\.meta\.json$' <(awk '{print $2}' "$SUMS")
rg -q '^chi6-package/SounioSatChi6ExternalCubeCoverReflect\.lean$' <(awk '{print $2}' "$SUMS")

BAD="$WORK/bad_handoff"
cp -a "$HANDOFF_OUT" "$BAD"
printf '\n# corruption\n' >> "$BAD/chi6-package/candidate.manifest"
if (cd "$BAD" && sha256sum -c SHA256SUMS > "$WORK/bad-sha.out" 2>&1); then
  echo "error: SHA256SUMS accepted a corrupted packaged manifest" >&2
  exit 1
fi
rg -q 'candidate.manifest: FAILED' "$WORK/bad-sha.out"

if "$HANDOFF_VALIDATOR" "$BAD" > "$WORK/bad-validator.out" 2>&1; then
  echo "error: handoff package validator accepted corrupted package" >&2
  exit 1
fi
rg -q 'chi6_manifest_sha256 mismatch|FAILED' "$WORK/bad-validator.out"

BAD_SUMS_FRESH="$WORK/bad_sums_fresh"
cp -a "$HANDOFF_OUT" "$BAD_SUMS_FRESH"
printf '\n# corruption\n' >> "$BAD_SUMS_FRESH/chi6-package/candidate.manifest"
(
  cd "$BAD_SUMS_FRESH"
  find chi6-package -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
new_sums_sha="$(sha256sum "$BAD_SUMS_FRESH/SHA256SUMS" | awk '{print $1}')"
sed -i "s/^artifact_sha256s_sha256: .*/artifact_sha256s_sha256: $new_sums_sha/" \
  "$BAD_SUMS_FRESH/handoff.txt"
if "$HANDOFF_VALIDATOR" "$BAD_SUMS_FRESH" > "$WORK/bad-sums-fresh.out" 2>&1; then
  echo "error: handoff package validator accepted stale chi6_manifest_sha256" >&2
  exit 1
fi
rg -q 'chi6_manifest_sha256 mismatch' "$WORK/bad-sums-fresh.out"

BAD_FIELD="$WORK/bad_field"
cp -a "$HANDOFF_OUT" "$BAD_FIELD"
sed -i 's/^sat_proof_route: .*/sat_proof_route: none/' "$BAD_FIELD/handoff.txt"
if "$HANDOFF_VALIDATOR" "$BAD_FIELD" > "$WORK/bad-field.out" 2>&1; then
  echo "error: handoff package validator accepted handoff/manifest route mismatch" >&2
  exit 1
fi
rg -q 'handoff/manifest mismatch for sat_proof_route' "$WORK/bad-field.out"

echo "chi6_foundry_handoff_package_gate: PASS"
