#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MSL="${1:?usage: pireus_apple_metal_xor_materializer_gate.sh <generated.metal>}"
SELECTION="tools/pireus/cross_arch_candidates.values.v1"
SEMANTICS="tools/pireus/xor_semantics.values.v1"
BASIS_SEMANTICS="tools/pireus/xor_basis4_semantics.values.v1"
RECEIPT="tools/pireus/evidence/pireus_apple_metal_xor_material.receipt.v1"

[[ -s "$MSL" ]] || { echo "FAIL: missing generated MSL: $MSL" >&2; exit 1; }
[[ "$(wc -l < "$SEMANTICS" | tr -d ' ')" == 16 ]] || {
  echo "FAIL: frozen Sounio semantics must contain 16 lanes" >&2
  exit 1
}
grep -qF 'selected=apple-silicon-metal/xor-shuffle+sign-xor+twofold-mul+compensated-reduce equivalence=APPROXIMATE' "$SELECTION" || {
  echo "FAIL: ontology did not select the Apple float2 compensated recipe" >&2
  exit 1
}
grep -qF 'kernel void sedenion_xor_product' "$MSL"
grep -qF 'pireus.recipe=xor-shuffle+sign-xor+twofold-mul+compensated-reduce' "$MSL"
grep -qF 'pireus.equivalence=APPROXIMATE pireus.storage=float2-hi-lo' "$MSL"
grep -qF 'pireus.source_semantics=f64x16 pireus.material_abi=float2x16' "$MSL"
grep -qF 'pireus.requires_pack=f64-to-twofold-float2' "$MSL"
grep -qF 'constant char pireus_xor_sign[256]' "$MSL"
grep -qF 'uint __ai = tid ^ __j;' "$MSL"
grep -qF 'fma(__av.x, __bv.x, -__p)' "$MSL"
if awk '/return;/{ret=NR} /pireus.recipe=/{recipe=NR} END{exit !(ret && recipe && ret < recipe)}' "$MSL"; then
  echo "FAIL: generated MSL returns before the Pireus materializer" >&2
  exit 1
fi
if grep -qF '// unhandled opcode' "$MSL"; then
  echo "FAIL: generated MSL contains an unhandled opcode" >&2
  exit 1
fi

sign_metrics="$(sed -n 's/^constant char pireus_xor_sign\[256\] = {\(.*\)};/\1/p' "$MSL" | awk -F, '
function bxor(a, b,    bit, out) {
  bit = 1; out = 0
  while (a > 0 || b > 0) {
    if ((a % 2) != (b % 2)) out += bit
    a = int(a / 2); b = int(b / 2); bit *= 2
  }
  return out
}
{
  negative = 0; checksum = 0
  for (k = 1; k <= NF; k++) {
    sign = $k + 0
    d = int((k - 1) / 16); j = (k - 1) % 16; a = bxor(d, j)
    if (sign < 0) negative++
    checksum += (a * 16 + j + 1) * sign * (d + 1)
  }
  print NF, negative, checksum
}')"
read -r sign_count negative_count sign_checksum <<<"$sign_metrics"
[[ "$sign_count" == 256 && "$negative_count" == 120 && "$sign_checksum" == 21336 ]] || {
  echo "FAIL: emitted sign table diverges from frozen Sounio basis semantics" >&2
  exit 1
}
grep -q '^basis_pairs=256$' "$BASIS_SEMANTICS"
grep -q '^negative_pairs=120$' "$BASIS_SEMANTICS"
grep -q '^signed_lane_checksum=21336$' "$BASIS_SEMANTICS"
msl_sha="$(shasum -a 256 "$MSL" | awk '{print $1}')"
grep -qx "msl_sha256=$msl_sha" "$RECEIPT"
grep -qx 'material_role=MATERIAL_PARITY' "$RECEIPT"
grep -qx 'equivalence=APPROXIMATE' "$RECEIPT"
grep -qx 'result=PASS' "$RECEIPT"

if [[ "${SOUNIO_PIREUS_METAL_STRUCTURAL_ONLY:-0}" == 1 ]]; then
  echo "PIREUS_APPLE_METAL_XOR_STRUCTURAL_PASS msl_sha256=$msl_sha"
  exit 0
fi

command -v xcrun >/dev/null || { echo "FAIL: xcrun unavailable; Apple runtime is required" >&2; exit 1; }
command -v swiftc >/dev/null || { echo "FAIL: swiftc unavailable; Apple runtime is required" >&2; exit 1; }
bash scripts/gpu-metal-validation/run_pireus_xor_metal.sh "$MSL"
echo "PIREUS_APPLE_METAL_XOR_MATERIALIZER_GATE_PASS msl_sha256=$msl_sha"
