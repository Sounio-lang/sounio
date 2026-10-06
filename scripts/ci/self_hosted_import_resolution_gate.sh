#!/usr/bin/env bash
# Every `use a::b::...` in self-hosted/ must name a module file that is
# committed. On 2026-06-15 (5c85634f3) two K-AXI dispatchers started importing
# gpu::erdos90_hc_smoke_emit, a file that only ever existed on an off-main
# snapshot (b8828063d6). Nothing noticed for four months; the K-AXI -> PTX golden
# gate went from 318 PASS to 318 FAIL (all dropped, not mismatched), because the
# driver could not compile. This gate is the cheap check that would have caught
# it on the first push: module resolution only, no compiler run.
#
# Resolution mirrors the module loader for self-hosted sources: some prefix of
# a::b::c must name <root>/a/b.sio, <root>/a/b/c.sio or <dir>/mod.sio, with root
# self-hosted/, stdlib/ or the importing file's own directory. Paths are taken from `git ls-files`, so an untracked file
# on the developer's disk does not satisfy the gate.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9

tracked="$(mktemp)"; trap 'rm -f "$tracked"' EXIT
git ls-files -- 'self-hosted/*.sio' 'stdlib/*.sio' > "$tracked"

# Known dangling, shrink-only. Each entry is a file that nothing imports (no
# driver, script or Makefile reaches it), so its broken imports cannot break a
# build; it is listed so the gate is honest about it rather than silent.
#   epistemic_tensor_core.sio: imports hardware::ptx/rtl/qir and
#     biochem::glycolysis_10step_atom_level, none committed (measured 2026-10-05;
#     only reference is scripts/ci/fixtures/madaros_self_parse_baseline.txt).
KNOWN_DANGLING=(
  self-hosted/gpu/epistemic_tensor_core.sio
)
is_known_dangling() { local f; for f in "${KNOWN_DANGLING[@]}"; do [[ "$1" == "$f" ]] && return 0; done; return 1; }

missing=0
checked=0
known=0
while IFS=: read -r file line rest; do
  mod="$(sed -E 's/^[[:space:]]*(pub[[:space:]]+)?use[[:space:]]+//; s/[[:space:]]*(\{.*|;.*|\*.*)?$//; s/::$//' <<<"$rest")"
  # A module path names a file by some prefix of its segments (a::b, or
  # a::b::c for nested directories such as gpu::opt::fusion); the rest are items.
  IFS=':' read -ra seg <<<"${mod//::/:}"
  [[ ${#seg[@]} -ge 2 ]] || continue
  case "${seg[0]}" in self|super|crate|std) continue ;; esac
  checked=$((checked + 1))
  found=0
  dir="$(dirname "$file")"
  # The first segment alone never counts: self-hosted/gpu/mod.sio exists, and
  # matching on it would accept any gpu::<anything>, including the import this
  # gate was written for.
  prefix="${seg[0]}"
  for s in "${seg[@]:1}"; do
    prefix="$prefix/$s"
    for root in self-hosted stdlib "$dir"; do
      if grep -qxF -e "$root/$prefix.sio" -e "$root/$prefix/mod.sio" "$tracked"; then found=1; break 2; fi
    done
  done
  a="${seg[0]}"; b="${seg[1]}"
  if [[ $found -eq 0 ]] && is_known_dangling "$file"; then
    known=$((known + 1))
    continue
  fi
  if [[ $found -eq 0 ]]; then
    echo "  $file:$line: use $mod — no committed module file" >&2
    missing=$((missing + 1))
  fi
done < <(git grep -n -E '^[[:space:]]*(pub[[:space:]]+)?use[[:space:]]+[a-z_][a-z0-9_]*::' -- 'self-hosted/gpu/*.sio')

if [[ $missing -gt 0 ]]; then
  echo "self_hosted_import_resolution_gate: FAIL ($missing of $checked imports in self-hosted/gpu name no committed file)" >&2
  exit 1
fi
echo "self_hosted_import_resolution_gate: PASS ($((checked - known)) of $checked imports in self-hosted/gpu resolve to committed files; $known in ${#KNOWN_DANGLING[@]} known-dangling unreachable file(s))"
