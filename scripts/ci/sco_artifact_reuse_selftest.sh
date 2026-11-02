#!/usr/bin/env bash
# Orchestration fixtures only; no compiler or scientific result is simulated as validation.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
repo="$work/repo"
mkdir -p "$repo"/{scripts/ci,self-hosted,stdlib,bin}
cp "$root/scripts/ci/sco_step4_runtime_gate.sh" "$root/scripts/ci/stdlib_deep_compiler_receipt.sh" "$repo/scripts/ci/"
printf '// fixture\n' > "$repo/self-hosted/a.sio"
printf '// fixture\n' > "$repo/stdlib/a.sio"
for gate in sco_corpus_gate sco_strategy_probe_gate; do
  printf '#!/usr/bin/env bash\nset -eu\ntest -x "${SOUC:-${SCO_MADAROS:?}}"\nprintf "%%s\\n" "%s" >> "$SCO_FIXTURE_CALLS"\n' "$gate" > "$repo/scripts/ci/$gate.sh"
done
printf '#!/usr/bin/env bash\necho unexpected-build > "$SCO_FIXTURE_BUILD"\nexit 99\n' > "$repo/scripts/ci/build_modular_madaros.sh"
printf '#!/usr/bin/env bash\nexit 0\n' > "$repo/bin/compiler-fixture"
chmod +x "$repo/bin/compiler-fixture"
git -C "$repo" init -q
git -C "$repo" -c user.name='CI Fixture' -c user.email='ci-fixture@example.invalid' add .
git -C "$repo" -c user.name='CI Fixture' -c user.email='ci-fixture@example.invalid' commit -qm fixture
sha=$(git -C "$repo" rev-parse HEAD)
bash "$repo/scripts/ci/stdlib_deep_compiler_receipt.sh" prepare "$repo" "$repo/bin/compiler-fixture" "$work/artifact" "$sha"
export SCO_FIXTURE_CALLS="$work/calls" SCO_FIXTURE_BUILD="$work/unexpected-build"
run_case() {
  local name=$1
  SCO_COMPILER_ARTIFACT="$work/artifact" SCO_MADAROS="$work/$name/madaros" SCO_OUTPUT_DIR="$work/$name" \
    bash "$repo/scripts/ci/sco_step4_runtime_gate.sh" > "$work/$name.log" 2>&1
}
run_case valid
cmp "$repo/bin/compiler-fixture" "$work/valid/madaros"
test -s "$work/valid/madaros.sco-source.sha256"
grep -qx sco_corpus_gate "$SCO_FIXTURE_CALLS"
grep -qx sco_strategy_probe_gate "$SCO_FIXTURE_CALLS"
printf 'tampered\n' >> "$work/artifact/madaros.elf"
if run_case tampered; then echo 'accepted tampered artifact' >&2; exit 1; fi
test ! -e "$work/tampered/madaros"
test ! -e "$SCO_FIXTURE_BUILD"
echo SCO_ARTIFACT_REUSE_SELFTEST_PASS
