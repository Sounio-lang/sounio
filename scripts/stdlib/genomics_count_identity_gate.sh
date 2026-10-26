#!/usr/bin/env bash
# Actual native + independent C++ genomics contracts. Allocated Linux CPU only.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?fresh absolute output directory required}
[[ "$out" == /* && -x ${MADAROS_RAW_BIN:?identified raw Madaros required} && -x ${SEED_RAW_BIN:?identified raw committed seed required} ]] || exit 64
mkdir -p "$(dirname "$out")"
if ! mkdir "$out";then echo 'fresh output directory required' >&2;exit 65;fi
trap 'rc=$?;printf "%s\n" "$rc" > "$out/STATUS"' EXIT
cd "$root"
unset SOUNIO_SOUC_BIN
export SOUC_BIN="$MADAROS_RAW_BIN" SOUNIO_SOUC_ENGINE=madaros SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_MADAROS_NOCACHE=1
ulimit -s 524288
id > "$out/identity.txt"
sha256sum "$MADAROS_RAW_BIN" "$SEED_RAW_BIN" > "$out/COMPILERS.sha256"
find stdlib tests/stdlib/genomics_counts -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum tests/reference/{genomics_count_identity,genomics_count_scale,genomics_id,genomics_id_coalesce,genomics_token,count_tsv,normalization_into}_reference.cpp tests/run-pass/stdlib_scale_normalization_into.sio scripts/stdlib/genomics_count_identity_gate.sh >> "$out/INPUTS.sha256"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
printf 'control\texpected\tactual\n' > "$out/CONTROLS.tsv"
step(){ local name=$1;shift;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
expected(){ local name=$1 want=$2;shift 2;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\t%s\n' "$name" "$want" "$rc" >> "$out/CONTROLS.tsv";[[ $rc == "$want" ]]; }
no_diagnostics(){ ! grep -Ei '(^|[[:space:]])error(\[|:)|typecheck: failed|parse failed' "$@" > "$out/last-diagnostic-scan.txt"; }
build_both(){
 local name=$1 src=$2
 sha256sum "$src" >> "$out/CALLERS.sha256"
 step "$name.check" timeout 180 bin/souc check "$src"
 no_diagnostics "$out/$name.check.stdout" "$out/$name.check.stderr"
 step "$name.compile" timeout 180 bin/souc compile "$src" -o "$out/$name.elf"
 no_diagnostics "$out/$name.compile.stdout" "$out/$name.compile.stderr"
 chmod +x "$out/$name.elf";sha256sum "$out/$name.elf" >> "$out/ELFS.sha256"
 step "$name.native" /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 300 "$out/$name.elf"
 step "$name.seed-compile" timeout 180 "$SEED_RAW_BIN" "$src" "$out/$name.seed.elf"
 no_diagnostics "$out/$name.seed-compile.stdout" "$out/$name.seed-compile.stderr"
 chmod +x "$out/$name.seed.elf";sha256sum "$out/$name.seed.elf" >> "$out/ELFS.sha256"
 step "$name.seed-native" /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.seed.resource" timeout 300 "$out/$name.seed.elf"
}
step reference-toolchain g++ --version
for stem in genomics_count_identity genomics_count_scale genomics_id genomics_id_coalesce genomics_token count_tsv normalization_into;do
 step "$stem.reference-build" g++ -std=c++23 -O2 -Wall -Wextra -Werror "tests/reference/${stem}_reference.cpp" -o "$out/$stem.reference"
 sha256sum "$out/$stem.reference" >> "$out/REFERENCES.sha256"
 sha256sum "$out/$stem.reference" >> "$out/ELFS.sha256"
done
build_both token-slot tests/stdlib/genomics_counts/token_main.sio
build_both named-smoke tests/stdlib/genomics_counts/named_smoke.sio
build_both csr-smoke tests/stdlib/genomics_counts/csr_smoke.sio
build_both csr-guards tests/stdlib/genomics_counts/csr_guards.sio
build_both ids tests/stdlib/genomics_counts/ids_main.sio
build_both coalesce tests/stdlib/genomics_counts/ids_coalesce_main.sio
build_both dense-original tests/run-pass/stdlib_scale_normalization_into.sio
step named-generate "$out/genomics_count_identity.reference" --generate-named
cp "$out/named-generate.stdout" "$out/named-generated.sio"
build_both named "$out/named-generated.sio"
step canonical-token-summary "$out/count_tsv.reference"
step canonical-token-fixture "$out/count_tsv.reference" --fixture
step canonical-token-adapt "$out/genomics_token.reference" --adapt "$out/canonical-token-fixture.stdout" into
cp "$out/canonical-token-adapt.stdout" "$out/canonical-token.sio"
build_both canonical-token "$out/canonical-token.sio"
for engine in native seed-native;do
 step "canonical-token.$engine.concordance" "$out/genomics_token.reference" --check "$out/canonical-token-summary.stdout" "$out/canonical-token.$engine.stdout"
 step "named.$engine.concordance" "$out/genomics_count_identity.reference" --named "$out/named.$engine.stdout"
 step "csr-guards.$engine.concordance" "$out/genomics_count_identity.reference" --csr-guards "$out/csr-guards.$engine.stdout"
 step "csr.$engine.concordance" "$out/genomics_count_identity.reference" --csr "$out/csr-smoke.$engine.stdout"
 step "ids.$engine.concordance" "$out/genomics_id.reference" "$out/ids.$engine.stdout"
 step "coalesce.$engine.concordance" "$out/genomics_id_coalesce.reference" "$out/coalesce.$engine.stdout"
 step "dense-original.$engine.concordance" "$out/normalization_into.reference" "$out/dense-original.$engine.stdout"
 step "token-slot.$engine.marker" grep -Fx TOKEN_SLOT_PASS "$out/token-slot.$engine.stdout"
 step "named-smoke.$engine.marker" grep -Fx NAMED_COUNT_SMOKE_PASS "$out/named-smoke.$engine.stdout"
 step "csr-guards.$engine.marker" grep -Fx CSR_GUARDS_PASS "$out/csr-guards.$engine.stdout"
done
# Default scale qualification executes both representations and both engines.
# Comparative AB/BA/AB measurements have their own frozen Slurm receipt.
if [[ ${GENOMICS_SCALE_GATE:-1} == 1 ]];then
 for representation in dense csr;do
  step "$representation-scale.generate" "$out/genomics_count_scale.reference" --generate "$representation" 1
  cp "$out/$representation-scale.generate.stdout" "$out/$representation-scale.sio"
  build_both "$representation-scale" "$out/$representation-scale.sio"
  for engine in native seed-native;do
   step "$representation-scale.$engine.concordance" "$out/genomics_count_scale.reference" "$out/$representation-scale.$engine.stdout"
  done
 done
 grep '^COUNT_SCALE_FACTOR ' "$out/dense-scale.native.stdout" > "$out/dense-scale.factors"
 grep '^COUNT_SCALE_FACTOR ' "$out/csr-scale.native.stdout" > "$out/csr-scale.factors"
 step scale-bits cmp "$out/dense-scale.factors" "$out/csr-scale.factors"
fi
# Each corrupted numerical stream is separate from the verifier's logs.
awk '$1=="NAMED"&&$2==0&&$3==1 {$9=0} {print}' "$out/named.native.stdout" > "$out/mutant-named.input.stdout"
expected mutant-named.verify 8 "$out/genomics_count_identity.reference" --named "$out/mutant-named.input.stdout"
grep -F 'named framing/identity/count disagreement 0 1' "$out/mutant-named.verify.stderr"
sed '/^NAMED_ROWS_COMPLETE$/d' "$out/named.native.stdout" > "$out/mutant-completion.input.stdout"
expected mutant-completion.verify 9 "$out/genomics_count_identity.reference" --named "$out/mutant-completion.input.stdout"
awk '$1=="CSR_FACTOR"&&$2==1&&$3==0 {$4=0} {print}' "$out/csr-smoke.native.stdout" > "$out/mutant-csr.input.stdout"
expected mutant-csr.verify 8 "$out/genomics_count_identity.reference" --csr "$out/mutant-csr.input.stdout"
grep -F 'CSR independent median ratio disagreement' "$out/mutant-csr.verify.stderr"
awk '$1=="CSR_GUARD"&&$2==1 {$3=0} {print}' "$out/csr-guards.native.stdout" > "$out/mutant-guard.input.stdout"
expected mutant-guard.verify 8 "$out/genomics_count_identity.reference" --csr-guards "$out/mutant-guard.input.stdout"
grep -F 'CSR guard status disagreement 1' "$out/mutant-guard.verify.stderr"
# Structural qualification is engine-specific. The raw committed bootstrap
# rejects repeated linear consumption but accepts these reachable private reads,
# including the unchanged pre-existing IntegerBuffer.size witness. Preserve the
# limitation explicitly; it is not counted as seed privacy enforcement.
printf 'source\tlimitation\tactual_exit\n' > "$out/SEED_LIMITATIONS.tsv"
for owner in buffer csr;do
 for probe in private linear;do
  src="tests/stdlib/genomics_counts/"$owner"_"$probe"_probe.sio"
  name="$owner-$probe"
  expected "$name.check" 1 timeout 180 bin/souc check "$src"
  if [[ $probe == private ]];then
   grep -F 'error[E259]' "$out/$name.check.stdout" "$out/$name.check.stderr"
   grep -F 'struct field is private in its defining module' "$out/$name.check.stdout" "$out/$name.check.stderr"
   expected "$name.seed-read-accepted" 0 timeout 180 "$SEED_RAW_BIN" "$src" "$out/$name.seed-unqualified.elf"
   grep -F 'compile: fns=' "$out/$name.seed-read-accepted.stdout"
   [[ -s "$out/$name.seed-unqualified.elf" ]]
   sha256sum "$out/$name.seed-unqualified.elf" >> "$out/ELFS.sha256"
   printf '%s\tPRIVATE_READ_NOT_ENFORCED_BY_BOOTSTRAP\t0\n' "$src" >> "$out/SEED_LIMITATIONS.tsv"
  else
   grep -E 'error\[E039\]|error\[E040\]' "$out/$name.check.stdout" "$out/$name.check.stderr"
   expected "$name.seed-check" 1 timeout 180 "$SEED_RAW_BIN" "$src" "$out/$name.seed-must-not-exist.elf"
   grep -F 'linear value used more than once' "$out/$name.seed-check.stdout" "$out/$name.seed-check.stderr"
  fi
 done
done
printf 'QUALIFICATION_SCOPE=both-numerical;canonical-private;both-linear;bootstrap-private-read-UNQUALIFIED\n'
step inputs-integrity sha256sum -c "$out/INPUTS.sha256"
step compilers-integrity sha256sum -c "$out/COMPILERS.sha256"
step references-integrity sha256sum -c "$out/REFERENCES.sha256"
step callers-integrity sha256sum -c "$out/CALLERS.sha256"
step executable-integrity sha256sum -c "$out/ELFS.sha256"
printf 'GENOMICS_COUNT_IDENTITY_GATE_PASS\n'
