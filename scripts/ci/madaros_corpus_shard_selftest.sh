#!/usr/bin/env bash
# Exercise the real inventory, dispatch, and baseline comparison using a compiler
# that rejects each fixture. No native compiler or platform runner is needed.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT
cd "$fixture"
mkdir -p scripts/ci tests/run-pass
gate=scripts/ci/madaros_corpus_regression_gate.sh
cp "$root/$gate" "$gate"
for name in a b c d e f g h i; do
    printf 'fn main() -> i32 { return 0 }\n' > "tests/run-pass/$name.sio"
    printf '%s.sio compile\n' "$name" >> tests/madaros_corpus_baseline.txt
done
printf '//@ ignore\nfn main() -> i32 { return 0 }\n' > tests/run-pass/ignored.sio
printf 'fn library() -> i32 { return 0 }\n' > tests/run-pass/library.sio
cat > compiler <<'COMPILER'
#!/usr/bin/env bash
[[ "$1" != --version ]] || { echo fixture; exit 0; }
echo "${2##*/}" >> "$SEEN"
exit "${COMPILE_STATUS:-1}"
COMPILER
chmod +x compiler
export SOUNIO_MADAROS_CORPUS_BIN="$PWD/compiler" SOUNIO_TEST_JOBS=2
export SOUNIO_MADAROS_CORPUS_STACK_KB="$(ulimit -s)" SEEN="$PWD/seen"
run_shard() {
    SOUNIO_MADAROS_CORPUS_SHARD="$1" SOUNIO_MADAROS_CORPUS_SHARDS="$2" bash "$gate"
}
bash "$gate" > all.log
sort seen > expected
[[ $(wc -l < expected) -eq 9 ]]
: > seen
for shard in {0..7}; do
    run_shard "$shard" 8 > "shard-$shard.log"
    grep -q '0 newly fixed' "shard-$shard.log"
done
sort seen > actual
cmp expected actual
reject() {
    local reason=$1
    shift
    if "$@" > refusal.log 2>&1; then echo "unexpected success: $*" >&2; exit 1; fi
    grep -Fq -- "$reason" refusal.log || { cat refusal.log >&2; exit 1; }
}
reject 'invalid shard' run_shard 8 8
reject 'invalid shard' run_shard 0 0
SOUNIO_MADAROS_CORPUS_SHARDS=8 reject 'set both shard index and count' bash "$gate"
cp tests/madaros_corpus_baseline.txt baseline.before
SOUNIO_MADAROS_CORPUS_REFRESH=1 reject 'full corpus' run_shard 0 8
cmp baseline.before tests/madaros_corpus_baseline.txt
COMPILE_STATUS=137 reject 'killed mid-run' run_shard 0 8
sed '/^a.sio /d' baseline.before > tests/madaros_corpus_baseline.txt
reject 'a.sio compile' run_shard 0 8
echo MADAROS_CORPUS_SHARD_SELFTEST_OK
