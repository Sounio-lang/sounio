#!/usr/bin/env bash
# Exact native demonstration resource observation, no compilation or tests.
set -euo pipefail
[[ $# == 6 && "$1" == --filter && "$2" == basic_math && "$3" == --format && "$4" == junit && "$5" == --jobs && "$6" == 2 ]] || exit 78
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
: "${SOUNIO_SCRATCH:?Forge scratch required}"
: "${SOUNIO_ARTIFACTS:?Forge artifacts required}"
printf '%s\n' 'purpose=nb2-exact-native-elf-resource-observation' 'corpus_acceptance=false' 'numerical_acceptance=false' 'comparative_benchmark=false'
expected=$(head -n 1 native-input/PROGRAM_IDENTITIES.sha256 | cut -d ' ' -f 1)
[[ "$expected" =~ ^[a-f0-9]{64}$ ]]
program="$SOUNIO_SCRATCH/nb2-diagnostic.elf"
gzip -dc native-input/nb2-diagnostic.elf.gz > "$program"
chmod +x "$program"
printf '%s  %s\n' "$expected" "$program" | sha256sum --check
out="$SOUNIO_ARTIFACTS/profile"
mkdir -p "$out"
cp native-input/PROGRAM_IDENTITIES.sha256 "$out/"
LC_ALL=C /usr/bin/time --version > "$out/TIME_VERSION.txt"
ulimit -s 524288
printf 'EXACT_NATIVE_PROFILE_BEGIN\n'
set +e
LC_ALL=C /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/RESOURCE.txt" "$program" > "$out/nb2-diagnostic.stdout" 2> "$out/native-stderr.txt"
program_status=$?
set -e
printf 'exit_code=%s\n' "$program_status" > "$out/EXECUTION.txt"
cat "$out/RESOURCE.txt"
sha256sum "$program" "$out/nb2-diagnostic.stdout"
[[ "$program_status" == 0 ]] || exit "$program_status"
printf 'EXACT_NATIVE_PROFILE_COMPLETE\n'
