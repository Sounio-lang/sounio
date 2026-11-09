#!/usr/bin/env bash
# Structural contract only; this gate cannot qualify directed arithmetic.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
out="${INTERVAL_CONTRACT_OUT:?set an absolute fresh output directory}"
[[ "$out" = /* && ! -e "$out" ]]
mkdir -p "$out"
trap 'rc=$?; printf "%s\n" "$rc" > "$out/STATUS"' EXIT
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" > "$out/compiler.sha256"
sha256sum tests/compile-fail/interval_contract_private.sio stdlib/epistemic/interval_contract.sio tests/run-pass/stdlib_interval_contract.sio tests/reference/interval_contract_reference.cpp scripts/stdlib/interval_contract_gate.sh > "$out/source.sha256"
bin/souc check tests/run-pass/stdlib_interval_contract.sio >"$out/check.stdout" 2>"$out/check.stderr"
bin/souc compile tests/run-pass/stdlib_interval_contract.sio -o "$out/contract.elf" >"$out/compile.stdout" 2>"$out/compile.stderr"
chmod +x "$out/contract.elf"
sha256sum "$out/contract.elf" > "$out/executable.sha256"
timeout 60 "$out/contract.elf" >"$out/native.stdout" 2>"$out/native.stderr"
g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/interval_contract_reference.cpp -o "$out/reference"
"$out/reference" "$out/native.stdout" >"$out/reference.stdout" 2>"$out/reference.stderr"
# Nonzero numerical mutation and truncated output must both be rejected.
sed '1s/^IC 0 0 0 2 /IC 0 0 0 0 /' "$out/native.stdout" > "$out/mutant.stdout"
rc=0; "$out/reference" "$out/mutant.stdout" >"$out/mutant.reference.stdout" 2>"$out/mutant.reference.stderr" || rc=$?
printf "wrong_value_expected8_actual\t%s\n" "$rc" > "$out/CONTROLS.tsv"
[[ $rc == 8 ]]
head -n 863 "$out/native.stdout" > "$out/truncated.stdout"
rc=0; "$out/reference" "$out/truncated.stdout" >"$out/truncated.reference.stdout" 2>"$out/truncated.reference.stderr" || rc=$?
printf "truncated_expected9_actual\t%s\n" "$rc" >> "$out/CONTROLS.tsv"
[[ $rc == 9 ]]
rc=0; bin/souc check tests/compile-fail/interval_contract_private.sio > "$out/private.stdout" 2> "$out/private.stderr" || rc=$?
printf "private_expected1_actual\t%s\n" "$rc" >> "$out/CONTROLS.tsv"
[[ $rc == 1 ]]
grep -qi private "$out/private.stdout" "$out/private.stderr"
sha256sum -c "$out/source.sha256" > "$out/source.recheck"
sha256sum -c "$out/compiler.sha256" > "$out/compiler.recheck"
echo INTERVAL_CONTRACT_GATE_PASS | tee "$out/PASS"
