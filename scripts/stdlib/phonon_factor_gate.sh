#!/usr/bin/env bash
# Run in a frozen source packet on a Slurm worker, under UID1000.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
: "${PHONON_GATE_OUTPUT:?select a new empty output directory}"
: "${MADAROS_RAW_BIN:?pin a verified absolute compiler ELF}"
[[ $MADAROS_RAW_BIN == /* && -x $MADAROS_RAW_BIN ]]
[[ $(id -u) == 1000 ]]
[[ ! -e $PHONON_GATE_OUTPUT ]] || { echo 'output must not exist' >&2;exit 2; }
mkdir -p "$PHONON_GATE_OUTPUT";out=$(cd "$PHONON_GATE_OUTPUT" && pwd)
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288
ulimit -c 0
find stdlib tests scripts/stdlib/phonon_factor_gate.sh -type f -print0|sort -z|xargs -0 sha256sum > "$out/SOURCE.sha256"
sha256sum "$MADAROS_RAW_BIN">"$out/COMPILER.sha256"
for name in core si status;do
 case $name in
 core) src=tests/run-pass/phonon_factor_transport.sio;;
 si) src=tests/stdlib/physics/test_phonon_quantity_parity.sio;;
 status) src=tests/run-pass/phonon_factor_status.sio;;
 esac
 timeout 180 bin/souc check "$src" >"$out/$name.check.stdout" 2>"$out/$name.check.stderr"
 timeout 240 bin/souc compile "$src" -o "$out/$name.elf" >"$out/$name.compile.stdout" 2>"$out/$name.compile.stderr"
 chmod +x "$out/$name.elf";sha256sum "$out/$name.elf">"$out/$name.ELF.sha256"
 /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 120 "$out/$name.elf" >"$out/$name.native.stdout" 2>"$out/$name.native.stderr"
done
timeout 180 g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/phonon_factor_reference.cpp -o "$out/reference" >"$out/cpp.stdout" 2>"$out/cpp.stderr"
"$out/reference" selftest > "$out/selftest.stdout"
for mode in core si status;do
 "$out/reference" "$mode" "$out/$mode.native.stdout" >"$out/$mode.reference.stdout" 2>"$out/$mode.reference.stderr"
 # The oracle must reject a missing row and a corrupted numerical value.
 sed '1d' "$out/$mode.native.stdout" >"$out/$mode.missing.stdout"
 if "$out/reference" "$mode" "$out/$mode.missing.stdout" >"$out/$mode.missing.reference.stdout" 2>"$out/$mode.missing.reference.stderr";then echo 'oracle accepted missing row' >&2;exit 1;fi
 awk 'BEGIN{done=0} /^PHONON_VALUE /&&!done{$3="0";done=1} {print}' "$out/$mode.native.stdout" >"$out/$mode.corrupt.stdout"
 if "$out/reference" "$mode" "$out/$mode.corrupt.stdout" >"$out/$mode.corrupt.reference.stdout" 2>"$out/$mode.corrupt.reference.stderr";then echo 'oracle accepted corrupt value' >&2;exit 1;fi
done
if timeout 120 bin/souc check tests/compile-fail/phonon_factor_private.sio >"$out/private.stdout" 2>"$out/private.stderr";then echo 'private field write accepted' >&2;exit 1;fi
grep -Eq 'error\[E259\].*struct field is private' "$out/private.stdout" "$out/private.stderr"
src=docs/evidence/stdlib-SIO-172/before_variance/variance_witness.sio
sha256sum "$src" docs/evidence/stdlib-SIO-172/before_variance/phonon_original.sio > "$out/BEFORE.sha256"
timeout 180 bin/souc check "$src" >"$out/before.check.stdout" 2>"$out/before.check.stderr"
timeout 240 bin/souc compile "$src" -o "$out/before.elf" >"$out/before.compile.stdout" 2>"$out/before.compile.stderr"
chmod +x "$out/before.elf";sha256sum "$out/before.elf">"$out/before.ELF.sha256"
timeout 30 "$out/before.elf" >"$out/before.native.stdout" 2>"$out/before.native.stderr"
grep -Eq '^LEGACY_WRONG_VARIANCE_RHS_WITNESS$' "$out/before.native.stdout"
sha256sum -c "$out/SOURCE.sha256">"$out/SOURCE.verify.txt"
echo 'PHONON_FACTOR_GATE_PASS'
