#!/usr/bin/env bash
# Re-run every measurement in README.md. Usage, from the repository root:
#
#   bash docs/audit/MADAROS_MEMORY_FOOTPRINT_2026-10-22/measure.sh [MADAROS_ELF]
#
# MADAROS_ELF defaults to artifacts/self-hosted/madaros (a Madaros built from
# source with `make build-madaros`). The committed ELF and the seed are always
# measured too. Needs g++ with -std=c++23 and readelf.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SRC_ELF="${1:-artifacts/self-hosted/madaros}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

g++ -std=c++23 -O2 -o "$WORK/maxrss" "$HERE/maxrss.cpp"
g++ -std=c++23 -O2 -o "$WORK/bsscensus" "$HERE/bsscensus.cpp"

# Only the .gz of the committed Madaros is tracked; materialise it for measurement.
COMMITTED="$WORK/madaros-linux-x86_64"
if [[ -x bin/madaros-linux-x86_64 ]]; then cp bin/madaros-linux-x86_64 "$COMMITTED"
else gzip -dc bin/madaros-linux-x86_64.gz > "$COMMITTED"; fi
chmod +x "$COMMITTED"

echo "== 1. ELF layout (file size; LOAD segments: offset vaddr paddr filesz memsz flags)"
for elf in "$SRC_ELF" "$COMMITTED" bin/souc-lean-single-x86_64; do
    [[ -x "$elf" ]] || { echo "missing: $elf"; continue; }
    echo "-- $elf $(stat -c%s "$elf") bytes, $(tr -dc '\000' < "$elf" | wc -c) zero bytes, $(gzip -c "$elf" | wc -c) gzip bytes"
    readelf -lW "$elf" | grep LOAD
done

echo "== 2. seed slot width per array element type (bss growth for 2^20 elements)"
printf 'fn main() with IO {\n    println("x")\n}\n' > "$WORK/base.sio"
base=$(bin/souc-lean-single-x86_64 "$WORK/base.sio" "$WORK/base.out" 2>&1 | grep -o 'bss=[0-9]*' | head -1 | cut -d= -f2)
for t in i8 u8 bool i16 i32 u32 f32 i64 f64; do
    init=0; [[ $t == bool ]] && init=false; [[ $t == f32 || $t == f64 ]] && init=0.0
    printf 'var A: [%s; 1048576] = [%s; 1048576]\nfn main() with IO, Mut {\n    println("x")\n}\n' "$t" "$init" > "$WORK/a.sio"
    b=$(bin/souc-lean-single-x86_64 "$WORK/a.sio" "$WORK/a.out" 2>&1 | grep -o 'bss=[0-9]*' | head -1 | cut -d= -f2)
    echo "$t: $(( (b - base) / 1048576 )) bytes/element"
done

echo "== 3. seed bss per by-value call of a 2 MiB struct"
for n in 1 2 4 8; do
    { printf 'struct Big { n: i64, xs: [i64; 262144] }\nfn bump(b: Big) -> Big with Mut {\n    var c = b\n    c.n = c.n + 1\n    c\n}\nfn main() with IO, Mut {\n    var b = Big { n: 0, xs: [0; 262144] }\n'
      for _ in $(seq 1 "$n"); do printf '    b = bump(b)\n'; done
      printf '    println(b.n)\n}\n'; } > "$WORK/big.sio"
    echo "calls=$n $(bin/souc-lean-single-x86_64 "$WORK/big.sio" "$WORK/big.out" 2>&1 | grep -o 'bss=[0-9]*' | tail -1)"
done

echo "== 4. static-array census over the Madaros module closure"
# The exact closure is the list of `source:`/`import:` lines the seed prints while
# compiling main.sio. Pass a saved build log as BUILD_LOG to use it; otherwise the
# census falls back to every self-hosted file except lean_single.sio (an upper bound).
if [[ -n "${BUILD_LOG:-}" ]]; then
    grep -oE '^(source|import): [^ ]+' "$BUILD_LOG" | awk '{print $2}' | sort -u > "$WORK/closure.txt"
    echo "(exact closure: $(wc -l < "$WORK/closure.txt") modules)"
else
    find self-hosted -name '*.sio' ! -name 'lean_single.sio' | sort > "$WORK/closure.txt"
    echo "(no BUILD_LOG: every self-hosted file except lean_single.sio, an upper bound)"
fi
SEED_LAYOUT=1 "$WORK/bsscensus" < "$WORK/closure.txt"

echo "== 5. peak RSS compiling hello world"
printf 'fn main() with IO {\n    println("hi")\n}\n' > "$WORK/hello.sio"
for elf in "$SRC_ELF" "$COMMITTED"; do
    [[ -x "$elf" ]] || continue
    echo -n "$elf: "
    "$WORK/maxrss" "$elf" build "$WORK/hello.sio" -o "$WORK/hello.out" 2>&1 >/dev/null | tail -1
done
echo -n "bin/souc-lean-single-x86_64: "
"$WORK/maxrss" bin/souc-lean-single-x86_64 "$WORK/hello.sio" "$WORK/hello2.out" 2>&1 >/dev/null | tail -1

echo "== 6. peak RSS of the seed compiling Madaros (optional: SEED_ELF=path to the derived seed)"
# build_modular_madaros.sh derives this seed from lean_single.sio and caches it;
# point SEED_ELF at it. Takes 12-17 min on 2 vCPU.
if [[ -n "${SEED_ELF:-}" ]]; then
    "$WORK/maxrss" "$SEED_ELF" self-hosted/compiler/main.sio "$WORK/madaros.out" 2>&1 >/dev/null | tail -1
else
    echo "(skipped: SEED_ELF not set)"
fi
