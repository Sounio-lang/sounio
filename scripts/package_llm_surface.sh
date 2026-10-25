#!/bin/bash
# Assemble dist/sounio-llm: the souc shim, the Madaros ELF it runs, and
# only the stdlib the LLM path imports. The dist is disposable.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
dest="$root/dist/sounio-llm"
rm -rf "$dest"
mkdir -p "$dest/bin" "$dest/scripts/lib" "$dest/stdlib" "$dest/examples" "$dest/docs/guide"

elf=""
for cand in \
    "$root/artifacts/self-hosted/madaros" \
    "$root/bin/madaros-linux-x86_64" \
    /workspace/sounio/artifacts/self-hosted/madaros \
    /workspace/sounio/bin/madaros-linux-x86_64
do
    if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
        elf="$cand"
        break
    fi
done
if [[ -z "$elf" ]]; then
    echo "package: no Madaros ELF found" >&2
    exit 1
fi
cp -a "$elf" "$dest/bin/madaros-linux-x86_64"
chmod +x "$dest/bin/madaros-linux-x86_64"

cp -a "$root/bin/madaros" "$dest/bin/madaros"
chmod +x "$dest/bin/madaros"
cp -a "$root/bin/souc" "$dest/bin/souc"
chmod +x "$dest/bin/souc"
# Pin the bundled ELF. Without this the shim may select a compiler
# that lives outside the dist.
pin='export MADAROS_RAW_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/bin/madaros-linux-x86_64}"'
awk -v ins="$pin" '
    { print }
    /^ROOT_DIR=/ && !done { print ins; done = 1 }
' "$dest/bin/souc" >"$dest/bin/souc.pinned"
if ! grep -q 'MADAROS_RAW_BIN:-' "$dest/bin/souc.pinned"; then
    echo "package: failed to pin the ELF in bin/souc" >&2
    exit 1
fi
mv "$dest/bin/souc.pinned" "$dest/bin/souc"
chmod +x "$dest/bin/souc"
cp -a "$root/scripts/lib/resolve_madaros.sh" "$dest/scripts/lib/resolve_madaros.sh"

for mod in llm hetero tensor2 math learn; do
    cp -a "$root/stdlib/$mod" "$dest/stdlib/$mod"
done
for ex in \
    tiny_train_demo.sio \
    tiny_gpt_demo.sio \
    llama_lora_demo.sio \
    llm_serve.sio \
    llama_load_bench.sio
do
    cp -a "$root/examples/$ex" "$dest/examples/$ex"
done
cp -a "$root/docs/guide/LLM_QUICKSTART.md" "$dest/docs/guide/LLM_QUICKSTART.md"
cp -a "$root/docs/guide/DEVICE_COMPATIBILITY.md" "$dest/docs/guide/DEVICE_COMPATIBILITY.md"
cat >"$dest/README.md" <<'EOF'
Linux x86-64.
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
./bin/souc check examples/tiny_train_demo.sio
EOF
echo "packed $dest"
echo "elf=$elf"
