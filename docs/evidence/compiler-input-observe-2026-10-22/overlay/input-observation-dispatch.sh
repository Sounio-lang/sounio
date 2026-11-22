#!/usr/bin/env bash
set -euo pipefail
source_root=$(pwd -P)
shared=${SOUNIO_ARTIFACTS:?}
local_root=/scratch/sounio-input-observation-20261022/run1
mkdir -- "$local_root"
prov=$local_root/provision
mkdir -p "$prov/lists/partial" "$prov/archives/partial" "$prov/downloads" "$prov/root" "$local_root/tmp" "$local_root/cache"
opts=(-o "Dir::State::lists=$prov/lists" -o "Dir::Cache::archives=$prov/archives" -o 'Dir::State::status=/var/lib/dpkg/status' -o 'Debug::NoLocking=1' -o 'Acquire::Languages=none' -o 'APT::Get::List-Cleanup=false')
printf '%s\n' 'purpose=instrumented-compiler-path-observation' 'tests_run=false' 'cache_authority=false' 'large_outputs=scratch-ext4' 'source_snapshot=reused-immutable-nb2-balanced-eeeab3025c' > "$local_root/SCOPE.txt"
timeout 240 apt-get "${opts[@]}" update > "$prov/update.log" 2>&1
(cd "$prov/downloads" && timeout 120 apt-get "${opts[@]}" download strace libunwind8 > "$prov/download.log" 2>&1)
apt-cache "${opts[@]}" policy strace libunwind8 > "$prov/PACKAGE_POLICY.txt"
apt-cache "${opts[@]}" show strace libunwind8 > "$prov/PACKAGE_METADATA.txt"
for package in "$prov/downloads/"*.deb; do
 sha256sum "$package" >> "$prov/PACKAGES.sha256"
 dpkg-deb --info "$package" >> "$prov/PACKAGES.info"
 dpkg-deb --extract "$package" "$prov/root"
done
loader=$(realpath -e /lib64/ld-linux-x86-64.so.2)
cat > "$prov/strace-launcher" <<LAUNCHER
#!/usr/bin/env bash
exec "$loader" --library-path "$prov/root/usr/lib/x86_64-linux-gnu" "$prov/root/usr/bin/strace" "\$@"
LAUNCHER
chmod 700 "$prov/strace-launcher"
sha256sum "$loader" "$prov/strace-launcher" "$prov/root/usr/bin/strace" > "$prov/TRACER_IDENTITIES.sha256"
find "$prov/root" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum --zero > "$prov/BUNDLE.sha256z"
"$prov/strace-launcher" --version > "$prov/TRACER_VERSION.txt"
unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_TEST_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN SOUNIO_REQUIRE_COMMITTED_MADAROS
export SOUNIO_STDLIB_PATH="$source_root/stdlib"
export SOUNIO_MADAROS_CACHE="$local_root/cache"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export SOUNIO_BUILD_LOCK="$local_root/build.lock" TMPDIR="$local_root/tmp"
ulimit -s 524288
printf '%s\n' 'OBSERVED_BUILD_BEGIN'
bash "$shared/overlay/observe_madaros_inputs.sh" "$source_root" "$local_root/observation" "$prov/strace-launcher" -- bash "$source_root/scripts/ci/build_modular_madaros.sh" "$local_root/madaros-observed.elf"
sha256sum "$local_root/madaros-observed.elf" > "$local_root/COMPILER.sha256"
gzip -n -c "$local_root/madaros-observed.elf" > "$local_root/madaros-observed.elf.gz"
sha256sum "$local_root/madaros-observed.elf.gz" > "$local_root/COMPILER_GZIP.sha256"
# The normal scientific example still compiles/runs; output stays off shared quota.
export MADAROS_RAW_BIN="$local_root/madaros-observed.elf" SOUNIO_SOUC_ENGINE=madaros
mkdir "$local_root/native"
"$source_root/bin/souc" check examples/viz_nb2_deviance.sio > "$local_root/native/typecheck.log" 2>&1
"$source_root/bin/souc" compile examples/viz_nb2_deviance.sio -o "$local_root/native/program.elf" > "$local_root/native/compile.log" 2>&1
/usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$local_root/native/RESOURCE.txt" "$local_root/native/program.elf" > "$local_root/native/native.stdout" 2> "$local_root/native/native.stderr"
sha256sum "$local_root/native/program.elf" "$local_root/native/native.stdout" > "$local_root/native/IDENTITIES.sha256"
gzip -n -c "$local_root/native/program.elf" > "$local_root/native/program.elf.gz"
cp "$local_root/observation/RESULT.txt" "$local_root/COMPILER.sha256" "$local_root/native/RESOURCE.txt" "$shared/"
printf '%s\n' "$local_root" > "$shared/LOCAL_ARTIFACT_ROOT.txt"
printf '%s\n' 'OBSERVED_COMPILER_AND_REAL_EXAMPLE_COMPLETE'
