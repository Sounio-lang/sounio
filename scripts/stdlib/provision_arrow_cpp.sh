#!/usr/bin/env bash
# Ubuntu 24.04 amd64 SDK extraction; no system package installation or Python.
# Usage: provision_arrow_cpp.sh ABS_FRESH_OUTPUT
# ARROW_CPP_DEB_CACHE may supply the four pinned official .deb files.
# ARROW_CPP_RUNTIME_CACHE may supply Ubuntu runtime .debs plus a SHA256 manifest.
set -euo pipefail
out="${1:?absolute fresh output required}"
[[ "$out" == /* && ! -e "$out" ]] || { echo 'require absolute fresh output' >&2; exit 2; }
[[ "$(dpkg --print-architecture)" == amd64 ]] || { echo 'SDK requires amd64' >&2; exit 2; }
. /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || { echo 'SDK requires Ubuntu 24.04' >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$out/cache" "$out/root"
out="$(cd "$out" && pwd)"
cp "$here/arrow_sdk_packages.tsv" "$out/cache/PINNED_PACKAGES.sha256"
while read -r expected package; do
 file="${package##*/}"
 if [[ -n "${ARROW_CPP_DEB_CACHE:-}" ]]; then
   cp "$ARROW_CPP_DEB_CACHE/$file" "$out/cache/$file"
 else
   curl --fail --location --retry 3 "https://packages.apache.org/artifactory/arrow/ubuntu/$package" -o "$out/cache/$file"
 fi
 printf '%s  %s\n' "$expected" "$out/cache/$file" | sha256sum -c -
 dpkg-deb -f "$out/cache/$file" Package Version Architecture >> "$out/cache/SDK_VERSIONS.txt"
 dpkg-deb -x "$out/cache/$file" "$out/root"
done < "$here/arrow_sdk_packages.tsv"
mkdir "$out/cache/runtime"
if [[ -n "${ARROW_CPP_RUNTIME_CACHE:-}" ]]; then
 cp "$ARROW_CPP_RUNTIME_CACHE"/*.deb "$out/cache/runtime/"
 cp "$ARROW_CPP_RUNTIME_CACHE/../UBUNTU_DEBS.sha256" "$out/cache/runtime/INPUTS.sha256"
 (cd "$out/cache/runtime";sha256sum -c INPUTS.sha256)
else
 (
 cd "$out/cache/runtime"
 for name in libgrpc++1.51t64 libgrpc29t64 libprotobuf32t64 libprotoc32t64 libabsl20220623t64 libcares2 libre2-10 liburiparser1 libsnappy1v5; do
   apt-cache show "$name" >> METADATA.txt
   apt-get download "$name"
 done
 sha256sum ./*.deb > INPUTS.sha256
 )
fi
for package in "$out/cache/runtime"/*.deb; do
 dpkg-deb -f "$package" Package Version Architecture >> "$out/cache/RUNTIME_VERSIONS.txt"
 dpkg-deb -x "$package" "$out/root"
done
lib="$out/root/usr/lib/x86_64-linux-gnu"
LD_LIBRARY_PATH="$lib" ldd "$lib/libarrow.so.2600" > "$out/ARROW_LINKAGE.txt"
LD_LIBRARY_PATH="$lib" ldd "$lib/libarrow_flight.so.2600" > "$out/FLIGHT_LINKAGE.txt"
if grep -q 'not found' "$out/"*LINKAGE.txt; then cat "$out/"*LINKAGE.txt >&2; exit 1; fi
grep -Fx '#define ARROW_VERSION_MAJOR 26' "$out/root/usr/include/arrow/util/config.h"
grep -Fx '#define ARROW_VERSION_MINOR 0' "$out/root/usr/include/arrow/util/config.h"
grep -Fx '#define ARROW_VERSION_PATCH 0' "$out/root/usr/include/arrow/util/config.h"
find "$out/root" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/SDK_FILES.sha256"
printf 'ARROW_CPP_PREFIX=%s\nARROW_CPP_VERSION=26.0.0\n' "$out/root" > "$out/SDK.env"
echo ARROW_CPP_SDK_EXTRACTED
