#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"

if ! command -v swift >/dev/null 2>&1 || ! command -v open >/dev/null 2>&1; then
    printf '%s\n' 'LOOM_LAUNCH_BLOCKED: Swift and macOS LaunchServices are required.' >&2
    exit 2
fi

swift build --product LoomSpatial
bin_path=$(swift build --show-bin-path)
resource_bundle="$bin_path/LoomApple_LoomSpatial.bundle"
app_path="$root/.build/LoomSpatial.app"

if [ ! -d "$resource_bundle" ]; then
    printf 'LOOM_LAUNCH_BLOCKED: resource bundle missing at %s\n' "$resource_bundle" >&2
    exit 2
fi

rm -rf "$app_path"
mkdir -p "$app_path/Contents/MacOS" "$app_path/Contents/Resources"
install -m 755 "$bin_path/LoomSpatial" "$app_path/Contents/MacOS/LoomSpatial"
install -m 644 Support/LoomSpatial-Info.plist "$app_path/Contents/Info.plist"
cp -R "$resource_bundle" "$app_path/Contents/Resources/"

open -n "$app_path" --args "$@"
printf 'LOOM_LAUNCHED=%s\n' "$app_path"
