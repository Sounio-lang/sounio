#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"

if ! command -v xcrun >/dev/null 2>&1; then
  printf '%s\n' 'APPLE_GATE_BLOCKED: xcrun is unavailable; run this gate on macOS with Xcode 27.' >&2
  exit 2
fi
if ! command -v swift >/dev/null 2>&1; then
  printf '%s\n' 'APPLE_GATE_BLOCKED: swift is unavailable.' >&2
  exit 2
fi

mkdir -p .build/metal-validation
xcrun metal -c Sources/LoomSpatial/Resources/LoomField.metal \
  -o .build/metal-validation/LoomField.air
xcrun metallib .build/metal-validation/LoomField.air \
  -o .build/metal-validation/LoomField.metallib

swift test
swift build --product LoomSpatial
