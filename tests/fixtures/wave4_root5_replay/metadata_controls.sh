#!/bin/bash
set -euo pipefail
[[ $# == 2 ]] || exit 2
helpers=$1; root=$2
mkdir "$root"
source "$helpers"
historical="$root/historical"; current="$root/current"
build_cwd="$root/build"; build_manifest="$build_cwd/SOURCE.sha256"
d="$root/fixture"
mkdir -p "$historical/sub" "$current/sub" "$build_cwd" "$d/binding"
printf 'historical bytes\n' > "$historical/unit.sio"
printf 'current bytes\n' > "$current/unit.sio"
printf 'build bytes\n' > "$build_cwd/unit.sio"
chmod 644 "$historical/unit.sio" "$current/unit.sio" "$build_cwd/unit.sio"
ln -s unit.sio "$historical/link"
ln -s unit.sio "$current/link"
sha256sum "$historical/unit.sio" "$current/unit.sio" > "$d/binding/IMMUTABLE.sha256"
(cd "$build_cwd";sha256sum unit.sio) > "$build_manifest"
sdk_inventory "$historical" > "$d/binding/historical.inventory.argv0"
sdk_inventory "$current" > "$d/binding/current.inventory.argv0"
failed=0
record(){ printf '%s %s\n' "$1" "$2" >> "$root/RESULTS.txt"; [[ "$2" == PASS ]] || failed=1; }
if verify_binding "$d" >"$root/initial.verify" 2>&1; then record initial PASS; else record initial FAIL; fi
if perl -0777 -ne 'exit(index($_,"\0")>=0 && index($_,"\\0")<0 ? 0 : 1)' "$d/binding/historical.inventory.argv0"; then record nul_records PASS; else record nul_records FAIL; fi
cp "$historical/unit.sio" "$root/saved.bytes"
printf 'different historical bytes\n' > "$historical/unit.sio"
sdk_inventory "$historical" > "$root/bytechange.inventory"
if cmp "$root/bytechange.inventory" "$d/binding/historical.inventory.argv0"; then record bytechange_same_census PASS; else record bytechange_same_census FAIL; fi
if ! verify_binding "$d" >"$root/bytechange.verify" 2>&1; then record bytechange_if_not PASS; else record bytechange_if_not FAIL; fi
cp "$root/saved.bytes" "$historical/unit.sio";chmod 644 "$historical/unit.sio"
printf 'addition\n' > "$historical/added"
if ! verify_binding "$d" >"$root/addition.verify" 2>&1; then record addition_if_not PASS; else record addition_if_not FAIL; fi
rm "$historical/added"
chmod 600 "$historical/unit.sio"
if ! verify_binding "$d" >"$root/mode.verify" 2>&1; then record mode_if_not PASS; else record mode_if_not FAIL; fi
chmod 644 "$historical/unit.sio"
rm "$historical/link";ln -s sub "$historical/link"
if ! verify_binding "$d" >"$root/link.verify" 2>&1; then record link_if_not PASS; else record link_if_not FAIL; fi
rm "$historical/link";ln -s unit.sio "$historical/link"
cp "$build_cwd/unit.sio" "$root/saved.build"
printf 'changed build bytes\n' > "$build_cwd/unit.sio"
if ! verify_binding "$d" >"$root/buildbytes.verify" 2>&1; then record buildbytes_if_not PASS; else record buildbytes_if_not FAIL; fi
cp "$root/saved.build" "$build_cwd/unit.sio"
saved_cwd=$build_cwd;build_cwd="$root/absent"
if ! verify_binding "$d" >"$root/buildcwd.verify" 2>&1; then record buildcwd_if_not PASS; else record buildcwd_if_not FAIL; fi
build_cwd=$saved_cwd
sha=$(printf '0%.0s' {1..64})
raw_file="$root/raw-placeholder"
printf 'metadata only; never executed\n' > "$raw_file";chmod 700 "$raw_file"
ready_file="$root/RAW_READY"
printf '%s  %s\n' "$sha" "$raw_file" > "$ready_file"
if parse_raw_ready "$ready_file" "$raw_file"; then record ready_valid PASS; else record ready_valid FAIL; fi
printf '%s  %s\nextra\n' "$sha" "$raw_file" > "$ready_file"
if ! parse_raw_ready "$ready_file" "$raw_file"; then record ready_multiline PASS; else record ready_multiline FAIL; fi
printf '%s  %s\n\n' "$sha" "$raw_file" > "$ready_file"
if ! parse_raw_ready "$ready_file" "$raw_file"; then record ready_blank_extra PASS; else record ready_blank_extra FAIL; fi
printf '%s  %s' "$sha" "$raw_file" > "$ready_file"
if ! parse_raw_ready "$ready_file" "$raw_file"; then record ready_no_newline PASS; else record ready_no_newline FAIL; fi
printf '%s  %s\0\n' "$sha" "$raw_file" > "$ready_file"
if ! parse_raw_ready "$ready_file" "$raw_file"; then record ready_nul PASS; else record ready_nul FAIL; fi
printf '%s  %s extra\n' "$sha" "$raw_file" > "$ready_file"
if ! parse_raw_ready "$ready_file" "$raw_file"; then record ready_extra_token PASS; else record ready_extra_token FAIL; fi
printf '%s  %s\n' "$sha" "$raw_file" > "$ready_file"
if ! parse_raw_ready "$ready_file" "$root/wrong"; then record ready_wrong_path PASS; else record ready_wrong_path FAIL; fi
printf '%s\n' "$failed" > "$root/STATUS"
exit "$failed"
