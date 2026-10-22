#!/usr/bin/env bash
# Observational only: never supplies a cache key or authorizes artifact reuse.
set -euo pipefail
umask 077
if [[ $# -lt 5 || "$4" != -- ]]; then
 printf '%s\n' 'usage: observe_madaros_inputs.sh ROOT NEW_OUTPUT_DIR TRACER -- COMMAND [ARGS...]' >&2
 exit 64
fi
observer=$(realpath -e -- "${BASH_SOURCE[0]}")
source_root=$(cd -- "$1" && pwd -P)
output=$(realpath -m -- "$2")
tracer=$(realpath -e -- "$3")
shift 4
[[ -x "$tracer" && -f "$tracer" ]] || { echo 'tracer unavailable' >&2; exit 69; }
case "$output/" in "$source_root/"*) echo 'output must be outside observed source' >&2; exit 64;; esac
[[ $(id -u) != 0 ]] || { echo 'run observations as the non-root build user' >&2; exit 65; }
mkdir -- "$output"
traced_rc=not_run
collector_status=incomplete
finish() {
 local collector_rc=$?
 trap - EXIT
 set +e
 if [[ "$traced_rc" == 0 && "$collector_rc" != 0 ]]; then collector_rc=70; fi
 if [[ "$traced_rc" == 0 && "$collector_status" != captured_endpoints_equal && "$collector_rc" == 0 ]]; then collector_rc=70; fi
 if ! printf 'schema=sounio.input-observation.v1\ntraced_command_status=%s\ncollector_status=%s\ncollector_exit_status=%s\ncache_authority=false\ncomplete_input_closure=false\n' "$traced_rc" "$collector_status" "$collector_rc" > "$output/RESULT.txt"; then
  if [[ "$traced_rc" != not_run && "$traced_rc" != 0 ]]; then exit "$traced_rc"; fi
  exit 70
 fi
 if [[ "$traced_rc" != not_run && "$traced_rc" != 0 ]]; then exit "$traced_rc"; fi
 if [[ "$traced_rc" == 0 && "$collector_rc" == 0 && "$collector_status" == captured_endpoints_equal ]]; then
  printf '%s\n' 'OBSERVATION_CAPTURED_CACHE_AUTHORITY_FALSE'
 fi
 exit "$collector_rc"
}
trap finish EXIT
cd -- "$source_root"
for tree in bin scripts self-hosted stdlib; do [[ -d "$tree" ]] || exit 66; done
printf '%s\n' "$source_root" > "$output/INITIAL_CWD.txt"
readlink /proc/self/ns/mnt /proc/self/ns/pid /proc/self/ns/user > "$output/NAMESPACES.txt"
id > "$output/IDENTITY.txt"
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}" > "$output/JOB.txt"
"$tracer" --version > "$output/TRACER_VERSION.txt"
sha256sum "$tracer" "$observer" > "$output/OBSERVER_IDENTITIES.sha256"
printf '%s\0' "$@" | sha256sum > "$output/COMMAND_ARGV.sha256"
printf '%s\n' '-f -q -yy -xx -s 0 -e trace=%file,%process,getcwd,chdir,fchdir' > "$output/TRACE_OPTIONS.txt"
printf '%s\n' 'ordinary argv strings intentionally truncated; pathname decoding separate' 'trace scope is selected syscalls, not content-consumption completeness' 'shell status only; termination signals are not re-raised by this wrapper' > "$output/POLICY.txt"
# Reuse the existing compiler environment fingerprint; never dump environment values.
bash --noprofile --norc -c 'unset _SOUNIO_MADAROS_CACHE_LOADED; source "$1/scripts/dev/madaros-cache.sh"; madaros_env_fingerprint' observer-env "$source_root" > "$output/BUILD_ENVIRONMENT.sha256"
inventory() {
 local prefix=$1
 find bin scripts self-hosted stdlib -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum --zero > "$output/$prefix.files.sha256z"
 # NUL-terminated metadata records retain symlinks and their literal targets.
 # Their contents/targets outside these trees are not silently followed.
 find bin scripts self-hosted stdlib -printf '%y %m %s %i %T@ %p -> %l\0' | LC_ALL=C sort -z > "$output/$prefix.paths.stat0"
}
inventory BEFORE
# A small shell shim separates child stdout/stderr from tracer diagnostics.
# The invocation and shim are fixed; raw paths may still be sensitive.
set +e
"$tracer" -f -q -yy -xx -s 0 -e trace=%file,%process,getcwd,chdir,fchdir \
 -o "$output/PATH_SYSCALLS.trace" -- /bin/bash --noprofile --norc -c \
 'stdout=$1; stderr=$2; shift 2; exec "$@" >"$stdout" 2>"$stderr"' \
 observation-shim "$output/command.stdout" "$output/command.stderr" "$@" \
 > "$output/shim.stdout" 2> "$output/tracer.stderr"
traced_rc=$?
set -e
inventory AFTER
[[ -s "$output/PATH_SYSCALLS.trace" ]] || { collector_status=missing_trace; exit 70; }
for name in PATH_SYSCALLS.trace command.stdout command.stderr tracer.stderr; do
 sha256sum "$output/$name" > "$output/$name.original.sha256"
 stat -c '%s' "$output/$name" > "$output/$name.bytes"
 gzip -n -c -- "$output/$name" > "$output/$name.gz"
 gzip -dc -- "$output/$name.gz" | sha256sum > "$output/$name.decompressed.sha256"
 [[ $(cut -d ' ' -f 1 "$output/$name.original.sha256") == $(cut -d ' ' -f 1 "$output/$name.decompressed.sha256") ]] || { collector_status=compression_mismatch; exit 70; }
 sha256sum "$output/$name.gz" > "$output/$name.gz.sha256"
 stat -c '%s' "$output/$name.gz" > "$output/$name.gz.bytes"
done
if [[ "$traced_rc" != 0 ]]; then collector_status=command_or_tracer_failed; exit "$traced_rc"; fi
if ! cmp -s "$output/BEFORE.files.sha256z" "$output/AFTER.files.sha256z" || ! cmp -s "$output/BEFORE.paths.stat0" "$output/AFTER.paths.stat0"; then
 collector_status=source_endpoints_changed
 exit 70
fi
# A nonempty diagnostic stream may indicate a partial tracer failure. Preserve it
# and require review rather than silently promoting it to a captured observation.
if [[ -s "$output/tracer.stderr" ]]; then collector_status=tracer_diagnostics_require_review; exit 70; fi
collector_status=captured_endpoints_equal
