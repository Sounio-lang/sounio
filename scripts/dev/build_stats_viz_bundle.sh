#!/usr/bin/env bash
# Build a reusable stats/viz executable from a prepared runtime package on Slurm.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: $0 tests/run-pass/ENTRY.sio NEW_BUNDLE_DIRECTORY" >&2
    exit 2
fi
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
    echo "Builds run in a Slurm allocation" >&2; exit 2;
}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ -f "$root/PACKAGE_READY" ]] || { echo "A prepared runtime package is required" >&2; exit 2; }
entry="$1"
case "$entry" in tests/run-pass/*.sio) ;; *) echo "Expected a packaged run-pass entry" >&2; exit 2;; esac
source_path="$(realpath -e -- "$root/$entry")"
case "$source_path" in "$root/tests/run-pass/"*) ;; *) echo "Entry escapes package" >&2; exit 2;; esac
dest="$(realpath -m -- "$2")"
case "$dest/" in "$root/"*) echo "Bundle must be outside the source package" >&2; exit 2;; esac
mkdir -m 700 -- "$dest"
export SOUNIO_REQUIRE_COMMITTED_MADAROS=1 SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_STDLIB_PATH="$root/stdlib"
(
    cd "$root"
    bash scripts/lib/materialize_madaros_prebuilt.sh
    sha256sum --check bin/madaros-linux-x86_64.sha256
)
inputs() {
    (
        cd "$root"
        # Prepared packages have regular files. Refuse symlinks rather than
        # omit external targets from the source receipt.
        if [[ -n "$(find bin scripts stdlib tests -type l -print -quit)" ]]; then
            echo "Symlinked package inputs are unsupported" >&2; exit 2
        fi
        # The materializer verification stamp is derived metadata, not an input.
        find bin scripts stdlib tests -type f ! -path "bin/.madaros-linux-x86_64.verified" -print0 |
            LC_ALL=C sort -z | xargs -0 sha256sum
    )
}
inputs > "$dest/inputs.sha256"
(
    cd "$root"
    bin/souc compile "$entry" -o "$dest/program.elf"
) > "$dest/compile.log" 2>&1
test -s "$dest/program.elf"
test -x "$dest/program.elf"
inputs > "$dest/inputs.after.sha256"
cmp "$dest/inputs.sha256" "$dest/inputs.after.sha256"
(
    cd "$dest"
    sha256sum program.elf > program.sha256
)
{
    printf 'schema=stats-viz-bundle-v1\nentry=%s\nslurm_job_id=%s\n' "$entry" "$SLURM_JOB_ID"
    printf 'compiler_sha256=%s\n' "$(sha256sum "$root/bin/madaros-linux-x86_64" | cut -c1-64)"
    printf 'inputs_manifest_sha256=%s\n' "$(sha256sum "$dest/inputs.sha256" | cut -c1-64)"
    printf 'policy=explicit compile-once artifact; no automatic source cache\n'
} > "$dest/provenance.txt"
cat > "$dest/run" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -f "$bundle/BUNDLE_READY" ]] || { echo "Bundle is incomplete" >&2; exit 78; }
(
    cd "$bundle"
    sha256sum --check --status program.sha256
) || { echo "Bundle executable checksum mismatch" >&2; exit 78; }
exec "$bundle/program.elf" "$@"
RUNNER
chmod +x "$dest/run"
printf 'ready\n' > "$dest/BUNDLE_READY"
printf 'Bundle ready: %s\n' "$dest"
