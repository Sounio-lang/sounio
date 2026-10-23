#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH=/tmp/sounio-stable-api-bin:$PATH
packet=/home/devsounio/sounio-artifact-archive/prepared/scientific-integration-20261023
custody=/home/devsounio/sounio-artifact-archive/scientific-integration-20261023
mkdir -p "$custody/attempt1"
cp "$packet/scripts/run_sio_test_suite.sh" "$custody/attempt1/dispatch.sh"
cp "$packet/SOURCE.txt" "$packet/MANIFEST.sha256" "$custody/attempt1/"
cp /tmp/scientific-integration-dispatch-20261023-v2.sh "$packet/scripts/run_sio_test_suite.sh"
cd "$packet"
find bin scripts self-hosted stdlib examples -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum --check --status MANIFEST.sha256
# Exact selected source packet: no broad directory-basename exclusions.
tar -czf "$custody/source.complete.tar.gz" -C "$packet" .
mkdir "$custody/readback"
tar -xzf "$custody/source.complete.tar.gz" -C "$custody/readback"
(cd "$custody/readback" && sha256sum --check --status MANIFEST.sha256)
sha256sum "$custody/source.complete.tar.gz" > "$custody/COMPLETE_SNAPSHOT.sha256"
stat -c '%s' "$custody/source.complete.tar.gz" > "$custody/COMPLETE_SNAPSHOT.bytes"
shared=/var/lib/cephfs-slurm/training/sounio-foundry/runs/scientific-integration-source-20261023
mkdir -- "$shared"
cp "$custody/source.complete.tar.gz" "$shared/source.tar.gz"
cp "$custody/COMPLETE_SNAPSHOT.sha256" "$custody/COMPLETE_SNAPSHOT.bytes" "$shared/"
exec /home/devsounio/projects/sounio/sounio-forge submit corpus-smoke --reuse-snapshot "$shared/source.tar.gz" --source-path "$packet" --gpu off --cpu-cpus 2 --cpu-memory-mib 16384 --cpu-time 00:25:00 --expected-job-uid 1000 --expected-job-gid 1000 --submit-user sounio --artifact-root /var/lib/cephfs-slurm/training/sounio-foundry/runs --run-id scientific-integration-20261023-v2
