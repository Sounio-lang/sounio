#!/bin/bash
#SBATCH --job-name=bidi-protocol-before
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/bidi-protocol50/before-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/bidi-protocol50/before-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/bidi-protocol50;S=$R/source;D=$R/results
export PATH=$B/services-tools-recovered/tools:/usr/bin:/bin
finish(){ r=$?;trap - EXIT;sha256sum -c "$D/BEFORE_INPUTS.sha256" >"$D/BEFORE_INPUTS.final.verify" 2>&1||r=1;echo "$r" >"$D/BEFORE_STATUS";exit "$r";};trap finish EXIT
find "$S" -type f -print0|sort -z|xargs -0 sha256sum >"$D/BEFORE_INPUTS.sha256"
node --check "$S/tests/fixtures/wave4_bidi_runtime/protocol_controls.mjs" >"$D/protocol.syntax.stdout" 2>&1
node "$S/tests/fixtures/wave4_bidi_runtime/protocol_controls.mjs" --module "$S/scripts/stdlib/runtime_fixtures.before.mjs" --golden "$S/tests/fixtures/wave4_bidi_runtime/official-corpus.json" --corpus "$B/root-bidi-inputs02/fixture-run/corpus" --original-source "$B/root-bidi-inputs02/source/tests/stdlib/wave3-bidi/conformance.sio" --out "$R/controls-before" >"$D/protocol.before.stdout" 2>&1
