#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;D=$B/compiler-resumed/trace-empty-component01
A=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/trace-empty-export01
mountpoint -q /orangefs/training;test ! -e "$A";mkdir -p "$A"
trap 'r=$?;echo "$r" >"$A/EXPORT_STATUS";exit "$r"' EXIT
findmnt -T /orangefs/training >"$A/MOUNT.txt"
sacct -X -j19692,19694,19696,19698,19700,19705,19706,19709,19712,19714,19717,19735 --format=JobID,State,ExitCode,Elapsed -P >"$A/JOBS.txt"
(cd "$D/source";sha256sum -c "$D/results/SOURCE.sha256") >"$A/SOURCE.before.verify"
sha256sum -c "$D/results/INPUTS.sha256" >"$A/INPUTS.before.verify"
printf '%s  %s\n' 22e7f121aff116216da7f1acc35221acdcbde5d92c63b36aa60b9d89f0e43b60 /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw04-export02/raw04-sources-seed-sdk-elfs-gates-v2.tar.gz >"$A/BASE_ARCHIVE.sha256"
sha256sum -c "$A/BASE_ARCHIVE.sha256" >"$A/BASE_ARCHIVE.verify"
tar -czf "$A/trace-empty-component-and-live-metadata-v1.tar.gz" -C "$B" compiler-resumed/trace-empty-component01 compiler-resumed/raw04-readonly-ir01 compiler-resumed/raw04-readonly-ir02 compiler-resumed/raw04-readonly-ir03 compiler-resumed/raw04-readonly-ir04 compiler-resumed/raw04-readonly-ir05 compiler-resumed/raw04-readonly-ir06 compiler-resumed/raw04-readonly-ir07 compiler-resumed/raw04-readonly-ir08 compiler-resumed/raw04-readonly-imported01 compiler-resumed/raw04-readonly-imported02 compiler-resumed/raw04-readonly-imported03
sha256sum "$A/trace-empty-component-and-live-metadata-v1.tar.gz" >"$A/ARCHIVE.sha256"
tar -tzf "$A/trace-empty-component-and-live-metadata-v1.tar.gz" >"$A/CONTENTS.txt"
sha256sum -c "$A/ARCHIVE.sha256" >"$A/READBACK.verify"
(cd "$D/source";sha256sum -c "$D/results/SOURCE.sha256") >"$A/SOURCE.final.verify"
sha256sum -c "$D/results/INPUTS.sha256" >"$A/INPUTS.final.verify"
echo TRACE_EMPTY_COMPONENT_AND_METADATA_EXPORT_OK
