#!/bin/bash
#SBATCH --job-name=frontier-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/frontier-export-%j.stdout
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/frontier-export-%j.stderr
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/root-independent-frontiers01
R=$B/root-integrated05
S=$R/source
E=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/independent-frontiers-export01
findmnt -T /orangefs/training
mkdir "$E"
finish(){ rc=$?;trap - EXIT;printf '%s\n' "$rc" > "$E/STATUS";exit "$rc"; };trap finish EXIT
cp "$D/input/export.sbatch" "$E/EXPORT.sbatch"
sha256sum "$E/EXPORT.sbatch" > "$E/RECIPE.sha256"
(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") > "$E/SOURCE.before.verify"
for n in 0 1 2;do test "$(cat "$D/case-$n/COLLECTOR_STATUS")" = 0;sha256sum -c "$D/case-$n/INPUTS.sha256" > "$E/case-$n.INPUTS.verify";done
test "$(cat "$D/native-retained01/COLLECTOR_STATUS")" = 0
test "$(cat "$D/observability03/STATUS")" = 0
sha256sum -c "$D/native-retained01/INPUTS.sha256" > "$E/native.INPUTS.verify"
sha256sum -c "$D/observability03/INPUTS.sha256" > "$E/observability.INPUTS.verify"
nodebin=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027/tools/node
"$nodebin" "$D/input/validate-final-modes.mjs" "$R/GIT_TREE_CENSUS.z" "$S" "$R/GIT_SOURCE_TREE" > "$E/MODES_LINKS.verify"
(cd "$D";find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$E/DIAGNOSTIC.before.sha256"
sha256sum "$B/root-integrated04/results/madaros" > "$E/RAW.sha256"
grep -q '^a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333 ' "$E/RAW.sha256"
cat > "$E/SOURCE_CUSTODY.txt" <<'REF'
Current SDK/source: root237ff2d674eb8e8f350d948c4f6f6393706cd67d plus declared snapshot overlays. Full immutable source/seed/recipes/failed-build custody:
/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw05-failed-export01/raw05-failed-full.tar.gz
SHA256 4a3008899c98a2e0aa491c2a33af6a54077d6ea7a15f5b50d183d85040b9bcf8
Independent readback19799. No failed R05 ELF executed; no RAW_READY created. This archive adds accepted old raw04, current SDK stdlib, original tests/reference, all diagnostic/native/observability inputs, retained ELFs, outputs and historical harness failures.
REF
sacct -X -j 19814,19822,19825,19828,19829 --format=JobID,State,ExitCode,Elapsed -P > "$E/JOBS.tsv"
tar -C "$B" -czf "$E/independent-frontiers.tar.gz" root-independent-frontiers01 root-integrated04/results/madaros root-integrated04/tools-env-v3.sh root-integrated04/private-tools-v3 root-integrated05/source/stdlib root-integrated05/source/tests/stdlib/os/test_env_present.sio root-integrated05/source/tests/stdlib/graph/test_graph_types_e2e.sio root-integrated05/source/tests/stdlib/ode/dp54_invalid_probe.sio root-integrated05/source/tests/reference/dp54_dense_events_reference.cpp root-integrated05/results/SOURCE.sha256 root-integrated05/GIT_TREE_CENSUS.z
(cd "$E";sha256sum independent-frontiers.tar.gz > ARCHIVE.sha256;sha256sum -c ARCHIVE.sha256 > ARCHIVE.verify;tar -tzf independent-frontiers.tar.gz > ARCHIVE.members)
(cd "$D";sha256sum -c "$E/DIAGNOSTIC.before.sha256") > "$E/DIAGNOSTIC.final.verify"
(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") > "$E/SOURCE.final.verify"
"$nodebin" "$D/input/validate-final-modes.mjs" "$R/GIT_TREE_CENSUS.z" "$S" "$R/GIT_SOURCE_TREE" > "$E/MODES_LINKS.final.verify"
echo FRONTIER_DURABLE_READBACK_OK
