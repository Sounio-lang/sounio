#!/bin/bash
#SBATCH --job-name=legacy-flow-final-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-bounds03/custody-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-bounds03/custody-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/legacy-flow-bounds03
S=$B/compiler-resumed/generic-module-native/source
A=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/legacy-flow-bounds03
cd "$D"
sha256sum -c SOURCE.before.sha256 > SOURCE.custody.verify
sha256sum -c SDK.before.sha256 > SDK.custody.verify
mkdir -p compact "$A"
perl -e '
use strict;use warnings;local $/;
sub readfile{open my$f,"<",$_[0]or die$!;my$s=<$f>;close$f;return$s;}
my$a=readfile($ARGV[0]);my$b=readfile($ARGV[1]);
for my$n(qw(flow_shadow_bit_x86 flow_shadow_mask_x86 flow_restore_shadows_x86)){
$a=~/^(fn \Q$n\E\(.*?^})/ms or die$n;my$x=$1;
$b=~/^(fn \Q$n\E\(.*?^})/ms or die$n;die"changed$n"if$x ne$1;
print "PRESENCE_HELPER_BYTE_IDENTICAL $n\n";}
' "$B/compiler-resumed/legacy-flow-presence01/source/self-hosted/compiler/lean_single.sio" source/self-hosted/compiler/lean_single.sio > compact/presence.helpers.guard.stdout
for f in worker.sbatch GATE_STATUS gate.stdout gate.stderr SOURCE.before.sha256 SOURCE.final.verify SDK.before.sha256 SDK.final.verify aarch64.guard.stdout; do cp "$f" compact/; done
for f in INPUTS.before.sha256 INPUTS.final.verify baseline.BUILD_STATUS candidate.BUILD_STATUS SELFCOMPILE_STATUS wrong_value.ORACLE_STATUS duplicate_output.ORACLE_STATUS missing_output.ORACLE_STATUS wrong_value.oracle.stdout duplicate_output.oracle.stdout missing_output.oracle.stderr; do cp "results/$f" compact/; done
for a in before after generation2; do
 mkdir -p "compact/$a"
 cp "results/$a/oracle.stdout" "results/$a/oracle.stderr" "results/$a/ORACLE_STATUS" "compact/$a/"
done
sha256sum results/baseline-producer.elf results/candidate-producer.elf results/generation2-producer.elf results/reference source/self-hosted/compiler/lean_single.sio > compact/QUALIFIED.sha256
mkdir compact/presence19760 compact/collision19759 compact/nominal19755 compact/failed19761 compact/bounds19783
P=$B/compiler-resumed/legacy-flow-presence01
for f in worker.sbatch INPUTS.before.sha256 INPUTS.final.verify before.compile.status before.native.status before.ORACLE_STATUS before.oracle.stdout after.compile.status after.native.status after.ORACLE_STATUS after.oracle.stdout after.native.stdout before.native.stdout; do cp "$P/$f" compact/presence19760/; done
C=$B/compiler-resumed/legacy-flow-collision03
cp "$C/collision.stdout" "$C/SHA256SUMS" "$C/worker.sbatch" compact/collision19759/
cp "$B/compiler-resumed/legacy-flow-nominal01/results/after/oracle.stdout" compact/nominal19755/
cp "$B/compiler-resumed/legacy-flow-unified01/results/after/oracle.stdout" compact/failed19761/
Q=$B/compiler-resumed/legacy-flow-bounds-component02
for f in worker.sbatch INPUTS.before.sha256 INPUTS.final.verify generator.stdout before.oracle.stdout before.oracle.stderr before.ORACLE_STATUS after.oracle.stdout after.oracle.stderr after.ORACLE_STATUS after.compile.status after.native.status after.native.stdout; do cp "$Q/$f" compact/bounds19783/; done
tar czf "$A/producer-sdk-metadata-matrices-receipts.tar.gz" -C "$B" compiler-resumed/legacy-flow-bounds03/source compiler-resumed/legacy-flow-bounds03/results compiler-resumed/legacy-flow-bounds03/compact compiler-resumed/legacy-flow-bounds03/worker.sbatch compiler-resumed/legacy-flow-bounds03/SOURCE.before.sha256 compiler-resumed/legacy-flow-bounds03/SDK.before.sha256 compiler-resumed/legacy-flow-bounds-component02 compiler-resumed/legacy-flow-presence01 compiler-resumed/legacy-flow-collision03 compiler-resumed/legacy-flow-baseline01/baseline.sio -C "$S" self-hosted bin/souc-lean-single-x86_64
sha256sum "$A/producer-sdk-metadata-matrices-receipts.tar.gz" > compact/CUSTODY.sha256
cp compact/CUSTODY.sha256 "$A/"
sha256sum -c SOURCE.before.sha256 > SOURCE.custody.final.verify
sha256sum -c SDK.before.sha256 > SDK.custody.final.verify
find compact -type f -print0 | sort -z | xargs -0 sha256sum > COMPACT.sha256
