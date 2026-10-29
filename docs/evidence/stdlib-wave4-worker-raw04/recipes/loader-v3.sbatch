#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
S=$R/source
SDK=$B/services/normal-integrated01/sealed-services04
D=$B/root-integrated04-loader
mkdir "$D"
source "$R/tools-env-v2.sh"
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") >"$D/ROOT_SOURCE.final.verify" 2>&1||r=1;(cd "$SDK";sha256sum -c SEALED_SOURCE.sha256) >"$D/SDK_SOURCE.final.verify" 2>&1||r=1;sha256sum -c "$D/INPUTS.sha256" >"$D/INPUTS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
printf '13f80e8c2afc2de5027ea2c070dd708c8dca8855c22e54cdb1b6bbcd90b10aaa  %s\n' "$SDK/SEALED_SOURCE.sha256" >"$D/EXPECTED_SDK.sha256"
sha256sum -c "$D/EXPECTED_SDK.sha256" >"$D/EXPECTED_SDK.verify"
(cd "$SDK";sha256sum -c SEALED_SOURCE.sha256) >"$D/SDK_SOURCE.before.verify"
(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") >"$D/ROOT_SOURCE.before.verify"
sha256sum "$R/RAW_READY" "$R/results/madaros" "$SDK/SEALED_SOURCE.sha256" "$SDK/SOURCE_HEAD" "$SDK/SEALED_SCOPE" "$0" >"$D/INPUTS.sha256"
printf '092cf6bd6cf1bfe54857b308b532a777b1edbffd8f7b592f8e637546583fd29f  %s\n8df264dce09a54f059227dc58d5d0a4c5ab870600c62e6c6e6bc9f4d74b3f008  %s\n' "$SDK/runtime/stdlib_platform.cpp" "$S/stdlib/stats/normal_scalar.sio" >"$D/EXPECTED_RUNTIME_KERNEL.sha256"
sha256sum -c "$D/EXPECTED_RUNTIME_KERNEL.sha256" >"$D/EXPECTED_RUNTIME_KERNEL.verify"
/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027/tools/time -v bash "$R/tool-overlays/loader_gate-v2/gate.sh" "$S" "$SDK" "$D/gate" "$R/RAW_READY" "$R/results/madaros" >"$D/gate.stdout" 2>"$D/gate.stderr"
echo ROOT_IMPORTED_WORKER_AND_NORMAL_NATIVE_QUALIFIED
