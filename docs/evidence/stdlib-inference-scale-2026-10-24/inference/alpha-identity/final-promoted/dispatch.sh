#!/bin/bash
set -uo pipefail
cd /scratch/sounio-stdlib-advanced-20261023/alpha_increment_cpp04
sha256sum source/* /scratch/sounio-stdlib-advanced-20261023/paired_likelihood01/artifacts/stdlib_scale_count_likelihood_difference.elf /scratch/sounio-stdlib-advanced-20261023/paired_likelihood01/artifacts/stdlib_scale_count_likelihood_difference.stdout /scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02/case-{3,11,13}.stdout > INPUTS.sha256
fail=0
/usr/bin/time -v -o compile.resource g++ -std=c++23 -O2 source/count_dispersion_increment_reference.cpp -o alpha_increment >compile.stdout 2>compile.stderr
rc=$?;echo "$rc" >compile.status
[[ $rc == 0 ]] || { echo 1 >STATUS;exit 1; }
/usr/bin/time -v -o native.resource ./alpha_increment >native.stdout 2>native.stderr
rc=$?;echo "$rc" >native.status
[[ $rc == 0 ]] || fail=1
./alpha_increment --mutant-sign >mutant.stdout 2>mutant.stderr
rc=$?;echo "$rc" >mutant.status
[[ $rc == 2 ]] || fail=1
g++ -std=c++23 -O2 source/count_likelihood_difference_reference.cpp -o paired_reference >paired.compile.stdout 2>paired.compile.stderr
rc=$?;echo "$rc" >paired.compile.status
if [[ $rc == 0 ]]; then
 ./paired_reference --native /scratch/sounio-stdlib-advanced-20261023/paired_likelihood01/artifacts/stdlib_scale_count_likelihood_difference.stdout /scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02/case-3.stdout /scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02/case-11.stdout /scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02/case-13.stdout >paired.concordance.stdout 2>paired.concordance.stderr
 rc=$?;echo "$rc" >paired.concordance.status
 [[ $rc == 0 ]] || fail=1
else fail=1;fi
sha256sum -c INPUTS.sha256 >INPUTS.after.verify 2>INPUTS.after.stderr || fail=1
sha256sum alpha_increment paired_reference native.stdout native.stderr mutant.stdout mutant.stderr paired.concordance.stdout paired.concordance.stderr >OUTPUTS.sha256
echo "$fail" >STATUS
exit "$fail"
