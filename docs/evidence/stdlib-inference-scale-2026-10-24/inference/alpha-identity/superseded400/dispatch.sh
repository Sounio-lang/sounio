#!/bin/bash
set -uo pipefail
cd /scratch/sounio-stdlib-advanced-20261023/alpha_increment_cpp03
sha256sum source/* > INPUTS.sha256
/usr/bin/time -v -o compile.resource g++ -std=c++23 -O2 source/count_dispersion_increment_reference.cpp -o alpha_increment >compile.stdout 2>compile.stderr
rc=$?; echo "$rc" > compile.status
if [[ $rc != 0 ]]; then echo 1 > STATUS; exit 1; fi
/usr/bin/time -v -o native.resource ./alpha_increment >native.stdout 2>native.stderr
rc=$?; echo "$rc" > native.status
./alpha_increment --mutant-sign >mutant.stdout 2>mutant.stderr
mrc=$?; echo "$mrc" > mutant.status
sha256sum alpha_increment native.stdout native.stderr mutant.stdout mutant.stderr >OUTPUTS.sha256
if [[ $rc == 0 && $mrc == 2 ]]; then echo 0 >STATUS; else echo 1 >STATUS; fi
