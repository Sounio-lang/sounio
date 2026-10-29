#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-collision03/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-collision03/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/legacy-flow-collision03
R=$B/root-integrated04
source "$R/tools-env-v2.sh"
cd "$D"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror tests/fixtures/wave4_legacy_flow/collision_reference.cpp -o reference
./reference hash_collision.sio > collision.stdout
sha256sum hash_collision.sio reference tests/fixtures/wave4_legacy_flow/collision_reference.cpp > SHA256SUMS
