#!/usr/bin/env bash
# scripts/ci/zd_fiber_spectra_witness_perturbed_gate.sh
#
# DEMONSTRATION TWIN of zd_fiber_spectra_witness_gate.sh, with the
# count-preserving sign flip sigma(H/2, H+H/2) applied.
#
# It exits 0 and emits the SAME verdict token as the real gate, because the
# proposition it states -- #spectra = 3*2^(n-5) -- is still true of the
# perturbed algebra. Every spectrum is different. Only the witness moves.
#
# A separate script rather than an environment flag: the claim executor runs
# gates with an EMPTY envp (CE_ENVP[0] = 0), deliberately, so nothing about the
# compiler's environment can reach a gate. Nothing here works around that.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFC_ZD_FIBER_PERTURB=1 exec bash "$SCRIPT_DIR/zd_fiber_spectra_witness_gate.sh"
