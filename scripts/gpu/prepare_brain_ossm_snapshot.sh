#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"
SRC_ROOT="${SOUNIO_DIR:-/home/devsounio/sounio}"
OUT_ROOT="${OUT_ROOT:-$(mktemp -d /tmp/brain-ossm-snapshot.XXXXXX)}"

mkdir -p "${OUT_ROOT}/bin" "${OUT_ROOT}/artifacts/self-hosted" "${OUT_ROOT}/examples"

rsync -a "${SRC_ROOT}/bin/souc" "${OUT_ROOT}/bin/"
rsync -a "${SRC_ROOT}/artifacts/self-hosted/souc-self-hosted-x86_64" "${OUT_ROOT}/artifacts/self-hosted/"
rsync -a "${SRC_ROOT}/stdlib/" "${OUT_ROOT}/stdlib/"
rsync -a "${SRC_ROOT}/examples/fractal_g2_ossm_v3.sio" "${OUT_ROOT}/examples/"
rsync -a "${SRC_ROOT}/examples/brain_ossm_classifier.sio" "${OUT_ROOT}/examples/"
rsync -a "${SRC_ROOT}/examples/hssm_native_algebra.sio" "${OUT_ROOT}/examples/"
rsync -a "${SRC_ROOT}/examples/multihead_unit_oct_benchmark.sio" "${OUT_ROOT}/examples/"
rsync -a "${SRC_ROOT}/examples/associativity_probe_benchmark.sio" "${OUT_ROOT}/examples/"

case "${MODE}" in
  baseline)
    ;;
  seed42)
    perl -0pi -e 's/rng_seed\(99999,11111\)/rng_seed(42,1042)/g; s/rng_seed\(99999\+15,11111\+15\)/rng_seed(57,1057)/g; s/var seeds_a:\[i64;10\]=\[[^\]]+\]/var seeds_a:[i64;10]=[42,42,42,42,42,42,42,42,42,42]/; s/var seeds_b:\[i64;10\]=\[[^\]]+\]/var seeds_b:[i64;10]=[1042,1042,1042,1042,1042,1042,1042,1042,1042,1042]/; s/10 Seeds \+ ListOps Downstream/SEED=42 Override + ListOps Downstream/g; s/=== PROBE \(10 seeds\) ===/=== PROBE (SEED=42 repeated x10) ===/g; s/PROBE SUMMARY \(10 seeds\)/PROBE SUMMARY (SEED=42 repeated x10)/g; s/=== LISTOPS L=15 \(3 seeds\) ===/=== LISTOPS L=15 (SEED=42 repeated x3) ===/g' "${OUT_ROOT}/examples/fractal_g2_ossm_v3.sio"
    perl -0pi -e 's/var seeds_a:\[i64;5\]=\[[^\]]+\]/var seeds_a:[i64;5]=[42,42,42,42,42]/; s/var seeds_b:\[i64;5\]=\[[^\]]+\]/var seeds_b:[i64;5]=[1042,1042,1042,1042,1042]/; s/30 subjects \(10 per class\), 200 epochs, 5 seeds/30 subjects (10 per class), 200 epochs, SEED=42 repeated x5/' "${OUT_ROOT}/examples/brain_ossm_classifier.sio"
    perl -0pi -e 's/rng_seed\(99999 \+ seq_len, 11111 \+ seq_len\)/rng_seed(42 + seq_len, 1042 + seq_len)/g; s/rng_seed\(99999 \+ slen, 11111 \+ slen\)/rng_seed(42 + slen, 1042 + slen)/g; s/rng_seed\(55555, 77777\)/rng_seed(42, 1042)/g; s/rng_seed\(12345, 67890\)/rng_seed(1042, 2042)/g' "${OUT_ROOT}/examples/hssm_native_algebra.sio"
    perl -0pi -e 's/rng_seed\(99999, 11111\)/rng_seed(42, 1042)/g; s/rng_seed\(99999 \+ 15, 11111 \+ 15\)/rng_seed(57, 1057)/g; s/rng_seed\(55555, 77777\)/rng_seed(42, 1042)/g; s/rng_seed\(12345, 67890\)/rng_seed(1042, 2042)/g' "${OUT_ROOT}/examples/multihead_unit_oct_benchmark.sio"
    perl -0pi -e 's/rng_seed\(54321, 98765\)/rng_seed(42, 1042)/g; s/rng_seed\(99999, 11111\)/rng_seed(42, 1042)/g; s/rng_seed\(42, 137\)/rng_seed(42, 1042)/g; s/rng_seed\(55555, 77777\)/rng_seed(42, 1042)/g; s/rng_seed\(12345, 67890\)/rng_seed(1042, 2042)/g' "${OUT_ROOT}/examples/associativity_probe_benchmark.sio"
    ;;
  fractal20)
    perl -0pi -e 's/10 Seeds \+ ListOps Downstream/20-Seed Probe + ListOps Downstream/g; s/var o_ov:\[f64;10\]=\[0\.0;10\];var o_na:\[f64;10\]=\[0\.0;10\];var o_an:\[f64;10\]=\[0\.0;10\]/var o_ov:[f64;20]=[0.0;20];var o_na:[f64;20]=[0.0;20];var o_an:[f64;20]=[0.0;20]/; s/var h_ov:\[f64;10\]=\[0\.0;10\];var h_na:\[f64;10\]=\[0\.0;10\]/var h_ov:[f64;20]=[0.0;20];var h_na:[f64;20]=[0.0;20]/; s/var seeds_a:\[i64;10\]=\[[^\]]+\]/var seeds_a:[i64;20]=[55555,11111,99887,22222,44444,66666,77777,88888,33333,12321,24680,13579,54321,67890,11223,44567,77889,99001,31415,27182]/; s/var seeds_b:\[i64;10\]=\[[^\]]+\]/var seeds_b:[i64;20]=[77777,33333,44556,55555,66666,88888,11111,22222,99999,45654,86420,97531,12345,98765,33211,76544,98877,10099,92653,58979]/; s/=== PROBE \(10 seeds\) ===/=== PROBE (20 seeds) ===/g; s/while si<10\{/while si<20{/g; s/o_ov_sum\/10\.0/o_ov_sum\/20.0/g; s/o_na_sum\/10\.0/o_na_sum\/20.0/g; s/o_an_sum\/10\.0/o_an_sum\/20.0/g; s/h_ov_sum\/10\.0/h_ov_sum\/20.0/g; s/h_na_sum\/10\.0/h_na_sum\/20.0/g; s/o_ov_v\/10\.0\)\/sqrt_q\(10\.0\)/o_ov_v\/20.0)\/sqrt_q(20.0)/g; s/o_na_v\/10\.0\)\/sqrt_q\(10\.0\)/o_na_v\/20.0)\/sqrt_q(20.0)/g; s/h_ov_v\/10\.0\)\/sqrt_q\(10\.0\)/h_ov_v\/20.0)\/sqrt_q(20.0)/g; s/h_na_v\/10\.0\)\/sqrt_q\(10\.0\)/h_na_v\/20.0)\/sqrt_q(20.0)/g; s/PROBE SUMMARY \(10 seeds\)/PROBE SUMMARY (20 seeds)/g' "${OUT_ROOT}/examples/fractal_g2_ossm_v3.sio"
    ;;
  *)
    echo "unknown snapshot mode: ${MODE}" >&2
    exit 1
    ;;
esac

echo "${OUT_ROOT}"
