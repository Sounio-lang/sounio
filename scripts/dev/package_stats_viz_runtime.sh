#!/usr/bin/env bash
# Linux runtime-only package for the bounded statistics/viz sprint.
# Does not compile, submit jobs, delete old packages, or change the checkout.
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "Usage: $0 NEW_DESTINATION_OUTSIDE_CHECKOUT" >&2
    exit 2
fi
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
dest="$(realpath -m -- "$1")"
case "$dest/" in "$src/"*) echo "Destination must be outside the checkout" >&2; exit 2;; esac
files=(
    bin/souc bin/souc.git-version bin/madaros
    bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256
    scripts/lib
    scripts/dev/build_stats_viz_bundle.sh
    scripts/run_sio_test_suite.sh
    scripts/dev/run_sio_test_suite.sh scripts/dev/run_sio_test_suite_v2.sh
    scripts/dev/sio_test_case.sh
    scripts/ci/souc-native-wrapper.sh
    stdlib
    tests/reference/statistical_interval_reference.cpp
    tests/reference/inferential_tail_reference.cpp
    tests/reference/multidomain_reference.cpp
    tests/reference/scientific_axis_reference.cpp
    tests/reference/scientific_clip_reference.cpp
    tests/reference/ols_influence_reference.cpp
    tests/reference/nice_axis_reference.cpp
    tests/reference/svg_path_reference.cpp
    tests/reference/scale_reference.cpp
    tests/reference/scale_scene_reference.cpp
    tests/reference/bootstrap_reference.cpp
    tests/reference/permutation_reference.cpp
    tests/reference/bca_reference.cpp
    tests/reference/normality_reference.cpp
    tests/reference/koenker_reference.cpp
    tests/reference/shapiro_reference.cpp
    tests/reference/kmer_reference.cpp
    tests/reference/kmer_embedding_reference.cpp
    tests/reference/correlation_svg_reference.cpp
    tests/reference/correlation_reference.cpp
    tests/reference/correlation_permutation_reference.cpp
    tests/reference/correlation_permutation_svg_reference.cpp
    tests/reference/least_squares_reference.cpp
    tests/reference/least_squares_svg_reference.cpp
    tests/reference/weighted_least_squares_reference.cpp
    tests/reference/weighted_least_squares_svg_reference.cpp
    tests/reference/weighted_inference_reference.cpp
    tests/reference/weighted_inference_svg_reference.cpp
    tests/reference/student_extreme_tail_reference.cpp
    tests/reference/student_small_tail_reference.cpp
    tests/reference/weighted_inference_scale_reference.cpp
    tests/reference/information_reference.cpp
    tests/reference/entropy_reference.cpp
    tests/reference/poisson_reference.cpp
    tests/reference/lorentz_reference.cpp
    tests/reference/poisson_offsets_svg_reference.cpp
    tests/reference/genomics_size_factors_reference.cpp
    tests/reference/entropy_covariance_reference.cpp
    tests/reference/entropy_covariance_svg_reference.cpp
    tests/reference/ks_reference.cpp
    tests/reference/anderson_darling_reference.cpp
    tests/reference/brown_forsythe_reference.cpp
    tests/reference/linear_hc3_reference.cpp
    tests/reference/linear_hc3_fixture.cpp
    tests/reference/count_tsv_reference.cpp
    tests/reference/count_matrix_qc_svg_reference.cpp
    tests/reference/nb_dispersion_reference.cpp
    tests/reference/linear_cluster_reference.cpp
    tests/reference/cluster_covariance_svg_reference.cpp
    tests/reference/scatter_projection_reference.cpp
    tests/reference/scatter_viewport_svg_reference.cpp
    tests/reference/bar_geometry_reference.cpp
    tests/reference/bar_viewport_svg_reference.cpp
)
tests=(
    stats_descriptive_mad stats_inferential_tails
    viz_chart_builders viz_statistical_interval_forest
    stats_t_test_ci stats_t_test_status stats_inference_forest
    math_cauchy_small_angle math_sqrt_range
    genomics_tpm physics_compton physics_sr_compton
    viz_scientific_axes viz_scientific_clip viz_series_legend
    stats_ols_influence stats_regression_diagnostics viz_regression_influence
    viz_nice_axis viz_auto_plot viz_svg_paths
    viz_scales viz_scaled_plot
    stats_bootstrap_reference stats_bootstrap_status stats_bootstrap_seed viz_bootstrap_forest
    stats_permutation_reference stats_permutation_status stats_permutation_units viz_permutation_mc stats_permutation_legacy
    stats_bca_smoke stats_bca_reference stats_bca_status stats_bca_legacy viz_bca_mean
    stats_normality_reference stats_normality_status stats_normality_units
    stats_koenker_reference stats_koenker_status stats_koenker_legacy viz_koenker
    stats_shapiro_reference stats_shapiro_status stats_shapiro_legacy viz_shapiro_qq
    stats_brown_forsythe_reference stats_brown_forsythe_status stats_brown_forsythe_tail stats_brown_forsythe_legacy viz_brown_forsythe
    stats_ad_reference stats_ad_status stats_ad_legacy viz_ad_ecdf
    stats_ks_reference stats_ks_status viz_ks_ecdf
    genomics_kmer_reference genomics_kmer_status genomics_kmer_legacy genomics_kmer_embedding
    physics_entropy_reference physics_entropy_status physics_entropy_legacy viz_standard_uncertainty viz_isothermal_entropy
    math_information_reference math_information_status math_information_legacy viz_information_channels
    stats_correlation_reference stats_correlation_status stats_correlation_legacy viz_correlation_diagnostics
    stats_correlation_permutation_reference viz_correlation_permutation stats_resampling_seed_guard
    linalg_least_squares_reference stats_multiple_qr_status viz_multivariable_qr
    linalg_weighted_least_squares_reference stats_weighted_qr_status viz_weighted_calibration
    stats_weighted_inference_reference stats_weighted_inference_status viz_weighted_inference stats_student_extreme_tail
    stats_poisson_reference stats_poisson_status viz_poisson_offsets
    physics_lorentz_reference physics_lorentz_status physics_lorentz_legacy
    genomics_size_factors_reference genomics_size_factors_status
    physics_entropy_covariance_reference physics_entropy_covariance_status viz_entropy_covariance
    stats_linear_hc3_reference stats_linear_hc3_status genomics_count_tsv viz_count_matrix_qc
    stats_nb_dispersion_status
    stats_linear_cluster_reference stats_linear_cluster_status viz_cluster_covariance
    viz_scatter_constant viz_scatter_projection_reference viz_scatter_viewport
    viz_bar_signed viz_bar_geometry_reference viz_bar_viewport
)
for name in "${tests[@]}"; do files+=("tests/run-pass/$name.sio"); done
for path in "${files[@]}"; do
    [[ -e "$src/$path" ]] || { echo "Missing source: $path" >&2; exit 1; }
done
mkdir -m 700 -- "$dest"
tar -C "$src" -cf - -- "${files[@]}" | tar -C "$dest" -xf -
if [[ -L "$dest/scripts/dev/run_sio_test_suite.sh" ]]; then
    cp --remove-destination -- "$src/scripts/dev/run_sio_test_suite.sh" "$dest/scripts/dev/run_sio_test_suite.sh"
fi
for name in "${tests[@]}"; do
    cp -- "$dest/tests/run-pass/$name.sio" "$dest/tests/run-pass/basic_math_$name.sio"
done
{
    printf 'scope=statistics-viz-focused-runtime\n'
    printf 'baseline=%s\n' "$(git -C "$src" rev-parse HEAD)"
    printf 'compiler_policy=committed-Madaros-gzip-and-checksum; no local ELF\n'
    printf 'selection=content-identical basic_math aliases for corpus-smoke\n'
    printf 'excluded=Git history, compiler source, local builds, legacy ELF backups, unrelated tests, docs, prior results\n'
    printf 'validation=package preparation only; runtime acceptance requires a new Slurm receipt\n'
} > "$dest/SOURCE.txt"
(
    cd "$dest"
    find bin scripts stdlib tests -type f -print0 |
        LC_ALL=C sort -z | xargs -0 sha256sum
) > "$dest/MANIFEST.sha256"
printf '%s\n' 'Package copy complete; runtime execution not performed.' > "$dest/PACKAGE_READY"
du -sh -- "$dest"
printf 'Package: %s\n' "$dest"
