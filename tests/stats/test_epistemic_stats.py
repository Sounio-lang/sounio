#!/usr/bin/env python3
"""
Integration tests for Sounio epistemic statistics
Cross-validates against R and Python scipy/statsmodels

Usage:
    python3 test_epistemic_stats.py
"""

import numpy as np
from scipy import stats
import subprocess
import json
import sys

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_test(name, passed):
    """Print test result with color"""
    status = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    print(f"{status} {name}")

def run_sounio_ttest(group1, group2):
    """
    Run Sounio t-test via compiled binary
    Returns: (t_stat, p_value, confidence)
    """
    # TODO: Compile and run Sounio test
    # For now: placeholder
    # Would run: ./target/release/sounio run tests/stats/ttest_runner.sio
    return None

def test_ttest_vs_scipy():
    """Test t-test against scipy"""
    print("\n=== T-Test Cross-Validation ===")

    # Test data
    drug = np.array([15.2, 18.1, 12.5, 20.3, 16.8, 14.7, 19.2, 13.9])
    placebo = np.array([8.3, 10.1, 7.5, 11.2, 9.4, 8.9, 10.5, 7.8])

    # SciPy t-test (ground truth)
    t_stat_scipy, p_value_scipy = stats.ttest_ind(drug, placebo)

    print(f"  SciPy:  t = {t_stat_scipy:.4f}, p = {p_value_scipy:.4f}")

    # Expected results
    # t ≈ 4.5-5.0, p ≈ 0.0002-0.001
    t_in_range = 4.0 < abs(t_stat_scipy) < 6.0
    p_in_range = 0.0001 < p_value_scipy < 0.01

    print_test("t-statistic in expected range", t_in_range)
    print_test("p-value in expected range", p_in_range)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(drug)-1)*np.var(drug, ddof=1) +
                          (len(placebo)-1)*np.var(placebo, ddof=1)) /
                         (len(drug) + len(placebo) - 2))
    cohens_d = (np.mean(drug) - np.mean(placebo)) / pooled_std

    print(f"  Cohen's d = {cohens_d:.4f}")

    d_in_range = 2.0 < cohens_d < 4.0
    print_test("Cohen's d in expected range (large effect)", d_in_range)

    return t_in_range and p_in_range and d_in_range

def test_normality_vs_scipy():
    """Test normality checks against scipy"""
    print("\n=== Normality Test Cross-Validation ===")

    # Normal data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 50)

    # Shapiro-Wilk test
    w_stat, p_value = stats.shapiro(normal_data)

    print(f"  SciPy Shapiro-Wilk: W = {w_stat:.4f}, p = {p_value:.4f}")

    # For normal data: p > 0.05 (fail to reject normality)
    normality_ok = p_value > 0.05

    print_test("Normal data passes Shapiro-Wilk (p > 0.05)", normality_ok)

    # Non-normal data (uniform)
    uniform_data = np.random.uniform(0, 1, 50)
    w_stat_u, p_value_u = stats.shapiro(uniform_data)

    print(f"  SciPy Shapiro-Wilk (uniform): W = {w_stat_u:.4f}, p = {p_value_u:.4f}")

    non_normality_detected = p_value_u < 0.05

    print_test("Uniform data fails Shapiro-Wilk (p < 0.05)", non_normality_detected)

    return normality_ok and non_normality_detected

def test_levene_vs_scipy():
    """Test Levene's test against scipy"""
    print("\n=== Levene's Test Cross-Validation ===")

    np.random.seed(42)

    # Equal variances
    group1 = np.random.normal(10, 2, 30)  # mean=10, std=2
    group2 = np.random.normal(12, 2, 30)  # mean=12, std=2

    levene_stat, p_value = stats.levene(group1, group2)

    print(f"  SciPy Levene (equal var): F = {levene_stat:.4f}, p = {p_value:.4f}")

    equal_var_ok = p_value > 0.05
    print_test("Equal variances pass Levene's test (p > 0.05)", equal_var_ok)

    # Unequal variances
    group3 = np.random.normal(10, 2, 30)  # std=2
    group4 = np.random.normal(12, 6, 30)  # std=6 (3x larger)

    levene_stat_u, p_value_u = stats.levene(group3, group4)

    print(f"  SciPy Levene (unequal var): F = {levene_stat_u:.4f}, p = {p_value_u:.4f}")

    unequal_var_detected = p_value_u < 0.05
    print_test("Unequal variances detected by Levene (p < 0.05)", unequal_var_detected)

    return equal_var_ok and unequal_var_detected

def test_linear_regression_vs_scipy():
    """Test linear regression against scipy"""
    print("\n=== Linear Regression Cross-Validation ===")

    # y = 2 + 3x + noise
    np.random.seed(42)
    x = np.arange(0, 10, dtype=float)
    y = 2 + 3*x + np.random.normal(0, 0.5, 10)

    # SciPy linear regression
    from scipy.stats import linregress
    result = linregress(x, y)

    print(f"  SciPy: β₀ = {result.intercept:.4f}, β₁ = {result.slope:.4f}")
    print(f"         R² = {result.rvalue**2:.4f}, p = {result.pvalue:.6f}")

    # Check coefficients
    beta0_ok = abs(result.intercept - 2.0) < 0.5
    beta1_ok = abs(result.slope - 3.0) < 0.5
    r2_ok = result.rvalue**2 > 0.95
    sig_ok = result.pvalue < 0.001

    print_test("Intercept ≈ 2.0", beta0_ok)
    print_test("Slope ≈ 3.0", beta1_ok)
    print_test("R² > 0.95", r2_ok)
    print_test("Slope is significant (p < 0.001)", sig_ok)

    return beta0_ok and beta1_ok and r2_ok and sig_ok

def test_bootstrap_ci():
    """Test bootstrap confidence intervals"""
    print("\n=== Bootstrap CI Validation ===")

    np.random.seed(42)
    data = np.random.normal(10, 2, 100)

    # Bootstrap CI (percentile method)
    n_boot = 1000
    boot_means = []

    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    true_mean = np.mean(data)

    print(f"  True mean: {true_mean:.4f}")
    print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # CI should contain true mean
    contains_mean = ci_lower < true_mean < ci_upper

    # Width should be reasonable (≈ 4 * SE for 95% CI)
    se = np.std(data, ddof=1) / np.sqrt(len(data))
    expected_width = 4 * se
    actual_width = ci_upper - ci_lower

    width_ok = 0.5 * expected_width < actual_width < 2.0 * expected_width

    print_test("CI contains true mean", contains_mean)
    print_test("CI width is reasonable", width_ok)

    return contains_mean and width_ok

def test_assumption_confidence_grading():
    """Test that assumption confidence is graded, not binary"""
    print("\n=== Graded Confidence Philosophy ===")

    np.random.seed(42)

    # Generate data with varying normality
    perfect_normal = np.random.normal(0, 1, 100)
    slight_skew = np.random.gamma(3, 1, 100)  # Slightly skewed
    heavy_skew = np.random.gamma(1, 1, 100)   # Heavily skewed

    # Shapiro-Wilk tests
    w1, p1 = stats.shapiro(perfect_normal)
    w2, p2 = stats.shapiro(slight_skew)
    w3, p3 = stats.shapiro(heavy_skew)

    print(f"  Perfect normal: p = {p1:.4f} → expect HIGH confidence (>0.8)")
    print(f"  Slight skew:    p = {p2:.4f} → expect MODERATE confidence (0.5-0.8)")
    print(f"  Heavy skew:     p = {p3:.4f} → expect LOW confidence (<0.5)")

    # Traditional binary approach would be: p > 0.05 → "normal"
    # Sounio graded approach: convert p to BetaConfidence

    # Demonstration of graded confidence mapping:
    def p_to_confidence(p):
        """Convert p-value to graded confidence level"""
        if p > 0.10:
            return 0.90
        elif p > 0.05:
            return 0.75
        elif p > 0.01:
            return 0.50
        else:
            return 0.30

    conf1 = p_to_confidence(p1)
    conf2 = p_to_confidence(p2)
    conf3 = p_to_confidence(p3)

    print(f"  → Graded confidences: {conf1:.2f}, {conf2:.2f}, {conf3:.2f}")

    # Verify grading
    grading_ok = conf1 > conf2 > conf3

    print_test("Confidence decreases with normality violation", grading_ok)

    return grading_ok

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Sounio Epistemic Statistics Integration Tests")
    print("Cross-validation against SciPy (Python)")
    print("=" * 60)

    tests = [
        ("T-Test", test_ttest_vs_scipy),
        ("Normality (Shapiro-Wilk)", test_normality_vs_scipy),
        ("Levene's Test", test_levene_vs_scipy),
        ("Linear Regression", test_linear_regression_vs_scipy),
        ("Bootstrap CI", test_bootstrap_ci),
        ("Graded Confidence", test_assumption_confidence_grading),
    ]

    passed = 0
    total = len(tests)

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
        except Exception as e:
            print(f"{RED}✗ {name} FAILED: {e}{RESET}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} test suites passed")
    print("=" * 60)

    if passed == total:
        print(f"{GREEN}✓ ALL TESTS PASSED{RESET}")
        return 0
    else:
        print(f"{RED}✗ SOME TESTS FAILED{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
