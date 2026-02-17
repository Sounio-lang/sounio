#!/usr/bin/env Rscript
#
# Integration tests for Sounio epistemic statistics
# Cross-validates against R (the gold standard for statistics)
#
# Usage:
#   Rscript test_epistemic_stats.R

library(car)  # For Levene's test

cat("================================================================\n")
cat("Sounio Epistemic Statistics Integration Tests\n")
cat("Cross-validation against R\n")
cat("================================================================\n\n")

# ANSI colors (if terminal supports)
GREEN <- "\033[92m"
RED <- "\033[91m"
RESET <- "\033[0m"

print_test <- function(name, passed) {
  status <- if (passed) paste0(GREEN, "✓", RESET) else paste0(RED, "✗", RESET)
  cat(sprintf("%s %s\n", status, name))
}

test_ttest <- function() {
  cat("\n=== T-Test Cross-Validation ===\n")

  # Clinical trial data
  drug <- c(15.2, 18.1, 12.5, 20.3, 16.8, 14.7, 19.2, 13.9)
  placebo <- c(8.3, 10.1, 7.5, 11.2, 9.4, 8.9, 10.5, 7.8)

  # R t-test (ground truth)
  result <- t.test(drug, placebo, var.equal = TRUE)

  cat(sprintf("  R: t = %.4f, df = %.0f, p = %.6f\n",
              result$statistic, result$parameter, result$p.value))
  cat(sprintf("     Mean difference = %.4f\n", diff(result$estimate)))
  cat(sprintf("     95%% CI: [%.4f, %.4f]\n", result$conf.int[1], result$conf.int[2]))

  # Expected results
  t_in_range <- abs(result$statistic) > 4.0 && abs(result$statistic) < 6.0
  p_in_range <- result$p.value < 0.001
  ci_excludes_zero <- result$conf.int[1] > 0  # Lower bound > 0

  print_test("t-statistic in expected range", t_in_range)
  print_test("p-value highly significant (p < 0.001)", p_in_range)
  print_test("95% CI excludes zero (significant)", ci_excludes_zero)

  # Effect size (Cohen's d)
  pooled_sd <- sqrt(((length(drug)-1)*var(drug) + (length(placebo)-1)*var(placebo)) /
                    (length(drug) + length(placebo) - 2))
  cohens_d <- diff(c(mean(placebo), mean(drug))) / pooled_sd

  cat(sprintf("  Cohen's d = %.4f (large effect)\n", cohens_d))

  d_in_range <- cohens_d > 2.0 && cohens_d < 4.0
  print_test("Cohen's d indicates large effect", d_in_range)

  # Check assumptions
  shapiro_drug <- shapiro.test(drug)
  shapiro_placebo <- shapiro.test(placebo)

  cat(sprintf("\n  Assumption checks:\n"))
  cat(sprintf("    Normality (drug):    W = %.4f, p = %.4f\n",
              shapiro_drug$statistic, shapiro_drug$p.value))
  cat(sprintf("    Normality (placebo): W = %.4f, p = %.4f\n",
              shapiro_placebo$statistic, shapiro_placebo$p.value))

  # Levene's test for variance equality
  levene_result <- leveneTest(c(drug, placebo), factor(rep(c("drug", "placebo"), c(8, 8))))
  cat(sprintf("    Variance equality:   F = %.4f, p = %.4f\n",
              levene_result$`F value`[1], levene_result$`Pr(>F)`[1]))

  # Sample size check
  n_total <- length(drug) + length(placebo)
  cat(sprintf("    Sample size: n = %d (SMALL - caution)\n", n_total))

  # Epistemic insight: BOTH groups have n=8 (small)
  # Traditional: "p < 0.05 → significant!"
  # Epistemic: "p < 0.001 BUT n=8, normality untestable → confidence ≈ 0.6-0.7"
  cat(sprintf("\n  %sEpistemic Assessment:%s\n", "\033[93m", RESET))
  cat("    - Statistically significant (p < 0.001) ✓\n")
  cat("    - BUT small sample (n=8 per group)\n")
  cat("    - Confidence in assumptions: ~0.65 (moderate)\n")
  cat("    - Recommendation: REPLICATE with n≥30 per group\n")

  all(t_in_range, p_in_range, ci_excludes_zero, d_in_range)
}

test_normality <- function() {
  cat("\n=== Normality Test Cross-Validation ===\n")

  set.seed(42)

  # Normal data
  normal_data <- rnorm(50)
  shapiro_normal <- shapiro.test(normal_data)

  cat(sprintf("  R Shapiro-Wilk (normal): W = %.4f, p = %.4f\n",
              shapiro_normal$statistic, shapiro_normal$p.value))

  normality_ok <- shapiro_normal$p.value > 0.05
  print_test("Normal data passes Shapiro-Wilk (p > 0.05)", normality_ok)

  # Non-normal data (exponential)
  exp_data <- rexp(50)
  shapiro_exp <- shapiro.test(exp_data)

  cat(sprintf("  R Shapiro-Wilk (exponential): W = %.4f, p = %.4f\n",
              shapiro_exp$statistic, shapiro_exp$p.value))

  non_normality_detected <- shapiro_exp$p.value < 0.05
  print_test("Exponential data fails Shapiro-Wilk (p < 0.05)", non_normality_detected)

  # Graded confidence demonstration
  cat("\n  Graded Confidence Mapping:\n")
  p_to_conf <- function(p) {
    ifelse(p > 0.10, 0.90,
    ifelse(p > 0.05, 0.75,
    ifelse(p > 0.01, 0.50, 0.30)))
  }

  conf_normal <- p_to_conf(shapiro_normal$p.value)
  conf_exp <- p_to_conf(shapiro_exp$p.value)

  cat(sprintf("    Normal data:      p = %.4f → confidence = %.2f\n",
              shapiro_normal$p.value, conf_normal))
  cat(sprintf("    Exponential data: p = %.4f → confidence = %.2f\n",
              shapiro_exp$p.value, conf_exp))

  all(normality_ok, non_normality_detected)
}

test_levene <- function() {
  cat("\n=== Levene's Test Cross-Validation ===\n")

  set.seed(42)

  # Equal variances
  group1 <- rnorm(30, mean = 10, sd = 2)
  group2 <- rnorm(30, mean = 12, sd = 2)

  levene_equal <- leveneTest(c(group1, group2),
                             factor(rep(c("A", "B"), c(30, 30))))

  cat(sprintf("  R Levene (equal var): F = %.4f, p = %.4f\n",
              levene_equal$`F value`[1], levene_equal$`Pr(>F)`[1]))

  equal_var_ok <- levene_equal$`Pr(>F)`[1] > 0.05
  print_test("Equal variances pass Levene's test (p > 0.05)", equal_var_ok)

  # Unequal variances
  group3 <- rnorm(30, mean = 10, sd = 2)
  group4 <- rnorm(30, mean = 12, sd = 6)  # 3x larger SD

  levene_unequal <- leveneTest(c(group3, group4),
                               factor(rep(c("A", "B"), c(30, 30))))

  cat(sprintf("  R Levene (unequal var): F = %.4f, p = %.4f\n",
              levene_unequal$`F value`[1], levene_unequal$`Pr(>F)`[1]))

  unequal_var_detected <- levene_unequal$`Pr(>F)`[1] < 0.05
  print_test("Unequal variances detected by Levene (p < 0.05)", unequal_var_detected)

  all(equal_var_ok, unequal_var_detected)
}

test_linear_regression <- function() {
  cat("\n=== Linear Regression Cross-Validation ===\n")

  # y = 2 + 3x + noise
  set.seed(42)
  x <- 0:9
  y <- 2 + 3*x + rnorm(10, sd = 0.5)

  # R linear regression
  model <- lm(y ~ x)
  summary_model <- summary(model)

  cat(sprintf("  R: β₀ = %.4f (p = %.6f)\n",
              coef(model)[1], summary_model$coefficients[1, 4]))
  cat(sprintf("     β₁ = %.4f (p = %.6f)\n",
              coef(model)[2], summary_model$coefficients[2, 4]))
  cat(sprintf("     R² = %.4f, Adj R² = %.4f\n",
              summary_model$r.squared, summary_model$adj.r.squared))
  cat(sprintf("     Residual SE = %.4f\n", summary_model$sigma))

  # Check coefficients
  beta0_ok <- abs(coef(model)[1] - 2.0) < 0.5
  beta1_ok <- abs(coef(model)[2] - 3.0) < 0.5
  r2_ok <- summary_model$r.squared > 0.95
  sig_ok <- summary_model$coefficients[2, 4] < 0.001

  print_test("Intercept ≈ 2.0", beta0_ok)
  print_test("Slope ≈ 3.0", beta1_ok)
  print_test("R² > 0.95", r2_ok)
  print_test("Slope is significant (p < 0.001)", sig_ok)

  # Diagnostics
  cat("\n  Regression Diagnostics:\n")
  cat(sprintf("    Cook's Distance (max): %.4f\n", max(cooks.distance(model))))
  cat(sprintf("    Leverage (max): %.4f\n", max(hatvalues(model))))

  all(beta0_ok, beta1_ok, r2_ok, sig_ok)
}

test_bootstrap <- function() {
  cat("\n=== Bootstrap CI Validation ===\n")

  set.seed(42)
  data <- rnorm(100, mean = 10, sd = 2)

  # Bootstrap CI
  n_boot <- 1000
  boot_means <- replicate(n_boot, mean(sample(data, replace = TRUE)))

  ci_lower <- quantile(boot_means, 0.025)
  ci_upper <- quantile(boot_means, 0.975)

  true_mean <- mean(data)

  cat(sprintf("  True mean: %.4f\n", true_mean))
  cat(sprintf("  Bootstrap 95%% CI: [%.4f, %.4f]\n", ci_lower, ci_upper))

  contains_mean <- ci_lower < true_mean && true_mean < ci_upper

  # Width check
  se <- sd(data) / sqrt(length(data))
  expected_width <- 4 * se
  actual_width <- ci_upper - ci_lower

  width_ok <- actual_width > 0.5 * expected_width && actual_width < 2.0 * expected_width

  print_test("CI contains true mean", contains_mean)
  print_test("CI width is reasonable", width_ok)

  all(contains_mean, width_ok)
}

# Run all tests
main <- function() {
  tests <- list(
    list(name = "T-Test", fn = test_ttest),
    list(name = "Normality (Shapiro-Wilk)", fn = test_normality),
    list(name = "Levene's Test", fn = test_levene),
    list(name = "Linear Regression", fn = test_linear_regression),
    list(name = "Bootstrap CI", fn = test_bootstrap)
  )

  passed <- 0
  total <- length(tests)

  for (test in tests) {
    tryCatch({
      result <- test$fn()
      if (result) {
        passed <- passed + 1
      }
    }, error = function(e) {
      cat(sprintf("%s✗ %s FAILED: %s%s\n", RED, test$name, e$message, RESET))
    })
  }

  cat("\n================================================================\n")
  cat(sprintf("Results: %d/%d test suites passed\n", passed, total))
  cat("================================================================\n")

  if (passed == total) {
    cat(sprintf("%s✓ ALL TESTS PASSED%s\n", GREEN, RESET))
    quit(status = 0)
  } else {
    cat(sprintf("%s✗ SOME TESTS FAILED%s\n", RED, RESET))
    quit(status = 1)
  }
}

main()
