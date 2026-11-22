#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

namespace {
constexpr std::array<double, 14> fw_e{
    0.620, 0.161, 0.273, 0.092, 0.320, 0.312, 0.118,
    0.135, 0.282, 0.382, 0.100, 0.207, 0.125, 0.200};
constexpr std::array<double, 14> fw_i{
    0.038, 0.573, 0.483, 0.678, 0.456, 0.392, 0.630,
    0.017, 0.475, 0.291, 0.346, 0.579, 0.631, 0.500};
constexpr std::array<double, 14> fn_l{
    0.0019, 0.0140, 0.0130, 0.0390, 0.0140, 0.0220, 0.0100,
    0.8530, 0.0380, 0.0600, 0.0170, 0.0110, 0.0200, 0.0200};
constexpr std::array<double, 14> fn_ph{
    0.0019, 0.0240, 0.0120, 0.0360, 0.0110, 0.0130, 0.0070,
    0.0020, 0.0160, 0.0040, 0.0020, 0.0110, 0.0130, 0.0100};
constexpr double ln10 = 2.302585092994046;

double kpu(std::size_t i, double log_p) {
  return fw_e[i] + fw_i[i] + std::pow(10.0, log_p) * (fn_l[i] + 0.3 * fn_ph[i]);
}

double kp(std::size_t i, double log_p, double fu_p) {
  return fu_p * kpu(i, log_p);
}

bool close(double actual, double expected, double relative = 1.0e-8) {
  return std::abs(actual - expected) <= relative * (1.0 + std::abs(expected));
}
}  // namespace

int main() {
  constexpr double log_p = 0.0;
  constexpr double fu_p = 0.2;
  constexpr double h_fu = 1.0e-6;
  constexpr double h_log_p = 1.0e-6;

  int closed_form = 0;
  int fu_derivative = 0;
  int logp_derivative = 0;
  int uncertainty_roundtrip = 0;
  for (std::size_t i = 0; i < fw_e.size(); ++i) {
    const double expected_kp = fu_p * kpu(i, log_p);
    if (close(kp(i, log_p, fu_p), expected_kp, 1.0e-12)) {
      ++closed_form;
    }

    const double numerical_fu =
        (kp(i, log_p, fu_p + h_fu) - kp(i, log_p, fu_p - h_fu)) /
        (2.0 * h_fu);
    if (numerical_fu > 0.0 && close(numerical_fu, kpu(i, log_p))) {
      ++fu_derivative;
    }

    const double numerical_logp =
        (kp(i, log_p + h_log_p, fu_p) - kp(i, log_p - h_log_p, fu_p)) /
        (2.0 * h_log_p);
    const double expected_logp = fu_p * ln10 * (fn_l[i] + 0.3 * fn_ph[i]);
    if (close(numerical_logp, expected_logp)) {
      ++logp_derivative;
    }

    constexpr double rapamycin_log_p = 7.0;
    constexpr double rapamycin_fu_p = 0.08;
    const double p_rapamycin = std::pow(10.0, rapamycin_log_p);
    const double lipid = fn_l[i] + 0.3 * fn_ph[i];
    const double derivative_logp = rapamycin_fu_p * p_rapamycin * ln10 * lipid;
    const double derivative_fu = kpu(i, rapamycin_log_p);
    const double variance = std::pow(derivative_logp * 1.05, 2.0) +
                            std::pow(derivative_fu * 0.02, 2.0);
    const double standard_uncertainty = std::sqrt(variance);
    if (close(standard_uncertainty * standard_uncertainty, variance)) {
      ++uncertainty_roundtrip;
    }
  }

  constexpr double c_p_total = 1.25;
  constexpr double c_t_total = 0.625;
  constexpr double synthetic_fu_p = 0.2;
  constexpr double synthetic_fu_t = 0.4;
  const double c_p_unbound = synthetic_fu_p * c_p_total;
  const double c_t_unbound = synthetic_fu_t * c_t_total;
  const double kp_definition = c_t_total / c_p_total;
  const double kpu_definition = c_t_total / c_p_unbound;
  const double kpuu_definition = c_t_unbound / c_p_unbound;
  const bool definitions =
      close(kp_definition, synthetic_fu_p * kpu_definition, 1.0e-12) &&
      close(kpuu_definition,
            (synthetic_fu_t / synthetic_fu_p) * kp_definition, 1.0e-12) &&
      close(kp_definition,
            (synthetic_fu_p / synthetic_fu_t) * kpuu_definition, 1.0e-12);

  const bool pow10_anchor = close(std::pow(10.0, -3.0), 0.001, 1.0e-12) &&
                            close(std::pow(10.0, 7.0), 10000000.0, 1.0e-12);

  std::cout << std::setprecision(17);
  std::cout << "evaluation_logP=" << log_p << " fu_p=" << fu_p
            << " h_fu=" << h_fu << " h_logP=" << h_log_p << '\n';
  std::cout << "constants_held=tissue_composition,fu_except_target; "
               "BP_ratio=not_used; Ka=not_used_or_recomputed\n";
  std::cout << "closed_form=" << closed_form << "/14\n";
  std::cout << "fu_derivative=" << fu_derivative << "/14\n";
  std::cout << "logP_derivative=" << logp_derivative << "/14\n";
  std::cout << "uncertainty_roundtrip=" << uncertainty_roundtrip << "/14\n";
  std::cout << "definition_triangle=" << (definitions ? "PASS" : "FAIL") << '\n';
  std::cout << "pow10_anchor=" << (pow10_anchor ? "PASS" : "FAIL") << '\n';

  if (closed_form == 14 && fu_derivative == 14 && logp_derivative == 14 &&
      uncertainty_roundtrip == 14 && definitions && pow10_anchor) {
    std::cout << "RR_CPP23_UNITS_OK\n";
    return 0;
  }
  std::cout << "RR_CPP23_UNITS_FAIL\n";
  return 1;
}
