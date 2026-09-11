#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kAllowPrefix =
    "SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW ";
constexpr std::string_view kRefusePrefix =
    "SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_REFUSE ";
constexpr std::string_view kSchema =
    "schema=sounio-spark-pair-decommission-plan-v1";
constexpr std::string_view kEffect = "effect=NONE";

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool has_token(std::string_view value, std::string_view token) {
  auto offset = value.find(token);
  while (offset != std::string_view::npos) {
    const auto end = offset + token.size();
    const bool left_boundary = offset == 0 || value[offset - 1] == ' ';
    const bool right_boundary = end == value.size() || value[end] == ' ';
    if (left_boundary && right_boundary) {
      return true;
    }
    offset = value.find(token, offset + 1);
  }
  return false;
}

bool has_single_effect_none(std::string_view value) {
  const auto first = value.find("effect=");
  if (first == std::string_view::npos || !has_token(value, kEffect)) {
    return false;
  }
  return value.find("effect=", first + kEffect.size()) == std::string_view::npos;
}

bool is_effect_free_sounio_plan(std::string_view value) {
  const bool prefix = starts_with(value, kAllowPrefix) ||
                      starts_with(value, kRefusePrefix);
  return prefix && has_token(value, kSchema) &&
         has_single_effect_none(value);
}

bool selftest() {
  const std::string allow =
      "SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW "
      "schema=sounio-spark-pair-decommission-plan-v1 action=BEGIN_DECOMMISSION "
      "from=SLURM_OWNED to=DECOMMISSION_DRAINING custody=SLURM "
      "effect=NONE reason=ALLOW code=0";
  const std::string refuse =
      "SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_REFUSE "
      "schema=sounio-spark-pair-decommission-plan-v1 action=BEGIN_DECOMMISSION "
      "state=SLURM_OWNED custody=SLURM effect=NONE reason=ACTIVE_JOBS";
  return is_effect_free_sounio_plan(allow) &&
         is_effect_free_sounio_plan(refuse) &&
         !is_effect_free_sounio_plan("SOUNIO_SPARK_PAIR_ALLOW effect=NONE") &&
         !is_effect_free_sounio_plan(
             "SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW "
             "schema=sounio-spark-pair-decommission-plan-v1 effect=EXEC") &&
         !is_effect_free_sounio_plan(
             "SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW "
             "schema=sounio-spark-pair-decommission-plan-v1 "
             "effect=NONE effect=EXEC") &&
         !is_effect_free_sounio_plan(
             "SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW "
             "schema=sounio-spark-pair-decommission-plan-v1 "
             "effect=NONE_EXEC") &&
         !is_effect_free_sounio_plan(
             "SOUNIO_SPARK_PAIR_DECOMMISSION_PLAN_ALLOW "
             "schema=sounio-spark-pair-decommission-plan-v1-evil "
             "effect=NONE");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
    if (!selftest()) {
      std::cout << "SOUNIO_SPARK_PAIR_DECOMMISSION_CPP_PARITY_FAIL\n";
      return 1;
    }
    std::cout << "SOUNIO_SPARK_PAIR_DECOMMISSION_CPP_PARITY_PASS "
                 "frame=9026 material_effect=NONE\n";
    return 0;
  }

  if (argc == 3 && std::string_view(argv[1]) == "--classify") {
    if (is_effect_free_sounio_plan(argv[2])) {
      std::cout << "NONE\n";
      return 0;
    }
    std::cout << "DENY\n";
    return 42;
  }

  std::cout << "DENY\n";
  return 64;
}
