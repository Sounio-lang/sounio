#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "capd/capdlib.h"

#ifndef CS6_WORKER_SOURCE_SHA256
#define CS6_WORKER_SOURCE_SHA256 "UNBOUND"
#endif

using capd::C0HOTripletonSet;
using capd::C0HORect2Set;
using capd::C0Rect2Set;
using capd::ICoordinateSection;
using capd::IMap;
using capd::IMatrix;
using capd::IOdeSolver;
using capd::IPoincareMap;
using capd::IVector;
using capd::interval;

namespace {

constexpr int kPhysicalDimension = 3;
constexpr int kLiouvilleDimension = 4;
constexpr int kSectionCoordinate = 2;
constexpr int kOrder = 8;
constexpr int kReturnCount = 2;
constexpr int kMaxDyadicDepth = 30;

enum class LiouvilleCarrier { kHOTripleton, kHORect2, kRect2 };

const char* carrier_name(LiouvilleCarrier carrier) {
  switch (carrier) {
    case LiouvilleCarrier::kHOTripleton:
      return "C0HOTripletonSet";
    case LiouvilleCarrier::kHORect2:
      return "C0HORect2Set";
    case LiouvilleCarrier::kRect2:
      return "C0Rect2Set";
  }
  throw std::runtime_error("unknown Liouville carrier");
}

LiouvilleCarrier parse_carrier(const std::string& value) {
  if (value == "C0HOTripletonSet") {
    return LiouvilleCarrier::kHOTripleton;
  }
  if (value == "C0HORect2Set") {
    return LiouvilleCarrier::kHORect2;
  }
  if (value == "C0Rect2Set") {
    return LiouvilleCarrier::kRect2;
  }
  throw std::runtime_error("unknown Liouville carrier");
}

constexpr char kLiouvilleField[] =
    "par:zs;var:x,y,w,ell;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs,"
    "x-y-(w+zs)/2-1;";

interval decimal(const char* value) { return interval(value, value); }

int parse_integer(const char* token, const char* name, int minimum,
                  int maximum) {
  const std::string text(token);
  const bool canonical =
      !text.empty() &&
      ((text == "0") ||
       (text[0] >= '1' && text[0] <= '9' &&
        std::all_of(text.begin() + 1, text.end(), [](char character) {
          return character >= '0' && character <= '9';
        })));
  if (!canonical) {
    throw std::runtime_error(std::string("noncanonical ") + name);
  }
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(token, &consumed, 10);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  if (consumed != text.size() || value < minimum || value > maximum) {
    throw std::runtime_error(std::string("out-of-range ") + name);
  }
  return static_cast<int>(value);
}

int dyadic_count(int depth) { return 1 << depth; }

bool lowercase_sha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool finite(const interval& value) {
  return std::isfinite(value.leftBound()) &&
         std::isfinite(value.rightBound()) &&
         value.leftBound() <= value.rightBound();
}

bool finite(const IVector& value) {
  for (int row = 0; row < value.dimension(); ++row) {
    if (!finite(value[row])) {
      return false;
    }
  }
  return true;
}

interval tile(const interval& center, const interval& radius, int index,
              int count) {
  const interval left = center - radius;
  const interval step = 2.0 * radius / static_cast<double>(count);
  const interval lower = left + static_cast<double>(index) * step;
  const interval upper = left + static_cast<double>(index + 1) * step;
  return interval(lower.leftBound(), upper.rightBound());
}

void write_interval(std::ostream& output, const std::string& name,
                    const interval& value) {
  const double lower = std::nextafter(
      value.leftBound(), -std::numeric_limits<double>::infinity());
  const double upper = std::nextafter(
      value.rightBound(), std::numeric_limits<double>::infinity());
  output << ' ' << name << "=[" << std::hexfloat << lower << ',' << upper
         << std::defaultfloat << ']';
}

void write_vector(std::ostream& output, const std::string& prefix,
                  const IVector& value) {
  for (int row = 0; row < value.dimension(); ++row) {
    write_interval(output, prefix + std::to_string(row), value[row]);
  }
}

struct FrozenInput {
  int u_depth;
  int u_index;
  int s_depth;
  int s_index;
  interval zs = decimal("22.3274637391");
  interval origin_x = decimal("15.186446520640786");
  interval origin_y = decimal("10.908543194765466");
  interval unstable_x = decimal("-0.67430316214199759");
  interval unstable_y = decimal("-0.73845463335624273");
  interval stable_x = decimal("-0.94170446778164518");
  interval stable_y = decimal("0.33644122125579123");
  interval radius_u = decimal("0.004");
  interval radius_s = decimal("0.3");

  FrozenInput(int u_depth_value, int u_index_value, int s_depth_value,
              int s_index_value)
      : u_depth(u_depth_value),
        u_index(u_index_value),
        s_depth(s_depth_value),
        s_index(s_index_value) {}

  int u_tiles() const { return dyadic_count(u_depth); }
  int s_tiles() const { return dyadic_count(s_depth); }

  interval source_u() const {
    return tile(interval(0.0), radius_u, u_index, u_tiles());
  }

  interval source_s() const {
    return tile(interval(0.0), radius_s, s_index, s_tiles());
  }

  IVector center() const {
    const interval u = source_u().mid();
    const interval s = source_s().mid();
    return IVector{origin_x + unstable_x * u + stable_x * s,
                   origin_y + unstable_y * u + stable_y * s,
                   interval(0.0)};
  }

  IMatrix frame() const {
    IMatrix result(kLiouvilleDimension, kLiouvilleDimension);
    result.setToIdentity();
    result[0][0] = unstable_x;
    result[1][0] = unstable_y;
    result[0][1] = stable_x;
    result[1][1] = stable_y;
    return result;
  }

  IVector radii() const {
    return IVector{source_u() - source_u().mid(),
                   source_s() - source_s().mid(), interval(0.0),
                   interval(0.0)};
  }

  interval oriented_q0_determinant() const {
    // Jacobian density in the global normalized (U,S) chart, not subtile area.
    const interval frame_determinant =
        unstable_x * stable_y - stable_x * unstable_y;
    return frame_determinant * radius_u * radius_s;
  }
};

struct LiouvilleData {
  interval time;
  IVector initial_hull = IVector(kLiouvilleDimension);
  IVector image = IVector(kLiouvilleDimension);
  interval initial_velocity;
  interval normal_velocity;
  interval ell;
  interval exp_ell;
  interval determinant;
};

template <typename Set>
LiouvilleData liouville_two_return_with(const FrozenInput& input) {
  IMap field(kLiouvilleField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(kLiouvilleDimension, kSectionCoordinate);
  IPoincareMap map(solver, section, capd::poincare::MinusPlus);
  const IVector physical_center = input.center();
  const IVector center{physical_center[0], physical_center[1], interval(0.0),
                       interval(0.0)};
  Set set(center, input.frame(), input.radii());
  LiouvilleData result;
  result.initial_hull = static_cast<IVector>(set);
  result.image = map(set, result.time, kReturnCount);
  const interval initial_x = input.origin_x +
                             input.unstable_x * input.source_u() +
                             input.stable_x * input.source_s();
  const interval initial_y = input.origin_y +
                             input.unstable_y * input.source_u() +
                             input.stable_y * input.source_s();
  result.initial_velocity = initial_x * initial_y - input.zs;
  result.normal_velocity =
      result.image[0] * result.image[1] - result.image[2] - input.zs;
  result.ell = result.image[3];
  result.exp_ell = exp(result.ell);
  result.determinant = result.exp_ell * result.initial_velocity /
                       result.normal_velocity *
                       input.oriented_q0_determinant();
  return result;
}

LiouvilleData liouville_two_return(const FrozenInput& input,
                                    LiouvilleCarrier carrier) {
  switch (carrier) {
    case LiouvilleCarrier::kHOTripleton:
      return liouville_two_return_with<C0HOTripletonSet>(input);
    case LiouvilleCarrier::kHORect2:
      return liouville_two_return_with<C0HORect2Set>(input);
    case LiouvilleCarrier::kRect2:
      return liouville_two_return_with<C0Rect2Set>(input);
  }
  throw std::runtime_error("unknown Liouville carrier");
}

}  // namespace

int main(int argc, char** argv) {
  std::string failure_carrier;
  std::string failure_binding;
  try {
    if (argc != 13) {
      throw std::runtime_error(
          "usage: U_DEPTH U_INDEX S_DEPTH S_INDEX INPUT_SHA256 "
          "RUN_CHALLENGE LIOUVILLE_CARRIER FROZEN_CONTRACT_SHA256 "
          "COORDINATE_MANIFEST_SHA256 RUN_CONTRACT_SHA256 "
          "MANIFEST_ROW_SHA256 ATTEMPT_BINDING");
    }
    const int u_depth =
        parse_integer(argv[1], "u_depth", 0, kMaxDyadicDepth);
    const int s_depth =
        parse_integer(argv[3], "s_depth", 0, kMaxDyadicDepth);
    const int u_index =
        parse_integer(argv[2], "u_index", 0, dyadic_count(u_depth) - 1);
    const int s_index =
        parse_integer(argv[4], "s_index", 0, dyadic_count(s_depth) - 1);
    const std::string input_sha256 = argv[5];
    const std::string run_challenge = argv[6];
    const LiouvilleCarrier carrier = parse_carrier(argv[7]);
    const std::string frozen_contract_sha256 = argv[8];
    const std::string coordinate_manifest_sha256 = argv[9];
    const std::string run_contract_sha256 = argv[10];
    const std::string manifest_row_sha256 = argv[11];
    const std::string attempt_binding = argv[12];
    const std::string worker_source_sha256 = CS6_WORKER_SOURCE_SHA256;
    for (const std::string* digest :
         {&worker_source_sha256, &input_sha256, &run_challenge,
          &frozen_contract_sha256, &coordinate_manifest_sha256,
          &run_contract_sha256, &manifest_row_sha256, &attempt_binding}) {
      if (!lowercase_sha256(*digest)) {
        throw std::runtime_error(
            "all binding digests must be lowercase SHA-256");
      }
    }
    failure_carrier = carrier_name(carrier);
    failure_binding = attempt_binding;

    const FrozenInput input(u_depth, u_index, s_depth, s_index);
    const LiouvilleData liouville = liouville_two_return(input, carrier);
    if (!finite(liouville.time) || !finite(liouville.initial_hull) ||
        !finite(liouville.image) || !finite(liouville.initial_velocity) ||
        !finite(liouville.normal_velocity) || !finite(liouville.ell) ||
        !finite(liouville.exp_ell) || !finite(liouville.determinant)) {
      throw std::runtime_error("nonfinite Liouville checkpoint");
    }

    std::ostringstream receipt;
    receipt << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::boolalpha;
    receipt << "V7A1_BINDING"
            << " WORKER_SOURCE_SHA256=" << worker_source_sha256
            << " INPUT_SHA256=" << input_sha256
            << " RUN_CHALLENGE=" << run_challenge
            << " LIOUVILLE_CARRIER=" << carrier_name(carrier)
            << " FROZEN_CONTRACT_SHA256=" << frozen_contract_sha256
            << " COORDINATE_MANIFEST_SHA256="
            << coordinate_manifest_sha256
            << " RUN_CONTRACT_SHA256=" << run_contract_sha256
            << " MANIFEST_ROW_SHA256=" << manifest_row_sha256
            << " ATTEMPT_BINDING=" << attempt_binding << '\n';
    receipt << "DECLARATIONS"
            << " SCHEMA=sounio.cs6.hapg-liouville-checkpoint.v1"
            << " CAPD_SOURCE_TREE_DECLARED=capd-5.3.0"
            << " INTERVAL_BACKEND_DECLARED=FILIB"
            << " INTERVAL_SERIALIZATION=ONE_ULP_OUTWARD_BINARY64_HEX"
            << " SOURCE=N0"
            << " U_DEPTH=" << u_depth
            << " U_INDEX=" << u_index
            << " S_DEPTH=" << s_depth
            << " S_INDEX=" << s_index
            << " U_TILES=" << input.u_tiles()
            << " S_TILES=" << input.s_tiles()
            << " ORDER=" << kOrder
            << " RETURN_COUNT=" << kReturnCount
            << " SECTION=COORDINATE_W_EQUALS_ZERO"
            << " CROSSING_DIRECTION=MINUS_PLUS"
            << " CHECKPOINT_SCOPE=LIOUVILLE_CARRIER_ONLY"
            << " C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED=false"
            << " DOWNSTREAM_SECTION_RESIDENT_EXECUTED=false"
            << " PROMOTION_ELIGIBLE=false\n";
    receipt << "SOURCE_TILE";
    write_interval(receipt, "U", input.source_u());
    write_interval(receipt, "S", input.source_s());
    write_interval(receipt, "Q0_DET", input.oriented_q0_determinant());
    receipt << '\n';
    receipt << "INITIAL_HULL";
    write_vector(receipt, "X", liouville.initial_hull);
    receipt << '\n';
    receipt << "LIOUVILLE";
    write_interval(receipt, "TIME", liouville.time);
    write_vector(receipt, "X", liouville.image);
    write_interval(receipt, "NU0", liouville.initial_velocity);
    write_interval(receipt, "NU2", liouville.normal_velocity);
    write_interval(receipt, "ELL", liouville.ell);
    write_interval(receipt, "EXP_ELL", liouville.exp_ell);
    write_interval(receipt, "DET", liouville.determinant);
    receipt << '\n';
    receipt << "CHECKPOINT COMPLETE=true\n";
    std::cout << receipt.str();
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    if (!failure_carrier.empty() && !failure_binding.empty()) {
      std::cerr << "V7A1_FAILURE_BINDING"
                << " LIOUVILLE_CARRIER=" << failure_carrier
                << " ATTEMPT_BINDING=" << failure_binding << '\n';
    }
    std::cerr << "checkpoint worker error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
