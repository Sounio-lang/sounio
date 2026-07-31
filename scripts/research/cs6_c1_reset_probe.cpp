#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "capd/capdlib.h"

#ifndef CS6_WORKER_SOURCE_SHA256
#define CS6_WORKER_SOURCE_SHA256 "UNBOUND"
#endif

using capd::C0HOTripletonSet;
using capd::C1Rect2Set;
using capd::ICoordinateSection;
using capd::IMap;
using capd::IMatrix;
using capd::IOdeSolver;
using capd::IPoincareMap;
using capd::IVector;
using capd::interval;

namespace {

constexpr int kDimension = 3;
constexpr int kReturns = 6;
constexpr char kVectorField[] =
    "par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs;";
constexpr char kLiouvilleField[] =
    "par:zs;var:x,y,w,ell;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs,"
    "x-y-(w+zs)/2-1;";

interval decimal(const char* value) { return interval(value, value); }

struct HSet {
  const char* name;
  interval center_u;
  interval center_s;
  interval radius_u;
  interval radius_s;
};

struct Matrix2 {
  interval a00;
  interval a01;
  interval a10;
  interval a11;
};

enum class Strategy { kDirect, kSequential, kCanonical, kScaled };

const char* strategy_name(Strategy strategy) {
  switch (strategy) {
    case Strategy::kDirect:
      return "direct";
    case Strategy::kSequential:
      return "sequential";
    case Strategy::kCanonical:
      return "canonical-rebox";
    case Strategy::kScaled:
      return "dyadic-right-rebox";
  }
  throw std::logic_error("unreachable strategy");
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

bool finite(const IMatrix& value) {
  for (int row = 0; row < value.numberOfRows(); ++row) {
    for (int column = 0; column < value.numberOfColumns(); ++column) {
      if (!finite(value[row][column])) {
        return false;
      }
    }
  }
  return true;
}

bool positive_finite(const interval& value) {
  return finite(value) && value.leftBound() > 0.0;
}

bool normal_power_of_two(double value) {
  if (!std::isnormal(value) || value <= 0.0) {
    return false;
  }
  int exponent = 0;
  return std::frexp(value, &exponent) == 0.5;
}

bool equal(const interval& left, const interval& right) {
  return left.leftBound() == right.leftBound() &&
         left.rightBound() == right.rightBound();
}

bool equal(const IVector& left, const IVector& right) {
  if (left.dimension() != right.dimension()) {
    return false;
  }
  for (int row = 0; row < left.dimension(); ++row) {
    if (!equal(left[row], right[row])) {
      return false;
    }
  }
  return true;
}

bool equal(const IMatrix& left, const IMatrix& right) {
  if (left.numberOfRows() != right.numberOfRows() ||
      left.numberOfColumns() != right.numberOfColumns()) {
    return false;
  }
  for (int row = 0; row < left.numberOfRows(); ++row) {
    for (int column = 0; column < left.numberOfColumns(); ++column) {
      if (!equal(left[row][column], right[row][column])) {
        return false;
      }
    }
  }
  return true;
}

bool subset(const IMatrix& inner, const IMatrix& outer) {
  if (inner.numberOfRows() != outer.numberOfRows() ||
      inner.numberOfColumns() != outer.numberOfColumns()) {
    return false;
  }
  for (int row = 0; row < inner.numberOfRows(); ++row) {
    for (int column = 0; column < inner.numberOfColumns(); ++column) {
      if (inner[row][column].leftBound() < outer[row][column].leftBound() ||
          inner[row][column].rightBound() > outer[row][column].rightBound()) {
        return false;
      }
    }
  }
  return true;
}

bool overlaps(const interval& left, const interval& right) {
  return left.leftBound() <= right.rightBound() &&
         right.leftBound() <= left.rightBound();
}

bool overlaps(const IVector& left, const IVector& right) {
  if (left.dimension() != right.dimension()) {
    return false;
  }
  for (int row = 0; row < left.dimension(); ++row) {
    if (!overlaps(left[row], right[row])) {
      return false;
    }
  }
  return true;
}

bool overlaps(const IMatrix& left, const IMatrix& right) {
  if (left.numberOfRows() != right.numberOfRows() ||
      left.numberOfColumns() != right.numberOfColumns()) {
    return false;
  }
  for (int row = 0; row < left.numberOfRows(); ++row) {
    for (int column = 0; column < left.numberOfColumns(); ++column) {
      if (!overlaps(left[row][column], right[row][column])) {
        return false;
      }
    }
  }
  return true;
}

bool joint_overlap(const std::vector<interval>& values) {
  if (values.empty()) {
    return false;
  }
  double lower = -std::numeric_limits<double>::infinity();
  double upper = std::numeric_limits<double>::infinity();
  for (const interval& value : values) {
    if (!finite(value)) {
      return false;
    }
    lower = std::max(lower, value.leftBound());
    upper = std::min(upper, value.rightBound());
  }
  return lower <= upper;
}

bool joint_overlap(const std::vector<IVector>& values) {
  if (values.empty()) {
    return false;
  }
  for (int row = 0; row < values.front().dimension(); ++row) {
    std::vector<interval> coordinate;
    for (const IVector& value : values) {
      if (value.dimension() != values.front().dimension()) {
        return false;
      }
      coordinate.push_back(value[row]);
    }
    if (!joint_overlap(coordinate)) {
      return false;
    }
  }
  return true;
}

bool joint_overlap(const std::vector<IMatrix>& values) {
  if (values.empty()) {
    return false;
  }
  for (int row = 0; row < values.front().numberOfRows(); ++row) {
    for (int column = 0; column < values.front().numberOfColumns(); ++column) {
      std::vector<interval> entry;
      for (const IMatrix& value : values) {
        if (value.numberOfRows() != values.front().numberOfRows() ||
            value.numberOfColumns() != values.front().numberOfColumns()) {
          return false;
        }
        entry.push_back(value[row][column]);
      }
      if (!joint_overlap(entry)) {
        return false;
      }
    }
  }
  return true;
}

bool identity(const IMatrix& matrix) {
  if (matrix.numberOfRows() != kDimension ||
      matrix.numberOfColumns() != kDimension) {
    return false;
  }
  for (int row = 0; row < kDimension; ++row) {
    for (int column = 0; column < kDimension; ++column) {
      const interval expected(row == column ? 1.0 : 0.0);
      if (!equal(matrix[row][column], expected)) {
        return false;
      }
    }
  }
  return true;
}

bool zero(const IMatrix& matrix) {
  if (matrix.numberOfRows() != kDimension ||
      matrix.numberOfColumns() != kDimension || !finite(matrix)) {
    return false;
  }
  for (int row = 0; row < matrix.numberOfRows(); ++row) {
    for (int column = 0; column < matrix.numberOfColumns(); ++column) {
      if (!equal(matrix[row][column], interval(0.0))) {
        return false;
      }
    }
  }
  return true;
}

bool centered(const IMatrix& matrix) {
  if (matrix.numberOfRows() != kDimension ||
      matrix.numberOfColumns() != kDimension || !finite(matrix)) {
    return false;
  }
  for (int row = 0; row < matrix.numberOfRows(); ++row) {
    for (int column = 0; column < matrix.numberOfColumns(); ++column) {
      if (!(matrix[row][column].leftBound() <= 0.0 &&
            matrix[row][column].rightBound() >= 0.0)) {
        return false;
      }
    }
  }
  return true;
}

bool zero_third_column(const IMatrix& matrix) {
  if (matrix.numberOfColumns() != kDimension) {
    return false;
  }
  for (int row = 0; row < matrix.numberOfRows(); ++row) {
    if (!equal(matrix[row][2], interval(0.0))) {
      return false;
    }
  }
  return true;
}

double max_width(const IMatrix& matrix) {
  double result = 0.0;
  for (int row = 0; row < matrix.numberOfRows(); ++row) {
    for (int column = 0; column < matrix.numberOfColumns(); ++column) {
      result = std::max(result, matrix[row][column].rightBound() -
                                    matrix[row][column].leftBound());
    }
  }
  return result;
}

double max_width(const Matrix2& matrix) {
  return std::max(
      {matrix.a00.rightBound() - matrix.a00.leftBound(),
       matrix.a01.rightBound() - matrix.a01.leftBound(),
       matrix.a10.rightBound() - matrix.a10.leftBound(),
       matrix.a11.rightBound() - matrix.a11.leftBound()});
}

IMatrix right_scale(const IMatrix& matrix,
                    const std::array<double, kDimension>& scale) {
  IMatrix result(matrix.numberOfRows(), matrix.numberOfColumns());
  for (int row = 0; row < matrix.numberOfRows(); ++row) {
    for (int column = 0; column < matrix.numberOfColumns(); ++column) {
      result[row][column] = matrix[row][column] * scale[column];
    }
  }
  return result;
}

std::array<double, kDimension> dyadic_scale(const IMatrix& matrix) {
  std::array<double, kDimension> result{1.0, 1.0, 1.0};
  for (int column = 0; column < 2; ++column) {
    double magnitude = 0.0;
    for (int row = 0; row < kDimension; ++row) {
      magnitude = std::max(
          magnitude,
          std::max(std::abs(matrix[row][column].leftBound()),
                   std::abs(matrix[row][column].rightBound())));
    }
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
      continue;
    }
    int exponent = 0;
    const double fraction = std::frexp(magnitude, &exponent);
    if (fraction == 0.5) {
      --exponent;
    }
    if (exponent < -500 || exponent > 500) {
      throw std::runtime_error("dyadic reset exponent outside frozen range");
    }
    result[column] = std::ldexp(1.0, exponent);
  }
  return result;
}

std::string fingerprint(const C1Rect2Set& set) {
  const auto& c0 = static_cast<const C1Rect2Set::C0BaseSet&>(set);
  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::hexfloat << static_cast<IVector>(set) << '|'
         << set.getCurrentTime() << '|' << set.getLastEnclosure() << '|'
         << c0.get_x() << '|' << c0.get_C() << '|' << c0.get_r0() << '|'
         << c0.get_B() << '|' << c0.get_invB() << '|' << c0.get_r();
  return output.str();
}

std::string scratch_fingerprint(const C1Rect2Set& set) {
  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::hexfloat << set.x << '|' << set.deltaX << '|' << set.y << '|'
         << set.deltaY << '|' << set.rem << '|' << set.enc << '|'
         << set.jacPhi << '|' << set.deltaC << '|' << set.B << '|'
         << set.jacRem << '|' << set.jacEnc << '|'
         << set.getC0Factor() << '|' << set.getC1Factor() << '|'
         << set.getC2Factor() << '|' << set.isReorganizationEnabled();
  return output.str();
}

struct ResetAudit {
  int return_index = 0;
  IMatrix pre_internal{3, 3};
  IMatrix post_current{3, 3};
  IMatrix post_doubleton{3, 3};
  std::array<double, kDimension> old_external{1.0, 1.0, 1.0};
  std::array<double, kDimension> scale{1.0, 1.0, 1.0};
  std::array<double, kDimension> new_external{1.0, 1.0, 1.0};
  bool c0_unchanged = false;
  bool scratch_unchanged = false;
  bool last_matrix_unchanged = false;
  bool current_exact = false;
  bool current_contains_candidate = false;
  bool doubleton_contains_candidate = false;
  bool physical_carrier_contains_pre = false;
  bool inverse_basis_identity = false;
  bool canonical_form = false;
  bool third_column_zero = false;
  bool scale_chain_valid = false;

  bool valid() const {
    return c0_unchanged && scratch_unchanged && last_matrix_unchanged &&
           current_exact && current_contains_candidate &&
           doubleton_contains_candidate && physical_carrier_contains_pre &&
           inverse_basis_identity && canonical_form && third_column_zero &&
           scale_chain_valid;
  }
};

class ResettableC1Rect2Set : public C1Rect2Set {
 public:
  using C1Rect2Set::C1Rect2Set;

  ResetAudit rebox(int return_index,
                   const std::array<double, kDimension>& scale,
                   const std::array<double, kDimension>& old_external) {
    ResetAudit audit;
    audit.return_index = return_index;
    audit.pre_internal = static_cast<IMatrix>(*this);
    audit.old_external = old_external;
    audit.scale = scale;
    for (int column = 0; column < kDimension; ++column) {
      if (!normal_power_of_two(old_external[column]) ||
          !normal_power_of_two(scale[column])) {
        throw std::runtime_error(
            "reset charts must be positive normal powers of two");
      }
      audit.new_external[column] = scale[column] * old_external[column];
      if (!normal_power_of_two(audit.new_external[column]) ||
          audit.new_external[column] / old_external[column] != scale[column]) {
        throw std::runtime_error("external reset chart underflow or overflow");
      }
    }
    if (old_external[2] != 1.0 || scale[2] != 1.0 ||
        audit.new_external[2] != 1.0) {
      throw std::runtime_error("normal chart coordinate must remain identity");
    }
    audit.scale_chain_valid = true;

    std::array<double, kDimension> inverse_scale{};
    for (int column = 0; column < kDimension; ++column) {
      inverse_scale[column] = 1.0 / scale[column];
      if (!normal_power_of_two(inverse_scale[column])) {
        throw std::runtime_error("inverse reset scale is outside normal range");
      }
    }
    const IMatrix candidate = right_scale(audit.pre_internal, inverse_scale);
    if (!finite(candidate)) {
      throw std::runtime_error("non-finite reset candidate");
    }

    const std::string c0_before = fingerprint(*this);
    const std::string scratch_before = scratch_fingerprint(*this);
    const IMatrix last_before = getLastMatrixEnclosure();

    C1BaseSet reset(candidate);
    static_cast<C1BaseSet&>(*this) = reset;
    m_invBjac = IMatrix::Identity(kDimension);
    m_currentMatrix = candidate;

    audit.post_current = static_cast<IMatrix>(*this);
    audit.post_doubleton =
        static_cast<IMatrix>(static_cast<const C1BaseSet&>(*this));
    audit.c0_unchanged = c0_before == fingerprint(*this);
    audit.scratch_unchanged = scratch_before == scratch_fingerprint(*this);
    audit.last_matrix_unchanged =
        equal(last_before, getLastMatrixEnclosure());
    audit.current_exact = equal(candidate, audit.post_current);
    audit.current_contains_candidate = subset(candidate, audit.post_current);
    audit.doubleton_contains_candidate = subset(candidate, audit.post_doubleton);
    const IMatrix physical_before =
        right_scale(audit.pre_internal, audit.old_external);
    const IMatrix physical_after =
        right_scale(audit.post_current, audit.new_external);
    audit.physical_carrier_contains_pre =
        finite(physical_before) && finite(physical_after) &&
        subset(physical_before, physical_after);
    audit.inverse_basis_identity = identity(m_invBjac);
    audit.canonical_form = identity(get_Cjac()) && identity(get_Bjac()) &&
                           identity(get_invBjac()) && centered(get_R0()) &&
                           zero(get_R());
    audit.third_column_zero = zero_third_column(candidate) &&
                              zero_third_column(audit.post_current) &&
                              zero_third_column(audit.post_doubleton);
    if (!audit.valid()) {
      throw std::runtime_error("atomic C1 rebox invariant failed");
    }
    return audit;
  }
};

struct Prefix {
  int return_index = 0;
  interval time;
  IVector image{3};
  interval normal_velocity;
  IMatrix dp{3, 3};
  interval determinant;
};

struct LiouvillePrefix {
  int return_index = 0;
  interval time;
  IVector image{4};
  interval normal_velocity;
  interval integral_divergence;
  interval exponential_divergence;
  interval determinant_with_source_frame;
};

struct LiouvilleRun {
  bool success = false;
  std::string error;
  std::vector<LiouvillePrefix> prefixes;
};

struct RunResult {
  explicit RunResult(Strategy value) : strategy(value) {}

  Strategy strategy;
  bool success = false;
  std::string error;
  std::vector<Prefix> prefixes;
  std::vector<ResetAudit> resets;
  IMatrix final_dp{3, 3};
  IVector final_image{3};
  interval final_time;
  Matrix2 normalized;
  double elapsed_seconds = 0.0;
};

bool valid_prefix_chain(const std::vector<Prefix>& prefixes) {
  if (prefixes.size() != kReturns) {
    return false;
  }
  for (int index = 0; index < kReturns; ++index) {
    const Prefix& prefix = prefixes[index];
    if (prefix.return_index != index + 1 || !positive_finite(prefix.time) ||
        !finite(prefix.image) || !positive_finite(prefix.normal_velocity) ||
        !finite(prefix.dp) || !finite(prefix.determinant) ||
        !zero_third_column(prefix.dp)) {
      return false;
    }
    if (index > 0 &&
        prefixes[index - 1].time.rightBound() >= prefix.time.leftBound()) {
      return false;
    }
  }
  return true;
}

class ProbeContext {
 public:
  explicit ProbeContext(int order)
      : zs_(decimal("22.3274637391")),
        origin_x_(decimal("15.186446520640786")),
        origin_y_(decimal("10.908543194765466")),
        unstable_x_(decimal("-0.67430316214199759")),
        unstable_y_(decimal("-0.73845463335624273")),
        stable_x_(decimal("-0.94170446778164518")),
        stable_y_(decimal("0.33644122125579123")),
        frame_determinant_(unstable_x_ * stable_y_ -
                           stable_x_ * unstable_y_),
        vector_field_(kVectorField),
        solver_(vector_field_, order),
        section_(3, 2),
        poincare_(solver_, section_, capd::poincare::MinusPlus),
        frame_(3, 3) {
    vector_field_.setParameter("zs", zs_);
    frame_.clear();
    frame_[0][0] = unstable_x_;
    frame_[1][0] = unstable_y_;
    frame_[0][1] = stable_x_;
    frame_[1][1] = stable_y_;
    frame_[2][2] = 1.0;
  }

  RunResult run(Strategy strategy, const HSet& source, const HSet& target,
                const interval& source_u, const interval& source_s) {
    RunResult result(strategy);
    const auto started = std::chrono::steady_clock::now();
    try {
      const interval midpoint_u = source_u.mid();
      const interval midpoint_s = source_s.mid();
      const interval radius_u = source_u - midpoint_u;
      const interval radius_s = source_s - midpoint_s;
      const IVector center{
          origin_x_ + unstable_x_ * midpoint_u + stable_x_ * midpoint_s,
          origin_y_ + unstable_y_ * midpoint_u + stable_y_ * midpoint_s,
          interval(0.0)};
      const IVector tile_radii{radius_u, radius_s, interval(0.0)};
      C1Rect2Set::C0BaseSet c0(center, frame_, tile_radii);
      IMatrix initial_tangent(3, 3);
      initial_tangent.clear();
      for (int row = 0; row < 3; ++row) {
        initial_tangent[row][0] = frame_[row][0] * source.radius_u;
        initial_tangent[row][1] = frame_[row][1] * source.radius_s;
      }
      C1Rect2Set::C1BaseSet c1(initial_tangent);
      ResettableC1Rect2Set set(c0, c1);
      std::array<double, kDimension> external{1.0, 1.0, 1.0};

      if (strategy == Strategy::kDirect) {
        IMatrix flow_derivative(3, 3);
        result.final_image =
            poincare_(set, flow_derivative, result.final_time, kReturns);
        result.final_dp = poincare_.computeDP(
            result.final_image, flow_derivative, result.final_time);
      } else {
        for (int index = 1; index <= kReturns; ++index) {
          IMatrix flow_derivative(3, 3);
          interval return_time;
          const IVector image =
              poincare_(set, flow_derivative, return_time, 1);
          const IMatrix internal_dp =
              poincare_.computeDP(image, flow_derivative, return_time);
          const IMatrix physical_dp = right_scale(internal_dp, external);
          Prefix prefix;
          prefix.return_index = index;
          prefix.time = return_time;
          prefix.image = image;
          prefix.normal_velocity = image[0] * image[1] - image[2] - zs_;
          prefix.dp = physical_dp;
          prefix.determinant = physical_dp[0][0] * physical_dp[1][1] -
                               physical_dp[0][1] * physical_dp[1][0];
          result.prefixes.push_back(prefix);

          if (index < kReturns &&
              (strategy == Strategy::kCanonical ||
               strategy == Strategy::kScaled)) {
            const IMatrix postsection = static_cast<IMatrix>(set);
            const std::array<double, kDimension> scale =
                strategy == Strategy::kScaled
                    ? dyadic_scale(postsection)
                    : std::array<double, kDimension>{1.0, 1.0, 1.0};
            ResetAudit audit = set.rebox(index, scale, external);
            external = audit.new_external;
            result.resets.push_back(audit);
          }
        }
        const Prefix& final = result.prefixes.back();
        result.final_dp = final.dp;
        result.final_image = final.image;
        result.final_time = final.time;
      }

      const interval local00 =
          (stable_y_ * result.final_dp[0][0] -
           stable_x_ * result.final_dp[1][0]) /
          frame_determinant_;
      const interval local01 =
          (stable_y_ * result.final_dp[0][1] -
           stable_x_ * result.final_dp[1][1]) /
          frame_determinant_;
      const interval local10 =
          (-unstable_y_ * result.final_dp[0][0] +
           unstable_x_ * result.final_dp[1][0]) /
          frame_determinant_;
      const interval local11 =
          (-unstable_y_ * result.final_dp[0][1] +
           unstable_x_ * result.final_dp[1][1]) /
          frame_determinant_;
      result.normalized = {local00 / target.radius_u,
                           local01 / target.radius_u,
                           local10 / target.radius_s,
                           local11 / target.radius_s};
      result.success = finite(result.final_dp) && finite(result.final_image) &&
                       positive_finite(result.final_time) &&
                       positive_finite(result.final_image[0] *
                                           result.final_image[1] -
                                       result.final_image[2] - zs_) &&
                       finite(result.normalized.a00) &&
                       finite(result.normalized.a01) &&
                       finite(result.normalized.a10) &&
                       finite(result.normalized.a11) &&
                       zero_third_column(result.final_dp) &&
                       (strategy == Strategy::kDirect ||
                        valid_prefix_chain(result.prefixes)) &&
                       (strategy != Strategy::kCanonical &&
                                strategy != Strategy::kScaled
                            ? result.resets.empty()
                            : result.resets.size() == kReturns - 1 &&
                                  std::all_of(
                                      result.resets.begin(), result.resets.end(),
                                      [](const ResetAudit& audit) {
                                        return audit.valid();
                                      }));
    } catch (const std::exception& error) {
      result.error = error.what();
      result.success = false;
    }
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count();
    return result;
  }

  const interval& zs() const { return zs_; }
  const interval& frame_determinant() const { return frame_determinant_; }
  const IMatrix& frame() const { return frame_; }

 private:
  interval zs_;
  interval origin_x_;
  interval origin_y_;
  interval unstable_x_;
  interval unstable_y_;
  interval stable_x_;
  interval stable_y_;
  interval frame_determinant_;
  IMap vector_field_;
  IOdeSolver solver_;
  ICoordinateSection section_;
  IPoincareMap poincare_;
  IMatrix frame_;
};

class LiouvilleContext {
 public:
  explicit LiouvilleContext(int order)
      : zs_(decimal("22.3274637391")),
        origin_x_(decimal("15.186446520640786")),
        origin_y_(decimal("10.908543194765466")),
        unstable_x_(decimal("-0.67430316214199759")),
        unstable_y_(decimal("-0.73845463335624273")),
        stable_x_(decimal("-0.94170446778164518")),
        stable_y_(decimal("0.33644122125579123")),
        frame_determinant_(unstable_x_ * stable_y_ -
                           stable_x_ * unstable_y_),
        vector_field_(kLiouvilleField),
        solver_(vector_field_, order),
        section_(4, 2),
        poincare_(solver_, section_, capd::poincare::MinusPlus),
        frame_(4, 4) {
    vector_field_.setParameter("zs", zs_);
    frame_.setToIdentity();
    frame_[0][0] = unstable_x_;
    frame_[1][0] = unstable_y_;
    frame_[0][1] = stable_x_;
    frame_[1][1] = stable_y_;
  }

  LiouvilleRun run(const HSet& source, const interval& source_u,
                   const interval& source_s) {
    LiouvilleRun result;
    try {
      const interval midpoint_u = source_u.mid();
      const interval midpoint_s = source_s.mid();
      const interval radius_u = source_u - midpoint_u;
      const interval radius_s = source_s - midpoint_s;
      const IVector center{
          origin_x_ + unstable_x_ * midpoint_u + stable_x_ * midpoint_s,
          origin_y_ + unstable_y_ * midpoint_u + stable_y_ * midpoint_s,
          interval(0.0), interval(0.0)};
      const IVector tile_radii{radius_u, radius_s, interval(0.0),
                               interval(0.0)};
      C0HOTripletonSet set(center, frame_, tile_radii);
      const interval initial_x =
          origin_x_ + unstable_x_ * source_u + stable_x_ * source_s;
      const interval initial_y =
          origin_y_ + unstable_y_ * source_u + stable_y_ * source_s;
      const interval initial_velocity = initial_x * initial_y - zs_;
      const interval source_frame_determinant =
          frame_determinant_ * source.radius_u * source.radius_s;
      if (!positive_finite(initial_velocity) ||
          !finite(source_frame_determinant) ||
          source_frame_determinant.rightBound() >= 0.0) {
        throw std::runtime_error("invalid Liouville source orientation");
      }
      for (int index = 1; index <= kReturns; ++index) {
        LiouvillePrefix prefix;
        prefix.return_index = index;
        prefix.image = poincare_(set, prefix.time, 1);
        prefix.normal_velocity =
            prefix.image[0] * prefix.image[1] - prefix.image[2] - zs_;
        prefix.integral_divergence = prefix.image[3];
        prefix.exponential_divergence = exp(prefix.integral_divergence);
        prefix.determinant_with_source_frame =
            prefix.exponential_divergence * initial_velocity /
            prefix.normal_velocity * source_frame_determinant;
        if (prefix.return_index != index || !positive_finite(prefix.time) ||
            !finite(prefix.image) || !positive_finite(prefix.normal_velocity) ||
            !finite(prefix.integral_divergence) ||
            !positive_finite(prefix.exponential_divergence) ||
            !finite(prefix.determinant_with_source_frame) ||
            prefix.determinant_with_source_frame.rightBound() >= 0.0 ||
            (!result.prefixes.empty() &&
             result.prefixes.back().time.rightBound() >=
                 prefix.time.leftBound())) {
          throw std::runtime_error("invalid Liouville prefix enclosure");
        }
        result.prefixes.push_back(prefix);
      }
      result.success = result.prefixes.size() == kReturns;
    } catch (const std::exception& error) {
      result.error = error.what();
      result.success = false;
    }
    return result;
  }

 private:
  interval zs_;
  interval origin_x_;
  interval origin_y_;
  interval unstable_x_;
  interval unstable_y_;
  interval stable_x_;
  interval stable_y_;
  interval frame_determinant_;
  IMap vector_field_;
  IOdeSolver solver_;
  ICoordinateSection section_;
  IPoincareMap poincare_;
  IMatrix frame_;
};

int positive_int(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 1 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

int nonnegative_int(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
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

void write_matrix(std::ostream& output, const std::string& prefix,
                  const IMatrix& matrix) {
  for (int row = 0; row < matrix.numberOfRows(); ++row) {
    for (int column = 0; column < matrix.numberOfColumns(); ++column) {
      write_interval(output,
                     prefix + std::to_string(row) + std::to_string(column),
                     matrix[row][column]);
    }
  }
}

void write_vector(std::ostream& output, const std::string& prefix,
                  const IVector& vector, int dimensions) {
  for (int row = 0; row < dimensions; ++row) {
    write_interval(output, prefix + std::to_string(row), vector[row]);
  }
}

void write_scale(std::ostream& output, const std::string& prefix,
                 const std::array<double, kDimension>& scale) {
  for (int column = 0; column < kDimension; ++column) {
    output << ' ' << prefix << column << '=' << std::hexfloat << scale[column]
           << std::defaultfloat;
  }
}

std::string safe_token(std::string value) {
  for (char& character : value) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (!(std::isalnum(byte) || character == '.' || character == '-' ||
          character == '_' || character == ':' || character == '/')) {
      character = '_';
    }
  }
  return value;
}

void emit_result(const RunResult& result) {
  std::cout << "RESULT STRATEGY=" << strategy_name(result.strategy)
            << " SUCCESS=" << result.success
            << " RESET_COUNT=" << result.resets.size()
            << " PREFIX_COUNT=" << result.prefixes.size()
            << " ELAPSED_SECONDS=" << result.elapsed_seconds;
  if (!result.success) {
    std::cout << " ERROR=" << safe_token(result.error) << '\n';
    return;
  }
  write_interval(std::cout, "TIME", result.final_time);
  write_vector(std::cout, "X", result.final_image, 3);
  write_matrix(std::cout, "DP", result.final_dp);
  write_interval(std::cout, "A00", result.normalized.a00);
  write_interval(std::cout, "A01", result.normalized.a01);
  write_interval(std::cout, "A10", result.normalized.a10);
  write_interval(std::cout, "A11", result.normalized.a11);
  std::cout << " DP_MAX_WIDTH=" << max_width(result.final_dp)
            << " A_MAX_WIDTH=" << max_width(result.normalized) << '\n';

  for (const Prefix& prefix : result.prefixes) {
    std::cout << "PREFIX STRATEGY=" << strategy_name(result.strategy)
              << " RETURN=" << prefix.return_index;
    write_interval(std::cout, "TIME", prefix.time);
    write_vector(std::cout, "X", prefix.image, 3);
    write_interval(std::cout, "NU", prefix.normal_velocity);
    write_matrix(std::cout, "DP", prefix.dp);
    write_interval(std::cout, "DET", prefix.determinant);
    std::cout << '\n';
  }

  for (const ResetAudit& audit : result.resets) {
    std::cout << "RESET STRATEGY=" << strategy_name(result.strategy)
              << " RETURN=" << audit.return_index
              << " CANDIDATE_SOURCE=POSTSECTION_CURRENT_MATRIX"
              << " C0_UNCHANGED=" << audit.c0_unchanged
              << " SCRATCH_POLICY_UNCHANGED=" << audit.scratch_unchanged
              << " LAST_MATRIX_UNCHANGED=" << audit.last_matrix_unchanged
              << " CURRENT_EXACT=" << audit.current_exact
              << " CURRENT_CONTAINS_CANDIDATE="
              << audit.current_contains_candidate
              << " DOUBLETON_CONTAINS_CANDIDATE="
              << audit.doubleton_contains_candidate
              << " PHYSICAL_CARRIER_CONTAINS_PRE="
              << audit.physical_carrier_contains_pre
              << " INVERSE_BASIS_IDENTITY=" << audit.inverse_basis_identity
              << " CANONICAL_FORM=" << audit.canonical_form
              << " THIRD_COLUMN_ZERO=" << audit.third_column_zero
              << " SCALE_CHAIN_VALID=" << audit.scale_chain_valid;
    write_scale(std::cout, "OLD_E", audit.old_external);
    write_scale(std::cout, "S", audit.scale);
    write_scale(std::cout, "NEW_E", audit.new_external);
    write_matrix(std::cout, "PRE", audit.pre_internal);
    write_matrix(std::cout, "POST", audit.post_current);
    write_matrix(std::cout, "BOX", audit.post_doubleton);
    std::cout << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::boolalpha;
  try {
    if (argc != 9 || std::string(argv[1]) != "probe") {
      throw std::runtime_error(
          "usage: probe N0|N1 N0|N1 U_INDEX S_INDEX U_TILES S_TILES ORDER");
    }
    const HSet n0{"N0", interval(0.0), interval(0.0), decimal("0.004"),
                  decimal("0.3")};
    const HSet n1{"N1", decimal("0.019771776972779206"), interval(0.0),
                  decimal("0.0015"), decimal("0.3")};
    const std::string source_name = argv[2];
    const std::string target_name = argv[3];
    if ((source_name != "N0" && source_name != "N1") ||
        (target_name != "N0" && target_name != "N1") ||
        (source_name == "N1" && target_name != "N0")) {
      throw std::runtime_error("edge is not in the frozen adjacency");
    }
    const HSet& source = source_name == "N1" ? n1 : n0;
    const HSet& target = target_name == "N1" ? n1 : n0;
    const int u_index = nonnegative_int(argv[4], "u_index");
    const int s_index = nonnegative_int(argv[5], "s_index");
    const int u_tiles = positive_int(argv[6], "u_tiles");
    const int s_tiles = positive_int(argv[7], "s_tiles");
    const int order = positive_int(argv[8], "order");
    if (u_index >= u_tiles || s_index >= s_tiles) {
      throw std::runtime_error("tile index outside partition");
    }
    const interval source_u =
        tile(source.center_u, source.radius_u, u_index, u_tiles);
    const interval source_s =
        tile(source.center_s, source.radius_s, s_index, s_tiles);

    std::cout
        << "SCHEMA=sounio.cs6.c1-representation-rebox-probe.v1\n"
        << "RESET_SEMANTICS=REPRESENTATION_PRESERVING_CUMULATIVE_JRAW_REBOX\n"
        << "CALL_PATTERN=6xP1_SAME_MUTABLE_SET\n"
        << "LOCAL_FACTOR_CHAIN=false\n"
        << "PREFIX_DP_PRODUCT_FORBIDDEN=true\n"
        << "FINAL_DP=PREFIX_6_ONLY\n"
        << "REBOX_COUNT=5\n"
        << "C0_CARRIER_RESET=false\n"
        << "LIOUVILLE_CARRIER_RESET=false\n"
        << "EVENT_DP_REINJECTION=false\n"
        << "RIGHT_REPARAMETERIZATION_ONLY=true\n"
        << "LAST_MATRIX_POLICY=PRESERVE_EXACTLY_NO_REPARAMETERIZATION\n"
        << "TARGET_NORMALIZATION_ONLY=true\n"
        << "LIOUVILLE_REJECT_ONLY=true\n"
        << "C1_CLIPPED_BY_LIOUVILLE=false\n"
        << "C1_SET_SUBTYPE=ResettableC1Rect2Set\n"
        << "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0\n"
        << "INTERVAL_BACKEND_DECLARED=FILIB\n"
        << "INTERVAL_SERIALIZATION=ONE_ULP_OUTWARD_BINARY64_HEX\n"
        << "WORKER_SOURCE_SHA256=" << CS6_WORKER_SOURCE_SHA256 << '\n'
        << "EXECUTION_SCOPE=BOUNDED_LOCAL_CAPD_CPU_PROBE\n"
        << "EXECUTION_PROVENANCE_ATTESTED=false\n"
        << "INDEPENDENT_REPLAY_REQUIRED=true\n"
        << "PROMOTION_ELIGIBLE=false\n"
        << "RESET_AUDIT_MODEL=HASHED_WORKER_REPLAY_TCB_SELF_REPORTED_FLAGS\n"
        << "VECTOR_FIELD_CAPD=" << kVectorField << '\n'
        << "LIOUVILLE_FIELD_CAPD=" << kLiouvilleField << '\n'
        << "SOURCE=" << source.name << '\n'
        << "TARGET=" << target.name << '\n'
        << "U_INDEX=" << u_index << '\n'
        << "S_INDEX=" << s_index << '\n'
        << "U_TILES=" << u_tiles << '\n'
        << "S_TILES=" << s_tiles << '\n'
        << "ORDER=" << order << '\n';
    write_interval(std::cout, "SOURCE_U", source_u);
    write_interval(std::cout, "SOURCE_S", source_s);
    std::cout << '\n';

    std::vector<RunResult> results;
    for (Strategy strategy : {Strategy::kDirect, Strategy::kSequential,
                              Strategy::kCanonical, Strategy::kScaled}) {
      ProbeContext context(order);
      results.push_back(
          context.run(strategy, source, target, source_u, source_s));
      emit_result(results.back());
    }

    LiouvilleContext liouville_context(order);
    const LiouvilleRun liouville =
        liouville_context.run(source, source_u, source_s);
    std::cout << "LIOUVILLE_STATUS SUCCESS=" << liouville.success
              << " PREFIX_COUNT=" << liouville.prefixes.size();
    if (!liouville.success) {
      std::cout << " ERROR=" << safe_token(liouville.error);
    }
    std::cout << '\n';
    for (const LiouvillePrefix& prefix : liouville.prefixes) {
      std::cout << "LIOUVILLE RETURN=" << prefix.return_index;
      write_interval(std::cout, "TIME", prefix.time);
      write_vector(std::cout, "X", prefix.image, 3);
      write_interval(std::cout, "NU", prefix.normal_velocity);
      write_interval(std::cout, "ELL", prefix.integral_divergence);
      write_interval(std::cout, "EXP_ELL", prefix.exponential_divergence);
      write_interval(std::cout, "DET_SOURCE_FRAME",
                     prefix.determinant_with_source_frame);
      std::cout << '\n';
    }

    bool all_success = liouville.success &&
                       liouville.prefixes.size() == kReturns;
    for (const RunResult& result : results) {
      all_success = all_success && result.success;
    }
    bool final_overlap = all_success;
    bool c0_overlap = all_success;
    bool time_overlap = all_success;
    if (all_success) {
      std::vector<IMatrix> final_matrices;
      std::vector<IVector> final_images;
      std::vector<interval> final_times;
      for (const RunResult& result : results) {
        final_matrices.push_back(result.final_dp);
        final_images.push_back(result.final_image);
        final_times.push_back(result.final_time);
      }
      final_overlap = joint_overlap(final_matrices);
      c0_overlap = joint_overlap(final_images);
      time_overlap = joint_overlap(final_times);
    }
    bool prefix_c1_dp_joint_overlap = all_success;
    bool prefix_liouville_overlap = all_success;
    bool prefix_liouville_determinant_overlap = all_success;
    if (all_success) {
      for (int index = 0; index < kReturns; ++index) {
        std::vector<IMatrix> c1_dp;
        std::vector<interval> times;
        std::vector<IVector> images;
        std::vector<interval> normal_velocities;
        std::vector<interval> determinants;
        for (std::size_t run = 1; run < results.size(); ++run) {
          const Prefix& prefix = results[run].prefixes[index];
          c1_dp.push_back(prefix.dp);
          times.push_back(prefix.time);
          images.push_back(prefix.image);
          normal_velocities.push_back(prefix.normal_velocity);
          determinants.push_back(prefix.determinant);
        }
        const LiouvillePrefix& independent = liouville.prefixes[index];
        times.push_back(independent.time);
        IVector independent_image(3);
        for (int row = 0; row < 3; ++row) {
          independent_image[row] = independent.image[row];
        }
        images.push_back(independent_image);
        normal_velocities.push_back(independent.normal_velocity);
        determinants.push_back(independent.determinant_with_source_frame);
        prefix_c1_dp_joint_overlap =
            prefix_c1_dp_joint_overlap && joint_overlap(c1_dp);
        prefix_liouville_overlap =
            prefix_liouville_overlap && joint_overlap(times) &&
            joint_overlap(images) && joint_overlap(normal_velocities);
        prefix_liouville_determinant_overlap =
            prefix_liouville_determinant_overlap &&
            joint_overlap(determinants);
      }
    }
    const bool probe_pass = all_success && final_overlap && c0_overlap &&
                            time_overlap && prefix_c1_dp_joint_overlap &&
                            prefix_liouville_overlap &&
                            prefix_liouville_determinant_overlap;
    std::cout << "SUMMARY ALL_STRATEGIES_SUCCESS=" << all_success
              << " FINAL_DP_OVERLAP=" << final_overlap
              << " FINAL_C0_OVERLAP=" << c0_overlap
              << " FINAL_TIME_OVERLAP=" << time_overlap
              << " PREFIX_C1_DP_JOINT_OVERLAP="
              << prefix_c1_dp_joint_overlap
              << " PREFIX_LIOUVILLE_OVERLAP=" << prefix_liouville_overlap
              << " PREFIX_LIOUVILLE_DETERMINANT_OVERLAP="
              << prefix_liouville_determinant_overlap
              << " PROBE_PASS=" << probe_pass << '\n'
              << "C1_REBOX_SCALING_BLOCKER_RESOLVED=false\n"
              << "FULL_SOURCE_C1_DERIVATIVE_ENCLOSURE_PROVED=false\n"
              << "GLOBAL_FULL_SOURCE_HULL_TESTED=false\n"
              << "PAIRWISE_CHORD_CONE_CONDITION_PROVED=false\n"
              << "UNIFORM_HYPERBOLICITY_PROVED=false\n"
              << "CHAOTIC_ATTRACTOR_PROVED=false\n";
    return probe_pass ? EXIT_SUCCESS : 2;
  } catch (const std::exception& error) {
    std::cerr << "CS6_C1_RESET_PROBE_ERROR=" << error.what() << '\n';
    return 3;
  }
}
