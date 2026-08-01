#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "capd/capdlib.h"

#ifndef CS6_WORKER_SOURCE_SHA256
#define CS6_WORKER_SOURCE_SHA256 "UNBOUND"
#endif

using capd::C0HOTripletonSet;
using capd::C1Rect2Set;
using capd::C2Rect2Set;
using capd::IC2OdeSolver;
using capd::IC2PoincareMap;
using capd::ICoordinateSection;
using capd::IHessian;
using capd::IMap;
using capd::IMatrix;
using capd::IOdeSolver;
using capd::IPoincareMap;
using capd::IVector;
using capd::interval;

namespace {

constexpr int kDimension = 3;
constexpr int kSectionCoordinate = 2;
constexpr int kOrder = 8;
constexpr int kMaxDyadicDepth = 30;

constexpr char kVectorField[] =
    "par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs;";
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
  if (consumed != std::string(token).size() || value < minimum ||
      value > maximum) {
    throw std::runtime_error(std::string("out-of-range ") + name);
  }
  return static_cast<int>(value);
}

int dyadic_count(int depth) { return 1 << depth; }

bool lowercase_sha256(const std::string& value) {
  if (value.size() != 64) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

double width(const interval& value) {
  return value.rightBound() - value.leftBound();
}

interval midpoint(const interval& value) {
  return interval(value.leftBound() + 0.5 * width(value));
}

interval centered_square(const interval& value) {
  return sqr(value);
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

bool finite(const IHessian& value) {
  for (int output = 0; output < kDimension; ++output) {
    for (int first = 0; first < kDimension; ++first) {
      for (int second = first; second < kDimension; ++second) {
        if (!finite(value(output, first, second))) {
          return false;
        }
      }
    }
  }
  return true;
}

bool overlaps(const interval& left, const interval& right) {
  return left.leftBound() <= right.rightBound() &&
         right.leftBound() <= left.rightBound();
}

bool overlaps(const IMatrix& left, const IMatrix& right) {
  for (int row = 0; row < left.numberOfRows(); ++row) {
    for (int column = 0; column < left.numberOfColumns(); ++column) {
      if (!overlaps(left[row][column], right[row][column])) {
        return false;
      }
    }
  }
  return true;
}

bool overlaps(const IVector& left, const IVector& right) {
  for (int row = 0; row < left.dimension(); ++row) {
    if (!overlaps(left[row], right[row])) {
      return false;
    }
  }
  return true;
}

bool contains(const interval& outer, const interval& inner) {
  return outer.leftBound() <= inner.leftBound() &&
         inner.rightBound() <= outer.rightBound();
}

bool contains(const IMatrix& outer, const IMatrix& inner) {
  for (int row = 0; row < outer.numberOfRows(); ++row) {
    for (int column = 0; column < outer.numberOfColumns(); ++column) {
      if (!contains(outer[row][column], inner[row][column])) {
        return false;
      }
    }
  }
  return true;
}

bool sign_definite(const interval& value) {
  return finite(value) && !value.contains(0.0);
}

bool same_strict_sign(const interval& left, const interval& right) {
  return (left.rightBound() < 0.0 && right.rightBound() < 0.0) ||
         (left.leftBound() > 0.0 && right.leftBound() > 0.0);
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

void write_matrix(std::ostream& output, const std::string& prefix,
                  const IMatrix& value) {
  for (int row = 0; row < value.numberOfRows(); ++row) {
    for (int column = 0; column < value.numberOfColumns(); ++column) {
      write_interval(output,
                     prefix + std::to_string(row) + std::to_string(column),
                     value[row][column]);
    }
  }
}

void write_hessian(std::ostream& output, const std::string& prefix,
                   const IHessian& value) {
  for (int image = 0; image < kDimension; ++image) {
    for (int first = 0; first < kDimension; ++first) {
      for (int second = first; second < kDimension; ++second) {
        write_interval(output,
                       prefix + std::to_string(image) +
                           std::to_string(first) + std::to_string(second),
                       value(image, first, second));
      }
    }
  }
}

interval determinant_xy(const IMatrix& value) {
  return value[0][0] * value[1][1] - value[0][1] * value[1][0];
}

IMatrix tangent_projection(const IMatrix& value) {
  IMatrix projected = value;
  for (int index = 0; index < kDimension; ++index) {
    if (!projected[kSectionCoordinate][index].contains(0.0) ||
        !projected[index][kSectionCoordinate].contains(0.0)) {
      throw std::runtime_error("coordinate section tangent was not zero");
    }
    projected[kSectionCoordinate][index] = interval(0.0);
    projected[index][kSectionCoordinate] = interval(0.0);
  }
  return projected;
}

IVector midpoint_vector(const IVector& value) {
  IVector result(value.dimension());
  for (int row = 0; row < value.dimension(); ++row) {
    result[row] = midpoint(value[row]);
  }
  return result;
}

IMatrix midpoint_matrix(const IMatrix& value) {
  IMatrix result(value.numberOfRows(), value.numberOfColumns());
  for (int row = 0; row < value.numberOfRows(); ++row) {
    for (int column = 0; column < value.numberOfColumns(); ++column) {
      result[row][column] = midpoint(value[row][column]);
    }
  }
  return result;
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

  IMatrix c0_frame() const {
    IMatrix frame(kDimension, kDimension);
    frame.clear();
    frame[0][0] = unstable_x;
    frame[1][0] = unstable_y;
    frame[0][1] = stable_x;
    frame[1][1] = stable_y;
    frame[2][2] = 1.0;
    return frame;
  }

  IVector c0_tile_radii() const {
    return IVector{source_u() - source_u().mid(),
                   source_s() - source_s().mid(), interval(0.0)};
  }

  IMatrix q0() const {
    IMatrix seed(kDimension, kDimension);
    seed.clear();
    seed[0][0] = unstable_x * radius_u;
    seed[1][0] = unstable_y * radius_u;
    seed[0][1] = stable_x * radius_s;
    seed[1][1] = stable_y * radius_s;
    return seed;
  }

  IVector normalized_delta() const {
    return IVector{(source_u() - source_u().mid()) / radius_u,
                   (source_s() - source_s().mid()) / radius_s,
                   interval(0.0)};
  }

  C1Rect2Set c1_set() const {
    C1Rect2Set::C0BaseSet c0(center(), c0_frame(), c0_tile_radii());
    C1Rect2Set::C1BaseSet c1(q0());
    return C1Rect2Set(c0, c1);
  }

  C2Rect2Set c2_set(bool point) const {
    C2Rect2Set::C0BaseSet c0 = point
        ? C2Rect2Set::C0BaseSet(center())
        : C2Rect2Set::C0BaseSet(center(), c0_frame(), c0_tile_radii());
    C2Rect2Set::C1BaseSet c1(q0());
    return C2Rect2Set(c0, c1);
  }
};

struct C1Data {
  interval time;
  IVector image = IVector(kDimension);
  IMatrix flow{kDimension, kDimension};
  IMatrix dp{kDimension, kDimension};
  interval normal_velocity;
};

C1Data c1_return(const FrozenInput& input, int return_count) {
  IMap field(kVectorField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(kDimension, kSectionCoordinate);
  IPoincareMap map(solver, section, capd::poincare::MinusPlus);
  C1Data result;
  C1Rect2Set set = input.c1_set();
  result.image = map(set, result.flow, result.time, return_count);
  result.dp = tangent_projection(
      map.computeDP(result.image, result.flow, result.time));
  result.normal_velocity =
      result.image[0] * result.image[1] - result.image[2] - input.zs;
  return result;
}

struct C2Data {
  interval time;
  IVector image = IVector(kDimension);
  IMatrix flow{kDimension, kDimension};
  IHessian flow_hessian{kDimension};
  IMatrix dp{kDimension, kDimension};
  IHessian d2p{kDimension};
  IVector dtime = IVector(kDimension);
  IMatrix d2time = IMatrix(kDimension, kDimension);
  IVector second_time_coefficient = IVector(kDimension);
  IVector reconstructed_dtime = IVector(kDimension);
  IMatrix reconstructed_d2time = IMatrix(kDimension, kDimension);
  interval normal_velocity;
};

void reconstruct_impact_time_derivatives(const FrozenInput& input,
                                         C2Data& result) {
  const interval x = result.image[0];
  const interval y = result.image[1];
  const interval w = result.image[2];
  IMatrix derivative(kDimension, kDimension);
  derivative.clear();
  derivative[0][0] = -y;
  derivative[0][1] = 4.0 * y - x;
  derivative[1][0] = y;
  derivative[1][1] = x - (w + input.zs) / 2.0;
  derivative[1][2] = -y / 2.0;
  derivative[2][0] = y;
  derivative[2][1] = x;
  derivative[2][2] = -1.0;
  const IMatrix derivative_times_flow = derivative * result.flow;
  result.reconstructed_d2time.clear();
  for (int first = 0; first < kDimension; ++first) {
    result.reconstructed_dtime[first] =
        -result.flow[kSectionCoordinate][first] / result.normal_velocity;
  }
  for (int first = 0; first < kDimension; ++first) {
    interval uncorrected =
        result.flow_hessian(kSectionCoordinate, first, first) +
        result.reconstructed_dtime[first] *
            (derivative_times_flow[kSectionCoordinate][first] +
             result.second_time_coefficient[kSectionCoordinate] *
                 result.reconstructed_dtime[first]);
    result.reconstructed_d2time[first][first] =
        -uncorrected / result.normal_velocity;
    for (int second = first + 1; second < kDimension; ++second) {
      uncorrected =
          result.flow_hessian(kSectionCoordinate, first, second) +
          derivative_times_flow[kSectionCoordinate][first] *
              result.reconstructed_dtime[second] +
          derivative_times_flow[kSectionCoordinate][second] *
              result.reconstructed_dtime[first] +
          2.0 * result.second_time_coefficient[kSectionCoordinate] *
              result.reconstructed_dtime[first] *
              result.reconstructed_dtime[second];
      result.reconstructed_d2time[first][second] =
          result.reconstructed_d2time[second][first] =
              -uncorrected / result.normal_velocity;
    }
  }
}

C2Data c2_return(const FrozenInput& input, bool point, int return_count) {
  IMap field(kVectorField);
  field.setParameter("zs", input.zs);
  IC2OdeSolver solver(field, kOrder);
  ICoordinateSection section(kDimension, kSectionCoordinate);
  IC2PoincareMap map(solver, section, capd::poincare::MinusPlus);
  C2Data result;
  C2Rect2Set set = input.c2_set(point);
  result.image = map(set, result.flow, result.flow_hessian, result.time,
                     return_count);
  map.computeDP(result.image, result.flow, result.flow_hessian, result.dp,
                result.d2p, result.dtime, result.d2time, result.time);
  solver.computeCoefficientsAtCenter(result.time, result.image, 2);
  for (int row = 0; row < kDimension; ++row) {
    result.second_time_coefficient[row] = solver.centerCoefficient(row, 2);
  }
  result.normal_velocity =
      result.image[0] * result.image[1] - result.image[2] - input.zs;
  reconstruct_impact_time_derivatives(input, result);
  result.dp = tangent_projection(result.dp);
  return result;
}

struct ResidentData {
  interval time;
  IVector image = IVector(kDimension);
  IMatrix flow{kDimension, kDimension};
  IMatrix dp{kDimension, kDimension};
  IVector postsection = IVector(kDimension);
  interval postsection_time;
  interval normal_velocity;
};

class SectionResidentMap : public IPoincareMap {
 public:
  using IPoincareMap::IPoincareMap;

  ResidentData one_return(C1Rect2Set source) {
    ResidentData result;
    C1Rect2Set before = source;
    C1Rect2Set after = source;
    this->sectionDerivativesEnclosure.init(
        &result.time, &result.flow, nullptr, nullptr);
    this->integrateUntilSectionCrossing(before, after, 1);
    interval local_time;
    result.image = static_cast<IVector>(before);
    if (!this->crossSectionInOneStep(before, after, local_time,
                                     result.image)) {
      throw std::runtime_error("one-step Newton crossing was not available");
    }
    this->sectionDerivativesEnclosure.computeOneStepSectionEnclosure(
        before, this->m_solver, result.image, local_time);
    result.dp = tangent_projection(
        this->computeDP(result.image, result.flow, result.time));
    result.postsection = static_cast<IVector>(after);
    result.postsection_time = after.getCurrentTime();
    return result;
  }
};

ResidentData resident_return(const FrozenInput& input, C1Rect2Set source) {
  IMap field(kVectorField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(kDimension, kSectionCoordinate);
  SectionResidentMap map(solver, section, capd::poincare::MinusPlus);
  ResidentData result = map.one_return(source);
  result.normal_velocity =
      result.image[0] * result.image[1] - result.image[2] - input.zs;
  return result;
}

struct LiouvilleData {
  interval time;
  IVector image = IVector(4);
  interval initial_velocity;
  interval normal_velocity;
  interval ell;
  interval exp_ell;
  interval determinant;
};

LiouvilleData liouville_two_return(const FrozenInput& input) {
  IMap field(kLiouvilleField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(4, kSectionCoordinate);
  IPoincareMap map(solver, section, capd::poincare::MinusPlus);
  IMatrix frame(4, 4);
  frame.setToIdentity();
  frame[0][0] = input.unstable_x;
  frame[1][0] = input.unstable_y;
  frame[0][1] = input.stable_x;
  frame[1][1] = input.stable_y;
  const IVector center{input.center()[0], input.center()[1], interval(0.0),
                       interval(0.0)};
  const IVector radii{input.c0_tile_radii()[0],
                      input.c0_tile_radii()[1], interval(0.0), interval(0.0)};
  C0HOTripletonSet set(center, frame, radii);
  LiouvilleData result;
  result.image = map(set, result.time, 2);
  const interval initial_x = input.origin_x + input.unstable_x * input.source_u() +
                             input.stable_x * input.source_s();
  const interval initial_y = input.origin_y + input.unstable_y * input.source_u() +
                             input.stable_y * input.source_s();
  result.initial_velocity = initial_x * initial_y - input.zs;
  result.normal_velocity =
      result.image[0] * result.image[1] - result.image[2] - input.zs;
  result.ell = result.image[3];
  result.exp_ell = exp(result.ell);
  const interval frame_det = input.unstable_x * input.stable_y -
                             input.stable_x * input.unstable_y;
  const interval q0_area =
      frame_det * input.radius_u * input.radius_s;
  result.determinant = result.exp_ell * result.initial_velocity /
                       result.normal_velocity * q0_area;
  return result;
}

struct AffineCarrier {
  IMatrix center{kDimension, kDimension};
  std::array<IMatrix, 2> coefficient{
      IMatrix(kDimension, kDimension), IMatrix(kDimension, kDimension)};
  IMatrix residual{kDimension, kDimension};
  IMatrix hull{kDimension, kDimension};
  interval determinant_polynomial;
  interval determinant_residual;
  interval determinant;
};

interval actual_second_derivative(const IHessian& hessian, int output,
                                  int first, int second) {
  const interval coefficient = hessian(output, first, second);
  return first == second ? 2.0 * coefficient : coefficient;
}

AffineCarrier build_affine_carrier(const C2Data& full,
                                   const C2Data& center,
                                   const IVector& delta) {
  AffineCarrier carrier;
  carrier.center.clear();
  carrier.residual.clear();
  carrier.hull.clear();
  carrier.coefficient[0].clear();
  carrier.coefficient[1].clear();
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      carrier.center[row][column] = midpoint(center.dp[row][column]);
      carrier.residual[row][column] =
          center.dp[row][column] - carrier.center[row][column];
      carrier.hull[row][column] = carrier.center[row][column];
      for (int variable = 0; variable < 2; ++variable) {
        const interval derivative = actual_second_derivative(
            full.d2p, row, column, variable);
        carrier.coefficient[variable][row][column] = midpoint(derivative);
        carrier.residual[row][column] +=
            (derivative - carrier.coefficient[variable][row][column]) *
            delta[variable];
        carrier.hull[row][column] +=
            carrier.coefficient[variable][row][column] * delta[variable];
      }
      carrier.hull[row][column] += carrier.residual[row][column];
    }
  }

  const interval c0 =
      carrier.center[0][0] * carrier.center[1][1] -
      carrier.center[0][1] * carrier.center[1][0];
  std::array<interval, 2> linear;
  std::array<interval, 2> square;
  for (int variable = 0; variable < 2; ++variable) {
    const IMatrix& a = carrier.coefficient[variable];
    linear[variable] =
        a[0][0] * carrier.center[1][1] +
        carrier.center[0][0] * a[1][1] -
        a[0][1] * carrier.center[1][0] -
        carrier.center[0][1] * a[1][0];
    square[variable] =
        a[0][0] * a[1][1] - a[0][1] * a[1][0];
  }
  const interval cross =
      carrier.coefficient[0][0][0] * carrier.coefficient[1][1][1] +
      carrier.coefficient[1][0][0] * carrier.coefficient[0][1][1] -
      carrier.coefficient[0][0][1] * carrier.coefficient[1][1][0] -
      carrier.coefficient[1][0][1] * carrier.coefficient[0][1][0];
  carrier.determinant_polynomial =
      c0 + linear[0] * delta[0] + linear[1] * delta[1] +
      square[0] * centered_square(delta[0]) +
      square[1] * centered_square(delta[1]) +
      cross * delta[0] * delta[1];

  IMatrix affine_hull(kDimension, kDimension);
  affine_hull.clear();
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      affine_hull[row][column] = carrier.center[row][column];
      for (int variable = 0; variable < 2; ++variable) {
        affine_hull[row][column] +=
            carrier.coefficient[variable][row][column] * delta[variable];
      }
    }
  }
  const IMatrix& e = carrier.residual;
  carrier.determinant_residual =
      affine_hull[0][0] * e[1][1] + e[0][0] * affine_hull[1][1] +
      e[0][0] * e[1][1] - affine_hull[0][1] * e[1][0] -
      e[0][1] * affine_hull[1][0] - e[0][1] * e[1][0];
  carrier.determinant =
      carrier.determinant_polynomial + carrier.determinant_residual;
  return carrier;
}

struct ProjectiveChart {
  bool eligible = false;
  interval first_slope;
  interval second_slope;
  interval separation;
  interval scale;
  interval determinant;
};

ProjectiveChart covector_projective_chart(
    const IMatrix& jacobian, const interval& l0, const interval& l1,
    const interval& m0, const interval& m1, const interval& transform_det) {
  ProjectiveChart result;
  const interval pivot0 = l0 * jacobian[0][0] + l1 * jacobian[1][0];
  const interval pivot1 = l0 * jacobian[0][1] + l1 * jacobian[1][1];
  const interval complement0 =
      m0 * jacobian[0][0] + m1 * jacobian[1][0];
  const interval complement1 =
      m0 * jacobian[0][1] + m1 * jacobian[1][1];
  result.eligible = !pivot0.contains(0.0) && !pivot1.contains(0.0) &&
                    !transform_det.contains(0.0);
  if (result.eligible) {
    result.first_slope = complement0 / pivot0;
    result.second_slope = complement1 / pivot1;
    result.separation = result.second_slope - result.first_slope;
    result.scale = pivot0 * pivot1 / transform_det;
    result.determinant = result.scale * result.separation;
  }
  return result;
}

ProjectiveChart x_projective_chart(const IMatrix& jacobian) {
  return covector_projective_chart(jacobian, 1.0, 0.0, 0.0, 1.0, 1.0);
}

ProjectiveChart y_projective_chart(const IMatrix& jacobian) {
  return covector_projective_chart(jacobian, 0.0, 1.0, 1.0, 0.0, -1.0);
}

ProjectiveChart plus_projective_chart(const IMatrix& jacobian) {
  return covector_projective_chart(jacobian, 1.0, 1.0, -1.0, 1.0, 2.0);
}

ProjectiveChart minus_projective_chart(const IMatrix& jacobian) {
  return covector_projective_chart(jacobian, 1.0, -1.0, 1.0, 1.0, 2.0);
}

void emit_projective(const char* marker, const ProjectiveChart& chart) {
  std::cout << marker << " ELIGIBLE=" << chart.eligible;
  write_interval(std::cout, "FIRST_SLOPE", chart.first_slope);
  write_interval(std::cout, "SECOND_SLOPE", chart.second_slope);
  write_interval(std::cout, "SEPARATION", chart.separation);
  write_interval(std::cout, "SCALE", chart.scale);
  write_interval(std::cout, "DET", chart.determinant);
  std::cout << '\n';
}

enum class RayChart { kNone, kX, kY, kPlus, kMinus };

const char* chart_name(RayChart chart) {
  switch (chart) {
    case RayChart::kX:
      return "X";
    case RayChart::kY:
      return "Y";
    case RayChart::kPlus:
      return "PLUS";
    case RayChart::kMinus:
      return "MINUS";
    case RayChart::kNone:
      return "NONE";
  }
  throw std::runtime_error("unknown projective chart");
}

struct NormalizedRay {
  bool eligible = false;
  RayChart chart = RayChart::kNone;
  std::array<interval, 4> candidate_pivots;
  interval pivot;
  interval slope;
  interval scale = interval(1.0);
  IVector normalized = IVector(kDimension);
};

bool finite(const NormalizedRay& ray) {
  return finite(ray.candidate_pivots[0]) && finite(ray.candidate_pivots[1]) &&
         finite(ray.candidate_pivots[2]) && finite(ray.candidate_pivots[3]) &&
         finite(ray.pivot) && finite(ray.slope) && finite(ray.scale) &&
         finite(ray.normalized);
}

double certified_pivot_score(const interval& pivot, double norm_squared) {
  if (!finite(pivot) || pivot.contains(0.0)) {
    return -1.0;
  }
  const double margin = std::min(std::abs(pivot.leftBound()),
                                 std::abs(pivot.rightBound()));
  return margin * margin / norm_squared;
}

NormalizedRay normalize_ray(const IMatrix& rays, int column) {
  NormalizedRay result;
  const interval x = rays[0][column];
  const interval y = rays[1][column];
  result.candidate_pivots = {x, y, x + y, x - y};
  constexpr std::array<double, 4> kNormSquared = {1.0, 1.0, 2.0, 2.0};
  constexpr std::array<RayChart, 4> kCharts = {
      RayChart::kX, RayChart::kY, RayChart::kPlus, RayChart::kMinus};
  int selected = -1;
  double selected_score = -1.0;
  for (int candidate = 0; candidate < 4; ++candidate) {
    const double score = certified_pivot_score(
        result.candidate_pivots[candidate], kNormSquared[candidate]);
    if (score > selected_score) {
      selected = candidate;
      selected_score = score;
    }
  }
  if (selected < 0 || selected_score < 0.0) {
    for (int row = 0; row < kDimension; ++row) {
      result.normalized[row] = rays[row][column];
    }
    return result;
  }

  result.eligible = true;
  result.chart = kCharts[selected];
  result.pivot = result.candidate_pivots[selected];
  result.scale = result.pivot;
  switch (result.chart) {
    case RayChart::kX:
      result.slope = y / x;
      result.normalized[0] = interval(1.0);
      result.normalized[1] = result.slope;
      break;
    case RayChart::kY:
      result.slope = x / y;
      result.normalized[0] = result.slope;
      result.normalized[1] = interval(1.0);
      break;
    case RayChart::kPlus:
      result.slope = (-x + y) / (x + y);
      result.normalized[0] = (interval(1.0) - result.slope) / 2.0;
      result.normalized[1] = (interval(1.0) + result.slope) / 2.0;
      break;
    case RayChart::kMinus:
      result.slope = (x + y) / (x - y);
      result.normalized[0] = (interval(1.0) + result.slope) / 2.0;
      result.normalized[1] = (-interval(1.0) + result.slope) / 2.0;
      break;
    case RayChart::kNone:
      throw std::runtime_error("selected an empty projective chart");
  }
  result.normalized[kSectionCoordinate] = interval(0.0);
  return result;
}

void emit_ray(const char* marker, const NormalizedRay& ray) {
  std::cout << marker << " ELIGIBLE=" << ray.eligible
            << " CHART=" << chart_name(ray.chart);
  write_interval(std::cout, "PIVOT_X", ray.candidate_pivots[0]);
  write_interval(std::cout, "PIVOT_Y", ray.candidate_pivots[1]);
  write_interval(std::cout, "PIVOT_PLUS", ray.candidate_pivots[2]);
  write_interval(std::cout, "PIVOT_MINUS", ray.candidate_pivots[3]);
  write_interval(std::cout, "PIVOT", ray.pivot);
  write_interval(std::cout, "SLOPE", ray.slope);
  write_interval(std::cout, "SCALE", ray.scale);
  write_vector(std::cout, "N", ray.normalized);
  std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::boolalpha;
  try {
    if (argc != 7) {
      throw std::runtime_error(
          "usage: U_DEPTH U_INDEX S_DEPTH S_INDEX INPUT_SHA256 RUN_CHALLENGE");
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
    if (!lowercase_sha256(input_sha256) || !lowercase_sha256(run_challenge)) {
      throw std::runtime_error(
          "input hash and challenge must be lowercase SHA-256");
    }
    const FrozenInput input(u_depth, u_index, s_depth, s_index);
    const C1Data c1_p1 = c1_return(input, 1);
    const C1Data c1_p2 = c1_return(input, 2);
    const C2Data c2_full_p1 = c2_return(input, false, 1);
    const C2Data c2_center_p1 = c2_return(input, true, 1);
    const C2Data c2_full_p2 = c2_return(input, false, 2);
    const C2Data c2_center_p2 = c2_return(input, true, 2);
    const LiouvilleData liouville = liouville_two_return(input);
    const IVector delta = input.normalized_delta();
    const AffineCarrier affine =
        build_affine_carrier(c2_full_p2, c2_center_p2, delta);
    const ProjectiveChart projective_x = x_projective_chart(affine.hull);
    const ProjectiveChart projective_y = y_projective_chart(affine.hull);
    const ProjectiveChart projective_plus =
        plus_projective_chart(affine.hull);
    const ProjectiveChart projective_minus =
        minus_projective_chart(affine.hull);
    const interval c1_determinant = determinant_xy(c1_p2.dp);
    const interval c2_hull_determinant = determinant_xy(c2_full_p2.dp);

    IVector event1_center = midpoint_vector(c2_center_p1.image);
    event1_center[kSectionCoordinate] = interval(0.0);
    const IMatrix event1_basis = midpoint_matrix(c2_full_p1.dp);
    IVector event1_residual =
        c2_center_p1.image - event1_center +
        (c2_full_p1.dp - event1_basis) * delta;
    event1_residual[kSectionCoordinate] = interval(0.0);
    IMatrix residual_basis(kDimension, kDimension);
    residual_basis.setToIdentity();

    const std::array<NormalizedRay, 2> event1_rays = {
        normalize_ray(c2_full_p1.dp, 0),
        normalize_ray(c2_full_p1.dp, 1)};
    IMatrix event1_normalized_seed(kDimension, kDimension);
    event1_normalized_seed.clear();
    for (int column = 0; column < 2; ++column) {
      for (int row = 0; row < kDimension; ++row) {
        event1_normalized_seed[row][column] =
            event1_rays[column].normalized[row];
      }
    }
    C1Rect2Set::C0BaseSet event1_c0(
        event1_center, event1_basis, delta, residual_basis, event1_residual);
    C1Rect2Set::C1BaseSet event1_c1(event1_normalized_seed);
    C1Rect2Set event1_carrier(event1_c0, event1_c1, c2_full_p1.time);
    event1_carrier.setC0Factor(std::numeric_limits<double>::infinity());
    const IVector event1_c0_hull = static_cast<IVector>(event1_carrier);
    const ResidentData homogeneous_p2 = resident_return(input, event1_carrier);

    const std::array<NormalizedRay, 2> event2_rays = {
        normalize_ray(homogeneous_p2.dp, 0),
        normalize_ray(homogeneous_p2.dp, 1)};
    IMatrix event2_normalized_rays(kDimension, kDimension);
    event2_normalized_rays.clear();
    IMatrix event1_reconstructed_rays(kDimension, kDimension);
    event1_reconstructed_rays.clear();
    IMatrix event2_reconstructed_rays(kDimension, kDimension);
    event2_reconstructed_rays.clear();
    IMatrix reconstructed_dp(kDimension, kDimension);
    reconstructed_dp.clear();
    for (int column = 0; column < 2; ++column) {
      for (int row = 0; row < kDimension; ++row) {
        event2_normalized_rays[row][column] =
            event2_rays[column].normalized[row];
        event1_reconstructed_rays[row][column] =
            event1_rays[column].normalized[row] *
            event1_rays[column].scale;
        event2_reconstructed_rays[row][column] =
            event2_rays[column].normalized[row] *
            event2_rays[column].scale;
        reconstructed_dp[row][column] =
            homogeneous_p2.dp[row][column] * event1_rays[column].scale;
      }
    }
    const interval total_scale0 =
        event1_rays[0].scale * event2_rays[0].scale;
    const interval total_scale1 =
        event1_rays[1].scale * event2_rays[1].scale;
    const interval normalized_exterior =
        determinant_xy(event2_normalized_rays);
    const interval homogeneous_determinant =
        total_scale0 * total_scale1 * normalized_exterior;
    const interval reconstructed_determinant =
        determinant_xy(reconstructed_dp);

    const bool all_finite =
        finite(c1_p1.time) && finite(c1_p1.image) && finite(c1_p1.flow) &&
        finite(c1_p1.dp) && finite(c1_p1.normal_velocity) &&
        finite(c1_p2.time) && finite(c1_p2.image) && finite(c1_p2.flow) &&
        finite(c1_p2.dp) && finite(c1_p2.normal_velocity) &&
        finite(c2_full_p1.time) && finite(c2_full_p1.image) &&
        finite(c2_full_p1.flow) && finite(c2_full_p1.flow_hessian) &&
        finite(c2_full_p1.dp) && finite(c2_full_p1.d2p) &&
        finite(c2_full_p1.dtime) && finite(c2_full_p1.d2time) &&
        finite(c2_full_p1.second_time_coefficient) &&
        finite(c2_full_p1.reconstructed_dtime) &&
        finite(c2_full_p1.reconstructed_d2time) &&
        finite(c2_full_p1.normal_velocity) && finite(c2_center_p1.time) &&
        finite(c2_center_p1.image) && finite(c2_center_p1.flow) &&
        finite(c2_center_p1.flow_hessian) && finite(c2_center_p1.dp) &&
        finite(c2_center_p1.d2p) && finite(c2_center_p1.dtime) &&
        finite(c2_center_p1.d2time) &&
        finite(c2_center_p1.second_time_coefficient) &&
        finite(c2_center_p1.reconstructed_dtime) &&
        finite(c2_center_p1.reconstructed_d2time) &&
        finite(c2_center_p1.normal_velocity) &&
        finite(c2_full_p2.time) && finite(c2_full_p2.image) &&
        finite(c2_full_p2.flow) && finite(c2_full_p2.flow_hessian) &&
        finite(c2_full_p2.dp) && finite(c2_full_p2.d2p) &&
        finite(c2_full_p2.dtime) && finite(c2_full_p2.d2time) &&
        finite(c2_full_p2.second_time_coefficient) &&
        finite(c2_full_p2.reconstructed_dtime) &&
        finite(c2_full_p2.reconstructed_d2time) &&
        finite(c2_full_p2.normal_velocity) && finite(c2_center_p2.time) &&
        finite(c2_center_p2.image) && finite(c2_center_p2.flow) &&
        finite(c2_center_p2.flow_hessian) && finite(c2_center_p2.dp) &&
        finite(c2_center_p2.d2p) && finite(c2_center_p2.dtime) &&
        finite(c2_center_p2.d2time) &&
        finite(c2_center_p2.second_time_coefficient) &&
        finite(c2_center_p2.reconstructed_dtime) &&
        finite(c2_center_p2.reconstructed_d2time) &&
        finite(c2_center_p2.normal_velocity) &&
        finite(liouville.time) && finite(liouville.image) &&
        finite(liouville.initial_velocity) &&
        finite(liouville.normal_velocity) && finite(liouville.ell) &&
        finite(liouville.exp_ell) && finite(liouville.determinant) &&
        finite(affine.center) &&
        finite(affine.coefficient[0]) && finite(affine.coefficient[1]) &&
        finite(affine.residual) && finite(affine.hull) &&
        finite(affine.determinant) && finite(event1_center) &&
        finite(event1_basis) && finite(event1_residual) &&
        finite(event1_c0_hull) && finite(event1_rays[0]) &&
        finite(event1_rays[1]) && finite(event1_normalized_seed) &&
        finite(homogeneous_p2.time) && finite(homogeneous_p2.image) &&
        finite(homogeneous_p2.flow) && finite(homogeneous_p2.dp) &&
        finite(homogeneous_p2.postsection) &&
        finite(homogeneous_p2.postsection_time) &&
        finite(homogeneous_p2.normal_velocity) && finite(event2_rays[0]) &&
        finite(event2_rays[1]) && finite(event2_normalized_rays) &&
        finite(reconstructed_dp) && finite(total_scale0) &&
        finite(total_scale1) && finite(normalized_exterior) &&
        finite(homogeneous_determinant) &&
        finite(reconstructed_determinant);
    const bool p1_center_full_dp_overlap =
        overlaps(c2_center_p1.dp, c2_full_p1.dp);
    const bool center_full_dp_overlap =
        overlaps(c2_center_p2.dp, c2_full_p2.dp);
    const bool c1_c2_dp_overlap = overlaps(c1_p2.dp, c2_full_p2.dp);
    const bool event1_mean_value_overlap = [&]() {
      for (int row = 0; row < kDimension; ++row) {
        if (std::max({event1_c0_hull[row].leftBound(),
                      c2_full_p1.image[row].leftBound(),
                      c2_center_p1.image[row].leftBound()}) >
            std::min({event1_c0_hull[row].rightBound(),
                      c2_full_p1.image[row].rightBound(),
                      c2_center_p1.image[row].rightBound()})) {
          return false;
        }
      }
      return true;
    }();
    const bool event1_ray_reconstruction =
        contains(event1_reconstructed_rays, c2_full_p1.dp);
    const bool event2_ray_reconstruction =
        contains(event2_reconstructed_rays, homogeneous_p2.dp);
    const bool cumulative_matrix_overlap =
        overlaps(reconstructed_dp, c1_p2.dp) &&
        overlaps(reconstructed_dp, c2_full_p2.dp);
    const bool event_order_certified =
        (homogeneous_p2.time - c2_full_p1.time).leftBound() > 0.0 &&
        homogeneous_p2.postsection_time.leftBound() >
            homogeneous_p2.time.rightBound();
    const bool postsection_plus_side =
        homogeneous_p2.postsection[kSectionCoordinate].leftBound() > 0.0;
    const bool event_transversality_certified =
        c1_p1.normal_velocity.leftBound() > 0.0 &&
        c1_p2.normal_velocity.leftBound() > 0.0 &&
        c2_full_p1.normal_velocity.leftBound() > 0.0 &&
        c2_center_p1.normal_velocity.leftBound() > 0.0 &&
        c2_full_p2.normal_velocity.leftBound() > 0.0 &&
        c2_center_p2.normal_velocity.leftBound() > 0.0 &&
        homogeneous_p2.normal_velocity.leftBound() > 0.0;
    const bool impact_time_crosscheck =
        overlaps(c2_full_p1.dtime, c2_full_p1.reconstructed_dtime) &&
        overlaps(c2_full_p1.d2time, c2_full_p1.reconstructed_d2time) &&
        overlaps(c2_center_p1.dtime, c2_center_p1.reconstructed_dtime) &&
        overlaps(c2_center_p1.d2time,
                 c2_center_p1.reconstructed_d2time) &&
        overlaps(c2_full_p2.dtime, c2_full_p2.reconstructed_dtime) &&
        overlaps(c2_full_p2.d2time, c2_full_p2.reconstructed_d2time) &&
        overlaps(c2_center_p2.dtime, c2_center_p2.reconstructed_dtime) &&
        overlaps(c2_center_p2.d2time,
                 c2_center_p2.reconstructed_d2time);
    const bool c1_orientation_unresolved = c1_determinant.contains(0.0);
    const bool c2_hull_orientation_unresolved =
        c2_hull_determinant.contains(0.0);
    const bool affine_orientation_certified =
        sign_definite(affine.determinant);
    const bool liouville_orientation_certified =
        sign_definite(liouville.determinant);
    const bool affine_liouville_overlap =
        overlaps(affine.determinant, liouville.determinant);
    const bool affine_liouville_same_sign =
        same_strict_sign(affine.determinant, liouville.determinant);
    const bool projective_x_orientation_certified =
        projective_x.eligible && sign_definite(projective_x.determinant);
    const bool projective_y_orientation_certified =
        projective_y.eligible && sign_definite(projective_y.determinant);
    const bool projective_plus_orientation_certified =
        projective_plus.eligible && sign_definite(projective_plus.determinant);
    const bool projective_minus_orientation_certified =
        projective_minus.eligible && sign_definite(projective_minus.determinant);
    const bool any_projective_orientation_certified =
        projective_x_orientation_certified ||
        projective_y_orientation_certified ||
        projective_plus_orientation_certified ||
        projective_minus_orientation_certified;
    const bool affine_strictly_narrower_than_c1 =
        width(affine.determinant) < width(c1_determinant);
    const bool affine_strictly_narrower_than_c2_hull =
        width(affine.determinant) < width(c2_hull_determinant);
    const bool event1_charts_certified =
        event1_rays[0].eligible && event1_rays[1].eligible;
    const bool event2_charts_certified =
        event2_rays[0].eligible && event2_rays[1].eligible;
    const bool homogeneous_orientation_certified =
        sign_definite(homogeneous_determinant);
    const bool homogeneous_liouville_overlap =
        overlaps(homogeneous_determinant, liouville.determinant);
    const bool homogeneous_liouville_same_sign =
        same_strict_sign(homogeneous_determinant, liouville.determinant);
    const bool homogeneous_joint_overlap =
        std::max({homogeneous_determinant.leftBound(),
                  reconstructed_determinant.leftBound(),
                  affine.determinant.leftBound(),
                  liouville.determinant.leftBound()}) <=
        std::min({homogeneous_determinant.rightBound(),
                  reconstructed_determinant.rightBound(),
                  affine.determinant.rightBound(),
                  liouville.determinant.rightBound()});
    double best_fixed_width = std::numeric_limits<double>::infinity();
    for (const ProjectiveChart* chart :
         {&projective_x, &projective_y, &projective_plus,
          &projective_minus}) {
      if (chart->eligible) {
        best_fixed_width = std::min(best_fixed_width, width(chart->determinant));
      }
    }
    const bool homogeneous_strictly_narrower_than_affine =
        width(homogeneous_determinant) < width(affine.determinant);
    const bool homogeneous_strictly_narrower_than_best_fixed =
        std::isfinite(best_fixed_width) &&
        width(homogeneous_determinant) < best_fixed_width;
    const bool structural_pass =
        all_finite && p1_center_full_dp_overlap && center_full_dp_overlap &&
        c1_c2_dp_overlap &&
        event_transversality_certified && impact_time_crosscheck &&
        liouville_orientation_certified;
    const bool homogeneous_computation_valid =
        structural_pass && event1_mean_value_overlap &&
        event1_ray_reconstruction && event2_ray_reconstruction &&
        cumulative_matrix_overlap && event_order_certified &&
        postsection_plus_side &&
        event1_charts_certified && event2_charts_certified &&
        homogeneous_joint_overlap;
    const bool affine_certificate_pass =
        structural_pass && affine.determinant.rightBound() < 0.0 &&
        affine_liouville_overlap && affine_liouville_same_sign;
    const bool projective_x_certificate_pass =
        structural_pass && projective_x_orientation_certified &&
        projective_x.determinant.rightBound() < 0.0 &&
        overlaps(projective_x.determinant, affine.determinant) &&
        overlaps(projective_x.determinant, liouville.determinant) &&
        same_strict_sign(projective_x.determinant, liouville.determinant);
    const bool projective_y_certificate_pass =
        structural_pass && projective_y_orientation_certified &&
        projective_y.determinant.rightBound() < 0.0 &&
        overlaps(projective_y.determinant, affine.determinant) &&
        overlaps(projective_y.determinant, liouville.determinant) &&
        same_strict_sign(projective_y.determinant, liouville.determinant);
    const bool projective_plus_certificate_pass =
        structural_pass && projective_plus_orientation_certified &&
        projective_plus.determinant.rightBound() < 0.0 &&
        overlaps(projective_plus.determinant, affine.determinant) &&
        overlaps(projective_plus.determinant, liouville.determinant) &&
        same_strict_sign(projective_plus.determinant, liouville.determinant);
    const bool projective_minus_certificate_pass =
        structural_pass && projective_minus_orientation_certified &&
        projective_minus.determinant.rightBound() < 0.0 &&
        overlaps(projective_minus.determinant, affine.determinant) &&
        overlaps(projective_minus.determinant, liouville.determinant) &&
        same_strict_sign(projective_minus.determinant, liouville.determinant);
    const bool homogeneous_certificate_pass =
        homogeneous_computation_valid && homogeneous_orientation_certified &&
        homogeneous_determinant.rightBound() < 0.0 &&
        homogeneous_liouville_overlap && homogeneous_liouville_same_sign;
    const bool certificate_pass =
        affine_certificate_pass || projective_x_certificate_pass ||
        projective_y_certificate_pass || projective_plus_certificate_pass ||
        projective_minus_certificate_pass || homogeneous_certificate_pass;
    const bool subdivision_required = !certificate_pass;
    const bool probe_pass = homogeneous_computation_valid;

    std::cout
        << "SCHEMA=sounio.cs6.plucker-cocycle-leaf.v1\n"
        << "WORKER_SOURCE_SHA256=" << CS6_WORKER_SOURCE_SHA256 << '\n'
        << "INPUT_SHA256=" << input_sha256 << '\n'
        << "RUN_CHALLENGE=" << run_challenge << '\n'
        << "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0\n"
        << "INTERVAL_BACKEND_DECLARED=FILIB\n"
        << "INTERVAL_SERIALIZATION=ONE_ULP_OUTWARD_BINARY64_HEX\n"
        << "SOURCE=N0\n"
        << "U_DEPTH=" << input.u_depth << '\n'
        << "U_INDEX=" << input.u_index << '\n'
        << "S_DEPTH=" << input.s_depth << '\n'
        << "S_INDEX=" << input.s_index << '\n'
        << "U_TILES=" << input.u_tiles() << '\n'
        << "S_TILES=" << input.s_tiles() << '\n'
        << "ORDER=8\n"
        << "RETURN_COUNT=2\n"
        << "SECTION=COORDINATE_W_EQUALS_ZERO\n"
        << "CROSSING_DIRECTION=MINUS_PLUS\n"
        << "C2_POINCARE_CONVERSION=CAPD_COMPUTE_DP_WITH_RETURN_TIME_CORRECTION\n"
        << "IMPACT_TIME_CROSSCHECK=CAPD_OUTPUT_VS_COORDINATE_SECTION_RECONSTRUCTION\n"
        << "C2_HESSIAN_ROLE=NORMALIZED_TAYLOR_COEFFICIENTS_OF_RETURN_MAP\n"
        << "DIAGONAL_TAYLOR_TO_DERIVATIVE_FACTOR=2\n"
        << "OFFDIAGONAL_TAYLOR_TO_DERIVATIVE_FACTOR=1\n"
        << "AFFINE_CARRIER_FORM=M_PLUS_A0_DELTA0_PLUS_A1_DELTA1_PLUS_R\n"
        << "AFFINE_REMAINDER_RULE=CENTER_DP_RADIUS_PLUS_HESSIAN_RADIUS_TIMES_DELTA\n"
        << "PROJECTIVE_CONTROL=FOUR_FIXED_COVECTOR_FINAL_COLUMN_SLOPE_CHARTS\n"
        << "PROJECTIVE_COVECTORS=X,Y,X_PLUS_Y,X_MINUS_Y\n"
        << "PROJECTIVE_TRANSFORMS=X_1_0_0_1_DET_1,Y_0_1_1_0_DET_NEG1,PLUS_1_1_NEG1_1_DET_2,MINUS_1_NEG1_1_1_DET_2\n"
        << "PROJECTIVE_DETERMINANT_IDENTITY=PIVOT0_TIMES_PIVOT1_TIMES_SLOPE_SEPARATION_OVER_TRANSFORM_DET\n"
        << "PROJECTIVE_RICCATI_INTEGRATED=false\n"
        << "HOMOGENEOUS_CARRIER=EVENT1_NORMALIZE_LOCAL_P2_EVENT2_NORMALIZE\n"
        << "EVENTWISE_NORMALIZATION=true\n"
        << "DISCRETE_POINCARE_COCYCLE=true\n"
        << "CHART_NORMALIZATION_SCOPE=EVENT_BOUNDARIES\n"
        << "GRASSMANN_OBJECT=GR_1_2_EQUALS_REAL_PROJECTIVE_LINE\n"
        << "GRASSMANN_SCOPE=GR_1_2_EQUALS_P1\n"
        << "PLUCKER_ROLE=HOMOGENEOUS_RAYS_WITH_FACTORED_SIGNED_EXTERIOR_RECONSTRUCTION\n"
        << "PLUCKER_RELATIONS_NONTRIVIAL=false\n"
        << "EVENT_NORMALIZATION_COUNT=2\n"
        << "DYNAMIC_CHART_SET=X,Y,PLUS,MINUS\n"
        << "DYNAMIC_CHART_SELECTION=MAX_CERTIFIED_PIVOT_MARGIN_SQUARED_OVER_COVECTOR_NORM_SQUARED\n"
        << "EVENT1_C0_CARRIER=MIDPOINT_MEAN_VALUE_CORRELATED_SOURCE_DELTA\n"
        << "TANGENT_DEPENDENCY_MODEL=INTERVAL_RAYS_WITH_EVENT_NORMALIZATION\n"
        << "TANGENT_SOURCE_DEPENDENCY_AT_SWITCH=BOXED_NOT_AFFINE\n"
        << "CONTINUOUS_RICCATI_INTEGRATED=false\n"
        << "CONTINUOUS_PROJECTIVE_FLOW_INTEGRATED=false\n"
        << "GENERAL_GRASSMANN_PLUCKER_INTEGRATOR=false\n"
        << "AUTONOMOUS_VECTOR_FIELD=true\n"
        << "EVENT_TIME_SENSITIVITY_PROPAGATED=false\n"
        << "NONAUTONOMOUS_GENERALIZATION_PROVED=false\n"
        << "LIOUVILLE_ROLE=INDEPENDENT_SIGN_CROSS_CHECK_ONLY\n"
        << "EXECUTION_SCOPE=DYADIC_LEAF_EVENT_NORMALIZED_CAPD_CPU_PROBE\n"
        << "FPGA_EXECUTION=false\n"
        << "EXECUTION_PROVENANCE_ATTESTED=false\n"
        << "INDEPENDENT_REPLAY_REQUIRED=true\n"
        << "PROMOTION_ELIGIBLE=false\n"
        << "FULL_SOURCE_CARRIER_PROVED=false\n"
        << "HYPERBOLICITY_PROVED=false\n"
        << "CHAOTIC_ATTRACTOR_PROVED=false\n";

    std::cout << "SOURCE_TILE";
    write_interval(std::cout, "U", input.source_u());
    write_interval(std::cout, "S", input.source_s());
    write_vector(std::cout, "DELTA", delta);
    write_matrix(std::cout, "Q0", input.q0());
    std::cout << '\n';

    std::cout << "C1_P1_TRANSVERSALITY";
    write_interval(std::cout, "TIME", c1_p1.time);
    write_vector(std::cout, "X", c1_p1.image);
    write_interval(std::cout, "NU", c1_p1.normal_velocity);
    std::cout << '\n';

    std::cout << "C1_P2_CONTROL";
    write_interval(std::cout, "TIME", c1_p2.time);
    write_vector(std::cout, "X", c1_p2.image);
    write_matrix(std::cout, "DP", c1_p2.dp);
    write_interval(std::cout, "NU", c1_p2.normal_velocity);
    write_interval(std::cout, "DET", c1_determinant);
    std::cout << '\n';

    std::cout << "C2_FULL_P1";
    write_interval(std::cout, "TIME", c2_full_p1.time);
    write_vector(std::cout, "X", c2_full_p1.image);
    write_matrix(std::cout, "FLOW", c2_full_p1.flow);
    write_hessian(std::cout, "FLOW_H", c2_full_p1.flow_hessian);
    write_vector(std::cout, "DT", c2_full_p1.dtime);
    write_matrix(std::cout, "D2T", c2_full_p1.d2time);
    write_vector(std::cout, "D2PHIDT2", c2_full_p1.second_time_coefficient);
    write_vector(std::cout, "DT_RECON", c2_full_p1.reconstructed_dtime);
    write_matrix(std::cout, "D2T_RECON",
                 c2_full_p1.reconstructed_d2time);
    write_matrix(std::cout, "DP", c2_full_p1.dp);
    write_hessian(std::cout, "D2P", c2_full_p1.d2p);
    write_interval(std::cout, "NU", c2_full_p1.normal_velocity);
    std::cout << '\n';

    std::cout << "C2_CENTER_P1";
    write_interval(std::cout, "TIME", c2_center_p1.time);
    write_vector(std::cout, "X", c2_center_p1.image);
    write_matrix(std::cout, "FLOW", c2_center_p1.flow);
    write_hessian(std::cout, "FLOW_H", c2_center_p1.flow_hessian);
    write_vector(std::cout, "DT", c2_center_p1.dtime);
    write_matrix(std::cout, "D2T", c2_center_p1.d2time);
    write_vector(std::cout, "D2PHIDT2", c2_center_p1.second_time_coefficient);
    write_vector(std::cout, "DT_RECON", c2_center_p1.reconstructed_dtime);
    write_matrix(std::cout, "D2T_RECON",
                 c2_center_p1.reconstructed_d2time);
    write_matrix(std::cout, "DP", c2_center_p1.dp);
    write_hessian(std::cout, "D2P", c2_center_p1.d2p);
    write_interval(std::cout, "NU", c2_center_p1.normal_velocity);
    std::cout << '\n';

    std::cout << "C2_FULL_P2";
    write_interval(std::cout, "TIME", c2_full_p2.time);
    write_vector(std::cout, "X", c2_full_p2.image);
    write_matrix(std::cout, "FLOW", c2_full_p2.flow);
    write_hessian(std::cout, "FLOW_H", c2_full_p2.flow_hessian);
    write_vector(std::cout, "DT", c2_full_p2.dtime);
    write_matrix(std::cout, "D2T", c2_full_p2.d2time);
    write_vector(std::cout, "D2PHIDT2", c2_full_p2.second_time_coefficient);
    write_vector(std::cout, "DT_RECON", c2_full_p2.reconstructed_dtime);
    write_matrix(std::cout, "D2T_RECON",
                 c2_full_p2.reconstructed_d2time);
    write_matrix(std::cout, "DP", c2_full_p2.dp);
    write_hessian(std::cout, "D2P", c2_full_p2.d2p);
    write_interval(std::cout, "NU", c2_full_p2.normal_velocity);
    write_interval(std::cout, "HULL_DET", c2_hull_determinant);
    std::cout << '\n';

    std::cout << "C2_CENTER_P2";
    write_interval(std::cout, "TIME", c2_center_p2.time);
    write_vector(std::cout, "X", c2_center_p2.image);
    write_matrix(std::cout, "FLOW", c2_center_p2.flow);
    write_hessian(std::cout, "FLOW_H", c2_center_p2.flow_hessian);
    write_vector(std::cout, "DT", c2_center_p2.dtime);
    write_matrix(std::cout, "D2T", c2_center_p2.d2time);
    write_vector(std::cout, "D2PHIDT2",
                 c2_center_p2.second_time_coefficient);
    write_vector(std::cout, "DT_RECON", c2_center_p2.reconstructed_dtime);
    write_matrix(std::cout, "D2T_RECON",
                 c2_center_p2.reconstructed_d2time);
    write_matrix(std::cout, "DP", c2_center_p2.dp);
    write_hessian(std::cout, "D2P", c2_center_p2.d2p);
    write_interval(std::cout, "NU", c2_center_p2.normal_velocity);
    std::cout << '\n';

    std::cout << "AFFINE_CARRIER";
    write_matrix(std::cout, "M", affine.center);
    write_matrix(std::cout, "A0", affine.coefficient[0]);
    write_matrix(std::cout, "A1", affine.coefficient[1]);
    write_matrix(std::cout, "R", affine.residual);
    write_matrix(std::cout, "HULL", affine.hull);
    write_interval(std::cout, "DET_POLYNOMIAL",
                   affine.determinant_polynomial);
    write_interval(std::cout, "DET_REMAINDER",
                   affine.determinant_residual);
    write_interval(std::cout, "DET", affine.determinant);
    std::cout << '\n';

    emit_projective("PROJECTIVE_X", projective_x);
    emit_projective("PROJECTIVE_Y", projective_y);
    emit_projective("PROJECTIVE_PLUS", projective_plus);
    emit_projective("PROJECTIVE_MINUS", projective_minus);

    std::cout << "EVENT1_MEAN_VALUE";
    write_interval(std::cout, "TIME", c2_full_p1.time);
    write_vector(std::cout, "CENTER", event1_center);
    write_matrix(std::cout, "BASIS", event1_basis);
    write_vector(std::cout, "DELTA", delta);
    write_vector(std::cout, "RESIDUAL", event1_residual);
    write_vector(std::cout, "HULL", event1_c0_hull);
    std::cout << '\n';

    emit_ray("HOMOGENEOUS_EVENT1_RAY0", event1_rays[0]);
    emit_ray("HOMOGENEOUS_EVENT1_RAY1", event1_rays[1]);

    std::cout << "HOMOGENEOUS_LOCAL_P2";
    write_interval(std::cout, "TIME", homogeneous_p2.time);
    write_interval(std::cout, "DURATION",
                   homogeneous_p2.time - c2_full_p1.time);
    write_vector(std::cout, "X", homogeneous_p2.image);
    write_matrix(std::cout, "FLOW", homogeneous_p2.flow);
    write_matrix(std::cout, "DP", homogeneous_p2.dp);
    write_interval(std::cout, "POST_TIME", homogeneous_p2.postsection_time);
    write_vector(std::cout, "POST_X", homogeneous_p2.postsection);
    write_interval(std::cout, "NU", homogeneous_p2.normal_velocity);
    write_matrix(std::cout, "RECON_DP", reconstructed_dp);
    write_interval(std::cout, "RECON_DET", reconstructed_determinant);
    std::cout << '\n';

    emit_ray("HOMOGENEOUS_EVENT2_RAY0", event2_rays[0]);
    emit_ray("HOMOGENEOUS_EVENT2_RAY1", event2_rays[1]);

    std::cout << "PLUCKER_COCYCLE";
    write_interval(std::cout, "TOTAL_SCALE0", total_scale0);
    write_interval(std::cout, "TOTAL_SCALE1", total_scale1);
    write_interval(std::cout, "NORMALIZED_EXTERIOR", normalized_exterior);
    write_interval(std::cout, "DET", homogeneous_determinant);
    std::cout << '\n';

    std::cout << "LIOUVILLE";
    write_interval(std::cout, "TIME", liouville.time);
    write_vector(std::cout, "X", liouville.image);
    write_interval(std::cout, "NU0", liouville.initial_velocity);
    write_interval(std::cout, "NU2", liouville.normal_velocity);
    write_interval(std::cout, "ELL", liouville.ell);
    write_interval(std::cout, "EXP_ELL", liouville.exp_ell);
    write_interval(std::cout, "DET", liouville.determinant);
    std::cout << '\n';

    std::cout << "LEAF_RESULT TERMINAL_CERTIFIED=" << certificate_pass
              << " SUBDIVISION_REQUIRED=" << subdivision_required
              << " AFFINE_CERTIFICATE_PASS=" << affine_certificate_pass
              << " PROJECTIVE_X_CERTIFICATE_PASS="
              << projective_x_certificate_pass
              << " PROJECTIVE_Y_CERTIFICATE_PASS="
              << projective_y_certificate_pass
              << " PROJECTIVE_PLUS_CERTIFICATE_PASS="
              << projective_plus_certificate_pass
              << " PROJECTIVE_MINUS_CERTIFICATE_PASS="
              << projective_minus_certificate_pass
              << " HOMOGENEOUS_CERTIFICATE_PASS="
              << homogeneous_certificate_pass << '\n';

    std::cout
        << "SUMMARY"
        << " ALL_FINITE=" << all_finite
        << " P1_CENTER_FULL_DP_OVERLAP=" << p1_center_full_dp_overlap
        << " CENTER_FULL_DP_OVERLAP=" << center_full_dp_overlap
        << " C1_C2_DP_OVERLAP=" << c1_c2_dp_overlap
        << " EVENT1_MEAN_VALUE_OVERLAP=" << event1_mean_value_overlap
        << " EVENT1_RAY_RECONSTRUCTION=" << event1_ray_reconstruction
        << " EVENT2_RAY_RECONSTRUCTION=" << event2_ray_reconstruction
        << " CUMULATIVE_MATRIX_OVERLAP=" << cumulative_matrix_overlap
        << " EVENT_ORDER_CERTIFIED=" << event_order_certified
        << " POSTSECTION_PLUS_SIDE=" << postsection_plus_side
        << " EVENT_TRANSVERSALITY_CERTIFIED="
        << event_transversality_certified
        << " IMPACT_TIME_CROSSCHECK=" << impact_time_crosscheck
        << " C1_ORIENTATION_UNRESOLVED=" << c1_orientation_unresolved
        << " C2_HULL_ORIENTATION_UNRESOLVED="
        << c2_hull_orientation_unresolved
        << " AFFINE_ORIENTATION_CERTIFIED=" << affine_orientation_certified
        << " LIOUVILLE_ORIENTATION_CERTIFIED="
        << liouville_orientation_certified
        << " AFFINE_LIOUVILLE_OVERLAP=" << affine_liouville_overlap
        << " AFFINE_LIOUVILLE_SAME_SIGN=" << affine_liouville_same_sign
        << " PROJECTIVE_X_ORIENTATION_CERTIFIED="
        << projective_x_orientation_certified
        << " PROJECTIVE_Y_ORIENTATION_CERTIFIED="
        << projective_y_orientation_certified
        << " PROJECTIVE_PLUS_ORIENTATION_CERTIFIED="
        << projective_plus_orientation_certified
        << " PROJECTIVE_MINUS_ORIENTATION_CERTIFIED="
        << projective_minus_orientation_certified
        << " ANY_PROJECTIVE_ORIENTATION_CERTIFIED="
        << any_projective_orientation_certified
        << " AFFINE_STRICTLY_NARROWER_THAN_C1="
        << affine_strictly_narrower_than_c1
        << " AFFINE_STRICTLY_NARROWER_THAN_C2_HULL="
        << affine_strictly_narrower_than_c2_hull
        << " EVENT1_CHARTS_CERTIFIED=" << event1_charts_certified
        << " EVENT2_CHARTS_CERTIFIED=" << event2_charts_certified
        << " HOMOGENEOUS_ORIENTATION_CERTIFIED="
        << homogeneous_orientation_certified
        << " HOMOGENEOUS_LIOUVILLE_OVERLAP="
        << homogeneous_liouville_overlap
        << " HOMOGENEOUS_LIOUVILLE_SAME_SIGN="
        << homogeneous_liouville_same_sign
        << " HOMOGENEOUS_JOINT_OVERLAP=" << homogeneous_joint_overlap
        << " HOMOGENEOUS_STRICTLY_NARROWER_THAN_AFFINE="
        << homogeneous_strictly_narrower_than_affine
        << " HOMOGENEOUS_STRICTLY_NARROWER_THAN_BEST_FIXED="
        << homogeneous_strictly_narrower_than_best_fixed
        << " HOMOGENEOUS_COMPUTATION_VALID="
        << homogeneous_computation_valid
        << " HOMOGENEOUS_CERTIFICATE_PASS="
        << homogeneous_certificate_pass
        << " STRUCTURAL_PASS=" << structural_pass
        << " CERTIFICATE_PASS=" << certificate_pass
        << " PROBE_PASS=" << probe_pass << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "probe error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
