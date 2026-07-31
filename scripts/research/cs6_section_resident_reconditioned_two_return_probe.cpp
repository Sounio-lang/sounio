#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "capd/capdlib.h"

#ifndef CS6_WORKER_SOURCE_SHA256
#define CS6_WORKER_SOURCE_SHA256 "UNBOUND"
#endif

#ifndef CS6_INPUT_SHA256
#define CS6_INPUT_SHA256 "UNBOUND"
#endif

#ifndef CS6_RUN_CHALLENGE
#define CS6_RUN_CHALLENGE "UNBOUND"
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
constexpr int kSectionCoordinate = 2;
constexpr int kOrder = 8;
constexpr int kUIndex = 20000;
constexpr int kSIndex = 15000;
constexpr int kUTiles = 40000;
constexpr int kSTiles = 30000;

constexpr char kVectorField[] =
    "par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs;";
constexpr char kLiouvilleField[] =
    "par:zs;var:x,y,w,ell;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs,"
    "x-y-(w+zs)/2-1;";

interval decimal(const char* value) { return interval(value, value); }

bool finite(const interval& value) {
  return std::isfinite(value.leftBound()) &&
         std::isfinite(value.rightBound()) &&
         value.leftBound() <= value.rightBound();
}

bool finite(const IVector& vector) {
  for (int row = 0; row < vector.dimension(); ++row) {
    if (!finite(vector[row])) {
      return false;
    }
  }
  return true;
}

bool finite(const IMatrix& matrix) {
  for (int row = 0; row < matrix.numberOfRows(); ++row) {
    for (int column = 0; column < matrix.numberOfColumns(); ++column) {
      if (!finite(matrix[row][column])) {
        return false;
      }
    }
  }
  return true;
}

bool positive(const interval& value) {
  return finite(value) && value.leftBound() > 0.0;
}

bool overlaps(const interval& left, const interval& right) {
  return left.leftBound() <= right.rightBound() &&
         right.leftBound() <= left.rightBound();
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
    std::vector<interval> coordinates;
    for (const IVector& value : values) {
      if (value.dimension() != values.front().dimension()) {
        return false;
      }
      coordinates.push_back(value[row]);
    }
    if (!joint_overlap(coordinates)) {
      return false;
    }
  }
  return true;
}

bool joint_overlap(const std::vector<IMatrix>& values) {
  if (values.empty()) {
    return false;
  }
  const int rows = values.front().numberOfRows();
  const int columns = values.front().numberOfColumns();
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      std::vector<interval> entries;
      for (const IMatrix& value : values) {
        if (value.numberOfRows() != rows ||
            value.numberOfColumns() != columns) {
          return false;
        }
        entries.push_back(value[row][column]);
      }
      if (!joint_overlap(entries)) {
        return false;
      }
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
                  const IVector& vector) {
  for (int row = 0; row < vector.dimension(); ++row) {
    write_interval(output, prefix + std::to_string(row), vector[row]);
  }
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

interval determinant_xy(const IMatrix& matrix) {
  return matrix[0][0] * matrix[1][1] -
         matrix[0][1] * matrix[1][0];
}

double width(const interval& value) {
  return value.rightBound() - value.leftBound();
}

IVector midpoint_vector(const IVector& value) {
  IVector result(value.dimension());
  for (int row = 0; row < value.dimension(); ++row) {
    const double midpoint =
        value[row].leftBound() + 0.5 * width(value[row]);
    result[row] = interval(midpoint);
  }
  return result;
}

IMatrix midpoint_tangent_basis(const IMatrix& value) {
  IMatrix result(kDimension, kDimension);
  result.clear();
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      const double midpoint = value[row][column].leftBound() +
                              0.5 * width(value[row][column]);
      result[row][column] = interval(midpoint);
    }
  }
  return result;
}

IMatrix oriented_qr_tangent_basis(const IMatrix& value) {
  const double first_x =
      value[0][0].leftBound() + 0.5 * width(value[0][0]);
  const double first_y =
      value[1][0].leftBound() + 0.5 * width(value[1][0]);
  const double norm = std::hypot(first_x, first_y);
  if (!std::isfinite(norm) || norm == 0.0) {
    throw std::runtime_error("cannot normalize first tangent column");
  }
  const double q0_x = first_x / norm;
  const double q0_y = first_y / norm;
  IMatrix basis(kDimension, kDimension);
  basis.clear();
  basis[0][0] = interval(q0_x);
  basis[1][0] = interval(q0_y);
  basis[0][1] = interval(-q0_y);
  basis[1][1] = interval(q0_x);
  return basis;
}

IMatrix inverse_tangent_basis(const IMatrix& basis) {
  const interval determinant = determinant_xy(basis);
  if (determinant.contains(0.0)) {
    throw std::runtime_error("midpoint tangent basis is singular");
  }
  IMatrix inverse(kDimension, kDimension);
  inverse.clear();
  inverse[0][0] = basis[1][1] / determinant;
  inverse[0][1] = -basis[0][1] / determinant;
  inverse[1][0] = -basis[1][0] / determinant;
  inverse[1][1] = basis[0][0] / determinant;
  return inverse;
}

IMatrix tangent_projector() {
  IMatrix projector(kDimension, kDimension);
  projector.clear();
  projector[0][0] = 1.0;
  projector[1][1] = 1.0;
  return projector;
}

bool contains_matrix(const IMatrix& outer, const IMatrix& inner) {
  if (outer.numberOfRows() != inner.numberOfRows() ||
      outer.numberOfColumns() != inner.numberOfColumns()) {
    return false;
  }
  for (int row = 0; row < outer.numberOfRows(); ++row) {
    for (int column = 0; column < outer.numberOfColumns(); ++column) {
      if (!outer[row][column].contains(inner[row][column])) {
        return false;
      }
    }
  }
  return true;
}

IMatrix tangent_seed() {
  IMatrix seed(kDimension, kDimension);
  seed.clear();
  seed[0][0] = 1.0;
  seed[1][1] = 1.0;
  return seed;
}

IVector project_state_to_section(const IVector& image) {
  if (image.dimension() != kDimension ||
      !image[kSectionCoordinate].contains(0.0)) {
    throw std::runtime_error("event image does not contain coordinate section");
  }
  IVector projected = image;
  projected[kSectionCoordinate] = interval(0.0);
  return projected;
}

IMatrix project_tangent_to_section(const IMatrix& derivative) {
  if (derivative.numberOfRows() != kDimension ||
      derivative.numberOfColumns() != kDimension) {
    throw std::runtime_error("unexpected derivative dimension");
  }
  IMatrix projected = derivative;
  for (int index = 0; index < kDimension; ++index) {
    if (!projected[kSectionCoordinate][index].contains(0.0) ||
        !projected[index][kSectionCoordinate].contains(0.0)) {
      throw std::runtime_error("tangent projection lost a nonzero enclosure");
    }
    projected[kSectionCoordinate][index] = interval(0.0);
    projected[index][kSectionCoordinate] = interval(0.0);
  }
  return projected;
}

struct FrozenInput {
  interval zs = decimal("22.3274637391");
  interval origin_x = decimal("15.186446520640786");
  interval origin_y = decimal("10.908543194765466");
  interval unstable_x = decimal("-0.67430316214199759");
  interval unstable_y = decimal("-0.73845463335624273");
  interval stable_x = decimal("-0.94170446778164518");
  interval stable_y = decimal("0.33644122125579123");
  interval radius_u = decimal("0.004");
  interval radius_s = decimal("0.3");

  interval source_u() const {
    return tile(interval(0.0), radius_u, kUIndex, kUTiles);
  }

  interval source_s() const {
    return tile(interval(0.0), radius_s, kSIndex, kSTiles);
  }

  IMatrix frame3() const {
    IMatrix frame(kDimension, kDimension);
    frame.clear();
    frame[0][0] = unstable_x;
    frame[1][0] = unstable_y;
    frame[0][1] = stable_x;
    frame[1][1] = stable_y;
    frame[2][2] = 1.0;
    return frame;
  }

  IMatrix source_tangent_seed() const {
    IMatrix seed(kDimension, kDimension);
    seed.clear();
    seed[0][0] = unstable_x * radius_u;
    seed[1][0] = unstable_y * radius_u;
    seed[0][1] = stable_x * radius_s;
    seed[1][1] = stable_y * radius_s;
    return seed;
  }

  C1Rect2Set source_set() const {
    const interval u = source_u();
    const interval s = source_s();
    const interval midpoint_u = u.mid();
    const interval midpoint_s = s.mid();
    const IVector center{
        origin_x + unstable_x * midpoint_u + stable_x * midpoint_s,
        origin_y + unstable_y * midpoint_u + stable_y * midpoint_s,
        interval(0.0)};
    const IVector tile_radii{u - midpoint_u, s - midpoint_s, interval(0.0)};
    C1Rect2Set::C0BaseSet c0(center, frame3(), tile_radii);
    C1Rect2Set::C1BaseSet c1(source_tangent_seed());
    return C1Rect2Set(c0, c1);
  }

  C1Rect2Set midpoint_source_set() const {
    const interval midpoint_u = source_u().mid();
    const interval midpoint_s = source_s().mid();
    const IVector center{
        origin_x + unstable_x * midpoint_u + stable_x * midpoint_s,
        origin_y + unstable_y * midpoint_u + stable_y * midpoint_s,
        interval(0.0)};
    C1Rect2Set::C0BaseSet c0(center);
    C1Rect2Set::C1BaseSet c1(source_tangent_seed());
    return C1Rect2Set(c0, c1);
  }

  IVector normalized_tile_delta() const {
    const interval u = source_u();
    const interval s = source_s();
    return IVector{(u - u.mid()) / radius_u,
                   (s - s.mid()) / radius_s, interval(0.0)};
  }
};

struct ReturnData {
  interval time;
  IVector image{kDimension};
  IMatrix flow{kDimension, kDimension};
  IMatrix dp{kDimension, kDimension};
  IVector postsection{kDimension};
  interval postsection_time;
};

class SectionResidentMap : public IPoincareMap {
 public:
  using IPoincareMap::IPoincareMap;

  ReturnData one_return(C1Rect2Set source) {
    ReturnData result;
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
    result.dp = this->computeDP(result.image, result.flow, result.time);
    result.postsection = static_cast<IVector>(after);
    result.postsection_time = after.getCurrentTime();
    return result;
  }
};

ReturnData direct_return(const FrozenInput& input, int return_count) {
  IMap field(kVectorField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(kDimension, kSectionCoordinate);
  IPoincareMap poincare(solver, section, capd::poincare::MinusPlus);
  ReturnData result;
  C1Rect2Set set = input.source_set();
  result.image = poincare(set, result.flow, result.time, return_count);
  result.dp = poincare.computeDP(result.image, result.flow, result.time);
  result.postsection = static_cast<IVector>(set);
  result.postsection_time = set.getCurrentTime();
  return result;
}

ReturnData resident_return(const FrozenInput& input, C1Rect2Set source) {
  IMap field(kVectorField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(kDimension, kSectionCoordinate);
  SectionResidentMap poincare(solver, section, capd::poincare::MinusPlus);
  return poincare.one_return(source);
}

struct LiouvilleData {
  interval time;
  IVector image{4};
  interval initial_velocity;
  interval normal_velocity;
  interval ell;
  interval exp_ell;
  interval determinant;
};

LiouvilleData liouville_return(const FrozenInput& input, int return_count) {
  IMap field(kLiouvilleField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(4, kSectionCoordinate);
  IPoincareMap poincare(solver, section, capd::poincare::MinusPlus);
  IMatrix frame(4, 4);
  frame.setToIdentity();
  frame[0][0] = input.unstable_x;
  frame[1][0] = input.unstable_y;
  frame[0][1] = input.stable_x;
  frame[1][1] = input.stable_y;
  const interval u = input.source_u();
  const interval s = input.source_s();
  const interval midpoint_u = u.mid();
  const interval midpoint_s = s.mid();
  const IVector center{
      input.origin_x + input.unstable_x * midpoint_u +
          input.stable_x * midpoint_s,
      input.origin_y + input.unstable_y * midpoint_u +
          input.stable_y * midpoint_s,
      interval(0.0), interval(0.0)};
  const IVector radii{u - midpoint_u, s - midpoint_s, interval(0.0),
                      interval(0.0)};
  C0HOTripletonSet set(center, frame, radii);
  LiouvilleData result;
  result.image = poincare(set, result.time, return_count);
  const interval initial_x = input.origin_x + input.unstable_x * u +
                             input.stable_x * s;
  const interval initial_y = input.origin_y + input.unstable_y * u +
                             input.stable_y * s;
  result.initial_velocity = initial_x * initial_y - input.zs;
  result.normal_velocity = result.image[0] * result.image[1] -
                           result.image[2] - input.zs;
  result.ell = result.image[3];
  result.exp_ell = exp(result.ell);
  const interval frame_determinant =
      input.unstable_x * input.stable_y -
      input.stable_x * input.unstable_y;
  const interval oriented_q0_area =
      frame_determinant * input.radius_u * input.radius_s;
  result.determinant = result.exp_ell * result.initial_velocity /
                       result.normal_velocity * oriented_q0_area;
  return result;
}

void emit_c0_components(const C1Rect2Set& carrier,
                        const std::string& prefix) {
  const auto& c0 = static_cast<const C1Rect2Set::C0BaseSet&>(carrier);
  write_vector(std::cout, prefix + "X", c0.get_x());
  write_matrix(std::cout, prefix + "C", c0.get_C());
  write_vector(std::cout, prefix + "R0", c0.get_r0());
  write_matrix(std::cout, prefix + "B", c0.get_B());
  write_vector(std::cout, prefix + "R", c0.get_r());
}

void emit_c1_components(const C1Rect2Set& carrier,
                        const std::string& prefix) {
  const auto& c1 = static_cast<const C1Rect2Set::C1BaseSet&>(carrier);
  write_matrix(std::cout, prefix + "D", c1.get_D());
  write_matrix(std::cout, prefix + "C", c1.get_Cjac());
  write_matrix(std::cout, prefix + "R0", c1.get_R0());
  write_matrix(std::cout, prefix + "B", c1.get_Bjac());
  write_matrix(std::cout, prefix + "R", c1.get_R());
}

void emit_gauge(const char* marker, const IMatrix& basis,
                const IMatrix& inverse, const IMatrix& transition,
                const IMatrix& basis_times_inverse,
                const IMatrix& inverse_times_basis,
                const IMatrix& basis_times_transition) {
  std::cout << marker;
  write_matrix(std::cout, "BASIS", basis);
  write_matrix(std::cout, "INVERSE_BASIS", inverse);
  write_matrix(std::cout, "TRANSITION", transition);
  write_matrix(std::cout, "BASIS_TIMES_INVERSE", basis_times_inverse);
  write_matrix(std::cout, "INVERSE_TIMES_BASIS", inverse_times_basis);
  write_matrix(std::cout, "BASIS_TIMES_TRANSITION",
               basis_times_transition);
  std::cout << '\n';
}

void emit_gauge_continuation(const char* marker,
                             const C1Rect2Set& carrier,
                             const IVector& c0_hull,
                             const IMatrix& c1_hull,
                             const IMatrix& incoming_j1) {
  std::cout << marker;
  write_interval(std::cout, "TIME", carrier.getCurrentTime());
  emit_c0_components(carrier, "C0_");
  emit_c1_components(carrier, "C1_");
  write_vector(std::cout, "C0_HULL", c0_hull);
  write_matrix(std::cout, "C1_HULL", c1_hull);
  write_matrix(std::cout, "INCOMING_J1", incoming_j1);
  std::cout << '\n';
}

void emit_gauge_local(const char* marker, const ReturnData& result,
                      const interval& duration,
                      const IVector& section_image,
                      const IMatrix& section_dp, const interval& nu,
                      const interval& det) {
  std::cout << marker;
  write_interval(std::cout, "TIME", result.time);
  write_interval(std::cout, "DURATION", duration);
  write_vector(std::cout, "X", result.image);
  write_matrix(std::cout, "FLOW_TANGENT", result.flow);
  write_matrix(std::cout, "DP", result.dp);
  write_vector(std::cout, "SECTION_X", section_image);
  write_matrix(std::cout, "SECTION_DP", section_dp);
  write_interval(std::cout, "NU", nu);
  write_interval(std::cout, "DET_IN_BASIS", det);
  std::cout << '\n';
}

void emit_gauge_composed(const char* marker, const IMatrix& local_dp,
                         const IMatrix& transition,
                         const IMatrix& fixed_dp, const interval& det) {
  std::cout << marker;
  write_matrix(std::cout, "J2_BASIS", local_dp);
  write_matrix(std::cout, "TRANSITION", transition);
  write_matrix(std::cout, "DP_FIXED_Q0", fixed_dp);
  write_interval(std::cout, "DET_FIXED_Q0", det);
  std::cout << '\n';
}

void emit_gauge_postsection(const char* marker, const ReturnData& result,
                            const interval& sign) {
  std::cout << marker;
  write_interval(std::cout, "TIME", result.postsection_time);
  write_vector(std::cout, "X", result.postsection);
  write_interval(std::cout, "SECTION_SIGN", sign);
  std::cout << '\n';
}

}  // namespace

int main() {
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::boolalpha;
  try {
    const FrozenInput input;
    const ReturnData direct1 = direct_return(input, 1);
    const ReturnData direct2 = direct_return(input, 2);
    const ReturnData local1 = resident_return(input, input.source_set());
    const ReturnData midpoint1 =
        resident_return(input, input.midpoint_source_set());
    const LiouvilleData liouville2 = liouville_return(input, 2);

    const IVector section_image1 = project_state_to_section(local1.image);
    const IMatrix direct_section_dp1 =
        project_tangent_to_section(direct1.dp);
    const IMatrix direct_section_dp2 =
        project_tangent_to_section(direct2.dp);
    const IMatrix local_section_dp1 =
        project_tangent_to_section(local1.dp);
    const IVector midpoint_section_image1 =
        project_state_to_section(midpoint1.image);
    const IVector reconditioned_center1 =
        midpoint_vector(midpoint_section_image1);
    const IMatrix event_affine_basis1 =
        midpoint_tangent_basis(local_section_dp1);
    const IMatrix inverse_event_affine_basis1 =
        inverse_tangent_basis(event_affine_basis1);
    const IMatrix reconditioned_basis1 =
        oriented_qr_tangent_basis(local_section_dp1);
    const IMatrix inverse_basis1 =
        inverse_tangent_basis(reconditioned_basis1);
    const IMatrix basis_bridge1 =
        reconditioned_basis1 * inverse_basis1;
    const IMatrix inverse_bridge1 =
        inverse_basis1 * reconditioned_basis1;
    const IVector normalized_delta1 = input.normalized_tile_delta();
    const IVector center_error1 =
        midpoint_section_image1 - reconditioned_center1;
    IVector section_center_error1 = center_error1;
    section_center_error1[kSectionCoordinate] = interval(0.0);
    IVector linearization_error1 =
        (local_section_dp1 - event_affine_basis1) * normalized_delta1;
    linearization_error1[kSectionCoordinate] = interval(0.0);
    const IVector reconditioned_residual1 =
        section_center_error1 + linearization_error1;
    IMatrix residual_basis1(kDimension, kDimension);
    residual_basis1.setToIdentity();
    const IMatrix seed = tangent_seed();
    C1Rect2Set::C0BaseSet event1_c0(section_image1);
    C1Rect2Set::C1BaseSet event1_c1(local_section_dp1);
    C1Rect2Set event1_carrier(event1_c0, event1_c1, local1.time);
    C1Rect2Set::C1BaseSet seed_c1(seed);
    C1Rect2Set continuation1_carrier(event1_c0, seed_c1, local1.time);

    C1Rect2Set::C0BaseSet reconditioned_event1_c0(
        reconditioned_center1, event_affine_basis1, normalized_delta1,
        residual_basis1, reconditioned_residual1);
    C1Rect2Set::C1BaseSet reconditioned_event1_c1(
        local_section_dp1);
    C1Rect2Set reconditioned_event1_carrier(
        reconditioned_event1_c0, reconditioned_event1_c1, local1.time);
    C1Rect2Set::C1BaseSet reconditioned_seed1_c1(
        reconditioned_basis1);
    C1Rect2Set reconditioned_continuation1_carrier(
        reconditioned_event1_c0, reconditioned_seed1_c1, local1.time);
    reconditioned_continuation1_carrier.setC0Factor(
        std::numeric_limits<double>::infinity());
    C1Rect2Set::C1BaseSet correlated_identity_seed1_c1(seed);
    C1Rect2Set correlated_identity_continuation1_carrier(
        reconditioned_event1_c0, correlated_identity_seed1_c1, local1.time);
    correlated_identity_continuation1_carrier.setC0Factor(
        std::numeric_limits<double>::infinity());
    C1Rect2Set::C1BaseSet correlated_affine_seed1_c1(
        event_affine_basis1);
    C1Rect2Set correlated_affine_continuation1_carrier(
        reconditioned_event1_c0, correlated_affine_seed1_c1, local1.time);
    correlated_affine_continuation1_carrier.setC0Factor(
        std::numeric_limits<double>::infinity());

    const ReturnData local2 = resident_return(input, continuation1_carrier);
    const ReturnData reconditioned_local2 =
        resident_return(input, reconditioned_continuation1_carrier);
    const ReturnData correlated_identity_local2 =
        resident_return(input, correlated_identity_continuation1_carrier);
    const ReturnData correlated_affine_local2 =
        resident_return(input, correlated_affine_continuation1_carrier);
    const IVector section_image2 = project_state_to_section(local2.image);
    const IMatrix local_section_dp2 =
        project_tangent_to_section(local2.dp);
    const IMatrix composed_dp2 = local_section_dp2 * local_section_dp1;
    const IMatrix reversed_dp2 = local_section_dp1 * local_section_dp2;
    const IMatrix reconditioned_section_dp2 =
        project_tangent_to_section(reconditioned_local2.dp);
    const IMatrix correlated_identity_section_dp2 =
        project_tangent_to_section(correlated_identity_local2.dp);
    const IMatrix correlated_affine_section_dp2 =
        project_tangent_to_section(correlated_affine_local2.dp);
    const IMatrix transition1 = inverse_basis1 * local_section_dp1;
    const IMatrix affine_transition1 =
        inverse_event_affine_basis1 * local_section_dp1;
    const IMatrix reconditioned_composed_dp2 =
        reconditioned_section_dp2 * transition1;
    const IMatrix correlated_identity_composed_dp2 =
        correlated_identity_section_dp2 * local_section_dp1;
    const IMatrix correlated_affine_composed_dp2 =
        correlated_affine_section_dp2 * affine_transition1;
    C1Rect2Set::C0BaseSet event2_c0(section_image2);
    C1Rect2Set::C1BaseSet event2_c1(local_section_dp2);
    C1Rect2Set event2_carrier(event2_c0, event2_c1, local2.time);
    C1Rect2Set continuation2_carrier(event2_c0, seed_c1, local2.time);

    const IVector event1_c0_hull = static_cast<IVector>(event1_carrier);
    const IMatrix event1_c1_hull = static_cast<IMatrix>(event1_carrier);
    const IVector continuation1_c0_hull =
        static_cast<IVector>(continuation1_carrier);
    const IMatrix continuation1_c1_hull =
        static_cast<IMatrix>(continuation1_carrier);
    const IVector reconditioned_event1_c0_hull =
        static_cast<IVector>(reconditioned_event1_carrier);
    const IMatrix reconditioned_event1_c1_hull =
        static_cast<IMatrix>(reconditioned_event1_carrier);
    const IVector reconditioned_continuation1_c0_hull =
        static_cast<IVector>(reconditioned_continuation1_carrier);
    const IMatrix reconditioned_continuation1_c1_hull =
        static_cast<IMatrix>(reconditioned_continuation1_carrier);
    const IVector correlated_identity_continuation1_c0_hull =
        static_cast<IVector>(correlated_identity_continuation1_carrier);
    const IMatrix correlated_identity_continuation1_c1_hull =
        static_cast<IMatrix>(correlated_identity_continuation1_carrier);
    const IVector correlated_affine_continuation1_c0_hull =
        static_cast<IVector>(correlated_affine_continuation1_carrier);
    const IMatrix correlated_affine_continuation1_c1_hull =
        static_cast<IMatrix>(correlated_affine_continuation1_carrier);
    const IVector event2_c0_hull = static_cast<IVector>(event2_carrier);
    const IMatrix event2_c1_hull = static_cast<IMatrix>(event2_carrier);
    const IVector continuation2_c0_hull =
        static_cast<IVector>(continuation2_carrier);
    const IMatrix continuation2_c1_hull =
        static_cast<IMatrix>(continuation2_carrier);

    const interval direct1_nu = direct1.image[0] * direct1.image[1] -
                                direct1.image[2] - input.zs;
    const interval local1_nu = local1.image[0] * local1.image[1] -
                               local1.image[2] - input.zs;
    const interval event1_nu = event1_c0_hull[0] * event1_c0_hull[1] -
                               event1_c0_hull[2] - input.zs;
    const interval direct2_nu = direct2.image[0] * direct2.image[1] -
                                direct2.image[2] - input.zs;
    const interval local2_nu = local2.image[0] * local2.image[1] -
                               local2.image[2] - input.zs;
    const interval reconditioned_local2_nu =
        reconditioned_local2.image[0] * reconditioned_local2.image[1] -
        reconditioned_local2.image[2] - input.zs;
    const interval correlated_identity_local2_nu =
        correlated_identity_local2.image[0] *
            correlated_identity_local2.image[1] -
        correlated_identity_local2.image[2] - input.zs;
    const interval correlated_affine_local2_nu =
        correlated_affine_local2.image[0] *
            correlated_affine_local2.image[1] -
        correlated_affine_local2.image[2] - input.zs;
    const interval event2_nu = event2_c0_hull[0] * event2_c0_hull[1] -
                               event2_c0_hull[2] - input.zs;
    const interval direct1_det = determinant_xy(direct_section_dp1);
    const interval local1_det = determinant_xy(local_section_dp1);
    const interval direct2_det = determinant_xy(direct_section_dp2);
    const interval local2_det = determinant_xy(local_section_dp2);
    const interval composed_det2 = determinant_xy(composed_dp2);
    const interval reconditioned_local2_det =
        determinant_xy(reconditioned_section_dp2);
    const interval reconditioned_composed_det2 =
        determinant_xy(reconditioned_composed_dp2);
    const interval correlated_identity_local2_det =
        determinant_xy(correlated_identity_section_dp2);
    const interval correlated_identity_composed_det2 =
        determinant_xy(correlated_identity_composed_dp2);
    const interval correlated_affine_local2_det =
        determinant_xy(correlated_affine_section_dp2);
    const interval correlated_affine_composed_det2 =
        determinant_xy(correlated_affine_composed_dp2);
    const interval duration2 = local2.time - local1.time;
    const interval correlated_identity_duration2 =
        correlated_identity_local2.time - local1.time;
    const interval correlated_affine_duration2 =
        correlated_affine_local2.time - local1.time;
    const interval reconditioned_duration2 =
        reconditioned_local2.time - local1.time;
    const IVector correlated_identity_section_image2 =
        project_state_to_section(correlated_identity_local2.image);
    const IVector correlated_affine_section_image2 =
        project_state_to_section(correlated_affine_local2.image);
    const IVector reconditioned_section_image2 =
        project_state_to_section(reconditioned_local2.image);
    const interval postsection1_sign =
        local1.postsection[kSectionCoordinate];
    const interval postsection2_sign =
        local2.postsection[kSectionCoordinate];
    const interval correlated_identity_postsection2_sign =
        correlated_identity_local2.postsection[kSectionCoordinate];
    const interval correlated_affine_postsection2_sign =
        correlated_affine_local2.postsection[kSectionCoordinate];
    const interval reconditioned_postsection2_sign =
        reconditioned_local2.postsection[kSectionCoordinate];

    const bool p1_state_overlap = joint_overlap(
        {direct1.image, local1.image, event1_c0_hull,
         continuation1_c0_hull});
    const bool p1_time_overlap = joint_overlap(
        {direct1.time, local1.time, event1_carrier.getCurrentTime(),
         continuation1_carrier.getCurrentTime()});
    const bool p1_dp_overlap = joint_overlap(
        {direct_section_dp1, local_section_dp1, event1_c1_hull});
    const bool p2_state_overlap = joint_overlap(
        {direct2.image, local2.image, event2_c0_hull,
         continuation2_c0_hull,
         IVector{liouville2.image[0], liouville2.image[1],
                 liouville2.image[2]}});
    const bool p2_time_overlap = joint_overlap(
        {direct2.time, local2.time, event2_carrier.getCurrentTime(),
         continuation2_carrier.getCurrentTime(), liouville2.time});
    const bool p2_dp_overlap = joint_overlap(
        {direct_section_dp2, composed_dp2});
    const bool p2_velocity_overlap = joint_overlap(
        {direct2_nu, local2_nu, event2_nu,
         liouville2.normal_velocity});
    const bool p2_determinant_overlap = joint_overlap(
        {direct2_det, composed_det2, liouville2.determinant});
    const bool reconditioned_p1_state_overlap = joint_overlap(
        {direct1.image, local1.image, reconditioned_event1_c0_hull,
         reconditioned_continuation1_c0_hull});
    const bool reconditioned_p1_dp_overlap = joint_overlap(
        {direct_section_dp1, local_section_dp1,
         reconditioned_event1_c1_hull});
    const bool reconditioned_p2_state_overlap = joint_overlap(
        {direct2.image, reconditioned_local2.image,
         IVector{liouville2.image[0], liouville2.image[1],
                 liouville2.image[2]}});
    const bool reconditioned_p2_time_overlap = joint_overlap(
        {direct2.time, reconditioned_local2.time, liouville2.time});
    const bool reconditioned_p2_dp_overlap = joint_overlap(
        {direct_section_dp2, reconditioned_composed_dp2});
    const bool reconditioned_p2_velocity_overlap = joint_overlap(
        {direct2_nu, reconditioned_local2_nu,
         liouville2.normal_velocity});
    const bool reconditioned_p2_determinant_overlap = joint_overlap(
        {direct2_det, reconditioned_composed_det2,
         liouville2.determinant});
    const bool identity_p2_joint_overlap =
        joint_overlap({direct2.image, correlated_identity_local2.image,
                       IVector{liouville2.image[0], liouville2.image[1],
                               liouville2.image[2]}}) &&
        joint_overlap({direct2.time, correlated_identity_local2.time,
                       liouville2.time}) &&
        joint_overlap({direct_section_dp2,
                       correlated_identity_composed_dp2}) &&
        joint_overlap({direct2_nu, correlated_identity_local2_nu,
                       liouville2.normal_velocity}) &&
        joint_overlap({direct2_det, correlated_identity_composed_det2,
                       liouville2.determinant});
    const bool affine_p2_joint_overlap =
        joint_overlap({direct2.image, correlated_affine_local2.image,
                       IVector{liouville2.image[0], liouville2.image[1],
                               liouville2.image[2]}}) &&
        joint_overlap({direct2.time, correlated_affine_local2.time,
                       liouville2.time}) &&
        joint_overlap({direct_section_dp2,
                       correlated_affine_composed_dp2}) &&
        joint_overlap({direct2_nu, correlated_affine_local2_nu,
                       liouville2.normal_velocity}) &&
        joint_overlap({direct2_det, correlated_affine_composed_det2,
                       liouville2.determinant});
    const IMatrix projector = tangent_projector();
    const IMatrix affine_basis_bridge1 =
        event_affine_basis1 * inverse_event_affine_basis1;
    const IMatrix affine_inverse_bridge1 =
        inverse_event_affine_basis1 * event_affine_basis1;
    const IMatrix identity_basis_bridge1 = seed * seed;
    const IMatrix identity_transition_image1 = seed * local_section_dp1;
    const IMatrix affine_transition_image1 =
        event_affine_basis1 * affine_transition1;
    const IMatrix qr_transition_image1 =
        reconditioned_basis1 * transition1;
    const bool qr_basis_inverse_certified =
        contains_matrix(basis_bridge1, projector) &&
        contains_matrix(inverse_bridge1, projector);
    const bool affine_basis_inverse_certified =
        contains_matrix(affine_basis_bridge1, projector) &&
        contains_matrix(affine_inverse_bridge1, projector);
    const bool transition_reconstruction_certified =
        contains_matrix(identity_transition_image1, local_section_dp1) &&
        contains_matrix(affine_transition_image1, local_section_dp1) &&
        contains_matrix(qr_transition_image1, local_section_dp1);
    const bool flat_determinant_crosses_zero =
        composed_det2.contains(0.0);
    const bool identity_determinant_crosses_zero =
        correlated_identity_composed_det2.contains(0.0);
    const bool affine_determinant_crosses_zero =
        correlated_affine_composed_det2.contains(0.0);
    const bool qr_determinant_crosses_zero =
        reconditioned_composed_det2.contains(0.0);
    const bool any_gauge_sign_definite =
        !identity_determinant_crosses_zero ||
        !affine_determinant_crosses_zero ||
        !qr_determinant_crosses_zero;
    const bool liouville_determinant_negative =
        liouville2.determinant.rightBound() < 0.0;
    const bool determinant_width_improved =
        width(reconditioned_composed_det2) < width(composed_det2);
    const bool identity_determinant_width_improved =
        width(correlated_identity_composed_det2) < width(composed_det2);
    const bool affine_determinant_width_improved =
        width(correlated_affine_composed_det2) < width(composed_det2);
    const bool correlated_state_componentwise_narrower =
        width(correlated_identity_local2.image[0]) < width(local2.image[0]) &&
        width(correlated_identity_local2.image[1]) < width(local2.image[1]) &&
        width(correlated_affine_local2.image[0]) < width(local2.image[0]) &&
        width(correlated_affine_local2.image[1]) < width(local2.image[1]) &&
        width(reconditioned_local2.image[0]) < width(local2.image[0]) &&
        width(reconditioned_local2.image[1]) < width(local2.image[1]);
    const bool correlated_carriers_c0_same =
        capd::vectalg::subset(reconditioned_event1_c0_hull,
                             reconditioned_continuation1_c0_hull) &&
        capd::vectalg::subset(reconditioned_continuation1_c0_hull,
                             reconditioned_event1_c0_hull) &&
        capd::vectalg::subset(correlated_identity_continuation1_c0_hull,
                             reconditioned_event1_c0_hull) &&
        capd::vectalg::subset(reconditioned_event1_c0_hull,
                             correlated_identity_continuation1_c0_hull) &&
        capd::vectalg::subset(correlated_affine_continuation1_c0_hull,
                             reconditioned_event1_c0_hull) &&
        capd::vectalg::subset(reconditioned_event1_c0_hull,
                             correlated_affine_continuation1_c0_hull);
    const bool carrier1_c0_same = joint_overlap(
        {event1_c0_hull, continuation1_c0_hull}) &&
        capd::vectalg::subset(event1_c0_hull, continuation1_c0_hull) &&
        capd::vectalg::subset(continuation1_c0_hull, event1_c0_hull);
    const bool carrier2_c0_same = joint_overlap(
        {event2_c0_hull, continuation2_c0_hull}) &&
        capd::vectalg::subset(event2_c0_hull, continuation2_c0_hull) &&
        capd::vectalg::subset(continuation2_c0_hull, event2_c0_hull);
    const bool event_sections_exact =
        event1_c0_hull[kSectionCoordinate] == interval(0.0) &&
        continuation1_c0_hull[kSectionCoordinate] == interval(0.0) &&
        event2_c0_hull[kSectionCoordinate] == interval(0.0) &&
        continuation2_c0_hull[kSectionCoordinate] == interval(0.0);
    const bool seeds_exact =
        capd::vectalg::subset(seed, continuation1_c1_hull) &&
        capd::vectalg::subset(continuation1_c1_hull, seed) &&
        capd::vectalg::subset(seed, continuation2_c1_hull) &&
        capd::vectalg::subset(continuation2_c1_hull, seed);
    bool order_discriminated = false;
    for (int row = 0; row < kDimension; ++row) {
      for (int column = 0; column < kDimension; ++column) {
        if (!overlaps(composed_dp2[row][column],
                      reversed_dp2[row][column])) {
          order_discriminated = true;
        }
      }
    }
    const bool second_event_after_first =
        local2.time.leftBound() > local1.time.rightBound() &&
        direct2.time.leftBound() > direct1.time.rightBound() &&
        correlated_identity_local2.time.leftBound() >
            local1.time.rightBound() &&
        correlated_affine_local2.time.leftBound() >
            local1.time.rightBound() &&
        reconditioned_local2.time.leftBound() > local1.time.rightBound() &&
        duration2.leftBound() > 0.0 &&
        correlated_identity_duration2.leftBound() > 0.0 &&
        correlated_affine_duration2.leftBound() > 0.0 &&
        reconditioned_duration2.leftBound() > 0.0;
    const bool postsections_after_events =
        local1.postsection_time.leftBound() > local1.time.rightBound() &&
        local2.postsection_time.leftBound() > local2.time.rightBound() &&
        correlated_identity_local2.postsection_time.leftBound() >
            correlated_identity_local2.time.rightBound() &&
        correlated_affine_local2.postsection_time.leftBound() >
            correlated_affine_local2.time.rightBound() &&
        reconditioned_local2.postsection_time.leftBound() >
            reconditioned_local2.time.rightBound();
    const bool postsections_plus = postsection1_sign.leftBound() > 0.0 &&
        postsection2_sign.leftBound() > 0.0 &&
        correlated_identity_postsection2_sign.leftBound() > 0.0 &&
        correlated_affine_postsection2_sign.leftBound() > 0.0 &&
        reconditioned_postsection2_sign.leftBound() > 0.0;
    const bool all_finite =
        finite(direct1.image) && finite(direct1.flow) && finite(direct1.dp) &&
        finite(direct2.image) && finite(direct2.flow) && finite(direct2.dp) &&
        finite(local1.image) && finite(local1.flow) && finite(local1.dp) &&
        finite(local2.image) && finite(local2.flow) && finite(local2.dp) &&
        finite(midpoint1.image) && finite(midpoint1.flow) &&
        finite(midpoint1.dp) && finite(reconditioned_local2.image) &&
        finite(reconditioned_local2.flow) &&
        finite(reconditioned_local2.dp) &&
        finite(correlated_identity_local2.image) &&
        finite(correlated_identity_local2.flow) &&
        finite(correlated_identity_local2.dp) &&
        finite(correlated_affine_local2.image) &&
        finite(correlated_affine_local2.flow) &&
        finite(correlated_affine_local2.dp) &&
        finite(event1_c0_hull) && finite(event1_c1_hull) &&
        finite(reconditioned_event1_c0_hull) &&
        finite(reconditioned_event1_c1_hull) &&
        finite(reconditioned_center1) && finite(event_affine_basis1) &&
        finite(reconditioned_basis1) &&
        finite(inverse_basis1) && finite(transition1) &&
        finite(inverse_event_affine_basis1) && finite(affine_transition1) &&
        finite(reconditioned_residual1) &&
        finite(event2_c0_hull) && finite(event2_c1_hull) &&
        finite(composed_dp2) && finite(reversed_dp2) &&
        finite(reconditioned_composed_dp2) &&
        finite(correlated_identity_composed_dp2) &&
        finite(correlated_affine_composed_dp2) &&
        finite(liouville2.image) && positive(direct1_nu) &&
        positive(local1_nu) && positive(event1_nu) &&
        positive(direct2_nu) && positive(local2_nu) &&
        positive(reconditioned_local2_nu) &&
        positive(correlated_identity_local2_nu) &&
        positive(correlated_affine_local2_nu) &&
        positive(event2_nu) && positive(liouville2.normal_velocity) &&
        positive(liouville2.exp_ell);
    const bool pass =
        all_finite && p1_state_overlap && p1_time_overlap &&
        p1_dp_overlap && p2_state_overlap && p2_time_overlap &&
        p2_dp_overlap && p2_velocity_overlap && p2_determinant_overlap &&
        carrier1_c0_same && carrier2_c0_same && event_sections_exact &&
        seeds_exact && order_discriminated && second_event_after_first &&
        postsections_after_events && postsections_plus &&
        reconditioned_p1_state_overlap && reconditioned_p1_dp_overlap &&
        reconditioned_p2_state_overlap && reconditioned_p2_time_overlap &&
        reconditioned_p2_dp_overlap &&
        reconditioned_p2_velocity_overlap &&
        reconditioned_p2_determinant_overlap && identity_p2_joint_overlap &&
        affine_p2_joint_overlap && qr_basis_inverse_certified &&
        affine_basis_inverse_certified &&
        transition_reconstruction_certified && correlated_carriers_c0_same &&
        flat_determinant_crosses_zero &&
        identity_determinant_crosses_zero &&
        affine_determinant_crosses_zero && qr_determinant_crosses_zero &&
        !any_gauge_sign_definite && liouville_determinant_negative &&
        correlated_state_componentwise_narrower;

    std::cout
        << "SCHEMA=sounio.cs6.section-resident-reconditioned-two-return.v1\n"
        << "WORKER_SOURCE_SHA256=" << CS6_WORKER_SOURCE_SHA256 << '\n'
        << "INPUT_SHA256=" << CS6_INPUT_SHA256 << '\n'
        << "RUN_CHALLENGE=" << CS6_RUN_CHALLENGE << '\n'
        << "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0\n"
        << "INTERVAL_BACKEND_DECLARED=FILIB\n"
        << "INTERVAL_SERIALIZATION=ONE_ULP_OUTWARD_BINARY64_HEX\n"
        << "DIRECT_FLOW_TANGENT_ROLE=D_FLOW_TIMES_Q0\n"
        << "LOCAL_P1_FLOW_TANGENT_ROLE=D_FLOW_TIMES_Q0\n"
        << "FLAT_LOCAL_P2_FLOW_TANGENT_ROLE=D_FLOW_LOCAL_TIMES_SECTION_IDENTITY\n"
        << "GAUGE_FLOW_TANGENT_ROLE=D_FLOW_LOCAL_TIMES_GAUGE_BASIS\n"
        << "WIDTH_COMPARISON_FRAME=FIXED_SOURCE_Q0_COORDINATES\n"
        << "SOURCE_TANGENT_SEED_ROLE=GLOBAL_FRAME_RADII_WITH_ZERO_DUMMY_NORMAL\n"
        << "Q0_AREA_ROLE=ORIENTED_XY_MINOR_OF_FIXED_GLOBAL_BASIS_NOT_TILE_AREA\n"
        << "TANGENT_ZERO_TIGHTENING=COORDINATE_OUTPUT_ROW_AND_Q0_DUMMY_INPUT_COLUMN\n"
        << "SOURCE=N0\n"
        << "U_INDEX=" << kUIndex << '\n'
        << "S_INDEX=" << kSIndex << '\n'
        << "U_TILES=" << kUTiles << '\n'
        << "S_TILES=" << kSTiles << '\n'
        << "ORDER=" << kOrder << '\n'
        << "CUMULATIVE_RETURN_COUNT=2\n"
        << "LOCAL_RETURN_COUNT=1\n"
        << "SECTION=COORDINATE_W_EQUALS_ZERO\n"
        << "CROSSING_DIRECTION=MINUS_PLUS\n"
        << "FAST_PATH_REQUIRED=true\n"
        << "EVENT1_CARRIER_ROLE=TERMINAL_J1_IN_EVIDENCE\n"
        << "CONTINUATION1_CARRIER_ROLE=LOCAL_TANGENT_SEED_WITH_J1_METADATA\n"
        << "EVENT2_CARRIER_ROLE=TERMINAL_J2_LOCAL_EVIDENCE\n"
        << "CONTINUATION2_CARRIER_ROLE=LOCAL_TANGENT_SEED_WITH_COMPOSED_METADATA\n"
        << "COMPOSITION_ORDER=J2_LOCAL_TIMES_J1_IN\n"
        << "GAUGE_COMPOSITION_ORDER=J2_BASIS_TIMES_BASIS_INVERSE_TIMES_J1_IN\n"
        << "EVENT1_C0_REPRESENTATION=MEAN_VALUE_DOUBLETON\n"
        << "EVENT1_C0_FORM=CENTER_PLUS_MID_J1_TIMES_NORMALIZED_DELTA_PLUS_RESIDUAL\n"
        << "EVENT1_C0_RESIDUAL_ROLE=POINT_INTEGRATION_ERROR_PLUS_J1_RADIUS\n"
        << "EVENT1_AFFINE_BASIS=MIDPOINT_J1\n"
        << "TANGENT_GAUGES=IDENTITY,MIDPOINT_M,ORIENTED_QR\n"
        << "PRIMARY_TANGENT_RECONDITIONING=ORIENTED_QR_OF_MIDPOINT_J1\n"
        << "C0_FACTOR_REORGANIZATION=DISABLED_TO_PRESERVE_SOURCE_R0\n"
        << "FLATTENED_TWO_RETURN_CONTROL_RETAINED=true\n"
        << "FLATTENED_BASELINE_RECEIPT_SHA256=14315dd35ada83d13bddaa1c653e0dea86a9da91379559e7f64d69b314077dba\n"
        << "FLATTENED_BASELINE_PHYSICAL_CHAIN_SHA256=536dea89d9f841e0afedaaeb9ef116f5237fb7dd96f7774340850833b5f4b0b1\n"
        << "SCIENTIFIC_RESULT_CLASS=CORRELATION_PRESERVED_ORIENTATION_UNRESOLVED\n"
        << "AUTONOMOUS_VECTOR_FIELD=true\n"
        << "EVENT_TIME_SENSITIVITY_PROPAGATED=false\n"
        << "NONAUTONOMOUS_GENERALIZATION_PROVED=false\n"
        << "INCOMING_DP_REINJECTED=false\n"
        << "POSTSECTION_STATE_REUSED=false\n"
        << "LIOUVILLE_REJECT_ONLY=true\n"
        << "EXECUTION_SCOPE=BOUNDED_LOCAL_CAPD_CPU_PROBE\n"
        << "EXECUTION_PROVENANCE_ATTESTED=false\n"
        << "INDEPENDENT_REPLAY_REQUIRED=true\n"
        << "PROMOTION_ELIGIBLE=false\n"
        << "FULL_SOURCE_CARRIER_PROVED=false\n"
        << "HYPERBOLICITY_PROVED=false\n"
        << "CHAOTIC_ATTRACTOR_PROVED=false\n";
    std::cout << "SOURCE_TILE";
    write_interval(std::cout, "SOURCE_U", input.source_u());
    write_interval(std::cout, "SOURCE_S", input.source_s());
    write_matrix(std::cout, "Q0", input.source_tangent_seed());
    std::cout << '\n';

    std::cout << "MIDPOINT_P1";
    write_interval(std::cout, "TIME", midpoint1.time);
    write_vector(std::cout, "X", midpoint1.image);
    write_vector(std::cout, "SECTION_X", midpoint_section_image1);
    std::cout << '\n';

    std::cout << "MEAN_VALUE_C0";
    write_vector(std::cout, "CENTER", reconditioned_center1);
    write_vector(std::cout, "NORMALIZED_DELTA", normalized_delta1);
    write_matrix(std::cout, "M", event_affine_basis1);
    write_matrix(std::cout, "RESIDUAL_BASIS", residual_basis1);
    write_vector(std::cout, "CENTER_ERROR", section_center_error1);
    write_vector(std::cout, "LINEARIZATION_ERROR", linearization_error1);
    write_vector(std::cout, "RESIDUAL", reconditioned_residual1);
    std::cout << '\n';

    std::cout << "DIRECT_P1";
    write_interval(std::cout, "TIME", direct1.time);
    write_vector(std::cout, "X", direct1.image);
    write_matrix(std::cout, "FLOW_TANGENT", direct1.flow);
    write_matrix(std::cout, "DP", direct1.dp);
    write_matrix(std::cout, "SECTION_DP", direct_section_dp1);
    write_interval(std::cout, "NU", direct1_nu);
    write_interval(std::cout, "DET", direct1_det);
    std::cout << '\n';

    std::cout << "LOCAL_P1";
    write_interval(std::cout, "TIME", local1.time);
    write_vector(std::cout, "X", local1.image);
    write_matrix(std::cout, "FLOW_TANGENT", local1.flow);
    write_matrix(std::cout, "DP", local1.dp);
    write_vector(std::cout, "SECTION_X", section_image1);
    write_matrix(std::cout, "SECTION_DP", local_section_dp1);
    write_interval(std::cout, "NU", local1_nu);
    write_interval(std::cout, "DET", local1_det);
    std::cout << '\n';

    std::cout << "EVENT1_CARRIER";
    write_interval(std::cout, "TIME", event1_carrier.getCurrentTime());
    emit_c0_components(event1_carrier, "C0_");
    emit_c1_components(event1_carrier, "C1_");
    write_vector(std::cout, "C0_HULL", event1_c0_hull);
    write_matrix(std::cout, "C1_HULL", event1_c1_hull);
    write_interval(std::cout, "NU", event1_nu);
    std::cout << '\n';

    std::cout << "CONTINUATION1_CARRIER";
    write_interval(std::cout, "TIME", continuation1_carrier.getCurrentTime());
    emit_c0_components(continuation1_carrier, "C0_");
    emit_c1_components(continuation1_carrier, "C1_");
    write_vector(std::cout, "C0_HULL", continuation1_c0_hull);
    write_matrix(std::cout, "C1_HULL", continuation1_c1_hull);
    write_matrix(std::cout, "INCOMING_J1", local_section_dp1);
    std::cout << '\n';

    std::cout << "RECONDITIONED_EVENT1_CARRIER";
    write_interval(std::cout, "TIME",
                   reconditioned_event1_carrier.getCurrentTime());
    emit_c0_components(reconditioned_event1_carrier, "C0_");
    emit_c1_components(reconditioned_event1_carrier, "C1_");
    write_vector(std::cout, "C0_HULL", reconditioned_event1_c0_hull);
    write_matrix(std::cout, "C1_HULL", reconditioned_event1_c1_hull);
    std::cout << '\n';

    emit_gauge("GAUGE_IDENTITY", seed, seed, local_section_dp1,
               identity_basis_bridge1, identity_basis_bridge1,
               identity_transition_image1);
    emit_gauge("GAUGE_MIDPOINT_M", event_affine_basis1,
               inverse_event_affine_basis1, affine_transition1,
               affine_basis_bridge1, affine_inverse_bridge1,
               affine_transition_image1);
    emit_gauge("GAUGE_ORIENTED_QR", reconditioned_basis1, inverse_basis1,
               transition1, basis_bridge1, inverse_bridge1,
               qr_transition_image1);

    emit_gauge_continuation(
        "GAUGE_IDENTITY_CONTINUATION1",
        correlated_identity_continuation1_carrier,
        correlated_identity_continuation1_c0_hull,
        correlated_identity_continuation1_c1_hull, local_section_dp1);
    emit_gauge_continuation(
        "GAUGE_MIDPOINT_M_CONTINUATION1",
        correlated_affine_continuation1_carrier,
        correlated_affine_continuation1_c0_hull,
        correlated_affine_continuation1_c1_hull, local_section_dp1);
    emit_gauge_continuation(
        "GAUGE_ORIENTED_QR_CONTINUATION1",
        reconditioned_continuation1_carrier,
        reconditioned_continuation1_c0_hull,
        reconditioned_continuation1_c1_hull, local_section_dp1);

    std::cout << "DIRECT_P2";
    write_interval(std::cout, "TIME", direct2.time);
    write_vector(std::cout, "X", direct2.image);
    write_matrix(std::cout, "FLOW_TANGENT", direct2.flow);
    write_matrix(std::cout, "DP", direct2.dp);
    write_matrix(std::cout, "SECTION_DP", direct_section_dp2);
    write_interval(std::cout, "NU", direct2_nu);
    write_interval(std::cout, "DET", direct2_det);
    std::cout << '\n';

    std::cout << "LOCAL_P2";
    write_interval(std::cout, "TIME", local2.time);
    write_interval(std::cout, "DURATION", duration2);
    write_vector(std::cout, "X", local2.image);
    write_matrix(std::cout, "FLOW_TANGENT", local2.flow);
    write_matrix(std::cout, "DP", local2.dp);
    write_vector(std::cout, "SECTION_X", section_image2);
    write_matrix(std::cout, "SECTION_DP", local_section_dp2);
    write_interval(std::cout, "NU", local2_nu);
    write_interval(std::cout, "DET", local2_det);
    std::cout << '\n';

    std::cout << "COMPOSED_P2";
    write_matrix(std::cout, "J1", local_section_dp1);
    write_matrix(std::cout, "J2_LOCAL", local_section_dp2);
    write_matrix(std::cout, "DP", composed_dp2);
    write_matrix(std::cout, "REVERSED_DP", reversed_dp2);
    write_interval(std::cout, "DET", composed_det2);
    std::cout << '\n';

    emit_gauge_local("GAUGE_IDENTITY_LOCAL_P2",
                     correlated_identity_local2,
                     correlated_identity_duration2,
                     correlated_identity_section_image2,
                     correlated_identity_section_dp2,
                     correlated_identity_local2_nu,
                     correlated_identity_local2_det);
    emit_gauge_local("GAUGE_MIDPOINT_M_LOCAL_P2",
                     correlated_affine_local2,
                     correlated_affine_duration2,
                     correlated_affine_section_image2,
                     correlated_affine_section_dp2,
                     correlated_affine_local2_nu,
                     correlated_affine_local2_det);
    emit_gauge_local("GAUGE_ORIENTED_QR_LOCAL_P2", reconditioned_local2,
                     reconditioned_duration2, reconditioned_section_image2,
                     reconditioned_section_dp2, reconditioned_local2_nu,
                     reconditioned_local2_det);

    emit_gauge_composed("GAUGE_IDENTITY_COMPOSED_P2",
                        correlated_identity_section_dp2,
                        local_section_dp1,
                        correlated_identity_composed_dp2,
                        correlated_identity_composed_det2);
    emit_gauge_composed("GAUGE_MIDPOINT_M_COMPOSED_P2",
                        correlated_affine_section_dp2, affine_transition1,
                        correlated_affine_composed_dp2,
                        correlated_affine_composed_det2);
    emit_gauge_composed("GAUGE_ORIENTED_QR_COMPOSED_P2",
                        reconditioned_section_dp2, transition1,
                        reconditioned_composed_dp2,
                        reconditioned_composed_det2);

    std::cout << "EVENT2_CARRIER";
    write_interval(std::cout, "TIME", event2_carrier.getCurrentTime());
    emit_c0_components(event2_carrier, "C0_");
    emit_c1_components(event2_carrier, "C1_");
    write_vector(std::cout, "C0_HULL", event2_c0_hull);
    write_matrix(std::cout, "C1_HULL", event2_c1_hull);
    write_interval(std::cout, "NU", event2_nu);
    std::cout << '\n';

    std::cout << "CONTINUATION2_CARRIER";
    write_interval(std::cout, "TIME", continuation2_carrier.getCurrentTime());
    emit_c0_components(continuation2_carrier, "C0_");
    emit_c1_components(continuation2_carrier, "C1_");
    write_vector(std::cout, "C0_HULL", continuation2_c0_hull);
    write_matrix(std::cout, "C1_HULL", continuation2_c1_hull);
    write_matrix(std::cout, "INCOMING_J2_LOCAL", local_section_dp2);
    write_matrix(std::cout, "INCOMING_COMPOSED_P2", composed_dp2);
    std::cout << '\n';

    std::cout << "POSTSECTION1";
    write_interval(std::cout, "TIME", local1.postsection_time);
    write_vector(std::cout, "X", local1.postsection);
    write_interval(std::cout, "SECTION_SIGN", postsection1_sign);
    std::cout << '\n';

    std::cout << "POSTSECTION2";
    write_interval(std::cout, "TIME", local2.postsection_time);
    write_vector(std::cout, "X", local2.postsection);
    write_interval(std::cout, "SECTION_SIGN", postsection2_sign);
    std::cout << '\n';

    emit_gauge_postsection("GAUGE_IDENTITY_POSTSECTION2",
                           correlated_identity_local2,
                           correlated_identity_postsection2_sign);
    emit_gauge_postsection("GAUGE_MIDPOINT_M_POSTSECTION2",
                           correlated_affine_local2,
                           correlated_affine_postsection2_sign);
    emit_gauge_postsection("GAUGE_ORIENTED_QR_POSTSECTION2",
                           reconditioned_local2,
                           reconditioned_postsection2_sign);

    std::cout << "LIOUVILLE_P2";
    write_interval(std::cout, "TIME", liouville2.time);
    write_vector(std::cout, "X", liouville2.image);
    write_interval(std::cout, "NU0", liouville2.initial_velocity);
    write_interval(std::cout, "NU2", liouville2.normal_velocity);
    write_interval(std::cout, "ELL", liouville2.ell);
    write_interval(std::cout, "EXP_ELL", liouville2.exp_ell);
    write_interval(std::cout, "DET", liouville2.determinant);
    std::cout << '\n';

    std::cout << "SUMMARY ALL_FINITE=" << all_finite
              << " P1_STATE_JOINT_OVERLAP=" << p1_state_overlap
              << " P1_TIME_JOINT_OVERLAP=" << p1_time_overlap
              << " P1_DP_JOINT_OVERLAP=" << p1_dp_overlap
              << " P2_STATE_JOINT_OVERLAP=" << p2_state_overlap
              << " P2_TIME_JOINT_OVERLAP=" << p2_time_overlap
              << " P2_DP_JOINT_OVERLAP=" << p2_dp_overlap
              << " P2_VELOCITY_JOINT_OVERLAP=" << p2_velocity_overlap
              << " P2_DETERMINANT_JOINT_OVERLAP="
              << p2_determinant_overlap
              << " MEAN_VALUE_P1_STATE_JOINT_OVERLAP="
              << reconditioned_p1_state_overlap
              << " MEAN_VALUE_P1_DP_JOINT_OVERLAP="
              << reconditioned_p1_dp_overlap
              << " IDENTITY_P2_JOINT_OVERLAP=" << identity_p2_joint_overlap
              << " MIDPOINT_M_P2_JOINT_OVERLAP=" << affine_p2_joint_overlap
              << " ORIENTED_QR_P2_JOINT_OVERLAP="
              << (reconditioned_p2_state_overlap &&
                  reconditioned_p2_time_overlap &&
                  reconditioned_p2_dp_overlap &&
                  reconditioned_p2_velocity_overlap &&
                  reconditioned_p2_determinant_overlap)
              << " MEAN_VALUE_C0_SHARED=" << correlated_carriers_c0_same
              << " MIDPOINT_M_INVERSE_CERTIFIED="
              << affine_basis_inverse_certified
              << " ORIENTED_QR_INVERSE_CERTIFIED="
              << qr_basis_inverse_certified
              << " TRANSITIONS_RECONSTRUCT_J1="
              << transition_reconstruction_certified
              << " FLAT_DETERMINANT_CROSSES_ZERO="
              << flat_determinant_crosses_zero
              << " IDENTITY_DETERMINANT_CROSSES_ZERO="
              << identity_determinant_crosses_zero
              << " MIDPOINT_M_DETERMINANT_CROSSES_ZERO="
              << affine_determinant_crosses_zero
              << " ORIENTED_QR_DETERMINANT_CROSSES_ZERO="
              << qr_determinant_crosses_zero
              << " ANY_GAUGE_SIGN_DEFINITE=" << any_gauge_sign_definite
              << " LIOUVILLE_DETERMINANT_NEGATIVE="
              << liouville_determinant_negative
              << " IDENTITY_DETERMINANT_WIDTH_IMPROVED="
              << identity_determinant_width_improved
              << " MIDPOINT_M_DETERMINANT_WIDTH_IMPROVED="
              << affine_determinant_width_improved
              << " ORIENTED_QR_DETERMINANT_WIDTH_IMPROVED="
              << determinant_width_improved
              << " CORRELATED_STATE_COMPONENTWISE_NARROWER="
              << correlated_state_componentwise_narrower
              << " CARRIER1_C0_IDENTICAL=" << carrier1_c0_same
              << " CARRIER2_C0_IDENTICAL=" << carrier2_c0_same
              << " EVENT_SECTIONS_EXACT=" << event_sections_exact
              << " CONTINUATION_SEEDS_EXACT=" << seeds_exact
              << " COMPOSITION_ORDER_DISCRIMINATED=" << order_discriminated
              << " SECOND_EVENT_STRICTLY_LATER=" << second_event_after_first
              << " POSTSECTIONS_STRICTLY_LATER=" << postsections_after_events
              << " POSTSECTIONS_PLUS=" << postsections_plus
              << " CERTIFICATE_PASS=" << pass
              << " PROBE_PASS=" << pass << '\n';
    return pass ? EXIT_SUCCESS : 2;
  } catch (const std::exception& error) {
    std::cerr << "CS6_SECTION_RESIDENT_RECONDITIONED_TWO_RETURN_PROBE_ERROR="
              << error.what() << '\n';
    return 2;
  }
}
