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

}  // namespace

int main() {
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::boolalpha;
  try {
    const FrozenInput input;
    const ReturnData direct1 = direct_return(input, 1);
    const ReturnData direct2 = direct_return(input, 2);
    const ReturnData local1 = resident_return(input, input.source_set());
    const LiouvilleData liouville2 = liouville_return(input, 2);

    const IVector section_image1 = project_state_to_section(local1.image);
    const IMatrix direct_section_dp1 =
        project_tangent_to_section(direct1.dp);
    const IMatrix direct_section_dp2 =
        project_tangent_to_section(direct2.dp);
    const IMatrix local_section_dp1 =
        project_tangent_to_section(local1.dp);
    const IMatrix seed = tangent_seed();
    C1Rect2Set::C0BaseSet event1_c0(section_image1);
    C1Rect2Set::C1BaseSet event1_c1(local_section_dp1);
    C1Rect2Set event1_carrier(event1_c0, event1_c1, local1.time);
    C1Rect2Set::C1BaseSet seed_c1(seed);
    C1Rect2Set continuation1_carrier(event1_c0, seed_c1, local1.time);

    const ReturnData local2 = resident_return(input, continuation1_carrier);
    const IVector section_image2 = project_state_to_section(local2.image);
    const IMatrix local_section_dp2 =
        project_tangent_to_section(local2.dp);
    const IMatrix composed_dp2 = local_section_dp2 * local_section_dp1;
    const IMatrix reversed_dp2 = local_section_dp1 * local_section_dp2;
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
    const interval event2_nu = event2_c0_hull[0] * event2_c0_hull[1] -
                               event2_c0_hull[2] - input.zs;
    const interval direct1_det = determinant_xy(direct_section_dp1);
    const interval local1_det = determinant_xy(local_section_dp1);
    const interval direct2_det = determinant_xy(direct_section_dp2);
    const interval local2_det = determinant_xy(local_section_dp2);
    const interval composed_det2 = determinant_xy(composed_dp2);
    const interval duration2 = local2.time - local1.time;
    const interval postsection1_sign =
        local1.postsection[kSectionCoordinate];
    const interval postsection2_sign =
        local2.postsection[kSectionCoordinate];

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
        duration2.leftBound() > 0.0;
    const bool postsections_after_events =
        local1.postsection_time.leftBound() > local1.time.rightBound() &&
        local2.postsection_time.leftBound() > local2.time.rightBound();
    const bool postsections_plus = postsection1_sign.leftBound() > 0.0 &&
                                   postsection2_sign.leftBound() > 0.0;
    const bool all_finite =
        finite(direct1.image) && finite(direct1.flow) && finite(direct1.dp) &&
        finite(direct2.image) && finite(direct2.flow) && finite(direct2.dp) &&
        finite(local1.image) && finite(local1.flow) && finite(local1.dp) &&
        finite(local2.image) && finite(local2.flow) && finite(local2.dp) &&
        finite(event1_c0_hull) && finite(event1_c1_hull) &&
        finite(event2_c0_hull) && finite(event2_c1_hull) &&
        finite(composed_dp2) && finite(reversed_dp2) &&
        finite(liouville2.image) && positive(direct1_nu) &&
        positive(local1_nu) && positive(event1_nu) &&
        positive(direct2_nu) && positive(local2_nu) &&
        positive(event2_nu) && positive(liouville2.normal_velocity) &&
        positive(liouville2.exp_ell);
    const bool pass =
        all_finite && p1_state_overlap && p1_time_overlap &&
        p1_dp_overlap && p2_state_overlap && p2_time_overlap &&
        p2_dp_overlap && p2_velocity_overlap && p2_determinant_overlap &&
        carrier1_c0_same && carrier2_c0_same && event_sections_exact &&
        seeds_exact && order_discriminated && second_event_after_first &&
        postsections_after_events && postsections_plus;

    std::cout
        << "SCHEMA=sounio.cs6.section-resident-two-return.v1\n"
        << "WORKER_SOURCE_SHA256=" << CS6_WORKER_SOURCE_SHA256 << '\n'
        << "INPUT_SHA256=" << CS6_INPUT_SHA256 << '\n'
        << "RUN_CHALLENGE=" << CS6_RUN_CHALLENGE << '\n'
        << "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0\n"
        << "INTERVAL_BACKEND_DECLARED=FILIB\n"
        << "INTERVAL_SERIALIZATION=ONE_ULP_OUTWARD_BINARY64_HEX\n"
        << "DIRECT_FLOW_TANGENT_ROLE=D_FLOW_TIMES_Q0\n"
        << "LOCAL_FLOW_TANGENT_ROLE=D_FLOW_LOCAL_TIMES_SECTION_IDENTITY\n"
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
              << " CARRIER1_C0_IDENTICAL=" << carrier1_c0_same
              << " CARRIER2_C0_IDENTICAL=" << carrier2_c0_same
              << " EVENT_SECTIONS_EXACT=" << event_sections_exact
              << " CONTINUATION_SEEDS_EXACT=" << seeds_exact
              << " COMPOSITION_ORDER_DISCRIMINATED=" << order_discriminated
              << " SECOND_EVENT_STRICTLY_LATER=" << second_event_after_first
              << " POSTSECTIONS_STRICTLY_LATER=" << postsections_after_events
              << " POSTSECTIONS_PLUS=" << postsections_plus
              << " PROBE_PASS=" << pass << '\n';
    return pass ? EXIT_SUCCESS : 2;
  } catch (const std::exception& error) {
    std::cerr << "CS6_SECTION_RESIDENT_TWO_RETURN_PROBE_ERROR="
              << error.what() << '\n';
    return 2;
  }
}
