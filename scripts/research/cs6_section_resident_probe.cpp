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

ReturnData direct_return(const FrozenInput& input) {
  IMap field(kVectorField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(kDimension, kSectionCoordinate);
  IPoincareMap poincare(solver, section, capd::poincare::MinusPlus);
  ReturnData result;
  C1Rect2Set set = input.source_set();
  result.image = poincare(set, result.flow, result.time, 1);
  result.dp = poincare.computeDP(result.image, result.flow, result.time);
  result.postsection = static_cast<IVector>(set);
  result.postsection_time = set.getCurrentTime();
  return result;
}

ReturnData resident_return(const FrozenInput& input) {
  IMap field(kVectorField);
  field.setParameter("zs", input.zs);
  IOdeSolver solver(field, kOrder);
  ICoordinateSection section(kDimension, kSectionCoordinate);
  SectionResidentMap poincare(solver, section, capd::poincare::MinusPlus);
  return poincare.one_return(input.source_set());
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

LiouvilleData liouville_return(const FrozenInput& input) {
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
  result.image = poincare(set, result.time, 1);
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
  const interval source_area =
      frame_determinant * input.radius_u * input.radius_s;
  result.determinant = result.exp_ell * result.initial_velocity /
                       result.normal_velocity * source_area;
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
    const ReturnData direct = direct_return(input);
    const ReturnData candidate = resident_return(input);
    const LiouvilleData liouville = liouville_return(input);

    const IVector section_image = project_state_to_section(candidate.image);
    const IMatrix section_dp = project_tangent_to_section(candidate.dp);
    const IMatrix seed = tangent_seed();
    C1Rect2Set::C0BaseSet event_c0(section_image);
    C1Rect2Set::C1BaseSet event_c1(section_dp);
    C1Rect2Set event_carrier(event_c0, event_c1, candidate.time);
    C1Rect2Set::C1BaseSet seed_c1(seed);
    C1Rect2Set continuation_carrier(event_c0, seed_c1, candidate.time);

    const IVector event_c0_hull = static_cast<IVector>(event_carrier);
    const IMatrix event_c1_hull = static_cast<IMatrix>(event_carrier);
    const IVector continuation_c0_hull =
        static_cast<IVector>(continuation_carrier);
    const IMatrix continuation_c1_hull =
        static_cast<IMatrix>(continuation_carrier);
    const interval direct_nu = direct.image[0] * direct.image[1] -
                               direct.image[2] - input.zs;
    const interval candidate_nu = candidate.image[0] * candidate.image[1] -
                                  candidate.image[2] - input.zs;
    const interval carrier_nu = event_c0_hull[0] * event_c0_hull[1] -
                                event_c0_hull[2] - input.zs;
    const interval direct_det = determinant_xy(direct.dp);
    const interval candidate_det = determinant_xy(section_dp);

    const bool state_overlap = joint_overlap(
        {direct.image, candidate.image, event_c0_hull,
         continuation_c0_hull,
         IVector{liouville.image[0], liouville.image[1],
                 liouville.image[2]}});
    const bool time_overlap = joint_overlap(
        {direct.time, candidate.time, event_carrier.getCurrentTime(),
         continuation_carrier.getCurrentTime(), liouville.time});
    const bool dp_overlap = joint_overlap(
        {direct.dp, candidate.dp, event_c1_hull, section_dp});
    const bool velocity_overlap = joint_overlap(
        {direct_nu, candidate_nu, carrier_nu, liouville.normal_velocity});
    const bool determinant_overlap = joint_overlap(
        {direct_det, candidate_det, liouville.determinant});
    const bool carrier_c0_same = joint_overlap(
        {event_c0_hull, continuation_c0_hull}) &&
        capd::vectalg::subset(event_c0_hull, continuation_c0_hull) &&
        capd::vectalg::subset(continuation_c0_hull, event_c0_hull);
    const bool event_section_exact =
        event_c0_hull[kSectionCoordinate] == interval(0.0) &&
        continuation_c0_hull[kSectionCoordinate] == interval(0.0);
    const bool seed_exact = capd::vectalg::subset(seed, continuation_c1_hull) &&
                            capd::vectalg::subset(continuation_c1_hull, seed);
    const bool postsection_after_event =
        candidate.postsection_time.leftBound() > candidate.time.rightBound();
    const interval postsection_sign =
        candidate.postsection[kSectionCoordinate];
    const bool postsection_plus = postsection_sign.leftBound() > 0.0;
    const bool all_finite =
        finite(direct.image) && finite(direct.flow) && finite(direct.dp) &&
        finite(candidate.image) && finite(candidate.flow) &&
        finite(candidate.dp) && finite(event_c0_hull) &&
        finite(event_c1_hull) && finite(liouville.image) &&
        positive(direct_nu) && positive(candidate_nu) &&
        positive(carrier_nu) && positive(liouville.normal_velocity) &&
        positive(liouville.exp_ell);
    const bool pass = all_finite && state_overlap && time_overlap &&
                      dp_overlap && velocity_overlap && determinant_overlap &&
                      carrier_c0_same && event_section_exact && seed_exact &&
                      postsection_after_event && postsection_plus;

    std::cout
        << "SCHEMA=sounio.cs6.section-resident-carrier.v1\n"
        << "WORKER_SOURCE_SHA256=" << CS6_WORKER_SOURCE_SHA256 << '\n'
        << "INPUT_SHA256=" << CS6_INPUT_SHA256 << '\n'
        << "RUN_CHALLENGE=" << CS6_RUN_CHALLENGE << '\n'
        << "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0\n"
        << "INTERVAL_BACKEND_DECLARED=FILIB\n"
        << "INTERVAL_SERIALIZATION=ONE_ULP_OUTWARD_BINARY64_HEX\n"
        << "FLOW_TANGENT_ROLE=D_FLOW_TIMES_Q0\n"
        << "SOURCE_TANGENT_SEED_ROLE=GLOBAL_FRAME_RADII_WITH_ZERO_DUMMY_NORMAL\n"
        << "TANGENT_ZERO_TIGHTENING=COORDINATE_OUTPUT_ROW_AND_Q0_DUMMY_INPUT_COLUMN\n"
        << "SOURCE=N0\n"
        << "U_INDEX=" << kUIndex << '\n'
        << "S_INDEX=" << kSIndex << '\n'
        << "U_TILES=" << kUTiles << '\n'
        << "S_TILES=" << kSTiles << '\n'
        << "ORDER=" << kOrder << '\n'
        << "RETURN_COUNT=1\n"
        << "SECTION=COORDINATE_W_EQUALS_ZERO\n"
        << "CROSSING_DIRECTION=MINUS_PLUS\n"
        << "FAST_PATH_REQUIRED=true\n"
        << "EVENT_CARRIER_ROLE=TERMINAL_INCOMING_DP_EVIDENCE\n"
        << "CONTINUATION_CARRIER_ROLE=LOCAL_TANGENT_SEED_ONLY\n"
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

    std::cout << "DIRECT";
    write_interval(std::cout, "TIME", direct.time);
    write_vector(std::cout, "X", direct.image);
    write_matrix(std::cout, "FLOW_TANGENT", direct.flow);
    write_matrix(std::cout, "DP", direct.dp);
    write_interval(std::cout, "NU", direct_nu);
    write_interval(std::cout, "DET", direct_det);
    std::cout << '\n';

    std::cout << "CANDIDATE";
    write_interval(std::cout, "TIME", candidate.time);
    write_vector(std::cout, "X", candidate.image);
    write_matrix(std::cout, "FLOW_TANGENT", candidate.flow);
    write_matrix(std::cout, "DP", candidate.dp);
    write_vector(std::cout, "SECTION_X", section_image);
    write_matrix(std::cout, "SECTION_DP", section_dp);
    write_interval(std::cout, "NU", candidate_nu);
    write_interval(std::cout, "DET", candidate_det);
    std::cout << '\n';

    std::cout << "EVENT_CARRIER";
    write_interval(std::cout, "TIME", event_carrier.getCurrentTime());
    emit_c0_components(event_carrier, "C0_");
    emit_c1_components(event_carrier, "C1_");
    write_vector(std::cout, "C0_HULL", event_c0_hull);
    write_matrix(std::cout, "C1_HULL", event_c1_hull);
    write_interval(std::cout, "NU", carrier_nu);
    std::cout << '\n';

    std::cout << "CONTINUATION_CARRIER";
    write_interval(std::cout, "TIME", continuation_carrier.getCurrentTime());
    emit_c0_components(continuation_carrier, "C0_");
    emit_c1_components(continuation_carrier, "C1_");
    write_vector(std::cout, "C0_HULL", continuation_c0_hull);
    write_matrix(std::cout, "C1_HULL", continuation_c1_hull);
    write_matrix(std::cout, "INCOMING_DP", section_dp);
    std::cout << '\n';

    std::cout << "POSTSECTION";
    write_interval(std::cout, "TIME", candidate.postsection_time);
    write_vector(std::cout, "X", candidate.postsection);
    write_interval(std::cout, "SECTION_SIGN", postsection_sign);
    std::cout << '\n';

    std::cout << "LIOUVILLE";
    write_interval(std::cout, "TIME", liouville.time);
    write_vector(std::cout, "X", liouville.image);
    write_interval(std::cout, "NU0", liouville.initial_velocity);
    write_interval(std::cout, "NU1", liouville.normal_velocity);
    write_interval(std::cout, "ELL", liouville.ell);
    write_interval(std::cout, "EXP_ELL", liouville.exp_ell);
    write_interval(std::cout, "DET", liouville.determinant);
    std::cout << '\n';

    std::cout << "SUMMARY ALL_FINITE=" << all_finite
              << " STATE_JOINT_OVERLAP=" << state_overlap
              << " TIME_JOINT_OVERLAP=" << time_overlap
              << " DP_JOINT_OVERLAP=" << dp_overlap
              << " VELOCITY_JOINT_OVERLAP=" << velocity_overlap
              << " DETERMINANT_JOINT_OVERLAP=" << determinant_overlap
              << " CARRIER_C0_IDENTICAL=" << carrier_c0_same
              << " EVENT_SECTION_EXACT=" << event_section_exact
              << " CONTINUATION_SEED_EXACT=" << seed_exact
              << " POSTSECTION_STRICTLY_LATER=" << postsection_after_event
              << " POSTSECTION_PLUS=" << postsection_plus
              << " PROBE_PASS=" << pass << '\n';
    return pass ? EXIT_SUCCESS : 2;
  } catch (const std::exception& error) {
    std::cerr << "CS6_SECTION_RESIDENT_PROBE_ERROR=" << error.what() << '\n';
    return 2;
  }
}
