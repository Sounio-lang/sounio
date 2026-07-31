#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "capd/capdlib.h"

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

constexpr int kReturns = 6;
constexpr int kCrossings = kReturns + 1;
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
  interval q_positive;
  interval q_negative;
};

struct Matrix2 {
  interval a00;
  interval a01;
  interval a10;
  interval a11;
};

struct ConeDiagnostic {
  interval m00;
  interval m01;
  interval m11;
  interval determinant_naive;
  interval determinant_expanded;
  bool positive_definite;
};

struct LiouvilleDiagnostic {
  interval integral_divergence;
  interval exponential_divergence;
  interval determinant;
  std::array<interval, kCrossings> normal_velocity;
  std::array<interval, kReturns> return_time;
  bool valid;
};

struct TileImage {
  interval c1_return_time;
  Matrix2 n0_target;
  Matrix2 n1_target;
  LiouvilleDiagnostic liouville;
};

struct TileSelector {
  int index;
  int count;

  bool owns(std::size_t linear_index) const {
    return linear_index % static_cast<std::size_t>(count) ==
           static_cast<std::size_t>(index);
  }

  std::size_t expected(std::size_t total) const {
    if (static_cast<std::size_t>(index) >= total) {
      return 0;
    }
    return (total - 1 - static_cast<std::size_t>(index)) /
               static_cast<std::size_t>(count) +
           1;
  }
};

bool finite(const interval& value) {
  return std::isfinite(value.leftBound()) &&
         std::isfinite(value.rightBound()) &&
         value.leftBound() <= value.rightBound();
}

bool finite(const Matrix2& value) {
  return finite(value.a00) && finite(value.a01) && finite(value.a10) &&
         finite(value.a11);
}

bool overlaps(const interval& left, const interval& right) {
  return left.leftBound() <= right.rightBound() &&
         right.leftBound() <= left.rightBound();
}

interval square(const interval& value) { return sqr(value); }

ConeDiagnostic cone_diagnostic(const Matrix2& map, const HSet& source,
                               const HSet& target) {
  const interval m00 = target.q_positive * square(map.a00) -
                       target.q_negative * square(map.a10) -
                       source.q_positive;
  const interval m01 = target.q_positive * map.a00 * map.a01 -
                       target.q_negative * map.a10 * map.a11;
  const interval m11 = target.q_positive * square(map.a01) -
                       target.q_negative * square(map.a11) +
                       source.q_negative;
  const interval determinant_naive = m00 * m11 - square(m01);
  const interval map_determinant =
      map.a00 * map.a11 - map.a01 * map.a10;
  const interval determinant_expanded =
      target.q_positive * source.q_negative * square(map.a00) -
      target.q_negative * source.q_negative * square(map.a10) -
      source.q_positive * target.q_positive * square(map.a01) +
      source.q_positive * target.q_negative * square(map.a11) -
      source.q_positive * source.q_negative -
      target.q_positive * target.q_negative * square(map_determinant);
  return {m00, m01, m11, determinant_naive, determinant_expanded,
          finite(m00) && finite(m01) && finite(m11) &&
              finite(determinant_naive) && finite(determinant_expanded) &&
              m00.leftBound() > 0.0 &&
              determinant_expanded.leftBound() > 0.0};
}

void write_interval(std::ostream& output, const char* name,
                    const interval& value) {
  const double negative_infinity =
      -std::numeric_limits<double>::infinity();
  const double positive_infinity =
      std::numeric_limits<double>::infinity();
  const double lower = std::nextafter(value.leftBound(), negative_infinity);
  const double upper = std::nextafter(value.rightBound(), positive_infinity);
  output << ' ' << name << "=[" << std::hexfloat << lower << ',' << upper
         << std::defaultfloat << ']';
}

void write_matrix(std::ostream& output, const Matrix2& value) {
  write_interval(output, "A00", value.a00);
  write_interval(output, "A01", value.a01);
  write_interval(output, "A10", value.a10);
  write_interval(output, "A11", value.a11);
}

class Ledger {
 public:
  explicit Ledger(const char* path) : enabled_(path != nullptr) {
    if (!enabled_) {
      return;
    }
    output_.open(path, std::ios::out | std::ios::trunc);
    if (!output_) {
      throw std::runtime_error(std::string("cannot open ledger: ") + path);
    }
    output_ << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::boolalpha;
  }

  void record(const HSet& source, const HSet& target, int u_index,
              int s_index, const interval& source_u,
              const interval& source_s, const TileImage& image,
              const Matrix2& normalized, const ConeDiagnostic& tile_cone) {
    if (!enabled_) {
      return;
    }
    output_ << "SOURCE=" << source.name << " TARGET=" << target.name
            << " EDGE=" << source.name << "->" << target.name
            << " U_INDEX=" << u_index << " S_INDEX=" << s_index;
    write_interval(output_, "SOURCE_U", source_u);
    write_interval(output_, "SOURCE_S", source_s);
    write_matrix(output_, normalized);
    write_interval(output_, "TILE_M00", tile_cone.m00);
    write_interval(output_, "TILE_DET_M_NAIVE",
                   tile_cone.determinant_naive);
    write_interval(output_, "TILE_DET_M_EXPANDED",
                   tile_cone.determinant_expanded);
    write_interval(output_, "C1_RETURN_TIME", image.c1_return_time);
    write_interval(output_, "INTEGRAL_DIVERGENCE",
                   image.liouville.integral_divergence);
    write_interval(output_, "EXP_INTEGRAL_DIVERGENCE",
                   image.liouville.exponential_divergence);
    write_interval(output_, "DET_LIOUVILLE", image.liouville.determinant);
    for (int crossing = 0; crossing < kCrossings; ++crossing) {
      const std::string name = "NU" + std::to_string(crossing);
      write_interval(output_, name.c_str(),
                     image.liouville.normal_velocity[crossing]);
    }
    for (int crossing = 0; crossing < kReturns; ++crossing) {
      const std::string name = "T" + std::to_string(crossing + 1);
      write_interval(output_, name.c_str(), image.liouville.return_time[crossing]);
    }
    output_ << " TILE_CONE_DIAGNOSTIC=" << tile_cone.positive_definite
            << " LIOUVILLE_INVERTIBLE=" << image.liouville.valid << '\n';
    ++records_;
  }

  bool enabled() const { return enabled_; }
  std::size_t records() const { return records_; }

 private:
  bool enabled_;
  std::ofstream output_;
  std::size_t records_ = 0;
};

class ProofContext {
 public:
  explicit ProofContext(int order)
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
        liouville_field_(kLiouvilleField),
        liouville_solver_(liouville_field_, order),
        liouville_section_(4, 2),
        liouville_poincare_(liouville_solver_, liouville_section_,
                            capd::poincare::MinusPlus),
        frame3_(3, 3),
        frame4_(4, 4) {
    vector_field_.setParameter("zs", zs_);
    liouville_field_.setParameter("zs", zs_);
    if (frame_determinant_.contains(0.0)) {
      throw std::runtime_error("frozen h-set frame is not invertible");
    }
    frame3_[0][0] = unstable_x_;
    frame3_[1][0] = unstable_y_;
    frame3_[2][0] = 0.0;
    frame3_[0][1] = stable_x_;
    frame3_[1][1] = stable_y_;
    frame3_[2][1] = 0.0;
    frame3_[0][2] = 0.0;
    frame3_[1][2] = 0.0;
    frame3_[2][2] = 1.0;
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        frame4_[row][column] = row == column ? interval(1.0) : interval(0.0);
      }
    }
    frame4_[0][0] = unstable_x_;
    frame4_[1][0] = unstable_y_;
    frame4_[0][1] = stable_x_;
    frame4_[1][1] = stable_y_;
  }

  TileImage image(const HSet& source, const interval& source_u,
                  const interval& source_s) {
    const interval midpoint_u = source_u.mid();
    const interval midpoint_s = source_s.mid();
    const interval radius_u = source_u - midpoint_u;
    const interval radius_s = source_s - midpoint_s;
    const IVector center3{
        origin_x_ + unstable_x_ * midpoint_u + stable_x_ * midpoint_s,
        origin_y_ + unstable_y_ * midpoint_u + stable_y_ * midpoint_s,
        interval(0.0)};
    const IVector tile_radii3{radius_u, radius_s, interval(0.0)};

    C1Rect2Set::C0BaseSet c0(center3, frame3_, tile_radii3);
    IMatrix initial_tangent(3, 3);
    initial_tangent.clear();
    for (int row = 0; row < 3; ++row) {
      initial_tangent[row][0] = frame3_[row][0] * source.radius_u;
      initial_tangent[row][1] = frame3_[row][1] * source.radius_s;
    }
    C1Rect2Set::C1BaseSet c1(initial_tangent);
    const IMatrix represented_tangent = static_cast<IMatrix>(c1);
    if (!subset(initial_tangent, represented_tangent)) {
      throw std::runtime_error("C1 initial tangent representation lost J0");
    }
    C1Rect2Set derivative_set(c0, c1);
    IMatrix flow_derivative(3, 3);
    interval c1_return_time;
    const IVector c1_image =
        poincare_(derivative_set, flow_derivative, c1_return_time, kReturns);
    const IMatrix chart_derivative =
        poincare_.computeDP(c1_image, flow_derivative, c1_return_time);

    const interval local00 =
        (stable_y_ * chart_derivative[0][0] -
         stable_x_ * chart_derivative[1][0]) /
        frame_determinant_;
    const interval local01 =
        (stable_y_ * chart_derivative[0][1] -
         stable_x_ * chart_derivative[1][1]) /
        frame_determinant_;
    const interval local10 =
        (-unstable_y_ * chart_derivative[0][0] +
         unstable_x_ * chart_derivative[1][0]) /
        frame_determinant_;
    const interval local11 =
        (-unstable_y_ * chart_derivative[0][1] +
         unstable_x_ * chart_derivative[1][1]) /
        frame_determinant_;

    const Matrix2 n0_target{local00 / decimal("0.004"),
                            local01 / decimal("0.004"),
                            local10 / decimal("0.3"),
                            local11 / decimal("0.3")};
    const Matrix2 n1_target{local00 / decimal("0.0015"),
                            local01 / decimal("0.0015"),
                            local10 / decimal("0.3"),
                            local11 / decimal("0.3")};
    LiouvilleDiagnostic liouville =
        liouville_image(source_u, source_s, midpoint_u, midpoint_s, radius_u,
                        radius_s);
    liouville.valid = liouville.valid &&
                      overlaps(c1_return_time,
                               liouville.return_time[kReturns - 1]);
    return {c1_return_time, n0_target, n1_target, liouville};
  }

  const interval& frame_determinant() const { return frame_determinant_; }
  const interval& zs() const { return zs_; }
  const interval& origin_x() const { return origin_x_; }
  const interval& origin_y() const { return origin_y_; }
  const interval& unstable_x() const { return unstable_x_; }
  const interval& unstable_y() const { return unstable_y_; }
  const interval& stable_x() const { return stable_x_; }
  const interval& stable_y() const { return stable_y_; }

 private:
  LiouvilleDiagnostic liouville_image(const interval& source_u,
                                      const interval& source_s,
                                      const interval& midpoint_u,
                                      const interval& midpoint_s,
                                      const interval& radius_u,
                                      const interval& radius_s) {
    const IVector center4{
        origin_x_ + unstable_x_ * midpoint_u + stable_x_ * midpoint_s,
        origin_y_ + unstable_y_ * midpoint_u + stable_y_ * midpoint_s,
        interval(0.0), interval(0.0)};
    const IVector tile_radii4{radius_u, radius_s, interval(0.0),
                              interval(0.0)};
    C0HOTripletonSet set(center4, frame4_, tile_radii4);
    LiouvilleDiagnostic result;
    const interval initial_x =
        origin_x_ + unstable_x_ * source_u + stable_x_ * source_s;
    const interval initial_y =
        origin_y_ + unstable_y_ * source_u + stable_y_ * source_s;
    result.normal_velocity[0] = initial_x * initial_y - zs_;
    IVector image(4);
    for (int crossing = 0; crossing < kReturns; ++crossing) {
      image = liouville_poincare_(set, result.return_time[crossing], 1);
      result.normal_velocity[crossing + 1] =
          image[0] * image[1] - image[2] - zs_;
    }
    result.integral_divergence = image[3];
    result.exponential_divergence = exp(result.integral_divergence);
    result.determinant = result.exponential_divergence *
                         result.normal_velocity[0] /
                         result.normal_velocity[kReturns];
    result.valid = finite(result.integral_divergence) &&
                   finite(result.determinant) &&
                   result.determinant.leftBound() > 0.0;
    for (const interval& normal : result.normal_velocity) {
      result.valid = result.valid && finite(normal) && normal.leftBound() > 0.0;
    }
    interval previous_time(0.0);
    for (const interval& return_time : result.return_time) {
      result.valid = result.valid && finite(return_time) &&
                     return_time.leftBound() > previous_time.rightBound();
      previous_time = return_time;
    }
    return result;
  }

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
  IMap liouville_field_;
  IOdeSolver liouville_solver_;
  ICoordinateSection liouville_section_;
  IPoincareMap liouville_poincare_;
  IMatrix frame3_;
  IMatrix frame4_;
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

bool valid_tile(const TileImage& image) {
  return finite(image.c1_return_time) && image.c1_return_time.leftBound() > 0.0 &&
         finite(image.n0_target) && finite(image.n1_target) &&
         image.liouville.valid;
}

void run_source(ProofContext& context, const HSet& source,
                const HSet& n0, const HSet& n1, int u_tiles, int s_tiles,
                const TileSelector& selector, Ledger& ledger,
                std::size_t& processed, std::size_t& valid,
                std::size_t& tile_cone_passes) {
  for (int u_index = 0; u_index < u_tiles; ++u_index) {
    const interval source_u =
        tile(source.center_u, source.radius_u, u_index, u_tiles);
    for (int s_index = 0; s_index < s_tiles; ++s_index) {
      const std::size_t linear =
          static_cast<std::size_t>(u_index) * s_tiles + s_index;
      if (!selector.owns(linear)) {
        continue;
      }
      ++processed;
      const interval source_s =
          tile(source.center_s, source.radius_s, s_index, s_tiles);
      const TileImage image = context.image(source, source_u, source_s);
      if (valid_tile(image)) {
        ++valid;
      }
      const auto emit = [&](const HSet& target, const Matrix2& normalized) {
        const ConeDiagnostic diagnostic =
            cone_diagnostic(normalized, source, target);
        if (diagnostic.positive_definite) {
          ++tile_cone_passes;
        }
        ledger.record(source, target, u_index, s_index, source_u, source_s,
                      image, normalized, diagnostic);
      };
      emit(n0, image.n0_target);
      if (std::string(source.name) == "N0") {
        emit(n1, image.n1_target);
      }
    }
  }
}

void print_selftest(const HSet& n0) {
  const Matrix2 hyperbolic{interval(2.0), interval(0.0), interval(0.0),
                           interval(0.5)};
  const Matrix2 identity{interval(1.0), interval(0.0), interval(0.0),
                         interval(1.0)};
  const Matrix2 singular{interval(2.0), interval(0.0), interval(0.0),
                         interval(0.0)};
  HSet unit{"U", interval(0.0), interval(0.0), interval(1.0), interval(1.0),
            interval(1.0), interval(1.0)};
  const bool hyperbolic_pass =
      cone_diagnostic(hyperbolic, unit, unit).positive_definite;
  const bool identity_rejected =
      !cone_diagnostic(identity, unit, unit).positive_definite;
  const bool singular_cone_pass =
      cone_diagnostic(singular, unit, unit).positive_definite;
  const bool q_constants_finite = finite(n0.q_positive) && finite(n0.q_negative);
  std::ostringstream encoded;
  write_interval(encoded, "X", interval(1.0));
  const bool exact_hex_endpoint_encoding = encoded.str() ==
      " X=[0x1.fffffffffffffp-1,0x1.0000000000001p+0]";
  const bool pass = hyperbolic_pass && identity_rejected && singular_cone_pass &&
                    q_constants_finite && exact_hex_endpoint_encoding;
  std::cout << "HYPERBOLIC_CONTROL_PASS=" << hyperbolic_pass << '\n'
            << "IDENTITY_CONTROL_REJECTED=" << identity_rejected << '\n'
            << "SINGULAR_CONE_CONTROL_PASS=" << singular_cone_pass << '\n'
            << "SINGULAR_CONTROL_INVERTIBILITY_PROVED=false\n"
            << "EXACT_HEX_ENDPOINT_ENCODING=" << exact_hex_endpoint_encoding
            << "\nOUTWARD_ONE_ULP_ENDPOINT_ENCODING="
            << exact_hex_endpoint_encoding
            << '\n'
            << "SELFTEST_PASS=" << pass << '\n';
  if (!pass) {
    throw std::runtime_error("cone algebra selftest failed");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::boolalpha;
  try {
    const HSet n0{"N0", interval(0.0), interval(0.0), decimal("0.004"),
                  decimal("0.3"), decimal("1"),
                  decimal("2.3023784599059653")};
    const HSet n1{"N1", decimal("0.019771776972779206"), interval(0.0),
                  decimal("0.0015"), decimal("0.3"),
                  decimal("0.06526711140171336"),
                  decimal("2.3023784599059653")};
    const std::string mode = argc > 1 ? argv[1] : "selftest";
    if (mode == "selftest") {
      print_selftest(n0);
      return EXIT_SUCCESS;
    }
    if (mode != "probe" && mode != "probe-ledger" && mode != "proof") {
      throw std::runtime_error(
          "usage: selftest | probe SOURCE TARGET U_INDEX S_INDEX U_TILES "
          "S_TILES ORDER | probe-ledger SOURCE TARGET U_INDEX S_INDEX "
          "U_TILES S_TILES ORDER LEDGER | proof N0_U N1_U S_TILES ORDER SHARD_ORDINAL "
          "SHARD_COUNT LEDGER");
    }

    const bool probe = mode == "probe" || mode == "probe-ledger";
    const bool probe_ledger = mode == "probe-ledger";
    if ((!probe_ledger && argc != 9) || (probe_ledger && argc != 10)) {
      throw std::runtime_error(
          "usage: probe N0|N1 N0|N1 U_INDEX S_INDEX U_TILES S_TILES ORDER | "
          "probe-ledger N0|N1 N0|N1 U_INDEX S_INDEX U_TILES S_TILES ORDER LEDGER | "
          "proof N0_U N1_U S_TILES ORDER SHARD_ORDINAL SHARD_COUNT LEDGER");
    }
    const int n0_u_tiles = probe ? 1 : positive_int(argv[2], "n0_u_tiles");
    const int n1_u_tiles = probe ? 1 : positive_int(argv[3], "n1_u_tiles");
    const int s_tiles = probe ? 1 : positive_int(argv[4], "s_tiles");
    const int order = probe ? positive_int(argv[8], "order")
                            : positive_int(argv[5], "order");
    const int shard_ordinal = probe ? 1 : positive_int(argv[6], "shard_ordinal");
    const int shard_count = probe ? 1 : positive_int(argv[7], "shard_count");
    if (!probe && shard_ordinal > shard_count) {
      throw std::runtime_error("shard_ordinal must be <= shard_count");
    }
    Ledger ledger(probe ? (probe_ledger ? argv[9] : nullptr) : argv[8]);
    ProofContext context(order);
    const bool hsets_disjoint =
        (n0.center_u + n0.radius_u).rightBound() <
        (n1.center_u - n1.radius_u).leftBound();

    std::cout << "SCHEMA=sounio.cs6.capd-c1-cone.v1\n"
              << "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0\n"
              << "INTERVAL_BACKEND_DECLARED=FILIB\n"
              << "C1_SET=C1Rect2Set\n"
              << "C1_INITIAL_DERIVATIVE=B*R_SOURCE_TANGENT_ZERO_NORMAL\n"
              << "MAP=P^6\nRETURNS_PER_MAP=6\n"
              << "SECTION_ORIENTATION=MinusPlus\nORDER=" << order << '\n'
              << "Q_DECIMAL_INTERPRETATION=exact-decimal-input-outward-interval\n"
              << "LEDGER_ENDPOINT_ENCODING=outward-one-ulp-exact-hexadecimal-binary64\n"
              << "Q_N0=1,-2.3023784599059653\n"
              << "Q_N1=0.06526711140171336,-2.3023784599059653\n"
              << "CONE_DETERMINANT_FORM=expanded-exact-cancellation-before-interval-evaluation\n"
              << "C1_LIOUVILLE_FINAL_RETURN_OVERLAP_REQUIRED=true\n"
              << "LIOUVILLE_EXPONENTIAL_OPERAND_EMITTED=true\n"
              << "C1_LIOUVILLE_NORMALIZED_DETERMINANT_OVERLAP_REQUIRED=true\n"
              << "VECTOR_FIELD_CAPD=" << kVectorField << '\n'
              << "LIOUVILLE_FIELD_CAPD=" << kLiouvilleField << '\n'
              << "ZSEC=" << context.zs() << "\nORIGIN={" << context.origin_x()
              << ',' << context.origin_y() << "}\nUNSTABLE={"
              << context.unstable_x() << ',' << context.unstable_y()
              << "}\nSTABLE={" << context.stable_x() << ','
              << context.stable_y() << "}\nN0_LOCAL={" << n0.center_u << ','
              << n0.center_s << ',' << n0.radius_u << ',' << n0.radius_s
              << "}\nN1_LOCAL={" << n1.center_u << ',' << n1.center_s << ','
              << n1.radius_u << ',' << n1.radius_s << "}\nHSETS_DISJOINT="
              << hsets_disjoint << '\n'
              << "FRAME_DETERMINANT=" << context.frame_determinant() << '\n'
              << "FRAME_RIGOROUSLY_INVERTIBLE="
              << !context.frame_determinant().contains(0.0) << '\n';

    if (probe) {
      const std::string source_name = argv[2];
      const std::string target_name = argv[3];
      if ((source_name != "N0" && source_name != "N1") ||
          (target_name != "N0" && target_name != "N1")) {
        throw std::runtime_error("probe source and target must be N0 or N1");
      }
      if (target_name == "N1" && source_name != "N0") {
        throw std::runtime_error("probe edge is not in the frozen adjacency");
      }
      const HSet& source = source_name == "N1" ? n1 : n0;
      const HSet& target = target_name == "N1" ? n1 : n0;
      const int u_index = nonnegative_int(argv[4], "u_index");
      const int s_index = nonnegative_int(argv[5], "s_index");
      const int u_tiles = positive_int(argv[6], "u_tiles");
      const int local_s_tiles = positive_int(argv[7], "s_tiles");
      if (u_index < 0 || u_index >= u_tiles || s_index < 0 ||
          s_index >= local_s_tiles) {
        throw std::runtime_error("probe tile index out of range");
      }
      const interval source_u =
          tile(source.center_u, source.radius_u, u_index, u_tiles);
      const interval source_s =
          tile(source.center_s, source.radius_s, s_index, local_s_tiles);
      const TileImage image = context.image(source, source_u, source_s);
      const Matrix2& normalized =
          target_name == "N1" ? image.n1_target : image.n0_target;
      const ConeDiagnostic diagnostic =
          cone_diagnostic(normalized, source, target);
      const bool probe_pass = valid_tile(image) && diagnostic.positive_definite;
      if (probe_ledger) {
        ledger.record(source, target, u_index, s_index, source_u, source_s,
                      image, normalized, diagnostic);
      }
      std::cout << "PROBE_SOURCE=" << source.name
                << " PROBE_TARGET=" << target.name << " PROBE_EDGE="
                << source.name << "->" << target.name << " U_INDEX=" << u_index
                << " S_INDEX=" << s_index << " U_TILES=" << u_tiles
                << " S_TILES=" << local_s_tiles << '\n';
      std::cout << "A=" << normalized.a00 << ',' << normalized.a01 << ','
                << normalized.a10 << ',' << normalized.a11 << '\n'
                << "TILE_M00=" << diagnostic.m00 << '\n'
                << "TILE_DET_M_NAIVE="
                << diagnostic.determinant_naive << '\n'
                << "TILE_DET_M_EXPANDED="
                << diagnostic.determinant_expanded << '\n'
                << "TILE_CONE_DIAGNOSTIC="
                << diagnostic.positive_definite << '\n'
                << "DET_LIOUVILLE=" << image.liouville.determinant << '\n'
                << "LIOUVILLE_INVERTIBLE=" << image.liouville.valid << '\n'
                << "PROBE_PASS=" << probe_pass << '\n'
                << "LEDGER_ENABLED=" << ledger.enabled() << '\n'
                << "EDGE_RECORDS_WRITTEN=" << ledger.records() << '\n'
                << "PAIRWISE_CHORD_CONE_CONDITION_PROVED=false\n"
                << "UNIFORM_HYPERBOLICITY_PROVED=false\n"
                << "CHAOTIC_ATTRACTOR_PROVED=false\n";
      return probe_pass ? EXIT_SUCCESS : 2;
    }

    const TileSelector selector{shard_ordinal - 1, shard_count};
    std::size_t processed = 0;
    std::size_t valid = 0;
    std::size_t tile_cone_passes = 0;
    run_source(context, n0, n0, n1, n0_u_tiles, s_tiles, selector, ledger,
               processed, valid, tile_cone_passes);
    run_source(context, n1, n0, n1, n1_u_tiles, s_tiles, selector, ledger,
               processed, valid, tile_cone_passes);
    const std::size_t expected_raw =
        selector.expected(static_cast<std::size_t>(n0_u_tiles) * s_tiles) +
        selector.expected(static_cast<std::size_t>(n1_u_tiles) * s_tiles);
    const std::size_t expected_records =
        2 * selector.expected(static_cast<std::size_t>(n0_u_tiles) * s_tiles) +
        selector.expected(static_cast<std::size_t>(n1_u_tiles) * s_tiles);
    const bool shard_pass = processed == expected_raw && valid == processed &&
                            ledger.records() == expected_records;
    std::cout << "GRID=N0_U:" << n0_u_tiles << ",N1_U:" << n1_u_tiles
              << ",S:" << s_tiles << '\n'
              << "SHARD=" << shard_ordinal << '/' << shard_count << '\n'
              << "LEDGER_ENABLED=" << ledger.enabled() << '\n'
              << "RAW_TILES_EXPECTED=" << expected_raw << '\n'
              << "RAW_TILES_PROCESSED=" << processed << '\n'
              << "RAW_TILES_VALID=" << valid << '\n'
              << "EDGE_RECORDS_EXPECTED=" << expected_records << '\n'
              << "EDGE_RECORDS_WRITTEN=" << ledger.records() << '\n'
              << "TILE_CONE_DIAGNOSTIC_PASSES=" << tile_cone_passes << '\n'
              << "SHARD_PASS=" << shard_pass << '\n'
              << "FULL_SOURCE_GLOBAL_HULL_TESTED=false\n"
              << "PAIRWISE_CHORD_CONE_CONDITION_PROVED=false\n"
              << "TANGENT_CONE_CONDITION_PROVED=false\n"
              << "LIOUVILLE_INVERTIBILITY_PROVED=false\n"
              << "UNIFORM_HYPERBOLICITY_PROVED=false\n"
              << "CHAOTIC_ATTRACTOR_PROVED=false\n";
    return shard_pass ? EXIT_SUCCESS : 2;
  } catch (const std::exception& error) {
    std::cerr << "CS6_CAPD_C1_CONE_ERROR=" << error.what() << '\n';
    return 3;
  }
}
