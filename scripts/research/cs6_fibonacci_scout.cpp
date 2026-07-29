#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "capd/capdlib.h"

namespace {

constexpr double kSectionZ = 22.3274637391;
constexpr double kFixedX = 15.186446520640786;
constexpr double kFixedY = 10.908543194765466;

struct Point {
  double x;
  double y;
};

struct Matrix2 {
  double a;
  double b;
  double c;
  double d;
};

struct MapResult {
  Point point;
  Matrix2 derivative;
  double return_time;
};

class Cs6Poincare {
 public:
  explicit Cs6Poincare(int order)
      : vector_field_(
            "par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,"
            "x*y-w-zs;"),
        solver_(vector_field_, order),
        section_(3, 2),
        poincare_(solver_, section_, capd::poincare::MinusPlus) {
    vector_field_.setParameter("zs", kSectionZ);
  }

  Point iterate(Point point, int returns, double* total_time = nullptr) {
    capd::DVector state{point.x, point.y, 0.0};
    double elapsed = 0.0;
    for (int i = 0; i < returns; ++i) {
      double crossing_time = 0.0;
      state = poincare_(state, crossing_time);
      elapsed += crossing_time;
      require_finite(state);
    }
    if (total_time != nullptr) {
      *total_time = elapsed;
    }
    return {state[0], state[1]};
  }

  MapResult iterate_with_derivative(Point point, int returns) {
    capd::DVector state{point.x, point.y, 0.0};
    Matrix2 product{1.0, 0.0, 0.0, 1.0};
    double elapsed = 0.0;
    for (int i = 0; i < returns; ++i) {
      capd::DMatrix flow_derivative(3, 3);
      double crossing_time = 0.0;
      state = poincare_(state, flow_derivative, crossing_time);
      const capd::DMatrix dp =
          poincare_.computeDP(state, flow_derivative, crossing_time);
      const Matrix2 step{dp[0][0], dp[0][1], dp[1][0], dp[1][1]};
      product = multiply(step, product);
      elapsed += crossing_time;
      require_finite(state);
    }
    return {{state[0], state[1]}, product, elapsed};
  }

  Point first_upward_crossing(double x, double y, double z,
                              double* return_time = nullptr) {
    capd::DVector state{x, y, z - kSectionZ};
    double elapsed = 0.0;
    state = poincare_(state, elapsed);
    require_finite(state);
    if (return_time != nullptr) {
      *return_time = elapsed;
    }
    return {state[0], state[1]};
  }

 private:
  static Matrix2 multiply(const Matrix2& left, const Matrix2& right) {
    return {
        left.a * right.a + left.b * right.c,
        left.a * right.b + left.b * right.d,
        left.c * right.a + left.d * right.c,
        left.c * right.b + left.d * right.d,
    };
  }

  static void require_finite(const capd::DVector& value) {
    for (std::size_t i = 0; i < value.dimension(); ++i) {
      if (!std::isfinite(value[i])) {
        throw std::runtime_error("non-finite Poincare image");
      }
    }
  }

  capd::DMap vector_field_;
  capd::DOdeSolver solver_;
  capd::DCoordinateSection section_;
  capd::DPoincareMap poincare_;
};

struct EigenData {
  double stable_value;
  double unstable_value;
  Point stable_vector;
  Point unstable_vector;
};

struct Frame {
  Point origin;
  Point unstable;
  Point stable;
  double determinant;
};

struct HSetCandidate {
  std::string name;
  double center_u;
  double center_s;
  double radius_u;
  double radius_s;
};

struct EdgeDiagnostics {
  std::string source;
  std::string target;
  int degree;
  double entry_margin;
  double stable_strip_margin;
  double left_exit_margin;
  double right_exit_margin;
  double min_image_u;
  double max_image_u;
  double min_image_s;
  double max_image_s;
  std::size_t samples;
};

Point normalize(Point value) {
  const double norm = std::hypot(value.x, value.y);
  if (!(norm > 0.0) || !std::isfinite(norm)) {
    throw std::runtime_error("cannot normalize zero or non-finite vector");
  }
  return {value.x / norm, value.y / norm};
}

Point eigenvector(const Matrix2& matrix, double eigenvalue) {
  Point candidate;
  if (std::abs(matrix.b) >= std::abs(matrix.c)) {
    candidate = {matrix.b, eigenvalue - matrix.a};
  } else {
    candidate = {eigenvalue - matrix.d, matrix.c};
  }
  return normalize(candidate);
}

EigenData eigendata(const Matrix2& matrix) {
  const double trace = matrix.a + matrix.d;
  const double determinant = matrix.a * matrix.d - matrix.b * matrix.c;
  const double discriminant = trace * trace - 4.0 * determinant;
  if (!(discriminant >= 0.0)) {
    throw std::runtime_error("Poincare derivative has non-real eigenvalues");
  }
  const double root = std::sqrt(discriminant);
  const double first = 0.5 * (trace - root);
  const double second = 0.5 * (trace + root);
  const double unstable = std::abs(first) >= std::abs(second) ? first : second;
  const double stable = std::abs(first) >= std::abs(second) ? second : first;
  return {stable, unstable, eigenvector(matrix, stable),
          eigenvector(matrix, unstable)};
}

Frame make_frame(const EigenData& eigen) {
  const double determinant = eigen.unstable_vector.x * eigen.stable_vector.y -
                             eigen.stable_vector.x * eigen.unstable_vector.y;
  if (!(std::abs(determinant) > 1e-12)) {
    throw std::runtime_error("stable and unstable frame is singular");
  }
  return {{kFixedX, kFixedY}, eigen.unstable_vector, eigen.stable_vector,
          determinant};
}

Point from_frame(const Frame& frame, double unstable, double stable) {
  return {frame.origin.x + unstable * frame.unstable.x +
              stable * frame.stable.x,
          frame.origin.y + unstable * frame.unstable.y +
              stable * frame.stable.y};
}

Point to_frame(const Frame& frame, Point point) {
  const double dx = point.x - frame.origin.x;
  const double dy = point.y - frame.origin.y;
  return {(frame.stable.y * dx - frame.stable.x * dy) / frame.determinant,
          (-frame.unstable.y * dx + frame.unstable.x * dy) /
              frame.determinant};
}

Point apply(const Matrix2& matrix, Point vector) {
  return {matrix.a * vector.x + matrix.b * vector.y,
          matrix.c * vector.x + matrix.d * vector.y};
}

int parse_int(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 1 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

double parse_double(const char* text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return value;
}

void print_probe(Cs6Poincare& map, int returns) {
  const MapResult fixed = map.iterate_with_derivative({kFixedX, kFixedY}, returns);
  const EigenData eigen = eigendata(fixed.derivative);
  std::cout << "MODE=probe\n";
  std::cout << "RETURNS_PER_MAP=" << returns << "\n";
  std::cout << "FIXED_CENTER=" << kFixedX << "," << kFixedY << "\n";
  std::cout << "FIXED_IMAGE=" << fixed.point.x << "," << fixed.point.y << "\n";
  std::cout << "FIXED_RESIDUAL=" << fixed.point.x - kFixedX << ","
            << fixed.point.y - kFixedY << "\n";
  std::cout << "RETURN_TIME=" << fixed.return_time << "\n";
  std::cout << "DERIVATIVE=" << fixed.derivative.a << "," << fixed.derivative.b
            << "," << fixed.derivative.c << "," << fixed.derivative.d << "\n";
  std::cout << "UNSTABLE_EIGENVALUE=" << eigen.unstable_value << "\n";
  std::cout << "STABLE_EIGENVALUE=" << eigen.stable_value << "\n";
  std::cout << "UNSTABLE_EIGENVECTOR=" << eigen.unstable_vector.x << ","
            << eigen.unstable_vector.y << "\n";
  std::cout << "STABLE_EIGENVECTOR=" << eigen.stable_vector.x << ","
            << eigen.stable_vector.y << "\n";
}

void print_manifold(Cs6Poincare& map, int returns, int steps, double epsilon) {
  const MapResult fixed = map.iterate_with_derivative({kFixedX, kFixedY}, returns);
  const EigenData eigen = eigendata(fixed.derivative);
  std::cout << "branch,step,x,y,distance_to_fixed\n";
  for (int sign : {-1, 1}) {
    Point point{kFixedX + sign * epsilon * eigen.unstable_vector.x,
                kFixedY + sign * epsilon * eigen.unstable_vector.y};
    for (int step = 0; step <= steps; ++step) {
      std::cout << sign << "," << step << "," << point.x << "," << point.y
                << "," << std::hypot(point.x - kFixedX, point.y - kFixedY)
                << "\n";
      if (step != steps) {
        point = map.iterate(point, returns);
      }
    }
  }
}

void print_orbit(Cs6Poincare& map, int returns, int steps) {
  double first_time = 0.0;
  Point point = map.first_upward_crossing(10.0, 1.0, 10.0, &first_time);
  std::cout << "step,x,y,map_time\n";
  std::cout << "0," << point.x << "," << point.y << "," << first_time << "\n";
  for (int step = 1; step <= steps; ++step) {
    double elapsed = 0.0;
    point = map.iterate(point, returns, &elapsed);
    std::cout << step << "," << point.x << "," << point.y << "," << elapsed
              << "\n";
  }
}

void print_slice(Cs6Poincare& map, int returns, int samples, double half_width) {
  if (samples < 2 || !(half_width > 0.0)) {
    throw std::runtime_error("slice needs samples >= 2 and half_width > 0");
  }
  const MapResult fixed = map.iterate_with_derivative({kFixedX, kFixedY}, returns);
  const Frame frame = make_frame(eigendata(fixed.derivative));
  std::cout << "source_u,image_u,image_s,du_image_u,du_image_s,map_time\n";
  for (int index = 0; index < samples; ++index) {
    const double alpha = static_cast<double>(index) / (samples - 1);
    const double source_u = -half_width + 2.0 * half_width * alpha;
    const Point source = from_frame(frame, source_u, 0.0);
    const MapResult image = map.iterate_with_derivative(source, returns);
    const Point image_local = to_frame(frame, image.point);
    const Point tangent_image = apply(image.derivative, frame.unstable);
    const Point tangent_local = {
        (frame.stable.y * tangent_image.x -
         frame.stable.x * tangent_image.y) /
            frame.determinant,
        (-frame.unstable.y * tangent_image.x +
         frame.unstable.x * tangent_image.y) /
            frame.determinant,
    };
    std::cout << source_u << "," << image_local.x << "," << image_local.y
              << "," << tangent_local.x << "," << tangent_local.y << ","
              << image.return_time << "\n";
  }
}

double image_u_on_unstable_axis(Cs6Poincare& map, const Frame& frame,
                                int returns, double source_u) {
  return to_frame(frame, map.iterate(from_frame(frame, source_u, 0.0), returns))
      .x;
}

double find_secondary_preimage(Cs6Poincare& map, const Frame& frame,
                               int returns) {
  double left = 0.019;
  double right = 0.021;
  double left_value = image_u_on_unstable_axis(map, frame, returns, left);
  double right_value = image_u_on_unstable_axis(map, frame, returns, right);
  if (!(left_value < 0.0 && right_value > 0.0)) {
    throw std::runtime_error("secondary preimage bracket lost");
  }
  for (int iteration = 0; iteration < 52; ++iteration) {
    const double midpoint = 0.5 * (left + right);
    const double value = image_u_on_unstable_axis(map, frame, returns, midpoint);
    if (value > 0.0) {
      right = midpoint;
      right_value = value;
    } else {
      left = midpoint;
      left_value = value;
    }
    if (right - left < 4.0 * std::numeric_limits<double>::epsilon() *
                           std::max(1.0, std::abs(midpoint))) {
      break;
    }
  }
  return std::abs(left_value) <= std::abs(right_value) ? left : right;
}

EdgeDiagnostics sample_edge(Cs6Poincare& map, const Frame& frame, int returns,
                            const HSetCandidate& source,
                            const HSetCandidate& target, int degree,
                            int grid) {
  if (grid < 2 || (degree != -1 && degree != 1)) {
    throw std::runtime_error("invalid candidate grid or degree");
  }
  EdgeDiagnostics result{source.name,
                         target.name,
                         degree,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         0};

  auto normalized_image = [&](double source_u, double source_s) {
    const Point image = map.iterate(from_frame(frame, source_u, source_s), returns);
    const Point local = to_frame(frame, image);
    return Point{(local.x - target.center_u) / target.radius_u,
                 (local.y - target.center_s) / target.radius_s};
  };

  for (int row = 0; row < grid; ++row) {
    const double alpha_s = static_cast<double>(row) / (grid - 1);
    const double source_s = source.center_s - source.radius_s +
                            2.0 * source.radius_s * alpha_s;
    for (int column = 0; column < grid; ++column) {
      const double alpha_u = static_cast<double>(column) / (grid - 1);
      const double source_u = source.center_u - source.radius_u +
                              2.0 * source.radius_u * alpha_u;
      const Point image = normalized_image(source_u, source_s);
      const double stable_margin = 1.0 - std::abs(image.y);
      const double entry_margin = std::abs(image.x) > 1.0
                                      ? std::abs(image.x) - 1.0
                                      : stable_margin;
      result.entry_margin = std::min(result.entry_margin, entry_margin);
      result.stable_strip_margin =
          std::min(result.stable_strip_margin, stable_margin);
      result.min_image_u = std::min(result.min_image_u, image.x);
      result.max_image_u = std::max(result.max_image_u, image.x);
      result.min_image_s = std::min(result.min_image_s, image.y);
      result.max_image_s = std::max(result.max_image_s, image.y);
      ++result.samples;
    }

    const Point left =
        normalized_image(source.center_u - source.radius_u, source_s);
    const Point right =
        normalized_image(source.center_u + source.radius_u, source_s);
    if (degree == 1) {
      result.left_exit_margin =
          std::min(result.left_exit_margin, -1.0 - left.x);
      result.right_exit_margin =
          std::min(result.right_exit_margin, right.x - 1.0);
    } else {
      result.left_exit_margin =
          std::min(result.left_exit_margin, left.x - 1.0);
      result.right_exit_margin =
          std::min(result.right_exit_margin, -1.0 - right.x);
    }
  }
  return result;
}

void print_edge(const EdgeDiagnostics& edge) {
  const bool pass = edge.entry_margin > 0.0 && edge.left_exit_margin > 0.0 &&
                    edge.right_exit_margin > 0.0;
  std::cout << "EDGE=" << edge.source << "->" << edge.target
            << " DEGREE=" << edge.degree << " SAMPLES=" << edge.samples
            << " ENTRY_MARGIN=" << edge.entry_margin
            << " STABLE_STRIP_MARGIN=" << edge.stable_strip_margin
            << " LEFT_EXIT_MARGIN=" << edge.left_exit_margin
            << " RIGHT_EXIT_MARGIN=" << edge.right_exit_margin
            << " IMAGE_U_RANGE=[" << edge.min_image_u << ","
            << edge.max_image_u << "] IMAGE_S_RANGE=[" << edge.min_image_s
            << "," << edge.max_image_s << "] CANDIDATE_PASS="
            << (pass ? "true" : "false") << "\n";
}

void print_candidate(Cs6Poincare& map, int returns, int grid, double radius_u0,
                     double radius_s0, double radius_u1, double radius_s1) {
  const MapResult fixed = map.iterate_with_derivative({kFixedX, kFixedY}, returns);
  const Frame frame = make_frame(eigendata(fixed.derivative));
  const double secondary_u = find_secondary_preimage(map, frame, returns);
  const HSetCandidate n0{"N0", 0.0, 0.0, radius_u0, radius_s0};
  const HSetCandidate n1{"N1", secondary_u, 0.0, radius_u1, radius_s1};
  const Point secondary_image =
      to_frame(frame, map.iterate(from_frame(frame, secondary_u, 0.0), returns));
  const double separation = secondary_u - radius_u0 - radius_u1;
  std::cout << "MODE=candidate\nRETURNS_PER_MAP=" << returns << "\n";
  std::cout << "FRAME_UNSTABLE=" << frame.unstable.x << ","
            << frame.unstable.y << "\nFRAME_STABLE=" << frame.stable.x << ","
            << frame.stable.y << "\n";
  std::cout << "N0=" << n0.center_u << "," << n0.center_s << ","
            << n0.radius_u << "," << n0.radius_s << "\n";
  std::cout << "N1=" << n1.center_u << "," << n1.center_s << ","
            << n1.radius_u << "," << n1.radius_s << "\n";
  std::cout << "N1_CENTER_IMAGE=" << secondary_image.x << ","
            << secondary_image.y << "\n";
  std::cout << "SUPPORT_U_SEPARATION=" << separation
            << " DISJOINT=" << (separation > 0.0 ? "true" : "false") << "\n";
  print_edge(sample_edge(map, frame, returns, n0, n0, -1, grid));
  print_edge(sample_edge(map, frame, returns, n0, n1, -1, grid));
  print_edge(sample_edge(map, frame, returns, n1, n0, 1, grid));
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [probe|manifold|orbit|slice|candidate] [returns=6] [steps=20]"
               " [epsilon_or_half_width=1e-9]"
               "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(17);
  try {
    const std::string mode = argc > 1 ? argv[1] : "probe";
    const int returns = argc > 2 ? parse_int(argv[2], "returns") : 6;
    const int steps = argc > 3 ? parse_int(argv[3], "steps") : 20;
    const double epsilon = argc > 4 ? parse_double(argv[4], "epsilon") : 1e-9;
    Cs6Poincare map(30);
    if (mode == "probe") {
      print_probe(map, returns);
    } else if (mode == "manifold") {
      print_manifold(map, returns, steps, epsilon);
    } else if (mode == "orbit") {
      print_orbit(map, returns, steps);
    } else if (mode == "slice") {
      print_slice(map, returns, steps, epsilon);
    } else if (mode == "candidate") {
      const double radius_s0 = argc > 5 ? parse_double(argv[5], "radius_s0")
                                         : 1e-5;
      const double radius_u1 = argc > 6 ? parse_double(argv[6], "radius_u1")
                                         : 0.0015;
      const double radius_s1 = argc > 7 ? parse_double(argv[7], "radius_s1")
                                         : 1e-5;
      print_candidate(map, returns, steps, epsilon, radius_s0, radius_u1,
                      radius_s1);
    } else {
      usage(argv[0]);
      return 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CS6_SCOUT_ERROR=" << error.what() << "\n";
    return 3;
  }
}
