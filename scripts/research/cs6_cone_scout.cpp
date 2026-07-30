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

namespace {

constexpr int kReturns = 6;
constexpr int kTaylorOrder = 30;
constexpr int kDirectionSamples = 257;
constexpr double kRateTolerance = 1e-12;
constexpr double kSectionZ = 22.3274637391;
constexpr double kOriginX = 15.186446520640786;
constexpr double kOriginY = 10.908543194765466;
constexpr double kUnstableX = -0.67430316214199759;
constexpr double kUnstableY = -0.73845463335624273;
constexpr double kStableX = -0.94170446778164518;
constexpr double kStableY = 0.33644122125579123;
constexpr double kN1CenterU = 0.019771776972779206;

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

struct HSet {
  const char* name;
  double center_u;
  double center_s;
  double radius_u;
  double radius_s;
};

struct QWeights {
  double positive;
  double negative;
};

struct DerivativeSample {
  Matrix2 derivative;
  double source_u;
  double source_s;
  double return_time;
};

struct EdgeSamples {
  const char* name;
  int source;
  int target;
  std::vector<DerivativeSample> samples;
};

struct ConeMatrix {
  long double m00;
  long double m01;
  long double m11;
  long double determinant;
  long double min_eigenvalue;
};

struct Interval {
  long double lower;
  long double upper;
};

struct MatrixHull {
  Interval a;
  Interval b;
  Interval c;
  Interval d;
};

struct HullConeDiagnostic {
  Interval m00;
  Interval m01;
  Interval m11;
  Interval determinant;
  bool positive_definite_sufficient;
};

struct Parameters {
  double log2_a1;
  double log2_b0;
  double log2_b1;
};

struct EdgeEvaluation {
  double min_normalized_margin = std::numeric_limits<double>::infinity();
  double min_raw_margin = std::numeric_limits<double>::infinity();
  double min_m00 = std::numeric_limits<double>::infinity();
  double min_determinant = std::numeric_limits<double>::infinity();
  double min_abs_map_determinant = std::numeric_limits<double>::infinity();
  double min_forward_expansion = std::numeric_limits<double>::infinity();
  double min_backward_expansion = std::numeric_limits<double>::infinity();
  std::size_t worst_index = 0;
  bool backward_rate_resolved = true;
};

struct Evaluation {
  double global_margin = -std::numeric_limits<double>::infinity();
  std::array<EdgeEvaluation, 3> edges;
};

struct SearchEvaluation {
  double sampled_hull_score;
  double point_margin;
};

Matrix2 multiply(const Matrix2& left, const Matrix2& right) {
  return {
      left.a * right.a + left.b * right.c,
      left.a * right.b + left.b * right.d,
      left.c * right.a + left.d * right.c,
      left.c * right.b + left.d * right.d,
  };
}

void require_finite_matrix(const Matrix2& matrix, const char* label) {
  if (!std::isfinite(matrix.a) || !std::isfinite(matrix.b) ||
      !std::isfinite(matrix.c) || !std::isfinite(matrix.d)) {
    throw std::runtime_error(std::string("non-finite ") + label);
  }
}

void require_positive_finite_time(double value) {
  if (!std::isfinite(value) || !(value > 0.0)) {
    throw std::runtime_error("non-positive or non-finite Poincare return time");
  }
}

Point apply(const Matrix2& matrix, Point vector) {
  return {matrix.a * vector.x + matrix.b * vector.y,
          matrix.c * vector.x + matrix.d * vector.y};
}

double determinant(const Matrix2& matrix) {
  const long double value =
      static_cast<long double>(matrix.a) * matrix.d -
      static_cast<long double>(matrix.b) * matrix.c;
  return static_cast<double>(value);
}

class Cs6Poincare {
 public:
  Cs6Poincare()
      : vector_field_(
            "par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,"
            "x*y-w-zs;"),
        solver_(vector_field_, kTaylorOrder),
        section_(3, 2),
        poincare_(solver_, section_, capd::poincare::MinusPlus) {
    vector_field_.setParameter("zs", kSectionZ);
  }

  Point iterate(Point point) {
    capd::DVector state{point.x, point.y, 0.0};
    for (int index = 0; index < kReturns; ++index) {
      double return_time = 0.0;
      state = poincare_(state, return_time);
      require_positive_finite_time(return_time);
      require_finite(state);
    }
    return {state[0], state[1]};
  }

  MapResult iterate_with_derivative(Point point) {
    capd::DVector state{point.x, point.y, 0.0};
    Matrix2 product{1.0, 0.0, 0.0, 1.0};
    double elapsed = 0.0;
    for (int index = 0; index < kReturns; ++index) {
      capd::DMatrix flow_derivative(3, 3);
      double return_time = 0.0;
      state = poincare_(state, flow_derivative, return_time);
      require_positive_finite_time(return_time);
      const capd::DMatrix dp =
          poincare_.computeDP(state, flow_derivative, return_time);
      const Matrix2 step{dp[0][0], dp[0][1], dp[1][0], dp[1][1]};
      require_finite_matrix(step, "Poincare derivative");
      product = multiply(step, product);
      require_finite_matrix(product, "composed Poincare derivative");
      elapsed += return_time;
      require_positive_finite_time(elapsed);
      require_finite(state);
    }
    return {{state[0], state[1]}, product, elapsed};
  }

 private:
  static void require_finite(const capd::DVector& value) {
    for (std::size_t index = 0; index < value.dimension(); ++index) {
      if (!std::isfinite(value[index])) {
        throw std::runtime_error("non-finite Poincare image");
      }
    }
  }

  capd::DMap vector_field_;
  capd::DOdeSolver solver_;
  capd::DCoordinateSection section_;
  capd::DPoincareMap poincare_;
};

const Matrix2 kFrame{kUnstableX, kStableX, kUnstableY, kStableY};

Matrix2 inverse(const Matrix2& matrix) {
  const double det = determinant(matrix);
  if (!(std::abs(det) > 1e-12)) {
    throw std::runtime_error("frozen frame is singular");
  }
  return {matrix.d / det, -matrix.b / det, -matrix.c / det,
          matrix.a / det};
}

const Matrix2 kFrameInverse = inverse(kFrame);
const std::array<HSet, 2> kSets{{
    {"N0", 0.0, 0.0, 0.004, 0.3},
    {"N1", kN1CenterU, 0.0, 0.0015, 0.3},
}};

Point from_local(double local_u, double local_s) {
  const Point physical = apply(kFrame, {local_u, local_s});
  return {kOriginX + physical.x, kOriginY + physical.y};
}

Point to_local(Point physical) {
  return apply(kFrameInverse, {physical.x - kOriginX, physical.y - kOriginY});
}

Point from_normalized(const HSet& set, double u, double s) {
  return from_local(set.center_u + set.radius_u * u,
                    set.center_s + set.radius_s * s);
}

Point normalized_image(Cs6Poincare& map, const HSet& source,
                       const HSet& target, double u, double s) {
  const Point local = to_local(map.iterate(from_normalized(source, u, s)));
  return {(local.x - target.center_u) / target.radius_u,
          (local.y - target.center_s) / target.radius_s};
}

Matrix2 normalized_derivative(const Matrix2& physical, const HSet& source,
                              const HSet& target) {
  const Matrix2 local = multiply(kFrameInverse, multiply(physical, kFrame));
  return {
      local.a * source.radius_u / target.radius_u,
      local.b * source.radius_s / target.radius_u,
      local.c * source.radius_u / target.radius_s,
      local.d * source.radius_s / target.radius_s,
  };
}

int parse_grid(const char* text) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 3 || value > 41 || value % 2 == 0) {
    throw std::runtime_error("grid must be an odd integer in [3,41]");
  }
  return static_cast<int>(value);
}

ConeMatrix cone_matrix(const Matrix2& map, const QWeights& source,
                       const QWeights& target) {
  const long double a = map.a;
  const long double b = map.b;
  const long double c = map.c;
  const long double d = map.d;
  const long double m00 = target.positive * a * a -
                          target.negative * c * c - source.positive;
  const long double m01 = target.positive * a * b -
                          target.negative * c * d;
  const long double m11 = target.positive * b * b -
                          target.negative * d * d + source.negative;
  const long double det = m00 * m11 - m01 * m01;
  const long double discriminant = std::hypotl(m00 - m11, 2.0L * m01);
  const long double max_eigenvalue = 0.5L * (m00 + m11 + discriminant);
  long double min_eigenvalue = 0.5L * (m00 + m11 - discriminant);
  if (det > 0.0L && max_eigenvalue > 0.0L) {
    min_eigenvalue = det / max_eigenvalue;
  }
  return {m00, m01, m11, det, min_eigenvalue};
}

Interval add(Interval left, Interval right) {
  return {left.lower + right.lower, left.upper + right.upper};
}

Interval subtract(Interval left, Interval right) {
  return {left.lower - right.upper, left.upper - right.lower};
}

Interval multiply(Interval left, Interval right) {
  const std::array<long double, 4> products{{
      left.lower * right.lower, left.lower * right.upper,
      left.upper * right.lower, left.upper * right.upper}};
  return {*std::min_element(products.begin(), products.end()),
          *std::max_element(products.begin(), products.end())};
}

Interval scale(Interval value, long double factor) {
  if (!(factor >= 0.0L)) {
    throw std::runtime_error("interval scale must be nonnegative");
  }
  return {factor * value.lower, factor * value.upper};
}

Interval square(Interval value) {
  if (value.lower <= 0.0L && value.upper >= 0.0L) {
    return {0.0L,
            std::max(value.lower * value.lower, value.upper * value.upper)};
  }
  const long double lower_squared = value.lower * value.lower;
  const long double upper_squared = value.upper * value.upper;
  return {std::min(lower_squared, upper_squared),
          std::max(lower_squared, upper_squared)};
}

MatrixHull sample_hull(const EdgeSamples& edge) {
  MatrixHull hull{{std::numeric_limits<long double>::infinity(),
                   -std::numeric_limits<long double>::infinity()},
                  {std::numeric_limits<long double>::infinity(),
                   -std::numeric_limits<long double>::infinity()},
                  {std::numeric_limits<long double>::infinity(),
                   -std::numeric_limits<long double>::infinity()},
                  {std::numeric_limits<long double>::infinity(),
                   -std::numeric_limits<long double>::infinity()}};
  const auto include = [](Interval& interval, double value) {
    interval.lower = std::min(interval.lower, static_cast<long double>(value));
    interval.upper = std::max(interval.upper, static_cast<long double>(value));
  };
  for (const DerivativeSample& sample : edge.samples) {
    include(hull.a, sample.derivative.a);
    include(hull.b, sample.derivative.b);
    include(hull.c, sample.derivative.c);
    include(hull.d, sample.derivative.d);
  }
  return hull;
}

HullConeDiagnostic hull_cone_diagnostic(const MatrixHull& map,
                                        const QWeights& source,
                                        const QWeights& target) {
  const Interval m00 = subtract(
      subtract(scale(square(map.a), target.positive),
               scale(square(map.c), target.negative)),
      {source.positive, source.positive});
  const Interval m01 = subtract(
      scale(multiply(map.a, map.b), target.positive),
      scale(multiply(map.c, map.d), target.negative));
  const Interval m11 = add(
      subtract(scale(square(map.b), target.positive),
               scale(square(map.d), target.negative)),
      {source.negative, source.negative});
  const Interval det = subtract(multiply(m00, m11), square(m01));
  return {m00, m01, m11, det, m00.lower > 0.0L && det.lower > 0.0L};
}

std::array<QWeights, 2> weights(const Parameters& parameters) {
  return {{{1.0, std::exp2(parameters.log2_b0)},
           {std::exp2(parameters.log2_a1), std::exp2(parameters.log2_b1)}}};
}

double normalization(const QWeights& source, const QWeights& target) {
  return std::max({source.positive, source.negative, target.positive,
                   target.negative});
}

Evaluation evaluate(const std::array<EdgeSamples, 3>& edges,
                    const Parameters& parameters, bool rates) {
  const auto forms = weights(parameters);
  Evaluation result;
  result.global_margin = std::numeric_limits<double>::infinity();
  for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const EdgeSamples& edge = edges[edge_index];
    const QWeights& source_q = forms[edge.source];
    const QWeights& target_q = forms[edge.target];
    const double scale = normalization(source_q, target_q);
    EdgeEvaluation& diagnostic = result.edges[edge_index];
    for (std::size_t sample_index = 0; sample_index < edge.samples.size();
         ++sample_index) {
      const DerivativeSample& sample = edge.samples[sample_index];
      const ConeMatrix matrix =
          cone_matrix(sample.derivative, source_q, target_q);
      const double raw = static_cast<double>(matrix.min_eigenvalue);
      const double normalized = raw / scale;
      if (!std::isfinite(raw) || !std::isfinite(normalized) ||
          !std::isfinite(static_cast<double>(matrix.m00)) ||
          !std::isfinite(static_cast<double>(matrix.determinant))) {
        throw std::runtime_error("non-finite cone diagnostic");
      }
      diagnostic.min_m00 =
          std::min(diagnostic.min_m00, static_cast<double>(matrix.m00));
      diagnostic.min_determinant = std::min(
          diagnostic.min_determinant, static_cast<double>(matrix.determinant));
      diagnostic.min_abs_map_determinant = std::min(
          diagnostic.min_abs_map_determinant,
          std::abs(determinant(sample.derivative)));
      if (normalized < diagnostic.min_normalized_margin) {
        diagnostic.min_normalized_margin = normalized;
        diagnostic.min_raw_margin = raw;
        diagnostic.worst_index = sample_index;
      }
    }
    result.global_margin =
        std::min(result.global_margin, diagnostic.min_normalized_margin);

    if (!rates) {
      continue;
    }
    const double forward_limit =
        std::sqrt(source_q.positive / source_q.negative);
    const double backward_limit =
        std::sqrt(target_q.negative / target_q.positive);
    for (const DerivativeSample& sample : edge.samples) {
      const Matrix2& map = sample.derivative;
      for (int slope_index = 0; slope_index < kDirectionSamples;
           ++slope_index) {
        const double alpha = static_cast<double>(slope_index) /
                             static_cast<double>(kDirectionSamples - 1);
        const double slope = -forward_limit + 2.0 * forward_limit * alpha;
        const Point image = apply(map, {1.0, slope});
        const double expansion =
            std::hypot(image.x, image.y) / std::hypot(1.0, slope);
        if (!std::isfinite(expansion)) {
          throw std::runtime_error("non-finite sampled forward expansion");
        }
        diagnostic.min_forward_expansion = std::min(
            diagnostic.min_forward_expansion, expansion);
      }
      const double det = determinant(map);
      if (!(std::abs(det) > 1e-12)) {
        diagnostic.backward_rate_resolved = false;
        diagnostic.min_backward_expansion =
            std::numeric_limits<double>::quiet_NaN();
        continue;
      }
      const Matrix2 inverse_map = inverse(map);
      for (int slope_index = 0; slope_index < kDirectionSamples;
           ++slope_index) {
        const double alpha = static_cast<double>(slope_index) /
                             static_cast<double>(kDirectionSamples - 1);
        const double slope = -backward_limit + 2.0 * backward_limit * alpha;
        const Point preimage = apply(inverse_map, {slope, 1.0});
        const double expansion =
            std::hypot(preimage.x, preimage.y) / std::hypot(slope, 1.0);
        if (!std::isfinite(expansion)) {
          throw std::runtime_error("non-finite sampled backward expansion");
        }
        diagnostic.min_backward_expansion = std::min(
            diagnostic.min_backward_expansion, expansion);
      }
    }
  }
  return result;
}

SearchEvaluation evaluate_search(
    const std::array<EdgeSamples, 3>& edges,
    const std::array<MatrixHull, 3>& hulls,
    const Parameters& parameters) {
  const auto forms = weights(parameters);
  double hull_score = std::numeric_limits<double>::infinity();
  for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const EdgeSamples& edge = edges[edge_index];
    const QWeights& source = forms[edge.source];
    const QWeights& target = forms[edge.target];
    const double form_scale = normalization(source, target);
    const HullConeDiagnostic diagnostic =
        hull_cone_diagnostic(hulls[edge_index], source, target);
    const double edge_score = std::min(
        static_cast<double>(diagnostic.m00.lower) / form_scale,
        static_cast<double>(diagnostic.determinant.lower) /
            (form_scale * form_scale));
    hull_score = std::min(hull_score, edge_score);
  }
  return {hull_score, evaluate(edges, parameters, false).global_margin};
}

bool metric_greater(double candidate, double incumbent) {
  if (!std::isfinite(incumbent)) {
    return std::isfinite(candidate);
  }
  const double tolerance =
      1e-14 * std::max({1.0, std::abs(candidate), std::abs(incumbent)});
  return candidate > incumbent + tolerance;
}

bool metric_tied(double left, double right) {
  if (!std::isfinite(left) || !std::isfinite(right)) {
    return left == right;
  }
  const double tolerance =
      1e-14 * std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance;
}

bool parameters_lexicographically_less(const Parameters& left,
                                       const Parameters& right) {
  if (left.log2_a1 != right.log2_a1) {
    return left.log2_a1 < right.log2_a1;
  }
  if (left.log2_b0 != right.log2_b0) {
    return left.log2_b0 < right.log2_b0;
  }
  return left.log2_b1 < right.log2_b1;
}

bool better(const SearchEvaluation& candidate,
            const Parameters& candidate_parameters,
            const SearchEvaluation& incumbent,
            const Parameters& incumbent_parameters) {
  if (metric_greater(candidate.sampled_hull_score,
                     incumbent.sampled_hull_score)) {
    return true;
  }
  if (!metric_tied(candidate.sampled_hull_score,
                   incumbent.sampled_hull_score)) {
    return false;
  }
  if (metric_greater(candidate.point_margin, incumbent.point_margin)) {
    return true;
  }
  return metric_tied(candidate.point_margin, incumbent.point_margin) &&
         parameters_lexicographically_less(candidate_parameters,
                                           incumbent_parameters);
}

Parameters search_parameters(const std::array<EdgeSamples, 3>& edges) {
  Parameters best_parameters{0.0, 0.0, 0.0};
  std::array<MatrixHull, 3> hulls;
  for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    hulls[edge_index] = sample_hull(edges[edge_index]);
  }
  SearchEvaluation best{-std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity()};
  for (int a1 = -12; a1 <= 12; ++a1) {
    for (int b0 = -12; b0 <= 12; ++b0) {
      for (int b1 = -12; b1 <= 12; ++b1) {
        const Parameters candidate{static_cast<double>(a1),
                                   static_cast<double>(b0),
                                   static_cast<double>(b1)};
        const SearchEvaluation value = evaluate_search(edges, hulls, candidate);
        if (better(value, candidate, best, best_parameters)) {
          best = value;
          best_parameters = candidate;
        }
      }
    }
  }
  for (double step : {0.25, 0.0625, 0.015625}) {
    const Parameters center = best_parameters;
    for (int da1 = -4; da1 <= 4; ++da1) {
      for (int db0 = -4; db0 <= 4; ++db0) {
        for (int db1 = -4; db1 <= 4; ++db1) {
          const Parameters candidate{center.log2_a1 + da1 * step,
                                     center.log2_b0 + db0 * step,
                                     center.log2_b1 + db1 * step};
          const SearchEvaluation value =
              evaluate_search(edges, hulls, candidate);
          if (better(value, candidate, best, best_parameters)) {
            best = value;
            best_parameters = candidate;
          }
        }
      }
    }
  }
  return best_parameters;
}

double sample_coordinate(int index, int count, bool cell_midpoints) {
  if (cell_midpoints) {
    return -1.0 + 2.0 * (index + 0.5) / static_cast<double>(count);
  }
  return -1.0 + 2.0 * index / static_cast<double>(count - 1);
}

std::array<EdgeSamples, 3> sample_derivatives(Cs6Poincare& map, int grid,
                                              bool cell_midpoints) {
  std::array<EdgeSamples, 3> edges{{
      {"N0->N0", 0, 0, {}},
      {"N0->N1", 0, 1, {}},
      {"N1->N0", 1, 0, {}},
  }};
  for (EdgeSamples& edge : edges) {
    edge.samples.reserve(static_cast<std::size_t>(grid) * grid);
  }
  for (int source_index = 0; source_index < 2; ++source_index) {
    const HSet& source = kSets[source_index];
    for (int row = 0; row < grid; ++row) {
      const double source_s = sample_coordinate(row, grid, cell_midpoints);
      for (int column = 0; column < grid; ++column) {
        const double source_u =
            sample_coordinate(column, grid, cell_midpoints);
        const MapResult mapped =
            map.iterate_with_derivative(from_normalized(source, source_u, source_s));
        for (EdgeSamples& edge : edges) {
          if (edge.source != source_index) {
            continue;
          }
          edge.samples.push_back(
              {normalized_derivative(mapped.derivative, source,
                                     kSets[edge.target]),
               source_u, source_s, mapped.return_time});
        }
      }
    }
  }
  return edges;
}

double finite_difference_error(Cs6Poincare& map, const EdgeSamples& edge,
                               double center_u, double center_s) {
  constexpr double step = 1e-5;
  const HSet& source = kSets[edge.source];
  const HSet& target = kSets[edge.target];
  const Point plus_u =
      normalized_image(map, source, target, center_u + step, center_s);
  const Point minus_u =
      normalized_image(map, source, target, center_u - step, center_s);
  const Point plus_s =
      normalized_image(map, source, target, center_u, center_s + step);
  const Point minus_s =
      normalized_image(map, source, target, center_u, center_s - step);
  const Matrix2 finite_difference{
      (plus_u.x - minus_u.x) / (2.0 * step),
      (plus_s.x - minus_s.x) / (2.0 * step),
      (plus_u.y - minus_u.y) / (2.0 * step),
      (plus_s.y - minus_s.y) / (2.0 * step),
  };
  const MapResult center =
      map.iterate_with_derivative(from_normalized(source, center_u, center_s));
  const Matrix2 exact = normalized_derivative(center.derivative, source, target);
  const std::array<double, 4> observed{{finite_difference.a, finite_difference.b,
                                        finite_difference.c, finite_difference.d}};
  const std::array<double, 4> expected{{exact.a, exact.b, exact.c, exact.d}};
  double error = 0.0;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    if (!std::isfinite(observed[index]) || !std::isfinite(expected[index])) {
      throw std::runtime_error("non-finite finite-difference diagnostic");
    }
    error = std::max(error, std::abs(observed[index] - expected[index]) /
                                std::max(1.0, std::abs(expected[index])));
  }
  return error;
}

void print_selftest() {
  const QWeights q{1.0, 1.0};
  const double hyperbolic = static_cast<double>(
      cone_matrix({2.0, 0.0, 0.0, 0.5}, q, q).min_eigenvalue);
  const double identity = static_cast<double>(
      cone_matrix({1.0, 0.0, 0.0, 1.0}, q, q).min_eigenvalue);
  const Matrix2 singular{2.0, 0.0, 0.0, 0.0};
  const double singular_margin =
      static_cast<double>(cone_matrix(singular, q, q).min_eigenvalue);
  const double swapped = static_cast<double>(
      cone_matrix({0.0, 2.0, 0.5, 0.0}, q, q).min_eigenvalue);
  const HSet synthetic_source{"S", 0.0, 0.0, 0.004, 0.3};
  const HSet synthetic_target{"T", 0.0, 0.0, 0.0015, 0.3};
  const Matrix2 scaled =
      normalized_derivative(multiply(kFrame, multiply(Matrix2{1, 2, 3, 4},
                                                        kFrameInverse)),
                            synthetic_source, synthetic_target);
  const bool scale_pass = std::abs(scaled.a - 8.0 / 3.0) < 1e-12 &&
                          std::abs(scaled.b - 400.0) < 1e-10 &&
                          std::abs(scaled.c - 0.04) < 1e-12 &&
                          std::abs(scaled.d - 4.0) < 1e-12;
  bool nonfinite_rejected = false;
  try {
    require_finite_matrix(
        {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0},
        "selftest derivative");
  } catch (const std::runtime_error&) {
    nonfinite_rejected = true;
  }
  const MatrixHull singleton_hyperbolic{{2.0L, 2.0L}, {0.0L, 0.0L},
                                        {0.0L, 0.0L}, {0.5L, 0.5L}};
  const MatrixHull singleton_identity{{1.0L, 1.0L}, {0.0L, 0.0L},
                                      {0.0L, 0.0L}, {1.0L, 1.0L}};
  const MatrixHull widened{{0.0L, 2.0L}, {0.0L, 0.0L},
                           {0.0L, 0.0L}, {0.5L, 1.5L}};
  const bool hull_pass =
      hull_cone_diagnostic(singleton_hyperbolic, q, q)
          .positive_definite_sufficient &&
      !hull_cone_diagnostic(singleton_identity, q, q)
           .positive_definite_sufficient &&
      !hull_cone_diagnostic(widened, q, q).positive_definite_sufficient;
  const SearchEvaluation exact_score{1.0, 1.0};
  const SearchEvaluation near_score{1.0 + 1e-15, 1.0};
  const Parameters lexical_first{-1.0, 0.0, 0.0};
  const Parameters lexical_second{0.0, 0.0, 0.0};
  const bool search_tiebreak_pass =
      better(exact_score, lexical_first, exact_score, lexical_second) &&
      better(near_score, lexical_first, exact_score, lexical_second) &&
      !better(exact_score, lexical_second, exact_score, lexical_first);
  const bool pass = hyperbolic > 0.0 && identity == 0.0 &&
                    singular_margin > 0.0 && determinant(singular) == 0.0 &&
                    swapped < 0.0 && scale_pass && nonfinite_rejected &&
                    hull_pass && search_tiebreak_pass;
  std::cout << "MODE=selftest\n"
            << "HYPERBOLIC_MARGIN=" << hyperbolic << "\n"
            << "IDENTITY_MARGIN=" << identity << "\n"
            << "SINGULAR_MARGIN=" << singular_margin << "\n"
            << "SINGULAR_DETERMINANT=" << determinant(singular) << "\n"
            << "SWAPPED_AXES_MARGIN=" << swapped << "\n"
            << "RADIUS_TRANSFORM=" << scaled.a << "," << scaled.b << ","
            << scaled.c << "," << scaled.d << "\n"
            << "RADIUS_TRANSFORM_PASS=" << (scale_pass ? "true" : "false")
            << "\nNONFINITE_DERIVATIVE_REJECTED="
            << (nonfinite_rejected ? "true" : "false")
            << "\nHULL_ARITHMETIC_SELFTEST_PASS="
            << (hull_pass ? "true" : "false")
            << "\nSEARCH_TIEBREAK_SELFTEST_PASS="
            << (search_tiebreak_pass ? "true" : "false")
            << "\nSELFTEST_PASS=" << (pass ? "true" : "false") << "\n";
  if (!pass) {
    throw std::runtime_error("cone algebra selftest failed");
  }
}

void print_scout(int grid) {
  Cs6Poincare map;
  const int holdout_grid = grid - 1;
  const auto discovery_edges = sample_derivatives(map, grid, false);
  const Parameters parameters = search_parameters(discovery_edges);
  const auto forms = weights(parameters);
  const Evaluation discovery = evaluate(discovery_edges, parameters, true);
  const auto holdout_edges = sample_derivatives(map, holdout_grid, true);
  const Evaluation holdout = evaluate(holdout_edges, parameters, true);
  double max_finite_difference_error = 0.0;
  constexpr std::array<Point, 5> derivative_probes{{
      {0.0, 0.0}, {-0.9, -0.9}, {-0.9, 0.9}, {0.9, -0.9}, {0.9, 0.9}}};
  for (const EdgeSamples& edge : discovery_edges) {
    for (const Point& probe : derivative_probes) {
      max_finite_difference_error =
          std::max(max_finite_difference_error,
                   finite_difference_error(map, edge, probe.x, probe.y));
    }
  }
  bool sample_records_valid = true;
  for (const EdgeSamples& edge : discovery_edges) {
    sample_records_valid = sample_records_valid &&
                           edge.samples.size() ==
                               static_cast<std::size_t>(grid) * grid;
  }
  for (const EdgeSamples& edge : holdout_edges) {
    sample_records_valid = sample_records_valid &&
                           edge.samples.size() ==
                               static_cast<std::size_t>(holdout_grid) *
                                   holdout_grid;
  }
  if (!sample_records_valid) {
    throw std::runtime_error("derivative sample record count mismatch");
  }
  bool determinant_resolved = true;
  bool rates_above_one = true;
  bool sampled_hulls_pd_sufficient = true;
  const bool search_boundary_hit =
      std::abs(parameters.log2_a1) >= 12.0 ||
      std::abs(parameters.log2_b0) >= 12.0 ||
      std::abs(parameters.log2_b1) >= 12.0;
  std::cout << "SCHEMA=sounio.cs6.cone-scout.v1\n"
            << "MODE=cone-scout\nMAP=P^6\nRETURNS_PER_MAP=" << kReturns
            << "\nTAYLOR_ORDER=" << kTaylorOrder << "\nDISCOVERY_GRID=" << grid
            << "x" << grid << "\nDISCOVERY_LAYOUT=endpoints"
            << "\nHOLDOUT_GRID=" << holdout_grid << "x" << holdout_grid
            << "\nHOLDOUT_LAYOUT=cell_midpoints"
            << "\nDISCOVERY_UNIQUE_SOURCE_SAMPLES=" << 2 * grid * grid
            << "\nHOLDOUT_UNIQUE_SOURCE_SAMPLES="
            << 2 * holdout_grid * holdout_grid
            << "\nDISCOVERY_EDGE_DERIVATIVE_RECORDS=" << 3 * grid * grid
            << "\nHOLDOUT_EDGE_DERIVATIVE_RECORDS="
            << 3 * holdout_grid * holdout_grid << "\n"
            << "ALL_SAMPLE_RECORDS_VALID=true\n"
            << "FRAME=" << kUnstableX << "," << kStableX << ","
            << kUnstableY << "," << kStableY << "\n"
            << "N0=0,0,0.004,0.3\nN1=" << kN1CenterU
            << ",0,0.0015,0.3\n"
            << "CONE_CRITERION=M=A^T*Q_TARGET*A-Q_SOURCE\n"
            << "Q_NORMALIZATION=A0:1\n"
            << "SEARCH_OBJECTIVE=sampled-entrywise-hull-sylvester-then-point-margin\n"
            << "SEARCH_LOG2_COARSE_RANGE=-12:12\n"
            << "SEARCH_REFINEMENT_STEPS=0.25,0.0625,0.015625\n"
            << "WEIGHTS_TUNED_ON_DISCOVERY_GRID=true\n"
            << "HOLDOUT_USED_FOR_TUNING=false\n"
            << "SAMPLED_HULL_ARITHMETIC=long-double-no-outward-rounding\n"
            << "Q_N0=" << forms[0].positive << "," << -forms[0].negative
            << "\nQ_N1=" << forms[1].positive << "," << -forms[1].negative
            << "\nLOG2_PARAMETERS=A1:" << parameters.log2_a1
            << ",B0:" << parameters.log2_b0 << ",B1:"
            << parameters.log2_b1 << "\nSEARCH_BOUNDARY_HIT="
            << (search_boundary_hit ? "true" : "false") << "\n"
            << "DIRECTION_SAMPLES_PER_CONE=" << kDirectionSamples << "\n"
            << "FINITE_DIFFERENCE_PROBES="
            << discovery_edges.size() * derivative_probes.size() << "\n"
            << "MAX_FINITE_DIFFERENCE_REL_ERROR="
            << max_finite_difference_error << "\n";
  const auto print_evaluation = [&](const char* phase,
                                    const std::array<EdgeSamples, 3>& edges,
                                    const Evaluation& evaluation) {
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
      const EdgeSamples& edge = edges[edge_index];
      const EdgeEvaluation& diagnostic = evaluation.edges[edge_index];
      const DerivativeSample& worst = edge.samples[diagnostic.worst_index];
      const MatrixHull hull = sample_hull(edge);
      const HullConeDiagnostic hull_diagnostic = hull_cone_diagnostic(
          hull, forms[edge.source], forms[edge.target]);
      sampled_hulls_pd_sufficient =
          sampled_hulls_pd_sufficient &&
          hull_diagnostic.positive_definite_sufficient;
      determinant_resolved = determinant_resolved &&
                             diagnostic.min_abs_map_determinant > 1e-12;
      rates_above_one = rates_above_one &&
                        diagnostic.min_forward_expansion >
                            1.0 + kRateTolerance &&
                        diagnostic.backward_rate_resolved &&
                        diagnostic.min_backward_expansion >
                            1.0 + kRateTolerance;
      std::cout << "PHASE=" << phase << " EDGE=" << edge.name
                << " SAMPLES=" << edge.samples.size()
                << " MIN_NORMALIZED_CONE_MARGIN="
                << diagnostic.min_normalized_margin
                << " MIN_RAW_CONE_MARGIN=" << diagnostic.min_raw_margin
                << " MIN_M00=" << diagnostic.min_m00
                << " MIN_DET_M=" << diagnostic.min_determinant
                << " DOUBLE_DP_MIN_ABS_DET_RESIDUAL="
                << diagnostic.min_abs_map_determinant
                << " MIN_SAMPLED_FORWARD_EXPANSION="
                << diagnostic.min_forward_expansion
                << " BACKWARD_RATE_RESOLVED="
                << (diagnostic.backward_rate_resolved ? "true" : "false")
                << " MIN_SAMPLED_BACKWARD_EXPANSION="
                << diagnostic.min_backward_expansion << " WORST_SOURCE="
                << worst.source_u << "," << worst.source_s << " WORST_A="
                << worst.derivative.a << "," << worst.derivative.b << ","
                << worst.derivative.c << "," << worst.derivative.d
                << " SAMPLED_A_HULL="
                << std::setprecision(
                       std::numeric_limits<long double>::max_digits10)
                << hull.a.lower << ":" << hull.a.upper
                << "," << hull.b.lower << ":" << hull.b.upper << ","
                << hull.c.lower << ":" << hull.c.upper << ","
                << hull.d.lower << ":" << hull.d.upper
                << " SAMPLED_HULL_M00_LOWER=" << hull_diagnostic.m00.lower
                << " SAMPLED_HULL_DET_M_LOWER="
                << hull_diagnostic.determinant.lower
                << " NONRIGOROUS_SAMPLED_HULL_PD_SUFFICIENT="
                << (hull_diagnostic.positive_definite_sufficient ? "true"
                                                                  : "false")
                << std::setprecision(17) << "\n";
    }
    std::cout << "PHASE=" << phase
              << " GLOBAL_MIN_NORMALIZED_CONE_MARGIN="
              << evaluation.global_margin << "\n";
  };
  print_evaluation("discovery", discovery_edges, discovery);
  print_evaluation("holdout", holdout_edges, holdout);
  const bool cone_candidate = discovery.global_margin > 0.0 &&
                              holdout.global_margin > 0.0 &&
                              max_finite_difference_error < 1e-4 &&
                              sample_records_valid &&
                              sampled_hulls_pd_sufficient;
  const bool sampled_hyperbolicity_candidate =
      cone_candidate && determinant_resolved && rates_above_one;
  std::cout << "DISCOVERY_GLOBAL_MIN_NORMALIZED_CONE_MARGIN="
            << discovery.global_margin
            << "\nHOLDOUT_GLOBAL_MIN_NORMALIZED_CONE_MARGIN="
            << holdout.global_margin
            << "\nDOUBLE_PRECISION_INVERTIBILITY_RESOLVED="
            << (determinant_resolved ? "true" : "false")
            << "\nDOUBLE_DP_DETERMINANT_DIAGNOSTIC="
            << "cancellation-sensitive-residual"
            << "\nSAMPLED_DIRECTION_RATES_ABOVE_ONE="
            << (rates_above_one ? "true" : "false")
            << "\nNONRIGOROUS_SAMPLED_ENTRYWISE_HULL_PD_SUFFICIENT="
            << (sampled_hulls_pd_sufficient ? "true" : "false")
            << "\nNUMERICAL_CONE_CANDIDATE_FOUND="
            << (cone_candidate ? "true" : "false")
            << "\nNUMERICAL_HYPERBOLICITY_CANDIDATE_FOUND="
            << (sampled_hyperbolicity_candidate ? "true" : "false")
            << "\nSAMPLED_POSITIVE_DEFINITE_MATRIX_CANDIDATE_FOUND="
            << (cone_candidate ? "true" : "false")
            << "\nCANDIDATE_ONLY=true\n"
            << "PAIRWISE_CHORD_CONE_CONDITION_PROVED=false\n"
            << "TANGENT_CONE_CONDITION_PROVED=false\n"
            << "UNIFORM_HYPERBOLICITY_PROVED=false\n"
            << "CHAOTIC_ATTRACTOR_PROVED=false\n";
}

void usage(const char* program) {
  std::cerr << "usage: " << program << " selftest | scout [odd_grid=17]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(17) << std::boolalpha;
  try {
    const std::string mode = argc > 1 ? argv[1] : "selftest";
    if (mode == "selftest") {
      print_selftest();
    } else if (mode == "scout") {
      print_scout(argc > 2 ? parse_grid(argv[2]) : 17);
    } else {
      usage(argv[0]);
      return 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CS6_CONE_SCOUT_ERROR=" << error.what() << "\n";
    return 3;
  }
}
