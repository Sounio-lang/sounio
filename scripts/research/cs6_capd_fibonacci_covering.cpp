#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "capd/capdlib.h"

using capd::C0HOTripletonSet;
using capd::ICoordinateSection;
using capd::IMap;
using capd::IMatrix;
using capd::IOdeSolver;
using capd::IPoincareMap;
using capd::IVector;
using capd::interval;

namespace {

constexpr int kReturns = 6;

interval decimal(const char* value) { return interval(value, value); }

struct HSet {
  const char* name;
  interval center_u;
  interval center_s;
  interval radius_u;
  interval radius_s;
};

struct LocalImage {
  interval u;
  interval s;
  interval initial_normal_velocity;
  interval normal_velocity;
  interval return_time;
  double physical_diameter;
};

struct RawImage {
  interval u;
  interval s;
  interval initial_normal_velocity;
  interval normal_velocity;
  interval return_time;
  double physical_diameter;
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

struct PredicateStats {
  std::size_t expected = 0;
  std::size_t processed = 0;
  std::size_t passed = 0;
  double min_margin = std::numeric_limits<double>::infinity();
  double min_return_time = std::numeric_limits<double>::infinity();
  double max_return_time = -std::numeric_limits<double>::infinity();
  double min_initial_normal_velocity = std::numeric_limits<double>::infinity();
  double min_normal_velocity = std::numeric_limits<double>::infinity();
  double max_physical_diameter = 0.0;
  bool first_failure_printed = false;
};

struct EdgeStats {
  const HSet* source;
  const HSet* target;
  int degree;
  PredicateStats support;
  PredicateStats left_exit;
  PredicateStats right_exit;
};

class Ledger {
 public:
  explicit Ledger(const char* path) : enabled_(path != nullptr) {
    if (enabled_) {
      output_.open(path, std::ios::out | std::ios::trunc);
      if (!output_) {
        throw std::runtime_error(std::string("cannot open ledger: ") + path);
      }
      output_ << std::setprecision(17) << std::boolalpha;
    }
  }

  void record(const EdgeStats& edge, const char* role, int u_index,
              int s_index, const interval& source_u, const interval& source_s,
              const LocalImage& image, double margin, bool pass) {
    if (!enabled_) {
      return;
    }
    output_ << "EDGE=" << edge.source->name << "->" << edge.target->name
            << " ROLE=" << role << " U_INDEX=" << u_index
            << " S_INDEX=" << s_index << " SOURCE_U=" << source_u
            << " SOURCE_S=" << source_s << " IMAGE_U=" << image.u
            << " IMAGE_S=" << image.s
            << " INITIAL_NORMAL_VELOCITY=" << image.initial_normal_velocity
            << " NORMAL_VELOCITY=" << image.normal_velocity
            << " RETURN_TIME=" << image.return_time
            << " PHYSICAL_DIAMETER=" << image.physical_diameter
            << " MARGIN=" << margin << " PASS=" << pass << "\n";
    ++records_;
  }

  void record_exception(const EdgeStats& edge, const char* role, int u_index,
                        int s_index, const interval& source_u,
                        const interval& source_s) {
    if (!enabled_) {
      return;
    }
    output_ << "EDGE=" << edge.source->name << "->" << edge.target->name
            << " ROLE=" << role << " U_INDEX=" << u_index
            << " S_INDEX=" << s_index << " SOURCE_U=" << source_u
            << " SOURCE_S=" << source_s << " PASS=false EXCEPTION=true\n";
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
        determinant_(unstable_x_ * stable_y_ - stable_x_ * unstable_y_),
        vector_field_(
            "par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,"
            "x*y-w-zs;"),
        solver_(vector_field_, order),
        section_(3, 2),
        poincare_(solver_, section_, capd::poincare::MinusPlus),
        basis_(3, 3) {
    vector_field_.setParameter("zs", zs_);
    if (determinant_.contains(0.0)) {
      throw std::runtime_error("frozen h-set frame is not rigorously invertible");
    }
    basis_[0][0] = unstable_x_;
    basis_[1][0] = unstable_y_;
    basis_[2][0] = 0.0;
    basis_[0][1] = stable_x_;
    basis_[1][1] = stable_y_;
    basis_[2][1] = 0.0;
    basis_[0][2] = 0.0;
    basis_[1][2] = 0.0;
    basis_[2][2] = 1.0;
  }

  RawImage image(const interval& source_u, const interval& source_s) {
    const interval midpoint_u = source_u.mid();
    const interval midpoint_s = source_s.mid();
    const interval radius_u = source_u - midpoint_u;
    const interval radius_s = source_s - midpoint_s;
    IVector center{
        origin_x_ + unstable_x_ * midpoint_u + stable_x_ * midpoint_s,
        origin_y_ + unstable_y_ * midpoint_u + stable_y_ * midpoint_s,
        interval(0.0)};
    IVector radii{radius_u, radius_s, interval(0.0)};
    const interval initial_x =
        origin_x_ + unstable_x_ * source_u + stable_x_ * source_s;
    const interval initial_y =
        origin_y_ + unstable_y_ * source_u + stable_y_ * source_s;
    const interval initial_normal_velocity = initial_x * initial_y - zs_;
    C0HOTripletonSet set(center, basis_, radii);
    interval return_time;
    const IVector physical = poincare_(set, return_time, kReturns);

    const interval dx = physical[0] - origin_x_;
    const interval dy = physical[1] - origin_y_;
    const interval local_u =
        (stable_y_ * dx - stable_x_ * dy) / determinant_;
    const interval local_s =
        (-unstable_y_ * dx + unstable_x_ * dy) / determinant_;
    const interval normal_velocity =
        physical[0] * physical[1] - physical[2] - zs_;
    const double diameter =
        std::max(physical[0].rightBound() - physical[0].leftBound(),
                 physical[1].rightBound() - physical[1].leftBound());
    return {local_u, local_s, initial_normal_velocity, normal_velocity,
            return_time, diameter};
  }

  const interval& determinant() const { return determinant_; }
  const interval& section_z() const { return zs_; }
  const interval& origin_x() const { return origin_x_; }
  const interval& origin_y() const { return origin_y_; }
  const interval& unstable_x() const { return unstable_x_; }
  const interval& unstable_y() const { return unstable_y_; }
  const interval& stable_x() const { return stable_x_; }
  const interval& stable_y() const { return stable_y_; }

 private:
  interval zs_;
  interval origin_x_;
  interval origin_y_;
  interval unstable_x_;
  interval unstable_y_;
  interval stable_x_;
  interval stable_y_;
  interval determinant_;
  IMap vector_field_;
  IOdeSolver solver_;
  ICoordinateSection section_;
  IPoincareMap poincare_;
  IMatrix basis_;
};

LocalImage normalize(const RawImage& image, const HSet& target) {
  return {(image.u - target.center_u) / target.radius_u,
          (image.s - target.center_s) / target.radius_s,
          image.initial_normal_velocity,
          image.normal_velocity,
          image.return_time,
          image.physical_diameter};
}

int positive_int(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 1 ||
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

double inside_margin(const interval& value) {
  return std::min(value.leftBound() + 1.0, 1.0 - value.rightBound());
}

double outside_margin(const interval& value) {
  if (value.rightBound() < -1.0) {
    return -1.0 - value.rightBound();
  }
  if (value.leftBound() > 1.0) {
    return value.leftBound() - 1.0;
  }
  return -std::numeric_limits<double>::infinity();
}

// CAPD HSet2D::across: the full image avoids the target entry boundary.
// Being outside in U or strictly inside the stable strip is sufficient.
double across_margin(const LocalImage& image) {
  return std::max(outside_margin(image.u), inside_margin(image.s));
}

bool finite_interval(const interval& value) {
  return std::isfinite(value.leftBound()) &&
         std::isfinite(value.rightBound());
}

double exit_margin(const interval& image_u, int degree, bool left_face) {
  if (degree == 1) {
    return left_face ? -1.0 - image_u.rightBound()
                     : image_u.leftBound() - 1.0;
  }
  return left_face ? image_u.leftBound() - 1.0
                   : -1.0 - image_u.rightBound();
}

void record(PredicateStats& stats, const LocalImage& image, double margin,
            const EdgeStats& edge, const char* role, int u_index, int s_index,
            const interval& source_u, const interval& source_s,
            Ledger& ledger) {
  ++stats.processed;
  stats.min_margin = std::min(stats.min_margin, margin);
  stats.min_return_time =
      std::min(stats.min_return_time, image.return_time.leftBound());
  stats.max_return_time =
      std::max(stats.max_return_time, image.return_time.rightBound());
  stats.min_initial_normal_velocity = std::min(
      stats.min_initial_normal_velocity,
      image.initial_normal_velocity.leftBound());
  stats.min_normal_velocity = std::min(
      stats.min_normal_velocity, image.normal_velocity.leftBound());
  stats.max_physical_diameter =
      std::max(stats.max_physical_diameter, image.physical_diameter);
  const bool pass =
      margin > 0.0 && finite_interval(image.return_time) &&
      image.return_time.leftBound() > 0.0 &&
      finite_interval(image.initial_normal_velocity) &&
      image.initial_normal_velocity.leftBound() > 0.0 &&
      finite_interval(image.normal_velocity) &&
      image.normal_velocity.leftBound() > 0.0 &&
      std::isfinite(image.physical_diameter);
  if (pass) {
    ++stats.passed;
  } else if (!stats.first_failure_printed) {
    stats.first_failure_printed = true;
    std::cout << "FIRST_FAILURE EDGE=" << edge.source->name << "->"
              << edge.target->name << " ROLE=" << role
              << " U_INDEX=" << u_index << " S_INDEX=" << s_index
              << " MARGIN=" << margin << " IMAGE_U=" << image.u
              << " IMAGE_S=" << image.s
              << " INITIAL_NORMAL_VELOCITY="
              << image.initial_normal_velocity
              << " NORMAL_VELOCITY=" << image.normal_velocity
              << " RETURN_TIME=" << image.return_time << "\n";
  }
  ledger.record(edge, role, u_index, s_index, source_u, source_s, image, margin,
                pass);
}

void record_exception(PredicateStats& stats, const EdgeStats& edge,
                      const char* role, int u_index, int s_index,
                      const interval& source_u, const interval& source_s,
                      const std::exception& error, Ledger& ledger) {
  ++stats.processed;
  stats.min_margin = -std::numeric_limits<double>::infinity();
  if (!stats.first_failure_printed) {
    stats.first_failure_printed = true;
    std::cout << "FIRST_EXCEPTION EDGE=" << edge.source->name << "->"
              << edge.target->name << " ROLE=" << role
              << " U_INDEX=" << u_index << " S_INDEX=" << s_index
              << " MESSAGE=" << error.what() << "\n";
  }
  ledger.record_exception(edge, role, u_index, s_index, source_u, source_s);
}

void check_supports(ProofContext& context, const HSet& source,
                    std::initializer_list<EdgeStats*> edges, int u_tiles,
                    int s_tiles, const TileSelector& selector, Ledger& ledger) {
  const std::size_t total = static_cast<std::size_t>(u_tiles) * s_tiles;
  for (EdgeStats* edge : edges) {
    edge->support.expected = selector.expected(total);
  }
  for (int u_index = 0; u_index < u_tiles; ++u_index) {
    const interval source_u =
        tile(source.center_u, source.radius_u, u_index, u_tiles);
    for (int s_index = 0; s_index < s_tiles; ++s_index) {
      const std::size_t linear =
          static_cast<std::size_t>(u_index) * s_tiles + s_index;
      if (!selector.owns(linear)) {
        continue;
      }
      const interval source_s =
          tile(source.center_s, source.radius_s, s_index, s_tiles);
      try {
        const RawImage raw = context.image(source_u, source_s);
        for (EdgeStats* edge : edges) {
          const LocalImage image = normalize(raw, *edge->target);
          record(edge->support, image, across_margin(image), *edge, "support",
                 u_index, s_index, source_u, source_s, ledger);
        }
      } catch (const std::exception& error) {
        for (EdgeStats* edge : edges) {
          record_exception(edge->support, *edge, "support", u_index, s_index,
                           source_u, source_s, error, ledger);
        }
      }
    }
  }
}

void check_exits(ProofContext& context, const HSet& source,
                 std::initializer_list<EdgeStats*> edges, bool left_face,
                 int s_tiles, const TileSelector& selector, Ledger& ledger) {
  for (EdgeStats* edge : edges) {
    PredicateStats& stats = left_face ? edge->left_exit : edge->right_exit;
    stats.expected = selector.expected(s_tiles);
  }
  const interval source_u =
      left_face ? source.center_u - source.radius_u
                : source.center_u + source.radius_u;
  for (int s_index = 0; s_index < s_tiles; ++s_index) {
    if (!selector.owns(s_index)) {
      continue;
    }
    const interval source_s =
        tile(source.center_s, source.radius_s, s_index, s_tiles);
    try {
      const RawImage raw = context.image(source_u, source_s);
      for (EdgeStats* edge : edges) {
        PredicateStats& stats =
            left_face ? edge->left_exit : edge->right_exit;
        const LocalImage image = normalize(raw, *edge->target);
        record(stats, image, exit_margin(image.u, edge->degree, left_face),
               *edge, left_face ? "left_exit" : "right_exit", 0, s_index,
               source_u, source_s, ledger);
      }
    } catch (const std::exception& error) {
      for (EdgeStats* edge : edges) {
        PredicateStats& stats =
            left_face ? edge->left_exit : edge->right_exit;
        record_exception(stats, *edge,
                         left_face ? "left_exit" : "right_exit", 0, s_index,
                         source_u, source_s, error, ledger);
      }
    }
  }
}

void print_stats(const EdgeStats& edge, const char* role,
                 const PredicateStats& stats) {
  std::cout << "EDGE=" << edge.source->name << "->" << edge.target->name
            << " DEGREE=" << edge.degree << " ROLE=" << role
            << " EXPECTED=" << stats.expected
            << " PROCESSED=" << stats.processed << " PASS=" << stats.passed
            << " MIN_MARGIN=" << stats.min_margin << " RETURN_TIME=["
            << stats.min_return_time << "," << stats.max_return_time
            << "] MIN_INITIAL_NORMAL_VELOCITY="
            << stats.min_initial_normal_velocity
            << " MIN_NORMAL_VELOCITY=" << stats.min_normal_velocity
            << " MAX_PHYSICAL_DIAMETER=" << stats.max_physical_diameter
            << "\n";
}

bool complete(const PredicateStats& stats) {
  if (stats.expected == 0) {
    return stats.processed == 0 && stats.passed == 0;
  }
  return stats.expected == stats.processed && stats.processed == stats.passed &&
         stats.min_margin > 0.0 && stats.min_return_time > 0.0 &&
         std::isfinite(stats.max_return_time) &&
         stats.min_initial_normal_velocity > 0.0 &&
         stats.min_normal_velocity > 0.0 &&
         std::isfinite(stats.max_physical_diameter);
}

bool complete(const EdgeStats& edge) {
  return complete(edge.support) && complete(edge.left_exit) &&
         complete(edge.right_exit);
}

bool rigorously_disjoint(const HSet& left, const HSet& right) {
  return (left.center_u + left.radius_u).rightBound() <
         (right.center_u - right.radius_u).leftBound();
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(17) << std::boolalpha;
  try {
    const bool probe_mode = argc > 1 && std::string(argv[1]) == "probe";
    const int n0_u_tiles =
        !probe_mode && argc > 1 ? positive_int(argv[1], "n0_u_tiles") : 200;
    const int n1_u_tiles =
        !probe_mode && argc > 2 ? positive_int(argv[2], "n1_u_tiles") : 75;
    const int support_s_tiles = !probe_mode && argc > 3
                                    ? positive_int(argv[3], "support_s_tiles")
                                    : 75;
    const int exit_s_tiles = !probe_mode && argc > 4
                                 ? positive_int(argv[4], "exit_s_tiles")
                                 : 1200;
    const int order = !probe_mode && argc > 5
                          ? positive_int(argv[5], "order")
                          : (probe_mode && argc > 7
                                 ? positive_int(argv[7], "order")
                                 : 8);
    const int shard_ordinal =
        !probe_mode && argc > 6 ? positive_int(argv[6], "shard_ordinal") : 1;
    const int shard_count =
        !probe_mode && argc > 7 ? positive_int(argv[7], "shard_count") : 1;
    if (!probe_mode && argc <= 8) {
      throw std::runtime_error(
          "proof usage: n0_u n1_u support_s exit_s order shard_ordinal "
          "shard_count ledger_path");
    }
    const char* ledger_path = !probe_mode ? argv[8] : nullptr;
    if (shard_ordinal > shard_count) {
      throw std::runtime_error("shard_ordinal must be <= shard_count");
    }
    const TileSelector selector{shard_ordinal - 1, shard_count};
    Ledger ledger(ledger_path);

    ProofContext context(order);
    const HSet n0{"N0", interval(0.0), interval(0.0), decimal("0.004"),
                  decimal("0.3")};
    const HSet n1{"N1", decimal("0.019771776972779206"), interval(0.0),
                  decimal("0.0015"), decimal("0.3")};
    EdgeStats n0_n0{&n0, &n0, -1};
    EdgeStats n0_n1{&n0, &n1, -1};
    EdgeStats n1_n0{&n1, &n0, 1};

    if (probe_mode) {
      if (argc < 7) {
        throw std::runtime_error(
            "probe usage: probe source_u source_s radius_u radius_s N0|N1 "
            "[order]");
      }
      const interval source_u = decimal(argv[2]);
      const interval source_s = decimal(argv[3]);
      const interval radius_u = decimal(argv[4]);
      const interval radius_s = decimal(argv[5]);
      const HSet& target = std::string(argv[6]) == "N1" ? n1 : n0;
      const RawImage raw =
          context.image(source_u + interval(-1.0, 1.0) * radius_u,
                        source_s + interval(-1.0, 1.0) * radius_s);
      const LocalImage image = normalize(raw, target);
      std::cout << "PROBE_TARGET=" << target.name << " IMAGE_U=" << image.u
                << " IMAGE_S=" << image.s
                << " SUPPORT_MARGIN=" << across_margin(image)
                << " INITIAL_NORMAL_VELOCITY="
                << image.initial_normal_velocity
                << " NORMAL_VELOCITY=" << image.normal_velocity
                << " RETURN_TIME=" << image.return_time
                << " PHYSICAL_DIAMETER=" << image.physical_diameter << "\n";
      return across_margin(image) > 0.0 &&
                     image.initial_normal_velocity.leftBound() > 0.0 &&
                     image.normal_velocity.leftBound() > 0.0 &&
                     image.return_time.leftBound() > 0.0 &&
                     finite_interval(image.return_time)
                 ? EXIT_SUCCESS
                 : 2;
    }

    std::cout << "CAPD_SOURCE_TREE_DECLARED=capd-5.3.0\n"
                 "INTERVAL_BACKEND_DECLARED=FILIB\n"
                 "MAP=P^6\nSECTION_ORIENTATION=MinusPlus\nORDER="
              << order << "\n";
    std::cout << "ZSEC=" << context.section_z() << "\nORIGIN={"
              << context.origin_x() << "," << context.origin_y() << "}\n";
    std::cout << "UNSTABLE={" << context.unstable_x() << ","
              << context.unstable_y() << "}\nSTABLE={" << context.stable_x()
              << "," << context.stable_y() << "}\nFRAME_DETERMINANT="
              << context.determinant() << "\n";
    std::cout << "N0_LOCAL={" << n0.center_u << "," << n0.center_s << ","
              << n0.radius_u << "," << n0.radius_s << "}\n";
    std::cout << "N1_LOCAL={" << n1.center_u << "," << n1.center_s << ","
              << n1.radius_u << "," << n1.radius_s << "}\n";
    const bool disjoint = rigorously_disjoint(n0, n1);
    std::cout << "HSETS_DISJOINT=" << disjoint << "\n";
    std::cout << "FRAME_RIGOROUSLY_INVERTIBLE="
              << !context.determinant().contains(0.0) << "\n";
    std::cout << "GRID=N0_U:" << n0_u_tiles << ",N1_U:" << n1_u_tiles
              << ",SUPPORT_S:" << support_s_tiles
              << ",EXIT_S:" << exit_s_tiles << "\n";
    std::cout << "SHARD=" << shard_ordinal << "/" << shard_count << "\n";
    std::cout << "LEDGER_ENABLED=" << ledger.enabled() << "\n";

    check_supports(context, n0, {&n0_n0, &n0_n1}, n0_u_tiles,
                   support_s_tiles, selector, ledger);
    check_supports(context, n1, {&n1_n0}, n1_u_tiles, support_s_tiles,
                   selector, ledger);
    check_exits(context, n0, {&n0_n0, &n0_n1}, true, exit_s_tiles,
                selector, ledger);
    check_exits(context, n0, {&n0_n0, &n0_n1}, false, exit_s_tiles,
                selector, ledger);
    check_exits(context, n1, {&n1_n0}, true, exit_s_tiles, selector, ledger);
    check_exits(context, n1, {&n1_n0}, false, exit_s_tiles, selector, ledger);

    for (const EdgeStats* edge : {&n0_n0, &n0_n1, &n1_n0}) {
      print_stats(*edge, "support", edge->support);
      print_stats(*edge, "left_exit", edge->left_exit);
      print_stats(*edge, "right_exit", edge->right_exit);
      std::cout << "PARTITION_RELATION=" << edge->source->name << "->"
                << edge->target->name << " DEGREE=" << edge->degree
                << " COMPLETE=" << complete(*edge) << "\n";
    }
    const bool shard_pass =
        disjoint && complete(n0_n0) && complete(n0_n1) && complete(n1_n0);
    std::cout << "SHARD_PASS=" << shard_pass << "\n";
    std::cout << "LEDGER_RECORDS=" << ledger.records() << "\n";
    std::cout << "FIBONACCI_COVERINGS_PROVED=false\n";
    std::cout << "POSITIVE_ENTROPY_PROVED=false\n";
    std::cout << "UNIFORM_HYPERBOLICITY_PROVED=false\n";
    std::cout << "CHAOTIC_ATTRACTOR_PROVED=false\n";
    std::cout << "FLOW_ENTROPY_BOUND_PROVED=false\n";
    return shard_pass ? EXIT_SUCCESS : 2;
  } catch (const std::exception& error) {
    std::cerr << "CAPD_EXCEPTION=" << error.what() << "\n";
    return 3;
  }
}
