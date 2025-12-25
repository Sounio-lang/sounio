// Demetrios Standard Library: Geometry Types
//
// Core geometric primitives for neuro-symbolic reasoning.
// Inspired by AlphaGeometry's representation but with native
// epistemic types, refinements, and ontology integration.
//
// All geometric objects carry Knowledge semantics:
// - Confidence in their existence/construction
// - Provenance tracking through proof chains
// - Ontology binding for semantic validation

import units::si::Length;
import units::angle::Angle;
import epistemic::Knowledge;
import epistemic::Confidence;
import epistemic::Source;

// =============================================================================
// Coordinate System
// =============================================================================

/// 2D coordinate with uncertainty
/// Uses Knowledge to track measurement/construction confidence
type Coord = Knowledge[f64];

/// A point in the Euclidean plane
///
/// Points are the fundamental primitives. They can be:
/// - Given (from problem statement) - high confidence, Source::Axiom
/// - Constructed (via intersections, etc.) - derived confidence
/// - Predicted (neural suggestion) - lower confidence, Source::ModelPrediction
@ontology("GEO:Point")
struct Point {
    x: Coord,
    y: Coord,

    /// Unique identifier for symbolic reasoning
    id: String,

    /// Whether this point is free (can vary) or constrained
    is_free: bool,
}

impl Point {
    /// Create a point from problem givens (axiomatic)
    fn given(id: String, x: f64, y: f64) -> Point {
        Point {
            x: Knowledge::axiom(x),
            y: Knowledge::axiom(y),
            id: id,
            is_free: false,
        }
    }

    /// Create a free point (for construction)
    fn free(id: String, x: f64, y: f64) -> Point {
        Point {
            x: Knowledge::uncertain(x, 0.9),
            y: Knowledge::uncertain(y, 0.9),
            id: id,
            is_free: true,
        }
    }

    /// Create a point from neural prediction
    fn predicted(id: String, x: f64, y: f64, confidence: f64) -> Point {
        Point {
            x: Knowledge::from_model("geometry_lm", x, confidence),
            y: Knowledge::from_model("geometry_lm", y, confidence),
            id: id,
            is_free: true,
        }
    }

    /// Euclidean distance to another point
    fn distance(self, other: Point) -> Knowledge[Length] {
        let dx = self.x.value - other.x.value;
        let dy = self.y.value - other.y.value;
        let dist = (dx * dx + dy * dy).sqrt();

        // Propagate epistemic status
        let combined_conf = self.x.confidence.min(self.y.confidence)
                           .min(other.x.confidence).min(other.y.confidence);

        Knowledge::derived(
            dist * Length::meter,
            combined_conf,
            Source::Derivation("euclidean_distance")
        )
    }

    /// Check if this point equals another (within tolerance)
    fn equals(self, other: Point, epsilon: f64) -> Knowledge[bool] {
        let dist = self.distance(other);
        let eq = dist.value.to_meters() < epsilon;

        Knowledge::derived(eq, dist.confidence, Source::Derivation("point_equality"))
    }
}

// =============================================================================
// Lines
// =============================================================================

/// A line in the plane (infinite extent)
/// Represented in general form: ax + by + c = 0
@ontology("GEO:Line")
struct Line {
    /// Coefficient a in ax + by + c = 0
    a: Coord,
    /// Coefficient b in ax + by + c = 0
    b: Coord,
    /// Coefficient c in ax + by + c = 0
    c: Coord,

    /// Identifier for symbolic reasoning
    id: String,
}

impl Line {
    /// Create line through two points
    fn through(p1: Point, p2: Point) -> Line {
        let a = p2.y.value - p1.y.value;
        let b = p1.x.value - p2.x.value;
        let c = p2.x.value * p1.y.value - p1.x.value * p2.y.value;

        let conf = p1.x.confidence.min(p1.y.confidence)
                   .min(p2.x.confidence).min(p2.y.confidence);

        Line {
            a: Knowledge::derived(a, conf, Source::Derivation("line_through")),
            b: Knowledge::derived(b, conf, Source::Derivation("line_through")),
            c: Knowledge::derived(c, conf, Source::Derivation("line_through")),
            id: format!("line_{}_{}", p1.id, p2.id),
        }
    }

    /// Create line with given slope through point
    fn with_slope(p: Point, slope: f64) -> Line {
        // y - y0 = m(x - x0) => mx - y + (y0 - m*x0) = 0
        let a = slope;
        let b = -1.0;
        let c = p.y.value - slope * p.x.value;

        let conf = p.x.confidence.min(p.y.confidence);

        Line {
            a: Knowledge::derived(a, conf, Source::Derivation("line_slope")),
            b: Knowledge::derived(b, conf, Source::Derivation("line_slope")),
            c: Knowledge::derived(c, conf, Source::Derivation("line_slope")),
            id: format!("line_slope_{}", p.id),
        }
    }

    /// Perpendicular line through a point
    fn perpendicular_through(self, p: Point) -> Line {
        // Perpendicular: swap a,b and negate one
        let new_a = self.b.value;
        let new_b = -self.a.value;
        let new_c = -new_a * p.x.value - new_b * p.y.value;

        let conf = self.a.confidence.min(self.b.confidence)
                   .min(p.x.confidence).min(p.y.confidence);

        Line {
            a: Knowledge::derived(new_a, conf, Source::Derivation("perpendicular")),
            b: Knowledge::derived(new_b, conf, Source::Derivation("perpendicular")),
            c: Knowledge::derived(new_c, conf, Source::Derivation("perpendicular")),
            id: format!("perp_{}_{}", self.id, p.id),
        }
    }

    /// Parallel line through a point
    fn parallel_through(self, p: Point) -> Line {
        // Parallel: same a,b, different c
        let new_c = -self.a.value * p.x.value - self.b.value * p.y.value;

        let conf = self.a.confidence.min(self.b.confidence)
                   .min(p.x.confidence).min(p.y.confidence);

        Line {
            a: Knowledge::derived(self.a.value, conf, Source::Derivation("parallel")),
            b: Knowledge::derived(self.b.value, conf, Source::Derivation("parallel")),
            c: Knowledge::derived(new_c, conf, Source::Derivation("parallel")),
            id: format!("para_{}_{}", self.id, p.id),
        }
    }

    /// Check if point lies on this line
    fn contains(self, p: Point) -> Knowledge[bool] {
        let val = self.a.value * p.x.value + self.b.value * p.y.value + self.c.value;
        let on_line = val.abs() < 1e-9;

        let conf = self.a.confidence.min(self.b.confidence).min(self.c.confidence)
                   .min(p.x.confidence).min(p.y.confidence);

        Knowledge::derived(on_line, conf, Source::Derivation("point_on_line"))
    }

    /// Distance from point to line
    fn distance_to(self, p: Point) -> Knowledge[Length] {
        let numer = (self.a.value * p.x.value + self.b.value * p.y.value + self.c.value).abs();
        let denom = (self.a.value * self.a.value + self.b.value * self.b.value).sqrt();
        let dist = numer / denom;

        let conf = self.a.confidence.min(self.b.confidence).min(self.c.confidence)
                   .min(p.x.confidence).min(p.y.confidence);

        Knowledge::derived(
            dist * Length::meter,
            conf,
            Source::Derivation("point_line_distance")
        )
    }
}

// =============================================================================
// Circles
// =============================================================================

/// A circle in the plane
@ontology("GEO:Circle")
struct Circle {
    /// Center point
    center: Point,
    /// Radius with unit
    radius: Knowledge[Length],
    /// Identifier
    id: String,
}

impl Circle {
    /// Create circle with center and radius
    fn new(center: Point, radius: Length) -> Circle {
        let conf = center.x.confidence.min(center.y.confidence);

        Circle {
            center: center,
            radius: Knowledge::derived(radius, conf, Source::Derivation("circle_new")),
            id: format!("circle_{}", center.id),
        }
    }

    /// Create circle through center and point on circumference
    fn through(center: Point, on_circle: Point) -> Circle {
        let r = center.distance(on_circle);

        Circle {
            center: center,
            radius: r,
            id: format!("circle_{}_{}", center.id, on_circle.id),
        }
    }

    /// Circumscribed circle of triangle (through 3 points)
    fn circumscribed(p1: Point, p2: Point, p3: Point) -> Option[Circle] {
        // Perpendicular bisector method
        let mid1 = Point::free(
            "mid12",
            (p1.x.value + p2.x.value) / 2.0,
            (p1.y.value + p2.y.value) / 2.0
        );
        let mid2 = Point::free(
            "mid23",
            (p2.x.value + p3.x.value) / 2.0,
            (p2.y.value + p3.y.value) / 2.0
        );

        let l1 = Line::through(p1, p2);
        let l2 = Line::through(p2, p3);

        let perp1 = l1.perpendicular_through(mid1);
        let perp2 = l2.perpendicular_through(mid2);

        // Find intersection of perpendicular bisectors
        match Line::intersect(perp1, perp2) {
            Some(center) => {
                let circle = Circle::through(center, p1);
                Some(circle)
            },
            None => None  // Collinear points
        }
    }

    /// Check if point lies on circle
    fn contains(self, p: Point) -> Knowledge[bool] {
        let dist = self.center.distance(p);
        let on_circle = (dist.value.to_meters() - self.radius.value.to_meters()).abs() < 1e-9;

        let conf = dist.confidence.min(self.radius.confidence);

        Knowledge::derived(on_circle, conf, Source::Derivation("point_on_circle"))
    }

    /// Check if point is inside circle
    fn contains_inside(self, p: Point) -> Knowledge[bool] {
        let dist = self.center.distance(p);
        let inside = dist.value.to_meters() < self.radius.value.to_meters();

        let conf = dist.confidence.min(self.radius.confidence);

        Knowledge::derived(inside, conf, Source::Derivation("point_inside_circle"))
    }
}

// =============================================================================
// Line Segment
// =============================================================================

/// A line segment (bounded between two points)
@ontology("GEO:LineSegment")
struct Segment {
    /// Start point
    start: Point,
    /// End point
    end: Point,
    /// Identifier
    id: String,
}

impl Segment {
    fn new(start: Point, end: Point) -> Segment {
        Segment {
            start: start,
            end: end,
            id: format!("seg_{}_{}", start.id, end.id),
        }
    }

    /// Length of segment
    fn length(self) -> Knowledge[Length] {
        self.start.distance(self.end)
    }

    /// Midpoint of segment
    fn midpoint(self) -> Point {
        let mx = (self.start.x.value + self.end.x.value) / 2.0;
        let my = (self.start.y.value + self.end.y.value) / 2.0;

        let conf = self.start.x.confidence.min(self.start.y.confidence)
                   .min(self.end.x.confidence).min(self.end.y.confidence);

        Point {
            x: Knowledge::derived(mx, conf, Source::Derivation("midpoint")),
            y: Knowledge::derived(my, conf, Source::Derivation("midpoint")),
            id: format!("mid_{}", self.id),
            is_free: false,
        }
    }

    /// Perpendicular bisector of segment
    fn perpendicular_bisector(self) -> Line {
        let mid = self.midpoint();
        let line = Line::through(self.start, self.end);
        line.perpendicular_through(mid)
    }

    /// Convert to infinite line
    fn to_line(self) -> Line {
        Line::through(self.start, self.end)
    }
}

// =============================================================================
// Angles
// =============================================================================

/// An angle formed by three points (vertex in middle)
@ontology("GEO:Angle")
struct AngleABC {
    /// First point (on one ray)
    a: Point,
    /// Vertex point
    b: Point,
    /// Third point (on other ray)
    c: Point,
    /// Identifier
    id: String,
}

impl AngleABC {
    fn new(a: Point, b: Point, c: Point) -> AngleABC {
        AngleABC {
            a: a,
            b: b,
            c: c,
            id: format!("angle_{}_{}{}", a.id, b.id, c.id),
        }
    }

    /// Measure of angle in radians
    fn measure(self) -> Knowledge[Angle] {
        let ba_x = self.a.x.value - self.b.x.value;
        let ba_y = self.a.y.value - self.b.y.value;
        let bc_x = self.c.x.value - self.b.x.value;
        let bc_y = self.c.y.value - self.b.y.value;

        let dot = ba_x * bc_x + ba_y * bc_y;
        let cross = ba_x * bc_y - ba_y * bc_x;

        let angle = cross.atan2(dot);

        let conf = self.a.x.confidence.min(self.a.y.confidence)
                   .min(self.b.x.confidence).min(self.b.y.confidence)
                   .min(self.c.x.confidence).min(self.c.y.confidence);

        Knowledge::derived(
            Angle::radians(angle.abs()),
            conf,
            Source::Derivation("angle_measure")
        )
    }

    /// Check if angle is right angle
    fn is_right(self) -> Knowledge[bool] {
        let measure = self.measure();
        let is_right = (measure.value.to_radians() - std::f64::consts::PI / 2.0).abs() < 1e-9;

        Knowledge::derived(is_right, measure.confidence, Source::Derivation("is_right_angle"))
    }

    /// Bisector ray from vertex
    fn bisector(self) -> Line {
        let measure = self.measure();
        let half = measure.value.to_radians() / 2.0;

        // Direction of BA
        let ba_x = self.a.x.value - self.b.x.value;
        let ba_y = self.a.y.value - self.b.y.value;
        let ba_len = (ba_x * ba_x + ba_y * ba_y).sqrt();

        // Normalize
        let ba_nx = ba_x / ba_len;
        let ba_ny = ba_y / ba_len;

        // Rotate by half angle
        let cos_h = half.cos();
        let sin_h = half.sin();
        let bis_x = ba_nx * cos_h - ba_ny * sin_h;
        let bis_y = ba_nx * sin_h + ba_ny * cos_h;

        // Point on bisector
        let bis_point = Point::free(
            format!("bis_{}", self.id),
            self.b.x.value + bis_x,
            self.b.y.value + bis_y
        );

        Line::through(self.b, bis_point)
    }
}

// =============================================================================
// Triangle
// =============================================================================

/// A triangle defined by three vertices
@ontology("GEO:Triangle")
struct Triangle {
    a: Point,
    b: Point,
    c: Point,
    id: String,
}

impl Triangle {
    fn new(a: Point, b: Point, c: Point) -> Triangle {
        Triangle {
            a: a,
            b: b,
            c: c,
            id: format!("tri_{}_{}{}", a.id, b.id, c.id),
        }
    }

    /// Side lengths
    fn sides(self) -> (Knowledge[Length], Knowledge[Length], Knowledge[Length]) {
        (
            self.b.distance(self.c),  // BC opposite to A
            self.a.distance(self.c),  // AC opposite to B
            self.a.distance(self.b),  // AB opposite to C
        )
    }

    /// Perimeter
    fn perimeter(self) -> Knowledge[Length] {
        let (a, b, c) = self.sides();
        let p = a.value + b.value + c.value;
        let conf = a.confidence.min(b.confidence).min(c.confidence);

        Knowledge::derived(p, conf, Source::Derivation("triangle_perimeter"))
    }

    /// Area using cross product
    fn area(self) -> Knowledge[f64] {
        let ab_x = self.b.x.value - self.a.x.value;
        let ab_y = self.b.y.value - self.a.y.value;
        let ac_x = self.c.x.value - self.a.x.value;
        let ac_y = self.c.y.value - self.a.y.value;

        let area = (ab_x * ac_y - ab_y * ac_x).abs() / 2.0;

        let conf = self.a.x.confidence.min(self.a.y.confidence)
                   .min(self.b.x.confidence).min(self.b.y.confidence)
                   .min(self.c.x.confidence).min(self.c.y.confidence);

        Knowledge::derived(area, conf, Source::Derivation("triangle_area"))
    }

    /// Centroid (intersection of medians)
    fn centroid(self) -> Point {
        let cx = (self.a.x.value + self.b.x.value + self.c.x.value) / 3.0;
        let cy = (self.a.y.value + self.b.y.value + self.c.y.value) / 3.0;

        let conf = self.a.x.confidence.min(self.a.y.confidence)
                   .min(self.b.x.confidence).min(self.b.y.confidence)
                   .min(self.c.x.confidence).min(self.c.y.confidence);

        Point {
            x: Knowledge::derived(cx, conf, Source::Derivation("centroid")),
            y: Knowledge::derived(cy, conf, Source::Derivation("centroid")),
            id: format!("centroid_{}", self.id),
            is_free: false,
        }
    }

    /// Circumcircle
    fn circumcircle(self) -> Option[Circle] {
        Circle::circumscribed(self.a, self.b, self.c)
    }

    /// Incircle center (incenter)
    fn incenter(self) -> Point {
        let (a_len, b_len, c_len) = self.sides();
        let a = a_len.value.to_meters();
        let b = b_len.value.to_meters();
        let c = c_len.value.to_meters();
        let p = a + b + c;

        let ix = (a * self.a.x.value + b * self.b.x.value + c * self.c.x.value) / p;
        let iy = (a * self.a.y.value + b * self.b.y.value + c * self.c.y.value) / p;

        let conf = a_len.confidence.min(b_len.confidence).min(c_len.confidence);

        Point {
            x: Knowledge::derived(ix, conf, Source::Derivation("incenter")),
            y: Knowledge::derived(iy, conf, Source::Derivation("incenter")),
            id: format!("incenter_{}", self.id),
            is_free: false,
        }
    }

    /// Orthocenter (intersection of altitudes)
    fn orthocenter(self) -> Point {
        // Altitude from A to BC
        let bc = Line::through(self.b, self.c);
        let alt_a = bc.perpendicular_through(self.a);

        // Altitude from B to AC
        let ac = Line::through(self.a, self.c);
        let alt_b = ac.perpendicular_through(self.b);

        match Line::intersect(alt_a, alt_b) {
            Some(h) => h,
            None => self.a  // Degenerate case
        }
    }
}

// =============================================================================
// Intersection Operations
// =============================================================================

impl Line {
    /// Intersection of two lines
    fn intersect(l1: Line, l2: Line) -> Option[Point] {
        let det = l1.a.value * l2.b.value - l2.a.value * l1.b.value;

        if det.abs() < 1e-12 {
            return None;  // Parallel or coincident
        }

        let x = (l1.b.value * l2.c.value - l2.b.value * l1.c.value) / det;
        let y = (l2.a.value * l1.c.value - l1.a.value * l2.c.value) / det;

        let conf = l1.a.confidence.min(l1.b.confidence).min(l1.c.confidence)
                   .min(l2.a.confidence).min(l2.b.confidence).min(l2.c.confidence);

        Some(Point {
            x: Knowledge::derived(x, conf, Source::Derivation("line_intersection")),
            y: Knowledge::derived(y, conf, Source::Derivation("line_intersection")),
            id: format!("inter_{}_{}", l1.id, l2.id),
            is_free: false,
        })
    }
}

impl Circle {
    /// Intersection of line and circle (0, 1, or 2 points)
    fn intersect_line(self, line: Line) -> Vec[Point] {
        let a = line.a.value;
        let b = line.b.value;
        let c = line.c.value;
        let cx = self.center.x.value;
        let cy = self.center.y.value;
        let r = self.radius.value.to_meters();

        // Distance from center to line
        let dist = (a * cx + b * cy + c).abs() / (a * a + b * b).sqrt();

        if dist > r + 1e-9 {
            return vec![];  // No intersection
        }

        // Closest point on line to center
        let t = -(a * cx + b * cy + c) / (a * a + b * b);
        let px = cx + a * t;
        let py = cy + b * t;

        let conf = self.center.x.confidence.min(self.center.y.confidence)
                   .min(self.radius.confidence)
                   .min(line.a.confidence).min(line.b.confidence).min(line.c.confidence);

        if (dist - r).abs() < 1e-9 {
            // Tangent - one point
            return vec![Point {
                x: Knowledge::derived(px, conf, Source::Derivation("circle_line_tangent")),
                y: Knowledge::derived(py, conf, Source::Derivation("circle_line_tangent")),
                id: format!("tangent_{}_{}", self.id, line.id),
                is_free: false,
            }];
        }

        // Two intersection points
        let half_chord = (r * r - dist * dist).sqrt();
        let norm = (a * a + b * b).sqrt();
        let dx = -b / norm * half_chord;
        let dy = a / norm * half_chord;

        vec![
            Point {
                x: Knowledge::derived(px + dx, conf, Source::Derivation("circle_line_inter")),
                y: Knowledge::derived(py + dy, conf, Source::Derivation("circle_line_inter")),
                id: format!("inter1_{}_{}", self.id, line.id),
                is_free: false,
            },
            Point {
                x: Knowledge::derived(px - dx, conf, Source::Derivation("circle_line_inter")),
                y: Knowledge::derived(py - dy, conf, Source::Derivation("circle_line_inter")),
                id: format!("inter2_{}_{}", self.id, line.id),
                is_free: false,
            },
        ]
    }

    /// Intersection of two circles (0, 1, or 2 points)
    fn intersect_circle(self, other: Circle) -> Vec[Point] {
        let c1x = self.center.x.value;
        let c1y = self.center.y.value;
        let r1 = self.radius.value.to_meters();

        let c2x = other.center.x.value;
        let c2y = other.center.y.value;
        let r2 = other.radius.value.to_meters();

        let dx = c2x - c1x;
        let dy = c2y - c1y;
        let d = (dx * dx + dy * dy).sqrt();

        // Check if circles are too far or one contains the other
        if d > r1 + r2 + 1e-9 || d < (r1 - r2).abs() - 1e-9 {
            return vec![];
        }

        let conf = self.center.x.confidence.min(self.center.y.confidence)
                   .min(self.radius.confidence)
                   .min(other.center.x.confidence).min(other.center.y.confidence)
                   .min(other.radius.confidence);

        // Distance from c1 to line through intersection points
        let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);

        // Point on line between centers at distance a from c1
        let px = c1x + a * dx / d;
        let py = c1y + a * dy / d;

        if (d - (r1 + r2)).abs() < 1e-9 || (d - (r1 - r2).abs()).abs() < 1e-9 {
            // Tangent - one point
            return vec![Point {
                x: Knowledge::derived(px, conf, Source::Derivation("circle_tangent")),
                y: Knowledge::derived(py, conf, Source::Derivation("circle_tangent")),
                id: format!("tangent_{}_{}", self.id, other.id),
                is_free: false,
            }];
        }

        // Two intersection points
        let h = (r1 * r1 - a * a).sqrt();

        vec![
            Point {
                x: Knowledge::derived(px + h * dy / d, conf, Source::Derivation("circle_inter")),
                y: Knowledge::derived(py - h * dx / d, conf, Source::Derivation("circle_inter")),
                id: format!("inter1_{}_{}", self.id, other.id),
                is_free: false,
            },
            Point {
                x: Knowledge::derived(px - h * dy / d, conf, Source::Derivation("circle_inter")),
                y: Knowledge::derived(py + h * dx / d, conf, Source::Derivation("circle_inter")),
                id: format!("inter2_{}_{}", self.id, other.id),
                is_free: false,
            },
        ]
    }
}

// =============================================================================
// Refinement Types for Geometric Constraints
// =============================================================================

/// A valid triangle (non-degenerate)
type ValidTriangle = { Triangle | not_collinear(t.a, t.b, t.c) };

/// An equilateral triangle
type EquilateralTriangle = {
    Triangle |
    eq_length(t.a.distance(t.b), t.b.distance(t.c)) &&
    eq_length(t.b.distance(t.c), t.c.distance(t.a))
};

/// An isoceles triangle (at least two equal sides)
type IsoscelesTriangle = {
    Triangle |
    eq_length(t.a.distance(t.b), t.b.distance(t.c)) ||
    eq_length(t.b.distance(t.c), t.c.distance(t.a)) ||
    eq_length(t.c.distance(t.a), t.a.distance(t.b))
};

/// A right triangle
type RightTriangle = {
    Triangle |
    is_right_angle(t.a, t.b, t.c) ||
    is_right_angle(t.b, t.c, t.a) ||
    is_right_angle(t.c, t.a, t.b)
};

/// Parallel lines
type ParallelLines = { (Line, Line) | is_parallel(l1, l2) };

/// Perpendicular lines
type PerpendicularLines = { (Line, Line) | is_perpendicular(l1, l2) };

/// Collinear points
type CollinearPoints = { (Point, Point, Point) | is_collinear(p1, p2, p3) };

/// Concyclic points (lie on same circle)
type ConcyclicPoints = { (Point, Point, Point, Point) | is_concyclic(p1, p2, p3, p4) };

// =============================================================================
// Helper Predicates for Refinements
// =============================================================================

fn not_collinear(a: Point, b: Point, c: Point) -> bool {
    let area = Triangle::new(a, b, c).area();
    area.value > 1e-9
}

fn eq_length(l1: Knowledge[Length], l2: Knowledge[Length]) -> bool {
    (l1.value.to_meters() - l2.value.to_meters()).abs() < 1e-9
}

fn is_right_angle(a: Point, vertex: Point, c: Point) -> bool {
    let angle = AngleABC::new(a, vertex, c);
    angle.is_right().value
}

fn is_parallel(l1: Line, l2: Line) -> bool {
    // Parallel if cross product of direction vectors is zero
    let cross = l1.a.value * l2.b.value - l2.a.value * l1.b.value;
    cross.abs() < 1e-12
}

fn is_perpendicular(l1: Line, l2: Line) -> bool {
    // Perpendicular if dot product of normal vectors is zero
    let dot = l1.a.value * l2.a.value + l1.b.value * l2.b.value;
    dot.abs() < 1e-12
}

fn is_collinear(p1: Point, p2: Point, p3: Point) -> bool {
    not not_collinear(p1, p2, p3)
}

fn is_concyclic(p1: Point, p2: Point, p3: Point, p4: Point) -> bool {
    // Four points are concyclic if the product of opposite sides equals
    // the sum of products of adjacent sides (Ptolemy's theorem for cyclic quad)
    // Or: check if p4 lies on circumcircle of p1, p2, p3
    match Circle::circumscribed(p1, p2, p3) {
        Some(circle) => circle.contains(p4).value,
        None => false,
    }
}
