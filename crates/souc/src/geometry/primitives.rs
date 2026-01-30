//! Geometric Primitives
//!
//! Core geometric objects with labels for symbolic reasoning.
//! These are the atoms of the predicate graph.

use std::fmt;

/// A labeled point in the plane
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Point {
    /// Unique label (e.g., "A", "B", "M")
    pub label: String,
    /// Whether this point is free (can vary) or constructed
    pub is_free: bool,
    /// Construction method if not free
    pub construction: Option<PointConstruction>,
}

impl Point {
    /// Create a free point (given in problem)
    pub fn free(label: impl Into<String>) -> Self {
        Point {
            label: label.into(),
            is_free: true,
            construction: None,
        }
    }

    /// Create a constructed point
    pub fn constructed(label: impl Into<String>, construction: PointConstruction) -> Self {
        Point {
            label: label.into(),
            is_free: false,
            construction: Some(construction),
        }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// How a point was constructed
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum PointConstruction {
    /// Intersection of two lines
    LineLineIntersection(String, String),
    /// Intersection of line and circle
    LineCircleIntersection(String, String, usize), // line, circle, which intersection (0 or 1)
    /// Intersection of two circles
    CircleCircleIntersection(String, String, usize),
    /// Midpoint of segment
    Midpoint(String, String),
    /// Foot of perpendicular from point to line
    PerpendicularFoot(String, String, String),
    /// Reflection of point over line
    Reflection(String, String, String),
    /// Center of circumcircle
    Circumcenter(String, String, String),
    /// Center of incircle
    Incenter(String, String, String),
    /// Centroid of triangle
    Centroid(String, String, String),
    /// Orthocenter of triangle
    Orthocenter(String, String, String),
}

/// A line defined by two points
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Line {
    /// First point on line
    pub p1: String,
    /// Second point on line
    pub p2: String,
}

impl Line {
    pub fn new(p1: impl Into<String>, p2: impl Into<String>) -> Self {
        let p1 = p1.into();
        let p2 = p2.into();
        // Canonical ordering for equality
        if p1 <= p2 {
            Line { p1, p2 }
        } else {
            Line { p1: p2, p2: p1 }
        }
    }

    /// Check if point is on this line (by label, symbolic)
    pub fn contains(&self, point: &str) -> bool {
        self.p1 == point || self.p2 == point
    }
}

impl fmt::Display for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line({}{})", self.p1, self.p2)
    }
}

/// A circle defined by center and point on circumference
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Circle {
    /// Center point
    pub center: String,
    /// A point on the circle (defines radius)
    pub on_circle: String,
    /// Optional label
    pub label: Option<String>,
}

impl Circle {
    pub fn new(center: impl Into<String>, on_circle: impl Into<String>) -> Self {
        Circle {
            center: center.into(),
            on_circle: on_circle.into(),
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl fmt::Display for Circle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref label) = self.label {
            write!(f, "{}(center={}, r={})", label, self.center, self.on_circle)
        } else {
            write!(f, "circle({}, {})", self.center, self.on_circle)
        }
    }
}

/// A line segment (bounded)
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Segment {
    /// First endpoint
    pub p1: String,
    /// Second endpoint
    pub p2: String,
}

impl Segment {
    pub fn new(p1: impl Into<String>, p2: impl Into<String>) -> Self {
        let p1 = p1.into();
        let p2 = p2.into();
        // Canonical ordering
        if p1 <= p2 {
            Segment { p1, p2 }
        } else {
            Segment { p1: p2, p2: p1 }
        }
    }
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.p1, self.p2)
    }
}

/// An angle defined by three points (vertex in middle)
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Angle {
    /// Point on first ray
    pub p1: String,
    /// Vertex
    pub vertex: String,
    /// Point on second ray
    pub p2: String,
}

impl Angle {
    pub fn new(p1: impl Into<String>, vertex: impl Into<String>, p2: impl Into<String>) -> Self {
        Angle {
            p1: p1.into(),
            vertex: vertex.into(),
            p2: p2.into(),
        }
    }

    /// Canonical form (p1 < p2 lexicographically)
    pub fn canonical(&self) -> Self {
        if self.p1 <= self.p2 {
            self.clone()
        } else {
            Angle {
                p1: self.p2.clone(),
                vertex: self.vertex.clone(),
                p2: self.p1.clone(),
            }
        }
    }
}

impl fmt::Display for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "∠{}{}{}", self.p1, self.vertex, self.p2)
    }
}

/// Union type for all geometry primitives
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum GeometryPrimitive {
    Point(Point),
    Line(Line),
    Circle(Circle),
    Segment(Segment),
    Angle(Angle),
}

impl GeometryPrimitive {
    /// Get all point labels referenced by this primitive
    pub fn referenced_points(&self) -> Vec<&str> {
        match self {
            GeometryPrimitive::Point(p) => vec![&p.label],
            GeometryPrimitive::Line(l) => vec![&l.p1, &l.p2],
            GeometryPrimitive::Circle(c) => vec![&c.center, &c.on_circle],
            GeometryPrimitive::Segment(s) => vec![&s.p1, &s.p2],
            GeometryPrimitive::Angle(a) => vec![&a.p1, &a.vertex, &a.p2],
        }
    }
}

impl fmt::Display for GeometryPrimitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeometryPrimitive::Point(p) => write!(f, "{}", p),
            GeometryPrimitive::Line(l) => write!(f, "{}", l),
            GeometryPrimitive::Circle(c) => write!(f, "{}", c),
            GeometryPrimitive::Segment(s) => write!(f, "{}", s),
            GeometryPrimitive::Angle(a) => write!(f, "{}", a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_canonical() {
        let l1 = Line::new("A", "B");
        let l2 = Line::new("B", "A");
        assert_eq!(l1, l2);
    }

    #[test]
    fn test_segment_canonical() {
        let s1 = Segment::new("X", "Y");
        let s2 = Segment::new("Y", "X");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_point_construction() {
        let p = Point::constructed("M", PointConstruction::Midpoint("A".into(), "B".into()));
        assert!(!p.is_free);
        assert!(p.construction.is_some());
    }
}
