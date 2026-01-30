//! IMO Problem Parser for AlphaGeometry Format
//!
//! Parses IMO geometry problems from AlphaGeometry's formalization format
//! into Sounio ProofState with epistemic priors.
//!
//! # AlphaGeometry Format
//!
//! The format uses simple declarations:
//! ```text
//! a b c = triangle a b c
//! o = circumcenter o a b c
//! d = on_circle d o a, on_line d a o
//! ? cong o b o d
//! ```
//!
//! Each line declares constructions or constraints. The `?` line is the goal.

use std::collections::HashMap;

use crate::epistemic::bayesian::BetaConfidence;

use super::predicates::{Predicate, PredicateEpistemic, PredicateKind};
use super::proof_state::ProofState;

// =============================================================================
// Parser Types
// =============================================================================

/// A parsed IMO problem
#[derive(Debug, Clone)]
pub struct IMOProblem {
    /// Problem ID (e.g., "imo_2000_p1")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Year of the IMO
    pub year: u32,
    /// Problem number within the year
    pub problem_num: u32,
    /// Initial proof state with premises
    pub initial_state: ProofState,
    /// Goal predicate to prove
    pub goal: Predicate,
    /// Difficulty estimate (0.0 - 1.0)
    pub difficulty: f64,
    /// Original AlphaGeometry format text
    pub original_text: String,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Token types for parsing
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Identifier(String),
    Equals,
    Comma,
    Question,
    Semicolon,
    Newline,
    Eof,
}

/// Parser for AlphaGeometry format
pub struct AGParser {
    /// Input text
    input: String,
    /// Current position
    pos: usize,
    /// Parsed points
    points: HashMap<String, PointInfo>,
    /// Current problem state
    state: ProofState,
    /// Goal predicate (if found)
    goal: Option<Predicate>,
}

/// Information about a parsed point
#[derive(Debug, Clone)]
struct PointInfo {
    name: String,
    construction: Option<AGConstruction>,
    constraints: Vec<AGConstraint>,
}

/// Parsed construction type
#[derive(Debug, Clone)]
enum AGConstruction {
    Free,
    Triangle(String, String, String),
    Circumcenter(String, String, String),
    Incenter(String, String, String),
    Centroid(String, String, String),
    Orthocenter(String, String, String),
    Midpoint(String, String),
    OnLine(String, String),
    OnCircle(String, String),
    OnTLine(String, String, String), // on tangent line
    OnPLine(String, String, String), // on perpendicular line
    OnBLine(String, String),         // on bisector
    Foot(String, String, String),    // foot of perpendicular
    Mirror(String, String, String),  // reflection
    Intersection(Vec<AGConstraint>),
    EqTriangle(String, String),              // equilateral triangle vertex
    IsoTriangle(String, String),             // isoceles triangle vertex
    EqAngle(String, String, String, String), // equal angle construction
    Ratio(String, String, String, f64),      // ratio point
}

/// Parsed constraint type
#[derive(Debug, Clone)]
enum AGConstraint {
    OnLine(String, String),
    OnCircle(String, String),
    Collinear(String, String, String),
    Perpendicular(String, String, String, String),
    Parallel(String, String, String, String),
    EqualLength(String, String, String, String),
    EqualAngle(String, String, String, String, String, String),
    Cyclic(Vec<String>),
    Tangent(String, String, String),
}

impl AGParser {
    /// Create a new parser
    pub fn new(input: &str) -> Self {
        AGParser {
            input: input.to_string(),
            pos: 0,
            points: HashMap::new(),
            state: ProofState::new(),
            goal: None,
        }
    }

    /// Parse the input into a ProofState and goal
    pub fn parse(&mut self) -> Result<(ProofState, Predicate), ParseError> {
        self.parse_declarations()?;

        let goal = self
            .goal
            .clone()
            .ok_or_else(|| ParseError::new("No goal found (missing '?' line)"))?;

        Ok((self.state.clone(), goal))
    }

    /// Parse all declarations
    fn parse_declarations(&mut self) -> Result<(), ParseError> {
        while self.pos < self.input.len() {
            self.skip_whitespace();
            if self.pos >= self.input.len() {
                break;
            }

            let line = self.read_line();
            if line.trim().is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }

            self.parse_line(&line)?;
        }
        Ok(())
    }

    /// Parse a single line
    fn parse_line(&mut self, line: &str) -> Result<(), ParseError> {
        let line = line.trim();

        // Goal line
        if line.starts_with('?') {
            return self.parse_goal(line[1..].trim());
        }

        // Declaration line: "names = construction constraints"
        if let Some(eq_pos) = line.find('=') {
            let names_part = line[..eq_pos].trim();
            let def_part = line[eq_pos + 1..].trim();

            let names: Vec<&str> = names_part.split_whitespace().collect();
            self.parse_definition(&names, def_part)?;
        } else {
            // Pure constraint line
            self.parse_constraint_line(line)?;
        }

        Ok(())
    }

    /// Parse a definition (names = construction, constraints)
    fn parse_definition(&mut self, names: &[&str], definition: &str) -> Result<(), ParseError> {
        // Split by comma for multiple constraints
        let parts: Vec<&str> = definition.split(',').map(|s| s.trim()).collect();

        if parts.is_empty() {
            return Err(ParseError::new("Empty definition"));
        }

        // First part is usually the main construction
        let main_construct = parts[0];
        let tokens: Vec<&str> = main_construct.split_whitespace().collect();

        if tokens.is_empty() {
            return Err(ParseError::new("Empty construction"));
        }

        let construct_type = tokens[0].to_lowercase();

        // Add points first
        for name in names {
            if !self.state.points.contains_key(*name) {
                self.state.add_point(*name);
            }
        }

        // Parse based on construction type
        match construct_type.as_str() {
            "triangle" => {
                // a b c = triangle a b c
                if names.len() >= 3 {
                    // Just marks them as free points in a triangle
                    self.add_triangle_predicates(names[0], names[1], names[2]);
                }
            }
            "circumcenter" | "circumcentre" => {
                // o = circumcenter o a b c
                if tokens.len() >= 4 {
                    let (a, b, c) = (tokens[1], tokens[2], tokens[3]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    self.ensure_point(c);
                    if !names.is_empty() {
                        self.add_circumcenter(names[0], a, b, c);
                    }
                }
            }
            "incenter" | "incentre" => {
                if tokens.len() >= 4 {
                    let (a, b, c) = (tokens[1], tokens[2], tokens[3]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    self.ensure_point(c);
                    if !names.is_empty() {
                        self.add_incenter(names[0], a, b, c);
                    }
                }
            }
            "centroid" => {
                if tokens.len() >= 4 {
                    let (a, b, c) = (tokens[1], tokens[2], tokens[3]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    self.ensure_point(c);
                    if !names.is_empty() {
                        self.add_centroid(names[0], a, b, c);
                    }
                }
            }
            "orthocenter" | "orthocentre" => {
                if tokens.len() >= 4 {
                    let (a, b, c) = (tokens[1], tokens[2], tokens[3]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    self.ensure_point(c);
                    if !names.is_empty() {
                        self.add_orthocenter(names[0], a, b, c);
                    }
                }
            }
            "midpoint" => {
                // m = midpoint m a b
                if tokens.len() >= 3 {
                    let (a, b) = (tokens[1], tokens[2]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    if !names.is_empty() {
                        self.add_midpoint(names[0], a, b);
                    }
                }
            }
            "on_line" => {
                // p = on_line p a b
                if tokens.len() >= 3 {
                    let (a, b) = (tokens[1], tokens[2]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    if !names.is_empty() {
                        self.add_collinear(names[0], a, b);
                    }
                }
            }
            "on_circle" => {
                // p = on_circle p o a (on circle with center o through a)
                if tokens.len() >= 3 {
                    let (center, through) = (tokens[1], tokens[2]);
                    self.ensure_point(center);
                    self.ensure_point(through);
                    if !names.is_empty() {
                        self.add_on_circle(names[0], center, through);
                    }
                }
            }
            "on_tline" => {
                // tangent line: p = on_tline p a b c (tangent at a to circle through b, c)
                if tokens.len() >= 4 {
                    let (p, a, b) = (tokens[1], tokens[2], tokens[3]);
                    self.ensure_point(p);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    // For now, just mark as on a line perpendicular construction
                    if !names.is_empty() {
                        self.add_tangent_point(names[0], p, a, b);
                    }
                }
            }
            "on_pline" => {
                // perpendicular line: p = on_pline p a b c
                if tokens.len() >= 4 {
                    let (pt, a, b) = (tokens[1], tokens[2], tokens[3]);
                    self.ensure_point(pt);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    if !names.is_empty() {
                        self.add_perpendicular_point(names[0], pt, a, b);
                    }
                }
            }
            "on_bline" => {
                // bisector line: p = on_bline p a b
                if tokens.len() >= 3 {
                    let (a, b) = (tokens[1], tokens[2]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    if !names.is_empty() {
                        self.add_bisector_point(names[0], a, b);
                    }
                }
            }
            "foot" => {
                // f = foot f p a b (perpendicular foot from p to line ab)
                if tokens.len() >= 4 {
                    let (p, a, b) = (tokens[1], tokens[2], tokens[3]);
                    self.ensure_point(p);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    if !names.is_empty() {
                        self.add_foot(names[0], p, a, b);
                    }
                }
            }
            "mirror" | "reflect" => {
                // m = mirror m p a b (reflection of p over line ab)
                if tokens.len() >= 4 {
                    let (p, a, b) = (tokens[1], tokens[2], tokens[3]);
                    self.ensure_point(p);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    if !names.is_empty() {
                        self.add_reflection(names[0], p, a, b);
                    }
                }
            }
            "eq_triangle" | "eqtriangle" => {
                // c = eq_triangle c a b (third vertex of equilateral triangle)
                if tokens.len() >= 3 {
                    let (a, b) = (tokens[1], tokens[2]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    if !names.is_empty() {
                        self.add_equilateral_vertex(names[0], a, b);
                    }
                }
            }
            "iso_triangle" | "isotriangle" => {
                // c = iso_triangle c a b (apex of isoceles triangle)
                if tokens.len() >= 3 {
                    let (a, b) = (tokens[1], tokens[2]);
                    self.ensure_point(a);
                    self.ensure_point(b);
                    if !names.is_empty() {
                        self.add_isoceles_apex(names[0], a, b);
                    }
                }
            }
            "segment" => {
                // Just declares two points form a segment
                if tokens.len() >= 3 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                }
            }
            "angle" => {
                // angle declaration (usually just informational)
                if tokens.len() >= 4 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                    self.ensure_point(tokens[3]);
                }
            }
            "free" => {
                // Free point - already added above
            }
            _ => {
                // Unknown construction - try to parse as constraint
                self.parse_constraint(main_construct)?;
            }
        }

        // Parse additional constraints
        for constraint in parts.iter().skip(1) {
            self.parse_constraint(constraint)?;
        }

        Ok(())
    }

    /// Parse a constraint
    fn parse_constraint(&mut self, constraint: &str) -> Result<(), ParseError> {
        let tokens: Vec<&str> = constraint.split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(());
        }

        let ctype = tokens[0].to_lowercase();

        match ctype.as_str() {
            "coll" | "collinear" => {
                if tokens.len() >= 4 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                    self.ensure_point(tokens[3]);
                    self.add_collinear(tokens[1], tokens[2], tokens[3]);
                }
            }
            "perp" | "perpendicular" => {
                if tokens.len() >= 5 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                    self.ensure_point(tokens[3]);
                    self.ensure_point(tokens[4]);
                    self.add_perpendicular(tokens[1], tokens[2], tokens[3], tokens[4]);
                }
            }
            "para" | "parallel" => {
                if tokens.len() >= 5 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                    self.ensure_point(tokens[3]);
                    self.ensure_point(tokens[4]);
                    self.add_parallel(tokens[1], tokens[2], tokens[3], tokens[4]);
                }
            }
            "cong" | "congruent" | "eqlen" | "eq_len" => {
                if tokens.len() >= 5 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                    self.ensure_point(tokens[3]);
                    self.ensure_point(tokens[4]);
                    self.add_equal_length(tokens[1], tokens[2], tokens[3], tokens[4]);
                }
            }
            "eqangle" | "eq_angle" => {
                if tokens.len() >= 7 {
                    for i in 1..7 {
                        self.ensure_point(tokens[i]);
                    }
                    self.add_equal_angle(
                        tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6],
                    );
                }
            }
            "cyclic" | "concyclic" => {
                if tokens.len() >= 5 {
                    for t in &tokens[1..] {
                        self.ensure_point(t);
                    }
                    self.add_cyclic(&tokens[1..]);
                }
            }
            "on_line" => {
                if tokens.len() >= 4 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                    self.ensure_point(tokens[3]);
                    self.add_collinear(tokens[1], tokens[2], tokens[3]);
                }
            }
            "on_circle" => {
                if tokens.len() >= 4 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                    self.ensure_point(tokens[3]);
                    self.add_on_circle(tokens[1], tokens[2], tokens[3]);
                }
            }
            "midp" | "midpoint" => {
                if tokens.len() >= 4 {
                    self.ensure_point(tokens[1]);
                    self.ensure_point(tokens[2]);
                    self.ensure_point(tokens[3]);
                    self.add_midpoint(tokens[1], tokens[2], tokens[3]);
                }
            }
            _ => {
                // Unknown constraint - ignore for now
            }
        }

        Ok(())
    }

    /// Parse a constraint line (without names =)
    fn parse_constraint_line(&mut self, line: &str) -> Result<(), ParseError> {
        self.parse_constraint(line)
    }

    /// Parse the goal predicate
    fn parse_goal(&mut self, goal_str: &str) -> Result<(), ParseError> {
        let tokens: Vec<&str> = goal_str.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(ParseError::new("Empty goal"));
        }

        let gtype = tokens[0].to_lowercase();

        let goal = match gtype.as_str() {
            "cong" | "eqlen" | "eq_len" => {
                if tokens.len() >= 5 {
                    Predicate::equal_length(tokens[1], tokens[2], tokens[3], tokens[4])
                } else {
                    return Err(ParseError::new("cong needs 4 points"));
                }
            }
            "perp" | "perpendicular" => {
                if tokens.len() >= 5 {
                    Predicate::perpendicular(tokens[1], tokens[2], tokens[3], tokens[4])
                } else {
                    return Err(ParseError::new("perp needs 4 points"));
                }
            }
            "para" | "parallel" => {
                if tokens.len() >= 5 {
                    Predicate::parallel(tokens[1], tokens[2], tokens[3], tokens[4])
                } else {
                    return Err(ParseError::new("para needs 4 points"));
                }
            }
            "coll" | "collinear" => {
                if tokens.len() >= 4 {
                    Predicate::collinear(tokens[1], tokens[2], tokens[3])
                } else {
                    return Err(ParseError::new("coll needs 3 points"));
                }
            }
            "cyclic" | "concyclic" => {
                if tokens.len() >= 5 {
                    Predicate::concyclic(tokens[1], tokens[2], tokens[3], tokens[4])
                } else {
                    return Err(ParseError::new("cyclic needs 4 points"));
                }
            }
            "eqangle" | "eq_angle" => {
                if tokens.len() >= 7 {
                    Predicate::new(
                        PredicateKind::EqualAngle,
                        vec![
                            tokens[1].to_string(),
                            tokens[2].to_string(),
                            tokens[3].to_string(),
                            tokens[4].to_string(),
                            tokens[5].to_string(),
                            tokens[6].to_string(),
                        ],
                    )
                } else {
                    return Err(ParseError::new("eqangle needs 6 points"));
                }
            }
            "midp" | "midpoint" => {
                if tokens.len() >= 4 {
                    Predicate::midpoint(tokens[1], tokens[2], tokens[3])
                } else {
                    return Err(ParseError::new("midp needs 3 points"));
                }
            }
            "simtri" | "similar" => {
                if tokens.len() >= 7 {
                    Predicate::new(
                        PredicateKind::Similar,
                        vec![
                            tokens[1].to_string(),
                            tokens[2].to_string(),
                            tokens[3].to_string(),
                            tokens[4].to_string(),
                            tokens[5].to_string(),
                            tokens[6].to_string(),
                        ],
                    )
                } else {
                    return Err(ParseError::new("simtri needs 6 points"));
                }
            }
            "contri" | "congruent_tri" => {
                if tokens.len() >= 7 {
                    Predicate::new(
                        PredicateKind::Congruent,
                        vec![
                            tokens[1].to_string(),
                            tokens[2].to_string(),
                            tokens[3].to_string(),
                            tokens[4].to_string(),
                            tokens[5].to_string(),
                            tokens[6].to_string(),
                        ],
                    )
                } else {
                    return Err(ParseError::new("contri needs 6 points"));
                }
            }
            _ => {
                return Err(ParseError::new(&format!("Unknown goal type: {}", gtype)));
            }
        };

        self.goal = Some(goal);
        Ok(())
    }

    // Helper methods to add predicates with epistemic priors

    fn ensure_point(&mut self, name: &str) {
        if !self.state.points.contains_key(name) {
            self.state.add_point(name);
        }
    }

    fn axiom_epistemic() -> PredicateEpistemic {
        PredicateEpistemic {
            confidence: BetaConfidence::new(100.0, 1.0), // High confidence premise
            source: crate::epistemic::Source::Axiom,
            revisability: crate::epistemic::Revisability::NonRevisable,
            depth: 0,
            derived_from: vec![],
            merkle_hash: None,
            parent_hashes: vec![],
        }
    }

    fn add_triangle_predicates(&mut self, a: &str, b: &str, c: &str) {
        // No specific predicate, just ensures points exist
        self.ensure_point(a);
        self.ensure_point(b);
        self.ensure_point(c);
    }

    fn add_collinear(&mut self, a: &str, b: &str, c: &str) {
        let pred = Predicate::collinear(a, b, c).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_perpendicular(&mut self, a: &str, b: &str, c: &str, d: &str) {
        let pred = Predicate::perpendicular(a, b, c, d).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_parallel(&mut self, a: &str, b: &str, c: &str, d: &str) {
        let pred = Predicate::parallel(a, b, c, d).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_equal_length(&mut self, a: &str, b: &str, c: &str, d: &str) {
        let pred = Predicate::equal_length(a, b, c, d).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_equal_angle(&mut self, a: &str, b: &str, c: &str, d: &str, e: &str, f: &str) {
        let pred = Predicate::new(
            PredicateKind::EqualAngle,
            vec![
                a.to_string(),
                b.to_string(),
                c.to_string(),
                d.to_string(),
                e.to_string(),
                f.to_string(),
            ],
        )
        .with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_cyclic(&mut self, points: &[&str]) {
        if points.len() >= 4 {
            let pred = Predicate::concyclic(points[0], points[1], points[2], points[3])
                .with_epistemic(Self::axiom_epistemic());
            self.state.add_axiom(pred);
        }
    }

    fn add_midpoint(&mut self, mid: &str, a: &str, b: &str) {
        let pred = Predicate::midpoint(mid, a, b).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_circumcenter(&mut self, o: &str, a: &str, b: &str, c: &str) {
        // Circumcenter: equidistant from all vertices
        let pred1 = Predicate::equal_length(o, a, o, b).with_epistemic(Self::axiom_epistemic());
        let pred2 = Predicate::equal_length(o, b, o, c).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred1);
        self.state.add_axiom(pred2);

        // Also add on_circle predicates
        let pred3 = Predicate::on_circle(a, o, a).with_epistemic(Self::axiom_epistemic());
        let pred4 = Predicate::on_circle(b, o, a).with_epistemic(Self::axiom_epistemic());
        let pred5 = Predicate::on_circle(c, o, a).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred3);
        self.state.add_axiom(pred4);
        self.state.add_axiom(pred5);
    }

    fn add_incenter(&mut self, i: &str, a: &str, b: &str, c: &str) {
        // Incenter lies on angle bisectors
        // For now, just mark the construction source
        let pred = Predicate::new(
            PredicateKind::Incenter,
            vec![i.to_string(), a.to_string(), b.to_string(), c.to_string()],
        )
        .with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_centroid(&mut self, g: &str, a: &str, b: &str, c: &str) {
        let pred = Predicate::new(
            PredicateKind::Centroid,
            vec![g.to_string(), a.to_string(), b.to_string(), c.to_string()],
        )
        .with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_orthocenter(&mut self, h: &str, a: &str, b: &str, c: &str) {
        let pred = Predicate::new(
            PredicateKind::Orthocenter,
            vec![h.to_string(), a.to_string(), b.to_string(), c.to_string()],
        )
        .with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_on_circle(&mut self, p: &str, center: &str, through: &str) {
        let pred = Predicate::on_circle(p, center, through).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_tangent_point(&mut self, p: &str, tangent_at: &str, a: &str, b: &str) {
        // Point on tangent line at tangent_at to circle through a, b
        // This implies perpendicularity to the radius
        let pred = Predicate::perpendicular(p, tangent_at, tangent_at, a)
            .with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_perpendicular_point(&mut self, p: &str, through: &str, a: &str, b: &str) {
        // Point on perpendicular to line ab through point
        let pred =
            Predicate::perpendicular(p, through, a, b).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
        // Also collinear with through on the perpendicular
        self.add_collinear(p, through, p);
    }

    fn add_bisector_point(&mut self, p: &str, a: &str, b: &str) {
        // Point on perpendicular bisector of ab
        let pred = Predicate::equal_length(p, a, p, b).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_foot(&mut self, f: &str, p: &str, a: &str, b: &str) {
        // Foot of perpendicular from p to line ab
        self.add_collinear(f, a, b);
        let pred = Predicate::perpendicular(p, f, a, b).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_reflection(&mut self, m: &str, p: &str, a: &str, b: &str) {
        // m is reflection of p over line ab
        // Midpoint of pm is on ab, and pm is perpendicular to ab
        // Create implicit midpoint
        let mid = format!("_mid_{}_{}", p, m);
        self.ensure_point(&mid);
        self.add_midpoint(&mid, p, m);
        self.add_collinear(&mid, a, b);
        let pred = Predicate::perpendicular(p, m, a, b).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn add_equilateral_vertex(&mut self, c: &str, a: &str, b: &str) {
        // c is third vertex of equilateral triangle abc
        let pred1 = Predicate::equal_length(a, b, b, c).with_epistemic(Self::axiom_epistemic());
        let pred2 = Predicate::equal_length(b, c, c, a).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred1);
        self.state.add_axiom(pred2);
    }

    fn add_isoceles_apex(&mut self, c: &str, a: &str, b: &str) {
        // c is apex of isoceles triangle with base ab
        let pred = Predicate::equal_length(c, a, c, b).with_epistemic(Self::axiom_epistemic());
        self.state.add_axiom(pred);
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            let c = self.input.chars().nth(self.pos).unwrap();
            if c == ' ' || c == '\t' || c == '\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn read_line(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input.chars().nth(self.pos).unwrap();
            if c == '\n' {
                self.pos += 1;
                break;
            }
            self.pos += 1;
        }
        self.input[start..self.pos].trim_end().to_string()
    }
}

// =============================================================================
// Parse Error
// =============================================================================

/// Error during parsing
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

impl ParseError {
    pub fn new(message: &str) -> Self {
        ParseError {
            message: message.to_string(),
            line: None,
            column: None,
        }
    }

    pub fn with_location(message: &str, line: usize, column: usize) -> Self {
        ParseError {
            message: message.to_string(),
            line: Some(line),
            column: Some(column),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let (Some(line), Some(col)) = (self.line, self.column) {
            write!(f, "Parse error at {}:{}: {}", line, col, self.message)
        } else {
            write!(f, "Parse error: {}", self.message)
        }
    }
}

impl std::error::Error for ParseError {}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Parse a single problem from AlphaGeometry format text
pub fn parse_ag_problem(text: &str) -> Result<(ProofState, Predicate), ParseError> {
    let mut parser = AGParser::new(text);
    parser.parse()
}

/// Parse a problem and wrap it in an IMOProblem struct
pub fn parse_imo_problem(
    id: &str,
    name: &str,
    year: u32,
    problem_num: u32,
    text: &str,
) -> Result<IMOProblem, ParseError> {
    let (state, goal) = parse_ag_problem(text)?;

    Ok(IMOProblem {
        id: id.to_string(),
        name: name.to_string(),
        year,
        problem_num,
        initial_state: state,
        goal,
        difficulty: estimate_difficulty(year, problem_num),
        original_text: text.to_string(),
        tags: vec![],
    })
}

/// Estimate difficulty based on problem position
fn estimate_difficulty(year: u32, problem_num: u32) -> f64 {
    // Problems 1, 4 are easier, 3, 6 are harder
    match problem_num {
        1 | 4 => 0.3,
        2 | 5 => 0.5,
        3 | 6 => 0.8,
        _ => 0.5,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_triangle() {
        let text = r#"
a b c = triangle a b c
? coll a b c
"#;
        let result = parse_ag_problem(text);
        // This should fail since a b c in a triangle aren't collinear
        // But parsing should succeed
        assert!(result.is_ok());
        let (state, goal) = result.unwrap();
        assert_eq!(state.points.len(), 3);
    }

    #[test]
    fn test_parse_midpoint() {
        let text = r#"
a b = segment a b
m = midpoint m a b
? cong a m m b
"#;
        let result = parse_ag_problem(text);
        assert!(result.is_ok());
        let (state, goal) = result.unwrap();
        assert!(state.points.contains_key("m"));
    }

    #[test]
    fn test_parse_circumcenter() {
        let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
? cong o a o b
"#;
        let result = parse_ag_problem(text);
        assert!(result.is_ok());
        let (state, _goal) = result.unwrap();
        assert!(state.points.contains_key("o"));
        // Should have equal_length predicates
        assert!(state.num_predicates() > 0);
    }

    #[test]
    fn test_parse_with_constraints() {
        let text = r#"
a b c = triangle a b c
d = on_line d b c
e = midpoint e a d
? para b c a e
"#;
        let result = parse_ag_problem(text);
        assert!(result.is_ok());
        let (state, goal) = result.unwrap();
        assert_eq!(state.points.len(), 5);
    }

    #[test]
    fn test_parse_perpendicular_goal() {
        let text = r#"
a b c = triangle a b c
m = midpoint m b c
? perp a m b c
"#;
        let result = parse_ag_problem(text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_cyclic() {
        let text = r#"
a b c d = free a b c d
? cyclic a b c d
"#;
        let result = parse_ag_problem(text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_imo_problem_creation() {
        let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
i = incenter i a b c
d = on_circle d o a, on_line d a i
? eqangle o b d c b d
"#;
        let result = parse_imo_problem("imo_2000_p1", "IMO 2000 Problem 1", 2000, 1, text);
        assert!(result.is_ok());
        let problem = result.unwrap();
        assert_eq!(problem.year, 2000);
        assert_eq!(problem.difficulty, 0.3); // Problem 1 = easier
    }
}
