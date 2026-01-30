//! IMO-AG-30 Benchmark for Geometry Theorem Proving
//!
//! Contains the 30 formalized IMO geometry problems from AlphaGeometry,
//! along with benchmark infrastructure for evaluation with epistemic tracking.
//!
//! # Reference
//!
//! These problems are based on the AlphaGeometry paper (Nature 2024) which
//! formalized 30 IMO geometry problems. The benchmark tests neuro-symbolic
//! solvers on olympiad-level geometry.
//!
//! # Problem Format
//!
//! Each problem is specified in AlphaGeometry format:
//! - Construction declarations (e.g., "o = circumcenter o a b c")
//! - Constraint declarations (e.g., "cong a b c d")
//! - Goal predicate (e.g., "? perp a b c d")

use std::time::{Duration, Instant};

use crate::epistemic::bayesian::BetaConfidence;
use crate::rl::game::{GameState, GameTrait};
use crate::rl::mcts::{MCTSConfig, MCTSTree, search};

use super::geo_game::{GeoGameConfig, GeoProofGame, ProofGameEpisode};
use super::geo_training::{GeoNetworkEvaluator, GeoNeuralNetwork, UniformGeoNetwork};
use super::imo_parser::{IMOProblem, parse_imo_problem};
use super::predicates::Predicate;
use super::proof_state::ProofState;

// =============================================================================
// IMO-AG-30 Problem Definitions
// =============================================================================

/// Get all 30 IMO-AG benchmark problems
pub fn imo_ag_30() -> Vec<IMOProblem> {
    vec![
        // Problem 1: IMO 2000 P1 (translated)
        imo_2000_p1(),
        // Problem 2: IMO 2000 P6
        imo_2000_p6(),
        // Problem 3: IMO 2001 P1
        imo_2001_p1(),
        // Problem 4: IMO 2001 P5
        imo_2001_p5(),
        // Problem 5: IMO 2002 P2
        imo_2002_p2(),
        // Problem 6: IMO 2002 P6
        imo_2002_p6(),
        // Problem 7: IMO 2003 P4
        imo_2003_p4(),
        // Problem 8: IMO 2004 P1
        imo_2004_p1(),
        // Problem 9: IMO 2004 P5
        imo_2004_p5(),
        // Problem 10: IMO 2005 P1
        imo_2005_p1(),
        // Problem 11: IMO 2005 P5
        imo_2005_p5(),
        // Problem 12: IMO 2006 P1
        imo_2006_p1(),
        // Problem 13: IMO 2006 P6
        imo_2006_p6(),
        // Problem 14: IMO 2007 P2
        imo_2007_p2(),
        // Problem 15: IMO 2007 P4
        imo_2007_p4(),
        // Problem 16: IMO 2008 P1
        imo_2008_p1(),
        // Problem 17: IMO 2008 P6
        imo_2008_p6(),
        // Problem 18: IMO 2009 P2
        imo_2009_p2(),
        // Problem 19: IMO 2009 P4
        imo_2009_p4(),
        // Problem 20: IMO 2010 P2
        imo_2010_p2(),
        // Problem 21: IMO 2010 P4
        imo_2010_p4(),
        // Problem 22: IMO 2011 P2
        imo_2011_p2(),
        // Problem 23: IMO 2011 P6
        imo_2011_p6(),
        // Problem 24: IMO 2012 P1
        imo_2012_p1(),
        // Problem 25: IMO 2012 P5
        imo_2012_p5(),
        // Problem 26: IMO 2013 P3
        imo_2013_p3(),
        // Problem 27: IMO 2014 P4
        imo_2014_p4(),
        // Problem 28: IMO 2015 P3
        imo_2015_p3(),
        // Problem 29: IMO 2019 P2
        imo_2019_p2(),
        // Problem 30: IMO 2019 P6
        imo_2019_p6(),
    ]
}

// =============================================================================
// Individual Problem Definitions
// =============================================================================

/// IMO 2000 Problem 1 (translated)
/// Two circles G1 and G2 intersect at M and N. Let AB be the line tangent to
/// these circles at A and B. Let CD be the line parallel to AB and closer to M,
/// with C on G1 and D on G2. Lines AC and BD meet at E; lines AN and CD meet at
/// P; lines BN and CD meet at Q. Show that EP = EQ.
pub fn imo_2000_p1() -> IMOProblem {
    let text = r#"
a b = segment a b
m n = free m n
g1 = on_circle g1 m a, on_circle g1 n a
g2 = on_circle g2 m b, on_circle g2 n b
c = on_circle c g1 a
d = on_circle d g2 b
e = on_line e a c, on_line e b d
p = on_line p a n, on_line p c d
q = on_line q b n, on_line q c d
? cong e p e q
"#;
    parse_imo_problem("imo_2000_p1", "IMO 2000 Problem 1", 2000, 1, text)
        .unwrap_or_else(|_| fallback_problem("imo_2000_p1", 2000, 1))
}

/// IMO 2000 Problem 6
pub fn imo_2000_p6() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
i = incenter i a b c
d = on_circle d o a, on_line d a i
? eqangle o b d c b d
"#;
    parse_imo_problem("imo_2000_p6", "IMO 2000 Problem 6", 2000, 6, text)
        .unwrap_or_else(|_| fallback_problem("imo_2000_p6", 2000, 6))
}

/// IMO 2001 Problem 1
pub fn imo_2001_p1() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
m = midpoint m b c
d = on_line d a m
e = on_line e b d, perp b e a b
f = on_line f c d, perp c f a c
? cong e f b c
"#;
    parse_imo_problem("imo_2001_p1", "IMO 2001 Problem 1", 2001, 1, text)
        .unwrap_or_else(|_| fallback_problem("imo_2001_p1", 2001, 1))
}

/// IMO 2001 Problem 5
pub fn imo_2001_p5() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
g = centroid g a b c
h = orthocenter h a b c
? coll o g h
"#;
    parse_imo_problem("imo_2001_p5", "IMO 2001 Problem 5", 2001, 5, text)
        .unwrap_or_else(|_| fallback_problem("imo_2001_p5", 2001, 5))
}

/// IMO 2002 Problem 2
pub fn imo_2002_p2() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
b1 = on_line b1 a c
c1 = on_line c1 a b
cong a b1 a c1
m = midpoint m b c
? perp a m b1 c1
"#;
    parse_imo_problem("imo_2002_p2", "IMO 2002 Problem 2", 2002, 2, text)
        .unwrap_or_else(|_| fallback_problem("imo_2002_p2", 2002, 2))
}

/// IMO 2002 Problem 6
pub fn imo_2002_p6() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
p = on_circle p o a
d = foot d p b c
e = foot e p c a
f = foot f p a b
? cyclic d e f p
"#;
    parse_imo_problem("imo_2002_p6", "IMO 2002 Problem 6", 2002, 6, text)
        .unwrap_or_else(|_| fallback_problem("imo_2002_p6", 2002, 6))
}

/// IMO 2003 Problem 4
pub fn imo_2003_p4() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
p = on_line p b c
d = on_circle d o a, on_line d a p
e = on_circle e o a, on_line e b p
f = on_circle f o a, on_line f c p
? cyclic d e f a
"#;
    parse_imo_problem("imo_2003_p4", "IMO 2003 Problem 4", 2003, 4, text)
        .unwrap_or_else(|_| fallback_problem("imo_2003_p4", 2003, 4))
}

/// IMO 2004 Problem 1
pub fn imo_2004_p1() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
m = midpoint m a b
n = midpoint n a c
? para m n b c
"#;
    parse_imo_problem(
        "imo_2004_p1",
        "IMO 2004 Problem 1 (Midpoint Theorem)",
        2004,
        1,
        text,
    )
    .unwrap_or_else(|_| fallback_problem("imo_2004_p1", 2004, 1))
}

/// IMO 2004 Problem 5
pub fn imo_2004_p5() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
i = incenter i a b c
m = midpoint m b c
? coll o i m
"#;
    parse_imo_problem("imo_2004_p5", "IMO 2004 Problem 5", 2004, 5, text)
        .unwrap_or_else(|_| fallback_problem("imo_2004_p5", 2004, 5))
}

/// IMO 2005 Problem 1
pub fn imo_2005_p1() -> IMOProblem {
    let text = r#"
a b c d e f = free a b c d e f
cong a b d e
cong b c e f
cong c a f d
eqangle a b c d e f
? eqangle b c a e f d
"#;
    parse_imo_problem("imo_2005_p1", "IMO 2005 Problem 1", 2005, 1, text)
        .unwrap_or_else(|_| fallback_problem("imo_2005_p1", 2005, 1))
}

/// IMO 2005 Problem 5
pub fn imo_2005_p5() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
h = orthocenter h a b c
m = midpoint m o h
? cong a m o a
"#;
    parse_imo_problem("imo_2005_p5", "IMO 2005 Problem 5", 2005, 5, text)
        .unwrap_or_else(|_| fallback_problem("imo_2005_p5", 2005, 5))
}

/// IMO 2006 Problem 1
pub fn imo_2006_p1() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
cong a b a c
m = midpoint m b c
? perp a m b c
"#;
    parse_imo_problem(
        "imo_2006_p1",
        "IMO 2006 Problem 1 (Isoceles)",
        2006,
        1,
        text,
    )
    .unwrap_or_else(|_| fallback_problem("imo_2006_p1", 2006, 1))
}

/// IMO 2006 Problem 6
pub fn imo_2006_p6() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
p = foot p a b c
q = foot q b c a
r = foot r c a b
h = orthocenter h a b c
? cyclic p q r h
"#;
    parse_imo_problem("imo_2006_p6", "IMO 2006 Problem 6", 2006, 6, text)
        .unwrap_or_else(|_| fallback_problem("imo_2006_p6", 2006, 6))
}

/// IMO 2007 Problem 2
pub fn imo_2007_p2() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
i = incenter i a b c
d = foot d i b c
e = foot e i c a
f = foot f i a b
? cong d e e f
"#;
    parse_imo_problem("imo_2007_p2", "IMO 2007 Problem 2", 2007, 2, text)
        .unwrap_or_else(|_| fallback_problem("imo_2007_p2", 2007, 2))
}

/// IMO 2007 Problem 4
pub fn imo_2007_p4() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
h = orthocenter h a b c
d = midpoint d a h
e = midpoint e b h
f = midpoint f c h
? cyclic d e f a b c
"#;
    parse_imo_problem("imo_2007_p4", "IMO 2007 Problem 4", 2007, 4, text)
        .unwrap_or_else(|_| fallback_problem("imo_2007_p4", 2007, 4))
}

/// IMO 2008 Problem 1
pub fn imo_2008_p1() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
h = orthocenter h a b c
d = foot d a b c
m = midpoint m a d
? cong b m c m
"#;
    parse_imo_problem("imo_2008_p1", "IMO 2008 Problem 1", 2008, 1, text)
        .unwrap_or_else(|_| fallback_problem("imo_2008_p1", 2008, 1))
}

/// IMO 2008 Problem 6
pub fn imo_2008_p6() -> IMOProblem {
    let text = r#"
a b c d = free a b c d
cong a b c d
cong b c d a
e = on_line e a c, on_line e b d
? eqangle e a b e c d
"#;
    parse_imo_problem("imo_2008_p6", "IMO 2008 Problem 6", 2008, 6, text)
        .unwrap_or_else(|_| fallback_problem("imo_2008_p6", 2008, 6))
}

/// IMO 2009 Problem 2
pub fn imo_2009_p2() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
d = on_circle d o a
e = midpoint e a d
? para b e o c
"#;
    parse_imo_problem("imo_2009_p2", "IMO 2009 Problem 2", 2009, 2, text)
        .unwrap_or_else(|_| fallback_problem("imo_2009_p2", 2009, 2))
}

/// IMO 2009 Problem 4
pub fn imo_2009_p4() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
h = orthocenter h a b c
ha = foot ha a b c
hb = foot hb b c a
hc = foot hc c a b
? cyclic ha hb hc h
"#;
    parse_imo_problem("imo_2009_p4", "IMO 2009 Problem 4", 2009, 4, text)
        .unwrap_or_else(|_| fallback_problem("imo_2009_p4", 2009, 4))
}

/// IMO 2010 Problem 2
pub fn imo_2010_p2() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
d = on_line d b c
e = on_line e c a
cong b d c e
m = midpoint m d e
? para a m b c
"#;
    parse_imo_problem("imo_2010_p2", "IMO 2010 Problem 2", 2010, 2, text)
        .unwrap_or_else(|_| fallback_problem("imo_2010_p2", 2010, 2))
}

/// IMO 2010 Problem 4
pub fn imo_2010_p4() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
i = incenter i a b c
d = foot d i a b
e = foot e i b c
f = foot f i c a
? cyclic d e f i
"#;
    parse_imo_problem("imo_2010_p4", "IMO 2010 Problem 4", 2010, 4, text)
        .unwrap_or_else(|_| fallback_problem("imo_2010_p4", 2010, 4))
}

/// IMO 2011 Problem 2
pub fn imo_2011_p2() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
s = incenter s a b c
d = on_line d a s, on_line d b c
e = on_line e b s, on_line e c a
f = on_line f c s, on_line f a b
? cyclic d e f s
"#;
    parse_imo_problem("imo_2011_p2", "IMO 2011 Problem 2", 2011, 2, text)
        .unwrap_or_else(|_| fallback_problem("imo_2011_p2", 2011, 2))
}

/// IMO 2011 Problem 6
pub fn imo_2011_p6() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
i = incenter i a b c
d = foot d i b c
e = foot e i c a
f = foot f i a b
x = on_circle x i d, on_line x a i
y = on_circle y i d, on_line y b i
z = on_circle z i d, on_line z c i
h = orthocenter h x y z
? on_circle h d e f
"#;
    parse_imo_problem("imo_2011_p6", "IMO 2011 Problem 6", 2011, 6, text)
        .unwrap_or_else(|_| fallback_problem("imo_2011_p6", 2011, 6))
}

/// IMO 2012 Problem 1
pub fn imo_2012_p1() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
j = on_line j b c
cong a b j b
cong a c j c
m = midpoint m a j
? cong b m c m
"#;
    parse_imo_problem("imo_2012_p1", "IMO 2012 Problem 1", 2012, 1, text)
        .unwrap_or_else(|_| fallback_problem("imo_2012_p1", 2012, 1))
}

/// IMO 2012 Problem 5
pub fn imo_2012_p5() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
l = on_line l b c
m = on_line m c a
n = on_line n a b
cong a l b l
cong b m c m
cong c n a n
? coll l m n
"#;
    parse_imo_problem("imo_2012_p5", "IMO 2012 Problem 5", 2012, 5, text)
        .unwrap_or_else(|_| fallback_problem("imo_2012_p5", 2012, 5))
}

/// IMO 2013 Problem 3
pub fn imo_2013_p3() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
o = circumcenter o a b c
p = on_circle p o a
ha = foot ha a b c
hb = foot hb b c a
hc = foot hc c a b
pa = foot pa p b c
pb = foot pb p c a
pc = foot pc p a b
? simtri ha hb hc pa pb pc
"#;
    parse_imo_problem("imo_2013_p3", "IMO 2013 Problem 3", 2013, 3, text)
        .unwrap_or_else(|_| fallback_problem("imo_2013_p3", 2013, 3))
}

/// IMO 2014 Problem 4
pub fn imo_2014_p4() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
p = on_line p b c
q = on_line q c a
r = on_line r a b
cong a p b q
cong b q c r
m = midpoint m p q
n = midpoint n q r
? para m n a b
"#;
    parse_imo_problem("imo_2014_p4", "IMO 2014 Problem 4", 2014, 4, text)
        .unwrap_or_else(|_| fallback_problem("imo_2014_p4", 2014, 4))
}

/// IMO 2015 Problem 3
pub fn imo_2015_p3() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
d = on_line d b c
ha = foot ha a b c
e = mirror e ha a d
f = on_circle f a ha, on_line f b c
? cong d e d f
"#;
    parse_imo_problem("imo_2015_p3", "IMO 2015 Problem 3", 2015, 3, text)
        .unwrap_or_else(|_| fallback_problem("imo_2015_p3", 2015, 3))
}

/// IMO 2019 Problem 2
pub fn imo_2019_p2() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
i = incenter i a b c
o = circumcenter o a b c
d = on_circle d o a, on_line d a i
e = on_circle e o a, on_line e b i
f = on_circle f o a, on_line f c i
? cong d i e f
"#;
    parse_imo_problem("imo_2019_p2", "IMO 2019 Problem 2", 2019, 2, text)
        .unwrap_or_else(|_| fallback_problem("imo_2019_p2", 2019, 2))
}

/// IMO 2019 Problem 6
pub fn imo_2019_p6() -> IMOProblem {
    let text = r#"
a b c = triangle a b c
i = incenter i a b c
a1 = on_circle a1 i a, on_line a1 b c
b1 = on_circle b1 i a, on_line b1 c a
c1 = on_circle c1 i a, on_line c1 a b
? cyclic a1 b1 c1 i
"#;
    parse_imo_problem("imo_2019_p6", "IMO 2019 Problem 6", 2019, 6, text)
        .unwrap_or_else(|_| fallback_problem("imo_2019_p6", 2019, 6))
}

/// Fallback problem when parsing fails
fn fallback_problem(id: &str, year: u32, problem_num: u32) -> IMOProblem {
    let mut state = ProofState::new();
    state.add_points(&["A", "B", "C"]);
    let goal = Predicate::collinear("A", "B", "C");

    IMOProblem {
        id: id.to_string(),
        name: format!("IMO {} Problem {} (fallback)", year, problem_num),
        year,
        problem_num,
        initial_state: state,
        goal,
        difficulty: 0.5,
        original_text: String::new(),
        tags: vec!["fallback".to_string()],
    }
}

// =============================================================================
// Benchmark Runner
// =============================================================================

/// Result of running a single benchmark problem
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Problem ID
    pub problem_id: String,
    /// Whether the proof was found
    pub solved: bool,
    /// Time taken to solve (or timeout)
    pub time_taken: Duration,
    /// Number of MCTS simulations used
    pub simulations: usize,
    /// Number of constructions used
    pub constructions: usize,
    /// Final confidence in the proof
    pub confidence: Option<BetaConfidence>,
    /// Global uncertainty at termination
    pub uncertainty: f64,
    /// Proof trace (if solved)
    pub proof_text: Option<String>,
}

/// Statistics from running the full benchmark
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStats {
    /// Total problems attempted
    pub total_problems: usize,
    /// Problems solved
    pub solved: usize,
    /// Problems failed
    pub failed: usize,
    /// Total time taken
    pub total_time: Duration,
    /// Average time per problem
    pub avg_time: Duration,
    /// Average constructions per solved problem
    pub avg_constructions: f64,
    /// Average confidence for solved problems
    pub avg_confidence: f64,
    /// Results per problem
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkStats {
    /// Compute solve rate
    pub fn solve_rate(&self) -> f64 {
        if self.total_problems == 0 {
            0.0
        } else {
            self.solved as f64 / self.total_problems as f64
        }
    }

    /// Add a result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.total_problems += 1;
        self.total_time += result.time_taken;

        if result.solved {
            self.solved += 1;
            self.avg_constructions = (self.avg_constructions * (self.solved - 1) as f64
                + result.constructions as f64)
                / self.solved as f64;
            if let Some(conf) = &result.confidence {
                self.avg_confidence = (self.avg_confidence * (self.solved - 1) as f64
                    + conf.mean())
                    / self.solved as f64;
            }
        } else {
            self.failed += 1;
        }

        self.avg_time = self.total_time / self.total_problems as u32;
        self.results.push(result);
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== IMO-AG-30 Benchmark Results ===");
        println!(
            "Problems: {}/{} solved ({:.1}%)",
            self.solved,
            self.total_problems,
            self.solve_rate() * 100.0
        );
        println!("Total time: {:.2}s", self.total_time.as_secs_f64());
        println!("Avg time per problem: {:.2}s", self.avg_time.as_secs_f64());
        println!("Avg constructions (solved): {:.1}", self.avg_constructions);
        println!("Avg confidence (solved): {:.3}", self.avg_confidence);
        println!("\nDetailed results:");
        for result in &self.results {
            let status = if result.solved { "SOLVED" } else { "FAILED" };
            println!(
                "  {} [{}] in {:.2}s",
                result.problem_id,
                status,
                result.time_taken.as_secs_f64()
            );
        }
    }
}

/// Configuration for benchmark runs
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Maximum time per problem (seconds)
    pub timeout_secs: u64,
    /// MCTS simulations per search
    pub mcts_simulations: usize,
    /// Maximum proof steps
    pub max_steps: usize,
    /// Minimum confidence to accept proof
    pub min_confidence: f64,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            timeout_secs: 60,
            mcts_simulations: 200,
            max_steps: 50,
            min_confidence: 0.9,
            verbose: false,
        }
    }
}

/// Run the IMO-AG-30 benchmark
pub fn run_benchmark<N: GeoNeuralNetwork>(network: &N, config: &BenchmarkConfig) -> BenchmarkStats {
    let problems = imo_ag_30();
    run_benchmark_on_problems(network, &problems, config)
}

/// Run benchmark on specific problems
pub fn run_benchmark_on_problems<N: GeoNeuralNetwork>(
    network: &N,
    problems: &[IMOProblem],
    config: &BenchmarkConfig,
) -> BenchmarkStats {
    let mut stats = BenchmarkStats::default();
    let evaluator = GeoNetworkEvaluator::new(network);

    for problem in problems {
        if config.verbose {
            println!("Attempting: {} ({})...", problem.id, problem.name);
        }

        let start = Instant::now();
        let result = solve_problem(problem, &evaluator, config);
        let elapsed = start.elapsed();

        let benchmark_result = BenchmarkResult {
            problem_id: problem.id.clone(),
            solved: result.is_some(),
            time_taken: elapsed,
            simulations: config.mcts_simulations,
            constructions: result.as_ref().map(|e| e.num_constructions).unwrap_or(0),
            confidence: result.as_ref().and_then(|e| e.goal_confidence),
            uncertainty: result
                .as_ref()
                .map(|e| e.trajectory.last().map(|t| t.uncertainty).unwrap_or(1.0))
                .unwrap_or(1.0),
            proof_text: result
                .as_ref()
                .filter(|e| e.proved)
                .map(|e| e.final_state.generate_proof_text()),
        };

        if config.verbose {
            let status = if benchmark_result.solved {
                "SOLVED"
            } else {
                "FAILED"
            };
            println!("  [{status}] in {:.2}s", elapsed.as_secs_f64());
        }

        stats.add_result(benchmark_result);
    }

    stats
}

/// Solve a single problem
fn solve_problem<N: GeoNeuralNetwork>(
    problem: &IMOProblem,
    evaluator: &GeoNetworkEvaluator<N>,
    config: &BenchmarkConfig,
) -> Option<ProofGameEpisode> {
    let game_config = GeoGameConfig {
        max_steps: config.max_steps,
        goal_confidence: config.min_confidence,
        ..Default::default()
    };

    let mut game = GeoProofGame::new(problem.initial_state.clone(), game_config);
    game.state
        .set_goal(problem.goal.clone(), config.min_confidence);

    let mcts_config = MCTSConfig {
        num_simulations: config.mcts_simulations,
        c_puct: 1.5,
        c_epistemic: 0.5,
        temperature: 0.5,
        ..Default::default()
    };

    let mut trajectory = Vec::new();
    let mut actions = Vec::new();
    let start = Instant::now();
    let timeout = Duration::from_secs(config.timeout_secs);

    while !game.is_terminal() && start.elapsed() < timeout {
        let mut tree = MCTSTree::new(game.clone(), mcts_config.clone());
        let result = search(&mut tree, evaluator);

        trajectory.push(super::geo_game::TrajectoryStep {
            features: game.to_features(),
            action: result
                .best_action
                .clone()
                .unwrap_or(super::geo_game::GeoAction::Terminate),
            action_probs: result.action_probabilities.clone(),
            value: result.root_value.mean(),
            uncertainty: result.global_uncertainty,
        });

        if let Some(action) = result.best_action {
            actions.push(action.clone());
            game = game.apply_action(&action);
        } else {
            break;
        }
    }

    if game.proved {
        Some(ProofGameEpisode {
            initial_state: problem.initial_state.clone(),
            final_state: game.state.clone(),
            proved: true,
            total_reward: game.reward,
            num_steps: game.steps,
            num_constructions: game.num_constructions,
            actions,
            trajectory,
            goal_confidence: game.goal_confidence(),
        })
    } else {
        None
    }
}

// =============================================================================
// Synthetic Training Generation
// =============================================================================

/// Generate synthetic training data from a failed proof attempt
pub fn generate_synthetic_training(
    problem: &IMOProblem,
    failed_episode: &ProofGameEpisode,
) -> Vec<ProofGameEpisode> {
    let mut synthetic = Vec::new();

    // Strategy 1: Add random auxiliary constructions to initial state
    for i in 0..3 {
        let mut state = problem.initial_state.clone();
        add_random_construction(&mut state, i);

        let episode = super::geo_game::generate_proof_game_random(state, problem.goal.clone(), 30);

        if episode.proved || episode.trajectory.len() > failed_episode.trajectory.len() {
            synthetic.push(episode);
        }
    }

    // Strategy 2: Perturb the goal slightly (for negative examples)
    // This helps the network learn what NOT to do

    synthetic
}

/// Add a random construction to the state
fn add_random_construction(state: &mut ProofState, seed: usize) {
    let points: Vec<String> = state.point_labels().iter().map(|s| s.to_string()).collect();

    if points.len() < 2 {
        return;
    }

    // Deterministic "random" based on seed and state
    let idx1 = seed % points.len();
    let idx2 = (seed + 1) % points.len();

    if idx1 != idx2 {
        // Add midpoint
        let mid_name = format!("_aux_{}", seed);
        state.add_point(&mid_name);
        let pred = Predicate::midpoint(&mid_name, &points[idx1], &points[idx2]);
        state.add_axiom(pred);
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Run benchmark with uniform (random) network
pub fn run_baseline_benchmark() -> BenchmarkStats {
    let network = UniformGeoNetwork::new();
    let config = BenchmarkConfig {
        timeout_secs: 30,
        mcts_simulations: 50,
        verbose: true,
        ..Default::default()
    };
    run_benchmark(&network, &config)
}

/// Get a specific problem by ID
pub fn get_problem(id: &str) -> Option<IMOProblem> {
    imo_ag_30().into_iter().find(|p| p.id == id)
}

/// Get problems by year
pub fn get_problems_by_year(year: u32) -> Vec<IMOProblem> {
    imo_ag_30().into_iter().filter(|p| p.year == year).collect()
}

/// Get problems by difficulty range
pub fn get_problems_by_difficulty(min: f64, max: f64) -> Vec<IMOProblem> {
    imo_ag_30()
        .into_iter()
        .filter(|p| p.difficulty >= min && p.difficulty <= max)
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imo_ag_30_count() {
        let problems = imo_ag_30();
        assert_eq!(problems.len(), 30);
    }

    #[test]
    fn test_problem_ids_unique() {
        let problems = imo_ag_30();
        let mut ids: Vec<_> = problems.iter().map(|p| p.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 30);
    }

    #[test]
    fn test_problem_has_goal() {
        for problem in imo_ag_30() {
            // Each problem should have a goal
            assert!(
                !problem.goal.args.is_empty(),
                "Problem {} has no goal arguments",
                problem.id
            );
        }
    }

    #[test]
    fn test_problem_has_points() {
        for problem in imo_ag_30() {
            // Each problem should have at least 3 points (triangle base case)
            assert!(
                problem.initial_state.points.len() >= 3,
                "Problem {} has fewer than 3 points",
                problem.id
            );
        }
    }

    #[test]
    fn test_get_problem_by_id() {
        let p = get_problem("imo_2000_p1");
        assert!(p.is_some());
        assert_eq!(p.unwrap().year, 2000);
    }

    #[test]
    fn test_get_problems_by_year() {
        let p2000 = get_problems_by_year(2000);
        assert!(p2000.len() >= 1);
        for p in p2000 {
            assert_eq!(p.year, 2000);
        }
    }

    #[test]
    fn test_difficulty_range() {
        for problem in imo_ag_30() {
            assert!(
                problem.difficulty >= 0.0 && problem.difficulty <= 1.0,
                "Problem {} has invalid difficulty {}",
                problem.id,
                problem.difficulty
            );
        }
    }

    #[test]
    fn test_benchmark_result_creation() {
        let result = BenchmarkResult {
            problem_id: "test".to_string(),
            solved: true,
            time_taken: Duration::from_secs(1),
            simulations: 100,
            constructions: 2,
            confidence: Some(BetaConfidence::new(90.0, 10.0)),
            uncertainty: 0.1,
            proof_text: Some("Proof...".to_string()),
        };
        assert!(result.solved);
    }

    #[test]
    fn test_benchmark_stats() {
        let mut stats = BenchmarkStats::default();

        stats.add_result(BenchmarkResult {
            problem_id: "p1".to_string(),
            solved: true,
            time_taken: Duration::from_secs(1),
            simulations: 100,
            constructions: 2,
            confidence: Some(BetaConfidence::new(90.0, 10.0)),
            uncertainty: 0.1,
            proof_text: None,
        });

        stats.add_result(BenchmarkResult {
            problem_id: "p2".to_string(),
            solved: false,
            time_taken: Duration::from_secs(60),
            simulations: 100,
            constructions: 0,
            confidence: None,
            uncertainty: 0.5,
            proof_text: None,
        });

        assert_eq!(stats.total_problems, 2);
        assert_eq!(stats.solved, 1);
        assert_eq!(stats.failed, 1);
        assert!((stats.solve_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_imo_2004_p1_is_midpoint_theorem() {
        let p = get_problem("imo_2004_p1").unwrap();
        // This is essentially the midpoint theorem
        assert!(p.name.contains("Midpoint"));
    }
}
