//! Locality Types: Type-level encoding of memory hierarchy position.
//!
//! Locality types capture where data lives in the memory hierarchy,
//! from CPU registers down to network storage. This allows the type
//! system to reason about and optimize data placement.

use std::fmt;

/// Memory hierarchy levels, ordered from fastest to slowest.
///
/// The ordering is: Register < L1 < L2 < L3 < Local < Remote < Persistent < Network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Locality {
    /// CPU register - fastest, most limited
    Register,
    /// L1 cache - very fast, small (32-64KB typical)
    L1,
    /// L2 cache - fast, medium (256KB-1MB typical)
    L2,
    /// L3 cache - shared, larger (2-64MB typical)
    L3,
    /// Local DRAM - main memory on this NUMA node
    Local,
    /// Remote DRAM - main memory on another NUMA node
    Remote,
    /// Persistent storage - NVMe, SSD, HDD
    Persistent,
    /// Network storage - remote filesystem, object store
    Network,
}

impl Locality {
    /// Get the relative latency multiplier compared to L1.
    /// These are approximate values for reasoning, not exact measurements.
    pub fn latency_multiplier(&self) -> f64 {
        match self {
            Locality::Register => 0.25,     // ~1 cycle
            Locality::L1 => 1.0,            // ~4 cycles (baseline)
            Locality::L2 => 3.0,            // ~12 cycles
            Locality::L3 => 10.0,           // ~40 cycles
            Locality::Local => 50.0,        // ~200 cycles
            Locality::Remote => 100.0,      // ~400 cycles (NUMA)
            Locality::Persistent => 5000.0, // ~20k cycles (NVMe)
            Locality::Network => 50000.0,   // ~200k cycles (1Gbps LAN)
        }
    }

    /// Get the typical capacity at this level (in bytes).
    /// Returns None for levels where capacity is highly variable.
    pub fn typical_capacity(&self) -> Option<usize> {
        match self {
            Locality::Register => Some(64 * 16), // ~64 general purpose registers * 8 bytes
            Locality::L1 => Some(64 * 1024),     // 64 KB
            Locality::L2 => Some(512 * 1024),    // 512 KB
            Locality::L3 => Some(16 * 1024 * 1024), // 16 MB
            Locality::Local => Some(16 * 1024 * 1024 * 1024), // 16 GB
            Locality::Remote => None,            // Varies by system
            Locality::Persistent => None,        // Varies widely
            Locality::Network => None,           // Unbounded
        }
    }

    /// Check if this locality is "hot" (frequently accessed, low latency).
    pub fn is_hot(&self) -> bool {
        matches!(self, Locality::Register | Locality::L1 | Locality::L2)
    }

    /// Check if this locality is "cold" (infrequently accessed, high latency).
    pub fn is_cold(&self) -> bool {
        matches!(self, Locality::Persistent | Locality::Network)
    }

    /// Check if prefetching is beneficial at this level.
    pub fn benefits_from_prefetch(&self) -> bool {
        matches!(self, Locality::L3 | Locality::Local | Locality::Remote)
    }

    /// Get the next slower locality level.
    pub fn slower(&self) -> Option<Locality> {
        match self {
            Locality::Register => Some(Locality::L1),
            Locality::L1 => Some(Locality::L2),
            Locality::L2 => Some(Locality::L3),
            Locality::L3 => Some(Locality::Local),
            Locality::Local => Some(Locality::Remote),
            Locality::Remote => Some(Locality::Persistent),
            Locality::Persistent => Some(Locality::Network),
            Locality::Network => None,
        }
    }

    /// Get the next faster locality level.
    pub fn faster(&self) -> Option<Locality> {
        match self {
            Locality::Register => None,
            Locality::L1 => Some(Locality::Register),
            Locality::L2 => Some(Locality::L1),
            Locality::L3 => Some(Locality::L2),
            Locality::Local => Some(Locality::L3),
            Locality::Remote => Some(Locality::Local),
            Locality::Persistent => Some(Locality::Remote),
            Locality::Network => Some(Locality::Persistent),
        }
    }

    /// Parse a locality from a string.
    pub fn parse(s: &str) -> Option<Locality> {
        match s.to_lowercase().as_str() {
            "register" | "reg" => Some(Locality::Register),
            "l1" | "l1cache" => Some(Locality::L1),
            "l2" | "l2cache" => Some(Locality::L2),
            "l3" | "l3cache" | "llc" => Some(Locality::L3),
            "local" | "dram" | "ram" => Some(Locality::Local),
            "remote" | "numa" => Some(Locality::Remote),
            "persistent" | "disk" | "ssd" | "nvme" => Some(Locality::Persistent),
            "network" | "net" | "remote_storage" => Some(Locality::Network),
            _ => None,
        }
    }
}

impl fmt::Display for Locality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Locality::Register => write!(f, "Register"),
            Locality::L1 => write!(f, "L1"),
            Locality::L2 => write!(f, "L2"),
            Locality::L3 => write!(f, "L3"),
            Locality::Local => write!(f, "Local"),
            Locality::Remote => write!(f, "Remote"),
            Locality::Persistent => write!(f, "Persistent"),
            Locality::Network => write!(f, "Network"),
        }
    }
}

/// A locality bound specifying a range of acceptable localities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalityBound {
    /// The fastest acceptable locality (upper bound)
    pub fastest: Locality,
    /// The slowest acceptable locality (lower bound)
    pub slowest: Locality,
}

impl LocalityBound {
    /// Create a new locality bound.
    pub fn new(fastest: Locality, slowest: Locality) -> Self {
        Self { fastest, slowest }
    }

    /// Create a bound that accepts only a single locality.
    pub fn exact(locality: Locality) -> Self {
        Self {
            fastest: locality,
            slowest: locality,
        }
    }

    /// Create a bound that accepts any locality.
    pub fn any() -> Self {
        Self {
            fastest: Locality::Register,
            slowest: Locality::Network,
        }
    }

    /// Create a bound for "hot" data (Register through L2).
    pub fn hot() -> Self {
        Self {
            fastest: Locality::Register,
            slowest: Locality::L2,
        }
    }

    /// Create a bound for "warm" data (L2 through Local).
    pub fn warm() -> Self {
        Self {
            fastest: Locality::L2,
            slowest: Locality::Local,
        }
    }

    /// Create a bound for "cold" data (Persistent through Network).
    pub fn cold() -> Self {
        Self {
            fastest: Locality::Persistent,
            slowest: Locality::Network,
        }
    }

    /// Check if a locality satisfies this bound.
    pub fn satisfied_by(&self, locality: Locality) -> bool {
        locality >= self.fastest && locality <= self.slowest
    }

    /// Check if this bound is valid (fastest <= slowest).
    pub fn is_valid(&self) -> bool {
        self.fastest <= self.slowest
    }

    /// Intersect two bounds, returning None if they don't overlap.
    pub fn intersect(&self, other: &LocalityBound) -> Option<LocalityBound> {
        let fastest = std::cmp::max(self.fastest, other.fastest);
        let slowest = std::cmp::min(self.slowest, other.slowest);

        if fastest <= slowest {
            Some(LocalityBound { fastest, slowest })
        } else {
            None
        }
    }

    /// Union two bounds (smallest encompassing bound).
    pub fn union(&self, other: &LocalityBound) -> LocalityBound {
        LocalityBound {
            fastest: std::cmp::min(self.fastest, other.fastest),
            slowest: std::cmp::max(self.slowest, other.slowest),
        }
    }
}

impl fmt::Display for LocalityBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.fastest == self.slowest {
            write!(f, "{}", self.fastest)
        } else {
            write!(f, "{}..{}", self.fastest, self.slowest)
        }
    }
}

/// A locality type parameter, used in generic types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalityParam {
    /// The parameter name (e.g., "L" in `Vector<T, L>`)
    pub name: String,
    /// The bound on this parameter
    pub bound: LocalityBound,
    /// Whether this is inferred or explicit
    pub inferred: bool,
}

impl LocalityParam {
    /// Create a new locality parameter.
    pub fn new(name: impl Into<String>, bound: LocalityBound) -> Self {
        Self {
            name: name.into(),
            bound,
            inferred: false,
        }
    }

    /// Create an inferred locality parameter.
    pub fn inferred(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bound: LocalityBound::any(),
            inferred: true,
        }
    }

    /// Check if a concrete locality satisfies this parameter.
    pub fn satisfied_by(&self, locality: Locality) -> bool {
        self.bound.satisfied_by(locality)
    }
}

impl fmt::Display for LocalityParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.inferred {
            write!(f, "?{}", self.name)
        } else if self.bound == LocalityBound::any() {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}: {}", self.name, self.bound)
        }
    }
}

/// A constraint on locality relationships between parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalityConstraint {
    /// L1 must be faster than or equal to L2
    FasterOrEqual(String, String),
    /// L1 must be strictly faster than L2
    Faster(String, String),
    /// L1 must be at the same level as L2
    Same(String, String),
    /// L must satisfy a specific bound
    Bound(String, LocalityBound),
}

impl LocalityConstraint {
    /// Check if this constraint is satisfied by a substitution.
    pub fn satisfied_by(&self, subst: &std::collections::HashMap<String, Locality>) -> bool {
        match self {
            LocalityConstraint::FasterOrEqual(l1, l2) => {
                match (subst.get(l1), subst.get(l2)) {
                    (Some(loc1), Some(loc2)) => loc1 <= loc2,
                    _ => true, // Unknown parameters don't fail
                }
            }
            LocalityConstraint::Faster(l1, l2) => match (subst.get(l1), subst.get(l2)) {
                (Some(loc1), Some(loc2)) => loc1 < loc2,
                _ => true,
            },
            LocalityConstraint::Same(l1, l2) => match (subst.get(l1), subst.get(l2)) {
                (Some(loc1), Some(loc2)) => loc1 == loc2,
                _ => true,
            },
            LocalityConstraint::Bound(l, bound) => match subst.get(l) {
                Some(loc) => bound.satisfied_by(*loc),
                None => true,
            },
        }
    }
}

impl fmt::Display for LocalityConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocalityConstraint::FasterOrEqual(l1, l2) => write!(f, "{} <= {}", l1, l2),
            LocalityConstraint::Faster(l1, l2) => write!(f, "{} < {}", l1, l2),
            LocalityConstraint::Same(l1, l2) => write!(f, "{} = {}", l1, l2),
            LocalityConstraint::Bound(l, bound) => write!(f, "{}: {}", l, bound),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locality_ordering() {
        assert!(Locality::Register < Locality::L1);
        assert!(Locality::L1 < Locality::L2);
        assert!(Locality::L2 < Locality::L3);
        assert!(Locality::L3 < Locality::Local);
        assert!(Locality::Local < Locality::Remote);
        assert!(Locality::Remote < Locality::Persistent);
        assert!(Locality::Persistent < Locality::Network);
    }

    #[test]
    fn test_locality_latency() {
        assert!(Locality::Register.latency_multiplier() < Locality::L1.latency_multiplier());
        assert!(Locality::L3.latency_multiplier() < Locality::Local.latency_multiplier());
        assert!(Locality::Local.latency_multiplier() < Locality::Persistent.latency_multiplier());
    }

    #[test]
    fn test_locality_parse() {
        assert_eq!(Locality::parse("register"), Some(Locality::Register));
        assert_eq!(Locality::parse("L1"), Some(Locality::L1));
        assert_eq!(Locality::parse("llc"), Some(Locality::L3));
        assert_eq!(Locality::parse("dram"), Some(Locality::Local));
        assert_eq!(Locality::parse("unknown"), None);
    }

    #[test]
    fn test_locality_traversal() {
        assert_eq!(Locality::Register.slower(), Some(Locality::L1));
        assert_eq!(Locality::L1.faster(), Some(Locality::Register));
        assert_eq!(Locality::Network.slower(), None);
        assert_eq!(Locality::Register.faster(), None);
    }

    #[test]
    fn test_bound_satisfaction() {
        let hot = LocalityBound::hot();
        assert!(hot.satisfied_by(Locality::Register));
        assert!(hot.satisfied_by(Locality::L1));
        assert!(hot.satisfied_by(Locality::L2));
        assert!(!hot.satisfied_by(Locality::L3));
        assert!(!hot.satisfied_by(Locality::Local));
    }

    #[test]
    fn test_bound_intersect() {
        let hot = LocalityBound::hot();
        let warm = LocalityBound::warm();

        let intersection = hot.intersect(&warm);
        assert!(intersection.is_some());
        let int = intersection.unwrap();
        assert_eq!(int.fastest, Locality::L2);
        assert_eq!(int.slowest, Locality::L2);
    }

    #[test]
    fn test_bound_no_intersect() {
        let hot = LocalityBound::new(Locality::Register, Locality::L1);
        let cold = LocalityBound::cold();

        assert!(hot.intersect(&cold).is_none());
    }

    #[test]
    fn test_bound_union() {
        let bound1 = LocalityBound::new(Locality::L1, Locality::L2);
        let bound2 = LocalityBound::new(Locality::L3, Locality::Local);

        let union = bound1.union(&bound2);
        assert_eq!(union.fastest, Locality::L1);
        assert_eq!(union.slowest, Locality::Local);
    }

    #[test]
    fn test_locality_param() {
        let param = LocalityParam::new("L", LocalityBound::hot());
        assert!(param.satisfied_by(Locality::L1));
        assert!(!param.satisfied_by(Locality::Local));

        let inferred = LocalityParam::inferred("M");
        assert!(inferred.inferred);
        assert!(inferred.satisfied_by(Locality::Network));
    }

    #[test]
    fn test_constraint_satisfaction() {
        use std::collections::HashMap;

        let mut subst = HashMap::new();
        subst.insert("L1".to_string(), Locality::L1);
        subst.insert("L2".to_string(), Locality::Local);

        let c1 = LocalityConstraint::FasterOrEqual("L1".to_string(), "L2".to_string());
        assert!(c1.satisfied_by(&subst));

        let c2 = LocalityConstraint::Faster("L2".to_string(), "L1".to_string());
        assert!(!c2.satisfied_by(&subst));

        let c3 = LocalityConstraint::Bound("L1".to_string(), LocalityBound::hot());
        assert!(c3.satisfied_by(&subst));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Locality::L1), "L1");
        assert_eq!(format!("{}", LocalityBound::exact(Locality::L2)), "L2");
        assert_eq!(format!("{}", LocalityBound::hot()), "Register..L2");

        let param = LocalityParam::new("L", LocalityBound::hot());
        assert_eq!(format!("{}", param), "L: Register..L2");

        let inferred = LocalityParam::inferred("M");
        assert_eq!(format!("{}", inferred), "?M");
    }
}
