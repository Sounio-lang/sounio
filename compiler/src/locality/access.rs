//! Access Pattern Analysis: Understanding how data is accessed at runtime.
//!
//! This module analyzes code to understand access patterns, enabling
//! better prefetching and cache optimization decisions.

use super::types::Locality;
use std::collections::{HashMap, HashSet};

/// A single field access observation.
#[derive(Debug, Clone)]
pub struct FieldAccess {
    /// The type containing the field
    pub type_name: String,
    /// The field name
    pub field_name: String,
    /// Source location (file:line:col)
    pub location: String,
    /// Access kind (read or write)
    pub kind: AccessKind,
    /// Estimated frequency (0.0 = rare, 1.0 = hot)
    pub frequency: f64,
    /// Whether this is in a loop
    pub in_loop: bool,
    /// Loop depth if in a loop
    pub loop_depth: u32,
}

impl FieldAccess {
    /// Create a new field access.
    pub fn new(type_name: impl Into<String>, field_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            field_name: field_name.into(),
            location: String::new(),
            kind: AccessKind::Read,
            frequency: 0.5,
            in_loop: false,
            loop_depth: 0,
        }
    }

    /// Set the location.
    pub fn at(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// Set as a write access.
    pub fn write(mut self) -> Self {
        self.kind = AccessKind::Write;
        self
    }

    /// Set as in a loop.
    pub fn in_loop(mut self, depth: u32) -> Self {
        self.in_loop = true;
        self.loop_depth = depth;
        self.frequency = (self.frequency * 10.0_f64.powi(depth as i32)).min(1.0);
        self
    }

    /// Get the "heat" of this access (for hot/cold classification).
    pub fn heat(&self) -> f64 {
        let base = self.frequency;
        let loop_factor = if self.in_loop {
            10.0_f64.powi(self.loop_depth as i32)
        } else {
            1.0
        };
        (base * loop_factor).min(1.0)
    }
}

/// Kind of access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKind {
    /// Read access
    Read,
    /// Write access
    Write,
}

/// Co-access information: fields accessed together.
#[derive(Debug, Clone)]
pub struct CoAccess {
    /// First field
    pub field_a: String,
    /// Second field
    pub field_b: String,
    /// How often they're accessed together (0.0 to 1.0)
    pub correlation: f64,
    /// Typical access order (A before B, B before A, or interleaved)
    pub order: CoAccessOrder,
    /// Distance in code between accesses
    pub code_distance: u32,
}

impl CoAccess {
    /// Create a new co-access relationship.
    pub fn new(field_a: impl Into<String>, field_b: impl Into<String>, correlation: f64) -> Self {
        Self {
            field_a: field_a.into(),
            field_b: field_b.into(),
            correlation: correlation.clamp(0.0, 1.0),
            order: CoAccessOrder::Unknown,
            code_distance: 0,
        }
    }

    /// Check if this is a strong co-access pattern.
    pub fn is_strong(&self) -> bool {
        self.correlation >= 0.7
    }
}

/// Order of co-access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoAccessOrder {
    /// First field accessed before second
    Sequential,
    /// Either order, varies
    Interleaved,
    /// Unknown order
    Unknown,
}

/// An access pattern for a type or module.
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Name of the pattern (usually type or function name)
    pub name: String,
    /// Individual field accesses
    pub accesses: Vec<FieldAccess>,
    /// Co-access relationships
    pub co_accesses: Vec<CoAccess>,
    /// Stride patterns for array accesses
    pub strides: HashMap<String, StridePattern>,
    /// Hotness classification
    pub hotness: Hotness,
}

impl AccessPattern {
    /// Create a new access pattern.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            accesses: Vec::new(),
            co_accesses: Vec::new(),
            strides: HashMap::new(),
            hotness: Hotness::Unknown,
        }
    }

    /// Add a field access.
    pub fn add_access(&mut self, access: FieldAccess) {
        self.accesses.push(access);
        self.update_hotness();
    }

    /// Add a co-access relationship.
    pub fn add_co_access(&mut self, co_access: CoAccess) {
        self.co_accesses.push(co_access);
    }

    /// Add a stride pattern.
    pub fn add_stride(&mut self, field: impl Into<String>, stride: StridePattern) {
        self.strides.insert(field.into(), stride);
    }

    /// Get fields that are frequently accessed together.
    pub fn get_hot_fields(&self) -> Vec<&str> {
        self.accesses
            .iter()
            .filter(|a| a.heat() >= 0.7)
            .map(|a| a.field_name.as_str())
            .collect()
    }

    /// Get strong co-access pairs.
    pub fn get_co_access_groups(&self) -> Vec<Vec<&str>> {
        // Union-find to group co-accessed fields
        let mut groups: Vec<HashSet<&str>> = Vec::new();

        for co in &self.co_accesses {
            if !co.is_strong() {
                continue;
            }

            let a = co.field_a.as_str();
            let b = co.field_b.as_str();

            // Find existing groups containing a or b
            let mut found_a = None;
            let mut found_b = None;

            for (i, group) in groups.iter().enumerate() {
                if group.contains(a) {
                    found_a = Some(i);
                }
                if group.contains(b) {
                    found_b = Some(i);
                }
            }

            match (found_a, found_b) {
                (None, None) => {
                    let mut group = HashSet::new();
                    group.insert(a);
                    group.insert(b);
                    groups.push(group);
                }
                (Some(i), None) => {
                    groups[i].insert(b);
                }
                (None, Some(i)) => {
                    groups[i].insert(a);
                }
                (Some(i), Some(j)) if i != j => {
                    // Merge groups
                    let group_j = groups.remove(j);
                    let idx = if j < i { i - 1 } else { i };
                    groups[idx].extend(group_j);
                }
                _ => {}
            }
        }

        groups
            .into_iter()
            .map(|g| g.into_iter().collect())
            .collect()
    }

    /// Update hotness classification.
    fn update_hotness(&mut self) {
        let max_heat = self
            .accesses
            .iter()
            .map(|a| a.heat())
            .fold(0.0_f64, f64::max);

        self.hotness = if max_heat >= 0.8 {
            Hotness::Hot
        } else if max_heat >= 0.3 {
            Hotness::Warm
        } else {
            Hotness::Cold
        };
    }

    /// Compute recommended locality for this pattern.
    pub fn recommended_locality(&self) -> Locality {
        match self.hotness {
            Hotness::Hot => Locality::L1,
            Hotness::Warm => Locality::L2,
            Hotness::Cold => Locality::Local,
            Hotness::Unknown => Locality::L3,
        }
    }
}

/// Stride pattern for array accesses.
#[derive(Debug, Clone)]
pub struct StridePattern {
    /// Primary stride in bytes
    pub stride: usize,
    /// Whether the stride is constant
    pub is_constant: bool,
    /// Secondary stride (for 2D access)
    pub secondary_stride: Option<usize>,
    /// Total access count
    pub count: usize,
}

impl StridePattern {
    /// Create a new stride pattern.
    pub fn new(stride: usize) -> Self {
        Self {
            stride,
            is_constant: true,
            secondary_stride: None,
            count: 1,
        }
    }

    /// Set secondary stride for 2D access.
    pub fn with_secondary(mut self, stride: usize) -> Self {
        self.secondary_stride = Some(stride);
        self
    }

    /// Check if this is a sequential access pattern.
    pub fn is_sequential(&self, element_size: usize) -> bool {
        self.stride == element_size
    }

    /// Get prefetch distance in elements.
    pub fn prefetch_distance(&self) -> usize {
        // Heuristic: prefetch 8-16 elements ahead
        if self.stride <= 8 {
            16
        } else if self.stride <= 64 {
            8
        } else {
            4
        }
    }
}

/// Hotness classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hotness {
    /// Unknown (not yet classified)
    Unknown,
    /// Cold (rarely accessed)
    Cold,
    /// Warm (moderately accessed)
    Warm,
    /// Hot (frequently accessed)
    Hot,
}

/// The access pattern analyzer.
pub struct AccessAnalyzer {
    /// Patterns by name
    patterns: HashMap<String, AccessPattern>,
    /// Current analysis context
    context: AnalysisContext,
}

/// Analysis context for tracking state during analysis.
#[derive(Debug, Clone, Default)]
pub struct AnalysisContext {
    /// Current function being analyzed
    pub current_function: Option<String>,
    /// Current loop depth
    pub loop_depth: u32,
    /// Recently accessed fields (for co-access detection)
    pub recent_accesses: Vec<(String, String)>, // (type, field)
    /// Window size for co-access detection
    pub window_size: usize,
}

impl AnalysisContext {
    /// Create a new context.
    pub fn new() -> Self {
        Self {
            current_function: None,
            loop_depth: 0,
            recent_accesses: Vec::new(),
            window_size: 10,
        }
    }

    /// Enter a loop.
    pub fn enter_loop(&mut self) {
        self.loop_depth += 1;
    }

    /// Exit a loop.
    pub fn exit_loop(&mut self) {
        self.loop_depth = self.loop_depth.saturating_sub(1);
    }

    /// Record an access.
    pub fn record_access(&mut self, type_name: &str, field_name: &str) {
        self.recent_accesses
            .push((type_name.to_string(), field_name.to_string()));

        // Keep window bounded
        while self.recent_accesses.len() > self.window_size {
            self.recent_accesses.remove(0);
        }
    }

    /// Get recent accesses for co-access detection.
    pub fn recent_for_type(&self, type_name: &str) -> Vec<&str> {
        self.recent_accesses
            .iter()
            .filter(|(t, _)| t == type_name)
            .map(|(_, f)| f.as_str())
            .collect()
    }
}

impl AccessAnalyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            context: AnalysisContext::new(),
        }
    }

    /// Start analyzing a function.
    pub fn enter_function(&mut self, name: &str) {
        self.context.current_function = Some(name.to_string());

        if !self.patterns.contains_key(name) {
            self.patterns
                .insert(name.to_string(), AccessPattern::new(name));
        }
    }

    /// Finish analyzing a function.
    pub fn exit_function(&mut self) {
        self.context.current_function = None;
        self.context.recent_accesses.clear();
    }

    /// Enter a loop.
    pub fn enter_loop(&mut self) {
        self.context.enter_loop();
    }

    /// Exit a loop.
    pub fn exit_loop(&mut self) {
        self.context.exit_loop();
    }

    /// Record a field access.
    pub fn record_access(&mut self, type_name: &str, field_name: &str, kind: AccessKind) {
        let access = FieldAccess::new(type_name, field_name).in_loop(self.context.loop_depth);

        let access = if kind == AccessKind::Write {
            access.write()
        } else {
            access
        };

        // Detect co-access with recent accesses
        // Collect recent fields first to avoid borrow conflict
        let recent: Vec<String> = self
            .context
            .recent_for_type(type_name)
            .iter()
            .map(|s| s.to_string())
            .collect();
        for recent_field in &recent {
            if recent_field != field_name {
                self.record_co_access(type_name, recent_field, field_name, 0.5);
            }
        }

        self.context.record_access(type_name, field_name);

        if let Some(func) = &self.context.current_function.clone()
            && let Some(pattern) = self.patterns.get_mut(func)
        {
            pattern.add_access(access);
        }
    }

    /// Record a co-access relationship.
    pub fn record_co_access(
        &mut self,
        type_name: &str,
        field_a: &str,
        field_b: &str,
        correlation: f64,
    ) {
        if let Some(func) = &self.context.current_function.clone()
            && let Some(pattern) = self.patterns.get_mut(func)
        {
            let co_access = CoAccess::new(
                format!("{}.{}", type_name, field_a),
                format!("{}.{}", type_name, field_b),
                correlation,
            );
            pattern.add_co_access(co_access);
        }
    }

    /// Record a stride pattern.
    pub fn record_stride(&mut self, type_name: &str, field_name: &str, stride: usize) {
        if let Some(func) = &self.context.current_function.clone()
            && let Some(pattern) = self.patterns.get_mut(func)
        {
            let key = format!("{}.{}", type_name, field_name);

            if let Some(existing) = pattern.strides.get_mut(&key) {
                if existing.stride != stride {
                    existing.is_constant = false;
                }
                existing.count += 1;
            } else {
                pattern.add_stride(key, StridePattern::new(stride));
            }
        }
    }

    /// Get the access pattern for a function.
    pub fn get_pattern(&self, name: &str) -> Option<&AccessPattern> {
        self.patterns.get(name)
    }

    /// Get all patterns.
    pub fn all_patterns(&self) -> impl Iterator<Item = &AccessPattern> {
        self.patterns.values()
    }

    /// Get co-access relationships for a specific type.
    /// Returns tuples of (field_a, field_b, correlation).
    pub fn co_access_for(&self, type_name: &str) -> Vec<(String, String, f64)> {
        let prefix = format!("{}.", type_name);
        let mut result = Vec::new();

        for pattern in self.patterns.values() {
            for co in &pattern.co_accesses {
                // Check if this co-access involves the requested type
                if co.field_a.starts_with(&prefix) || co.field_b.starts_with(&prefix) {
                    // Extract just the field names
                    let field_a = co
                        .field_a
                        .strip_prefix(&prefix)
                        .unwrap_or(&co.field_a)
                        .to_string();
                    let field_b = co
                        .field_b
                        .strip_prefix(&prefix)
                        .unwrap_or(&co.field_b)
                        .to_string();
                    result.push((field_a, field_b, co.correlation));
                }
            }
        }

        result
    }

    /// Generate optimization recommendations.
    pub fn recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recs = Vec::new();

        for pattern in self.patterns.values() {
            // Recommend prefetching for hot fields
            let hot_fields = pattern.get_hot_fields();
            if !hot_fields.is_empty() {
                recs.push(OptimizationRecommendation {
                    pattern_name: pattern.name.clone(),
                    kind: RecommendationKind::Prefetch,
                    fields: hot_fields.iter().map(|s| s.to_string()).collect(),
                    estimated_benefit: 0.3,
                    description: format!("Prefetch hot fields: {}", hot_fields.join(", ")),
                });
            }

            // Recommend packing for co-accessed fields
            let groups = pattern.get_co_access_groups();
            for group in groups {
                if group.len() >= 2 {
                    recs.push(OptimizationRecommendation {
                        pattern_name: pattern.name.clone(),
                        kind: RecommendationKind::Packing,
                        fields: group.iter().map(|s| s.to_string()).collect(),
                        estimated_benefit: 0.2,
                        description: format!(
                            "Pack co-accessed fields together: {}",
                            group.join(", ")
                        ),
                    });
                }
            }

            // Recommend software prefetch for stride patterns
            for (field, stride) in &pattern.strides {
                if stride.is_constant && stride.count >= 3 {
                    recs.push(OptimizationRecommendation {
                        pattern_name: pattern.name.clone(),
                        kind: RecommendationKind::StridePrefetch,
                        fields: vec![field.clone()],
                        estimated_benefit: 0.25,
                        description: format!(
                            "Use software prefetch with stride {} for {}",
                            stride.stride, field
                        ),
                    });
                }
            }
        }

        // Sort by benefit
        recs.sort_by(|a, b| {
            b.estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap()
        });
        recs
    }
}

impl Default for AccessAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// An optimization recommendation.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// The pattern this applies to
    pub pattern_name: String,
    /// Kind of optimization
    pub kind: RecommendationKind,
    /// Fields involved
    pub fields: Vec<String>,
    /// Estimated performance benefit (0.0 to 1.0)
    pub estimated_benefit: f64,
    /// Human-readable description
    pub description: String,
}

/// Kind of optimization recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationKind {
    /// Software prefetch
    Prefetch,
    /// Cache-line packing
    Packing,
    /// Stride-based prefetch
    StridePrefetch,
    /// NUMA placement
    NumaPlacement,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_access_heat() {
        let access = FieldAccess::new("Patient", "name");
        assert!(access.heat() < 1.0);

        let loop_access = FieldAccess::new("Patient", "name").in_loop(2);
        assert!(loop_access.heat() >= access.heat());
    }

    #[test]
    fn test_co_access() {
        let co = CoAccess::new("name", "age", 0.8);
        assert!(co.is_strong());

        let weak = CoAccess::new("name", "notes", 0.3);
        assert!(!weak.is_strong());
    }

    #[test]
    fn test_access_pattern() {
        let mut pattern = AccessPattern::new("process_patient");

        pattern.add_access(FieldAccess::new("Patient", "name").in_loop(1));
        pattern.add_access(FieldAccess::new("Patient", "age").in_loop(1));
        pattern.add_co_access(CoAccess::new("name", "age", 0.9));

        let hot = pattern.get_hot_fields();
        assert!(!hot.is_empty());

        let groups = pattern.get_co_access_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn test_stride_pattern() {
        let stride = StridePattern::new(8);
        assert!(stride.is_sequential(8));
        assert!(!stride.is_sequential(4));

        let stride_2d = StridePattern::new(8).with_secondary(1024);
        assert!(stride_2d.secondary_stride.is_some());
    }

    #[test]
    fn test_analyzer() {
        let mut analyzer = AccessAnalyzer::new();

        analyzer.enter_function("process");
        analyzer.enter_loop();
        analyzer.record_access("Data", "x", AccessKind::Read);
        analyzer.record_access("Data", "y", AccessKind::Read);
        analyzer.record_stride("Data", "values", 8);
        analyzer.record_stride("Data", "values", 8);
        analyzer.record_stride("Data", "values", 8);
        analyzer.exit_loop();
        analyzer.exit_function();

        let pattern = analyzer.get_pattern("process");
        assert!(pattern.is_some());

        let p = pattern.unwrap();
        assert_eq!(p.accesses.len(), 2);
        assert!(p.strides.contains_key("Data.values"));
    }

    #[test]
    fn test_recommendations() {
        let mut analyzer = AccessAnalyzer::new();

        analyzer.enter_function("hot_loop");
        analyzer.enter_loop();
        analyzer.enter_loop();
        analyzer.record_access("Vec", "data", AccessKind::Read);
        analyzer.exit_loop();
        analyzer.exit_loop();
        analyzer.exit_function();

        let recs = analyzer.recommendations();
        assert!(!recs.is_empty());
    }

    #[test]
    fn test_analysis_context() {
        let mut ctx = AnalysisContext::new();

        ctx.enter_loop();
        assert_eq!(ctx.loop_depth, 1);
        ctx.enter_loop();
        assert_eq!(ctx.loop_depth, 2);
        ctx.exit_loop();
        assert_eq!(ctx.loop_depth, 1);

        ctx.record_access("Type", "field1");
        ctx.record_access("Type", "field2");

        let recent = ctx.recent_for_type("Type");
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_recommended_locality() {
        let mut hot = AccessPattern::new("hot");
        hot.hotness = Hotness::Hot;
        assert_eq!(hot.recommended_locality(), Locality::L1);

        let mut cold = AccessPattern::new("cold");
        cold.hotness = Hotness::Cold;
        assert_eq!(cold.recommended_locality(), Locality::Local);
    }
}
