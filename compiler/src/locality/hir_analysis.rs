//! HIR-based Locality Analysis
//!
//! This module provides real access pattern analysis by traversing HIR.
//! It extracts field accesses, detects co-access patterns, and generates
//! locality optimization recommendations.

use super::access::{AccessAnalyzer, AccessKind, AccessPattern, Hotness};
use super::types::Locality;
use crate::hir::{Hir, HirBlock, HirExpr, HirExprKind, HirFn, HirItem, HirStmt, HirStruct};
use std::collections::{HashMap, HashSet};

/// HIR-based locality analyzer.
///
/// Traverses HIR to extract actual field access patterns for locality optimization.
pub struct HirLocalityAnalyzer {
    /// Underlying access analyzer
    analyzer: AccessAnalyzer,
    /// Struct definitions for type resolution
    structs: HashMap<String, StructInfo>,
    /// Current function being analyzed
    current_function: Option<String>,
    /// Loop nesting depth (for hotness estimation)
    loop_depth: usize,
}

/// Information about a struct for analysis.
#[derive(Debug, Clone)]
pub struct StructInfo {
    /// Struct name
    pub name: String,
    /// Field names and their types
    pub fields: Vec<(String, String)>,
    /// Whether the struct is linear
    pub is_linear: bool,
}

/// Result of HIR locality analysis.
#[derive(Debug)]
pub struct LocalityAnalysisResult {
    /// Access patterns per function
    pub patterns: HashMap<String, AccessPattern>,
    /// Recommended locality for each type
    pub recommended_localities: HashMap<String, Locality>,
    /// Field packing recommendations
    pub packing_recommendations: Vec<PackingRecommendation>,
    /// Prefetch insertion points
    pub prefetch_points: Vec<PrefetchPoint>,
}

/// Recommendation for struct field packing.
#[derive(Debug, Clone)]
pub struct PackingRecommendation {
    /// Struct name
    pub struct_name: String,
    /// Fields that should be grouped together
    pub hot_fields: Vec<String>,
    /// Fields that can be separated
    pub cold_fields: Vec<String>,
    /// Estimated benefit (0.0 to 1.0)
    pub estimated_benefit: f64,
}

/// A point where prefetch instructions could be inserted.
#[derive(Debug, Clone)]
pub struct PrefetchPoint {
    /// Function containing the prefetch point
    pub function: String,
    /// Type being accessed
    pub type_name: String,
    /// Field to prefetch
    pub field: String,
    /// Fields likely to be accessed after
    pub prefetch_fields: Vec<String>,
    /// Priority (High, Medium, Low)
    pub priority: PrefetchPriority,
}

/// Prefetch priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchPriority {
    High,
    Medium,
    Low,
}

impl HirLocalityAnalyzer {
    /// Create a new HIR locality analyzer.
    pub fn new() -> Self {
        Self {
            analyzer: AccessAnalyzer::new(),
            structs: HashMap::new(),
            current_function: None,
            loop_depth: 0,
        }
    }

    /// Analyze a complete HIR.
    pub fn analyze(&mut self, hir: &Hir) -> LocalityAnalysisResult {
        // First pass: collect struct definitions
        for item in &hir.items {
            if let HirItem::Struct(s) = item {
                self.register_struct(s);
            }
        }

        // Second pass: analyze functions
        for item in &hir.items {
            if let HirItem::Function(f) = item {
                self.analyze_function(f);
            }
        }

        // Generate results
        self.generate_results()
    }

    /// Register a struct for analysis.
    fn register_struct(&mut self, s: &HirStruct) {
        let fields = s
            .fields
            .iter()
            .map(|f| (f.name.clone(), format!("{:?}", f.ty)))
            .collect();

        self.structs.insert(
            s.name.clone(),
            StructInfo {
                name: s.name.clone(),
                fields,
                is_linear: s.is_linear,
            },
        );
    }

    /// Analyze a function.
    fn analyze_function(&mut self, f: &HirFn) {
        self.current_function = Some(f.name.clone());
        self.analyzer.enter_function(&f.name);
        self.loop_depth = 0;

        self.analyze_block(&f.body);

        self.analyzer.exit_function();
        self.current_function = None;
    }

    /// Analyze a block.
    fn analyze_block(&mut self, block: &HirBlock) {
        for stmt in &block.stmts {
            self.analyze_stmt(stmt);
        }
    }

    /// Analyze a statement.
    fn analyze_stmt(&mut self, stmt: &HirStmt) {
        match stmt {
            HirStmt::Let { value, .. } => {
                if let Some(expr) = value {
                    self.analyze_expr(expr, AccessKind::Read);
                }
            }
            HirStmt::Expr(expr) => {
                self.analyze_expr(expr, AccessKind::Read);
            }
            HirStmt::Assign { target, value } => {
                self.analyze_expr(target, AccessKind::Write);
                self.analyze_expr(value, AccessKind::Read);
            }
        }
    }

    /// Analyze an expression, tracking field accesses.
    fn analyze_expr(&mut self, expr: &HirExpr, access_kind: AccessKind) {
        match &expr.kind {
            HirExprKind::Field { base, field } => {
                // This is a field access - record it
                if let Some(type_name) = self.infer_type_name(base) {
                    let kind = if self.loop_depth > 0 {
                        // Inside loop = likely hot - mark as write for higher weight
                        AccessKind::Write
                    } else {
                        access_kind
                    };
                    self.analyzer.record_access(&type_name, field, kind);
                }
                // Also analyze the base expression
                self.analyze_expr(base, AccessKind::Read);
            }

            HirExprKind::Binary { left, right, .. } => {
                self.analyze_expr(left, AccessKind::Read);
                self.analyze_expr(right, AccessKind::Read);
            }

            HirExprKind::Unary { expr, .. } => {
                self.analyze_expr(expr, AccessKind::Read);
            }

            HirExprKind::Call { func, args } => {
                self.analyze_expr(func, AccessKind::Read);
                for arg in args {
                    self.analyze_expr(arg, AccessKind::Read);
                }
            }

            HirExprKind::MethodCall { receiver, args, .. } => {
                self.analyze_expr(receiver, AccessKind::Read);
                for arg in args {
                    self.analyze_expr(arg, AccessKind::Read);
                }
            }

            HirExprKind::Index { base, index } => {
                self.analyze_expr(base, AccessKind::Read);
                self.analyze_expr(index, AccessKind::Read);
            }

            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.analyze_expr(condition, AccessKind::Read);
                self.analyze_block(then_branch);
                if let Some(else_expr) = else_branch {
                    self.analyze_expr(else_expr, AccessKind::Read);
                }
            }

            HirExprKind::Loop(block) => {
                self.loop_depth += 1;
                self.analyze_block(block);
                self.loop_depth -= 1;
            }

            HirExprKind::While { condition, body } => {
                self.analyze_expr(condition, AccessKind::Read);
                self.loop_depth += 1;
                self.analyze_block(body);
                self.loop_depth -= 1;
            }

            HirExprKind::Block(block) => {
                self.analyze_block(block);
            }

            HirExprKind::Match { scrutinee, arms } => {
                self.analyze_expr(scrutinee, AccessKind::Read);
                for arm in arms {
                    self.analyze_expr(&arm.body, AccessKind::Read);
                }
            }

            HirExprKind::Struct { fields, .. } => {
                for (_, expr) in fields {
                    self.analyze_expr(expr, AccessKind::Read);
                }
            }

            HirExprKind::Tuple(exprs) | HirExprKind::Array(exprs) => {
                for expr in exprs {
                    self.analyze_expr(expr, AccessKind::Read);
                }
            }

            HirExprKind::Ref { expr, .. } => {
                self.analyze_expr(expr, AccessKind::Read);
            }

            HirExprKind::Deref(expr) => {
                self.analyze_expr(expr, AccessKind::Read);
            }

            HirExprKind::Cast { expr, .. } => {
                self.analyze_expr(expr, AccessKind::Read);
            }

            HirExprKind::Return(Some(expr)) => {
                self.analyze_expr(expr, AccessKind::Read);
            }

            HirExprKind::Break(Some(expr)) => {
                self.analyze_expr(expr, AccessKind::Read);
            }

            HirExprKind::Closure { body, .. } => {
                self.analyze_expr(body, AccessKind::Read);
            }

            // Leaf expressions - no nested analysis needed
            HirExprKind::Literal(_)
            | HirExprKind::Local(_)
            | HirExprKind::Global(_)
            | HirExprKind::Continue
            | HirExprKind::Return(None)
            | HirExprKind::Break(None) => {}

            // Handle other cases
            _ => {}
        }
    }

    /// Try to infer the type name from an expression.
    fn infer_type_name(&self, expr: &HirExpr) -> Option<String> {
        match &expr.ty {
            crate::hir::HirType::Named { name, .. } => Some(name.clone()),
            crate::hir::HirType::Ref { inner, .. } => {
                if let crate::hir::HirType::Named { name, .. } = inner.as_ref() {
                    Some(name.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Generate analysis results.
    fn generate_results(&self) -> LocalityAnalysisResult {
        let mut patterns = HashMap::new();
        let mut recommended_localities = HashMap::new();
        let mut packing_recommendations = Vec::new();
        let mut prefetch_points = Vec::new();

        // Collect patterns
        for pattern in self.analyzer.all_patterns() {
            patterns.insert(pattern.name.clone(), pattern.clone());
        }

        // Generate recommendations for each struct
        for (name, info) in &self.structs {
            // Recommend locality based on usage patterns
            let locality = self.recommend_locality(name);
            recommended_localities.insert(name.clone(), locality);

            // Generate packing recommendation
            if let Some(rec) = self.generate_packing_recommendation(name, info) {
                packing_recommendations.push(rec);
            }

            // Generate prefetch points
            prefetch_points.extend(self.generate_prefetch_points(name));
        }

        LocalityAnalysisResult {
            patterns,
            recommended_localities,
            packing_recommendations,
            prefetch_points,
        }
    }

    /// Recommend locality for a type based on access patterns.
    fn recommend_locality(&self, type_name: &str) -> Locality {
        // Check if this type is accessed in hot loops
        for pattern in self.analyzer.all_patterns() {
            for access in &pattern.accesses {
                if access.type_name == type_name {
                    return match pattern.hotness {
                        Hotness::Hot => Locality::L1,
                        Hotness::Warm => Locality::L2,
                        Hotness::Cold => Locality::L3,
                        Hotness::Unknown => Locality::Local,
                    };
                }
            }
        }
        Locality::Local
    }

    /// Generate packing recommendation for a struct.
    fn generate_packing_recommendation(
        &self,
        struct_name: &str,
        info: &StructInfo,
    ) -> Option<PackingRecommendation> {
        let field_names: HashSet<_> = info.fields.iter().map(|(n, _)| n.as_str()).collect();
        let mut hot_fields = Vec::new();
        let mut cold_fields = Vec::new();

        // Analyze co-access patterns
        let co_access = self.analyzer.co_access_for(struct_name);

        // Find frequently accessed fields
        let mut field_heat: HashMap<&str, f64> = HashMap::new();
        for (f1, f2, correlation) in &co_access {
            *field_heat.entry(f1.as_str()).or_default() += correlation;
            *field_heat.entry(f2.as_str()).or_default() += correlation;
        }

        // Classify fields
        let threshold = 0.5;
        for (name, _) in &info.fields {
            let heat = field_heat.get(name.as_str()).copied().unwrap_or(0.0);
            if heat > threshold {
                hot_fields.push(name.clone());
            } else {
                cold_fields.push(name.clone());
            }
        }

        // Only recommend if there's a meaningful split
        if hot_fields.is_empty() || cold_fields.is_empty() {
            return None;
        }

        let estimated_benefit = hot_fields.len() as f64 / info.fields.len() as f64 * 0.3;

        Some(PackingRecommendation {
            struct_name: struct_name.to_string(),
            hot_fields,
            cold_fields,
            estimated_benefit,
        })
    }

    /// Generate prefetch points for a type.
    fn generate_prefetch_points(&self, type_name: &str) -> Vec<PrefetchPoint> {
        let mut points = Vec::new();
        let co_access = self.analyzer.co_access_for(type_name);

        // Group by first field
        let mut by_field: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for (f1, f2, correlation) in co_access {
            by_field
                .entry(f1.clone())
                .or_default()
                .push((f2.clone(), correlation));
        }

        // Create prefetch points
        for (field, related) in by_field {
            if related.is_empty() {
                continue;
            }

            let prefetch_fields: Vec<_> = related
                .iter()
                .filter(|(_, c)| *c > 0.3)
                .map(|(f, _)| f.clone())
                .collect();

            if prefetch_fields.is_empty() {
                continue;
            }

            let max_correlation = related.iter().map(|(_, c)| *c).fold(0.0, f64::max);
            let priority = if max_correlation > 0.8 {
                PrefetchPriority::High
            } else if max_correlation > 0.5 {
                PrefetchPriority::Medium
            } else {
                PrefetchPriority::Low
            };

            points.push(PrefetchPoint {
                function: self.current_function.clone().unwrap_or_default(),
                type_name: type_name.to_string(),
                field,
                prefetch_fields,
                priority,
            });
        }

        points
    }
}

impl Default for HirLocalityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::*;

    fn make_field_access(base_name: &str, field: &str) -> HirExpr {
        HirExpr {
            id: crate::common::NodeId(0),
            kind: HirExprKind::Field {
                base: Box::new(HirExpr {
                    id: crate::common::NodeId(0),
                    kind: HirExprKind::Local(base_name.to_string()),
                    ty: HirType::Named {
                        name: "TestStruct".to_string(),
                        args: vec![],
                    },
                }),
                field: field.to_string(),
            },
            ty: HirType::I32,
        }
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = HirLocalityAnalyzer::new();
        assert!(analyzer.structs.is_empty());
    }

    #[test]
    fn test_struct_registration() {
        let mut analyzer = HirLocalityAnalyzer::new();
        let s = HirStruct {
            id: crate::common::NodeId(0),
            name: "Point".to_string(),
            fields: vec![
                HirField {
                    id: crate::common::NodeId(1),
                    name: "x".to_string(),
                    ty: HirType::F64,
                },
                HirField {
                    id: crate::common::NodeId(2),
                    name: "y".to_string(),
                    ty: HirType::F64,
                },
            ],
            is_linear: false,
            is_affine: false,
        };

        analyzer.register_struct(&s);
        assert!(analyzer.structs.contains_key("Point"));
        assert_eq!(analyzer.structs["Point"].fields.len(), 2);
    }

    #[test]
    fn test_empty_hir_analysis() {
        let mut analyzer = HirLocalityAnalyzer::new();
        let hir = Hir {
            items: vec![],
            externs: Vec::new(),
        };
        let result = analyzer.analyze(&hir);
        assert!(result.patterns.is_empty());
    }

    #[test]
    fn test_prefetch_priority() {
        assert_eq!(PrefetchPriority::High, PrefetchPriority::High);
        assert_ne!(PrefetchPriority::High, PrefetchPriority::Low);
    }
}
