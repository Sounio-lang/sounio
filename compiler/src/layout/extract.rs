//! Concept Extraction
//!
//! Extracts ontology concepts from types and tracks co-occurrence patterns.
//! Supports both string-based extraction and HIR-based extraction.

use std::collections::{HashMap, HashSet};

use crate::hir::{Hir, HirBlock, HirExpr, HirExprKind, HirItem, HirStmt, HirType};

/// Extracted concept usage information
#[derive(Debug, Default, Clone)]
pub struct ConceptUsage {
    /// All concepts used (as CURIE strings, e.g., "CHEBI:15365")
    pub concepts: HashSet<String>,

    /// Co-occurrence: concepts that appear in the same scope
    /// (concept_a, concept_b) -> count (canonical order: a < b)
    pub co_occurrences: HashMap<(String, String), u32>,

    /// Access frequency per concept
    pub access_counts: HashMap<String, u32>,
}

impl ConceptUsage {
    /// Create a new empty usage tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a concept access
    pub fn record_access(&mut self, concept: &str) {
        self.concepts.insert(concept.to_string());
        *self.access_counts.entry(concept.to_string()).or_insert(0) += 1;
    }

    /// Record co-occurrence of two concepts
    pub fn record_co_occurrence(&mut self, a: &str, b: &str) {
        if a == b {
            return;
        }
        let key = if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        *self.co_occurrences.entry(key).or_insert(0) += 1;
    }

    /// Record all co-occurrences in a scope
    pub fn record_scope(&mut self, concepts_in_scope: &[&str]) {
        for concept in concepts_in_scope {
            self.record_access(concept);
        }

        // Record pairwise co-occurrences
        for i in 0..concepts_in_scope.len() {
            for j in (i + 1)..concepts_in_scope.len() {
                self.record_co_occurrence(concepts_in_scope[i], concepts_in_scope[j]);
            }
        }
    }

    /// Get the total access count
    pub fn total_accesses(&self) -> u32 {
        self.access_counts.values().sum()
    }

    /// Get the maximum co-occurrence count
    pub fn max_co_occurrence(&self) -> u32 {
        self.co_occurrences.values().copied().max().unwrap_or(0)
    }

    /// Merge another usage into this one
    pub fn merge(&mut self, other: &ConceptUsage) {
        for concept in &other.concepts {
            self.concepts.insert(concept.clone());
        }
        for (key, count) in &other.co_occurrences {
            *self.co_occurrences.entry(key.clone()).or_insert(0) += count;
        }
        for (concept, count) in &other.access_counts {
            *self.access_counts.entry(concept.clone()).or_insert(0) += count;
        }
    }
}

/// Extract concepts from a list of type strings
///
/// This is a simplified extractor that looks for Knowledge type patterns.
/// In a full implementation, this would walk the HIR.
pub fn extract_concepts_from_types(type_strings: &[&str]) -> ConceptUsage {
    let mut usage = ConceptUsage::new();

    for type_str in type_strings {
        // Look for Knowledge[ONTOLOGY:ID, ...] patterns
        if let Some(concepts) = extract_knowledge_concepts(type_str) {
            for concept in &concepts {
                usage.record_access(concept);
            }
            // Record co-occurrences within the same type
            let refs: Vec<&str> = concepts.iter().map(|s| s.as_str()).collect();
            usage.record_scope(&refs);
        }
    }

    usage
}

/// Extract concept CURIEs from a Knowledge type string
fn extract_knowledge_concepts(type_str: &str) -> Option<Vec<String>> {
    // Simple pattern matching for Knowledge[CURIE, ...]
    // Full implementation would parse the type properly

    if !type_str.contains("Knowledge") {
        return None;
    }

    let mut concepts = Vec::new();

    // Look for CURIE patterns: PREFIX:ID (e.g., CHEBI:15365)
    let curie_pattern = |s: &str| -> bool {
        s.contains(':')
            && s.split(':')
                .next()
                .map(|p| p.chars().all(|c| c.is_ascii_uppercase()))
                .unwrap_or(false)
    };

    // Split by common delimiters and check each token
    for token in type_str.split(&[',', '[', ']', ' ', '<', '>'][..]) {
        let token = token.trim();
        if curie_pattern(token) {
            concepts.push(token.to_string());
        }
    }

    if concepts.is_empty() {
        None
    } else {
        Some(concepts)
    }
}

/// Concept extractor that can be used with HIR
pub struct ConceptExtractor {
    usage: ConceptUsage,
    current_scope: Vec<String>,
}

impl ConceptExtractor {
    pub fn new() -> Self {
        Self {
            usage: ConceptUsage::new(),
            current_scope: Vec::new(),
        }
    }

    /// Enter a new scope (function, block, etc.)
    pub fn enter_scope(&mut self) {
        // Record co-occurrences from current scope before entering new one
        if !self.current_scope.is_empty() {
            let refs: Vec<&str> = self.current_scope.iter().map(|s| s.as_str()).collect();
            self.usage.record_scope(&refs);
        }
        self.current_scope.clear();
    }

    /// Exit the current scope
    pub fn exit_scope(&mut self) {
        if !self.current_scope.is_empty() {
            let refs: Vec<&str> = self.current_scope.iter().map(|s| s.as_str()).collect();
            self.usage.record_scope(&refs);
        }
        self.current_scope.clear();
    }

    /// Record a concept in the current scope
    pub fn record_concept(&mut self, concept: &str) {
        self.current_scope.push(concept.to_string());
    }

    /// Get the final usage
    pub fn finish(mut self) -> ConceptUsage {
        self.exit_scope();
        self.usage
    }
}

impl Default for ConceptExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== HIR-Based Extraction ====================

/// Extract concepts from HIR (High-level IR)
///
/// This walks the entire HIR looking for Knowledge types and tracks
/// their co-occurrence patterns based on scope.
pub fn extract_concepts_from_hir(hir: &Hir) -> ConceptUsage {
    let mut extractor = HirConceptExtractor::new();
    extractor.visit_hir(hir);
    extractor.finish()
}

/// HIR visitor that extracts concepts from Knowledge types
struct HirConceptExtractor {
    usage: ConceptUsage,
    /// Current scope's concepts (for co-occurrence tracking)
    scope_stack: Vec<Vec<String>>,
}

impl HirConceptExtractor {
    fn new() -> Self {
        Self {
            usage: ConceptUsage::new(),
            scope_stack: vec![Vec::new()],
        }
    }

    fn enter_scope(&mut self) {
        self.scope_stack.push(Vec::new());
    }

    fn exit_scope(&mut self) {
        if let Some(scope_concepts) = self.scope_stack.pop() {
            // Record co-occurrences within this scope
            let refs: Vec<&str> = scope_concepts.iter().map(|s| s.as_str()).collect();
            self.usage.record_scope(&refs);
        }
    }

    fn record_concept(&mut self, concept: &str) {
        self.usage.record_access(concept);
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.push(concept.to_string());
        }
    }

    fn finish(mut self) -> ConceptUsage {
        // Flush remaining scopes
        while !self.scope_stack.is_empty() {
            self.exit_scope();
        }
        self.usage
    }

    fn visit_hir(&mut self, hir: &Hir) {
        for item in &hir.items {
            self.visit_item(item);
        }
    }

    fn visit_item(&mut self, item: &HirItem) {
        match item {
            HirItem::Function(func) => {
                self.enter_scope();
                // Visit parameters
                for param in &func.ty.params {
                    self.visit_type(&param.ty);
                }
                // Visit return type
                self.visit_type(&func.ty.return_type);
                // Visit body
                self.visit_block(&func.body);
                self.exit_scope();
            }
            HirItem::Struct(s) => {
                self.enter_scope();
                for field in &s.fields {
                    self.visit_type(&field.ty);
                }
                self.exit_scope();
            }
            HirItem::Enum(e) => {
                self.enter_scope();
                for variant in &e.variants {
                    for field_ty in &variant.fields {
                        self.visit_type(field_ty);
                    }
                }
                self.exit_scope();
            }
            HirItem::Global(g) => {
                self.visit_type(&g.ty);
                self.visit_expr(&g.value);
            }
            HirItem::TypeAlias(ta) => {
                self.visit_type(&ta.ty);
            }
            HirItem::Impl(impl_) => {
                self.enter_scope();
                self.visit_type(&impl_.self_ty);
                for method in &impl_.methods {
                    self.enter_scope();
                    for param in &method.ty.params {
                        self.visit_type(&param.ty);
                    }
                    self.visit_type(&method.ty.return_type);
                    self.visit_block(&method.body);
                    self.exit_scope();
                }
                self.exit_scope();
            }
            HirItem::Trait(t) => {
                for method in &t.methods {
                    for param in &method.ty.params {
                        self.visit_type(&param.ty);
                    }
                    self.visit_type(&method.ty.return_type);
                }
            }
            HirItem::Effect(_) | HirItem::Handler(_) => {
                // Effects don't contain Knowledge types directly
            }
        }
    }

    fn visit_block(&mut self, block: &HirBlock) {
        self.enter_scope();
        for stmt in &block.stmts {
            self.visit_stmt(stmt);
        }
        self.exit_scope();
    }

    fn visit_stmt(&mut self, stmt: &HirStmt) {
        match stmt {
            HirStmt::Let { ty, value, .. } => {
                self.visit_type(ty);
                if let Some(expr) = value {
                    self.visit_expr(expr);
                }
            }
            HirStmt::Expr(expr) => {
                self.visit_expr(expr);
            }
            HirStmt::Assign { target, value } => {
                self.visit_expr(target);
                self.visit_expr(value);
            }
        }
    }

    fn visit_expr(&mut self, expr: &HirExpr) {
        // Visit the expression's type
        self.visit_type(&expr.ty);

        // Visit child expressions
        match &expr.kind {
            HirExprKind::Binary { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            HirExprKind::Unary { expr, .. } => {
                self.visit_expr(expr);
            }
            HirExprKind::Call { func, args } => {
                self.visit_expr(func);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                self.visit_expr(receiver);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            HirExprKind::Field { base, .. } | HirExprKind::TupleField { base, .. } => {
                self.visit_expr(base);
            }
            HirExprKind::Index { base, index } => {
                self.visit_expr(base);
                self.visit_expr(index);
            }
            HirExprKind::Cast { expr, target } => {
                self.visit_expr(expr);
                self.visit_type(target);
            }
            HirExprKind::Block(block) => {
                self.visit_block(block);
            }
            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.visit_expr(condition);
                self.visit_block(then_branch);
                if let Some(else_) = else_branch {
                    self.visit_expr(else_);
                }
            }
            HirExprKind::Match { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.visit_expr(&arm.body);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                }
            }
            HirExprKind::Loop(block) => {
                self.visit_block(block);
            }
            HirExprKind::While { condition, body } => {
                self.visit_expr(condition);
                self.visit_block(body);
            }
            HirExprKind::Return(expr) => {
                if let Some(e) = expr {
                    self.visit_expr(e);
                }
            }
            HirExprKind::Break(expr) => {
                if let Some(e) = expr {
                    self.visit_expr(e);
                }
            }
            HirExprKind::Closure { params, body } => {
                self.enter_scope();
                for param in params {
                    self.visit_type(&param.ty);
                }
                self.visit_expr(body);
                self.exit_scope();
            }
            HirExprKind::Tuple(exprs) | HirExprKind::Array(exprs) => {
                for e in exprs {
                    self.visit_expr(e);
                }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.visit_expr(s);
                }
                if let Some(e) = end {
                    self.visit_expr(e);
                }
            }
            HirExprKind::Struct { fields, .. } => {
                for (_, e) in fields {
                    self.visit_expr(e);
                }
            }
            HirExprKind::Variant { fields, .. } => {
                for e in fields {
                    self.visit_expr(e);
                }
            }
            HirExprKind::Ref { expr, .. }
            | HirExprKind::Deref(expr)
            | HirExprKind::Sample(expr) => {
                self.visit_expr(expr);
            }
            HirExprKind::Perform { args, .. } => {
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            HirExprKind::Handle { expr, .. } => {
                self.visit_expr(expr);
            }
            HirExprKind::Literal(_)
            | HirExprKind::Local(_)
            | HirExprKind::Global(_)
            | HirExprKind::Continue => {}

            // Epistemic expressions
            HirExprKind::Knowledge {
                value,
                epsilon,
                validity,
                ..
            } => {
                self.visit_expr(value);
                self.visit_expr(epsilon);
                if let Some(v) = validity {
                    self.visit_expr(v);
                }
            }
            HirExprKind::Do { value, .. } => {
                self.visit_expr(value);
            }
            HirExprKind::Counterfactual {
                factual,
                intervention,
                outcome,
            } => {
                self.visit_expr(factual);
                self.visit_expr(intervention);
                self.visit_expr(outcome);
            }
            HirExprKind::Query {
                target,
                given,
                interventions,
            } => {
                self.visit_expr(target);
                for g in given {
                    self.visit_expr(g);
                }
                for i in interventions {
                    self.visit_expr(i);
                }
            }
            HirExprKind::Observe { value, .. } => {
                self.visit_expr(value);
            }
            HirExprKind::EpsilonOf(e)
            | HirExprKind::ProvenanceOf(e)
            | HirExprKind::ValidityOf(e)
            | HirExprKind::Unwrap(e) => {
                self.visit_expr(e);
            }
            HirExprKind::OntologyTerm { namespace, term } => {
                // Register ontology term as a concept reference
                let curie = format!("{}:{}", namespace, term);
                self.usage.concepts.insert(curie);
            }

            // Async expressions - visit inner expressions
            HirExprKind::Await { future } => {
                self.visit_expr(future);
            }
            HirExprKind::Spawn { expr } => {
                self.visit_expr(expr);
            }
            HirExprKind::AsyncBlock { body } => {
                for stmt in &body.stmts {
                    self.visit_stmt(stmt);
                }
            }
            HirExprKind::Join { futures } => {
                for f in futures {
                    self.visit_expr(f);
                }
            }
            HirExprKind::Select { arms } => {
                for arm in arms {
                    self.visit_expr(&arm.future);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                }
            }
        }
    }

    fn visit_type(&mut self, ty: &HirType) {
        // Extract concepts from Knowledge types
        // Knowledge types appear as Named { name: "Knowledge", args: [...] }
        match ty {
            HirType::Named { name, args } => {
                if name == "Knowledge" {
                    // Extract ontology binding from args
                    // Convention: first arg with CURIE pattern is the domain
                    for arg in args {
                        if let Some(curie) = extract_curie_from_type(arg) {
                            self.record_concept(&curie);
                        }
                        // Recursively visit nested types
                        self.visit_type(arg);
                    }
                } else {
                    // Visit type arguments
                    for arg in args {
                        self.visit_type(arg);
                    }
                }
            }
            HirType::Ref { inner, .. } => {
                self.visit_type(inner);
            }
            HirType::Array { element, .. } => {
                self.visit_type(element);
            }
            HirType::Tuple(tys) => {
                for t in tys {
                    self.visit_type(t);
                }
            }
            HirType::Fn {
                params,
                return_type,
            } => {
                for p in params {
                    self.visit_type(p);
                }
                self.visit_type(return_type);
            }
            _ => {}
        }
    }
}

/// Extract a CURIE string from a type if it represents an ontology term
fn extract_curie_from_type(ty: &HirType) -> Option<String> {
    match ty {
        HirType::Named { name, .. } => {
            // Check if name is a CURIE (PREFIX:ID pattern)
            if is_curie(name) {
                Some(name.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if a string is a CURIE (PREFIX:ID pattern)
fn is_curie(s: &str) -> bool {
    if let Some(colon_pos) = s.find(':') {
        let prefix = &s[..colon_pos];
        let local = &s[colon_pos + 1..];

        // Prefix should be uppercase letters
        !prefix.is_empty()
            && prefix
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            && !local.is_empty()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_usage_basic() {
        let mut usage = ConceptUsage::new();
        usage.record_access("CHEBI:15365");
        usage.record_access("CHEBI:15365");
        usage.record_access("GO:0008150");

        assert_eq!(usage.concepts.len(), 2);
        assert_eq!(usage.access_counts.get("CHEBI:15365"), Some(&2));
        assert_eq!(usage.access_counts.get("GO:0008150"), Some(&1));
    }

    #[test]
    fn test_co_occurrence() {
        let mut usage = ConceptUsage::new();
        usage.record_scope(&["CHEBI:15365", "GO:0008150", "DOID:0001"]);

        // Should have 3 co-occurrences: (CHEBI, DOID), (CHEBI, GO), (DOID, GO)
        assert_eq!(usage.co_occurrences.len(), 3);

        // Check canonical ordering
        assert!(
            usage
                .co_occurrences
                .contains_key(&("CHEBI:15365".to_string(), "GO:0008150".to_string()))
        );
    }

    #[test]
    fn test_extract_from_types() {
        let types = vec![
            "Knowledge[CHEBI:15365, 0.9, PHARMA]",
            "Knowledge[GO:0008150, 0.8, BIOLOGY]",
            "int",
            "string",
        ];

        let usage = extract_concepts_from_types(&types);

        assert!(usage.concepts.contains("CHEBI:15365"));
        assert!(usage.concepts.contains("GO:0008150"));
        assert_eq!(usage.concepts.len(), 2);
    }

    #[test]
    fn test_concept_extractor() {
        let mut extractor = ConceptExtractor::new();

        extractor.enter_scope();
        extractor.record_concept("CHEBI:15365");
        extractor.record_concept("GO:0008150");
        extractor.exit_scope();

        extractor.enter_scope();
        extractor.record_concept("DOID:0001");
        extractor.exit_scope();

        let usage = extractor.finish();

        assert_eq!(usage.concepts.len(), 3);
        // CHEBI and GO should have co-occurrence
        assert!(
            usage
                .co_occurrences
                .contains_key(&("CHEBI:15365".to_string(), "GO:0008150".to_string()))
        );
    }
}
