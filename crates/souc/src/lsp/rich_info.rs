//! Rich Type Information for LSP
//!
//! This module provides a unified `RichTypeInfo` structure that combines:
//! - Base type information
//! - Effect annotations
//! - Units of measure
//! - Epistemic status (confidence, provenance)
//! - Refinement predicates
//!
//! This enables the LSP to provide comprehensive hover, inlay hints,
//! and diagnostics that surface Sounio's advanced type system features.

use tower_lsp::lsp_types::*;

use crate::ast::{Ast, FnDef, Item, TypeExpr};
use crate::common::Span;
use crate::resolve::SymbolTable;

use super::query_bridge::{
    CausalIdMethod, CausalInfo, EffectInfo, EpistemicInfo, RefinementInfo, SmtStatus, UnitInfo,
};

/// Rich type information combining all Sounio type system features
#[derive(Debug, Clone)]
pub struct RichTypeInfo {
    /// Base type (from type checker)
    pub base_type: Option<TypeDescription>,

    /// Effect set for this expression/function
    pub effects: Option<EffectInfo>,

    /// Unit of measure (for numeric types)
    pub unit: Option<UnitInfo>,

    /// Epistemic status (confidence, provenance)
    pub epistemic: Option<EpistemicInfo>,

    /// Refinement predicate (for refined types)
    pub refinement: Option<RefinementInfo>,

    /// Causal graph information (for CausalKnowledge types)
    pub causal: Option<CausalInfo>,

    /// Source span
    pub span: Option<Span>,

    /// Symbol name (if this is a named entity)
    pub name: Option<String>,

    /// Kind of entity (function, variable, type, etc.)
    pub kind: RichInfoKind,
}

/// Kind of entity for RichTypeInfo
#[derive(Debug, Clone, Default)]
pub enum RichInfoKind {
    #[default]
    Unknown,
    Function,
    Variable,
    Parameter,
    Type,
    Effect,
    Struct,
    Enum,
    Constant,
    Field,
    Module,
}

/// Description of a type (simpler than full HirType for display)
#[derive(Debug, Clone)]
pub struct TypeDescription {
    /// Formatted type string
    pub formatted: String,

    /// Is this a tensor type?
    pub is_tensor: bool,

    /// Tensor shape (if tensor)
    pub tensor_shape: Option<Vec<String>>,

    /// Is this an epistemic type (Knowledge<T>)?
    pub is_epistemic: bool,

    /// Is this a unit-annotated type?
    pub has_unit: bool,

    /// Is this a refined type?
    pub is_refined: bool,
}

impl RichTypeInfo {
    /// Create empty rich info
    pub fn new() -> Self {
        Self {
            base_type: None,
            effects: None,
            unit: None,
            epistemic: None,
            refinement: None,
            causal: None,
            span: None,
            name: None,
            kind: RichInfoKind::Unknown,
        }
    }

    /// Create rich info for a function
    pub fn for_function(f: &FnDef) -> Self {
        let mut info = Self::new();
        info.name = Some(f.name.clone());
        info.kind = RichInfoKind::Function;
        info.span = Some(f.span.clone());

        // Extract effects
        info.effects = Some(EffectInfo {
            effects: f.effects.iter().map(|e| e.name.to_string()).collect(),
            sources: vec![],
        });

        // Format return type
        if let Some(ref ret_type) = f.return_type {
            info.base_type = Some(TypeDescription::from_type_expr(ret_type));
        } else {
            info.base_type = Some(TypeDescription {
                formatted: "()".to_string(),
                is_tensor: false,
                tensor_shape: None,
                is_epistemic: false,
                has_unit: false,
                is_refined: false,
            });
        }

        info
    }

    /// Create rich info for a variable/binding
    pub fn for_variable(name: &str, ty: Option<&TypeExpr>, mutable: bool) -> Self {
        let mut info = Self::new();
        info.name = Some(name.to_string());
        info.kind = RichInfoKind::Variable;

        if let Some(ty) = ty {
            info.base_type = Some(TypeDescription::from_type_expr(ty));
        }

        info
    }

    /// Format as hover content
    pub fn format_hover(&self) -> Vec<MarkedString> {
        let mut parts = Vec::new();

        // Type signature in code block
        if let Some(ref name) = self.name {
            let sig = self.format_signature(name);
            parts.push(MarkedString::LanguageString(LanguageString {
                language: "sounio".to_string(),
                value: sig,
            }));
        }

        // Base type description
        if let Some(ref ty) = self.base_type {
            if ty.is_tensor {
                if let Some(ref shape) = ty.tensor_shape {
                    parts.push(MarkedString::String(format!(
                        "**Shape:** `[{}]`",
                        shape.join(", ")
                    )));
                }
            }
        }

        // Effects section
        if let Some(ref effects) = self.effects {
            if effects.effects.is_empty() {
                parts.push(MarkedString::String(
                    "**Effects:** *pure* (no side effects)".to_string(),
                ));
            } else {
                let effects_str = effects.effects.join(", ");
                parts.push(MarkedString::String(format!(
                    "**Effects:** `{}`",
                    effects_str
                )));
            }
        }

        // Units section
        if let Some(ref unit) = self.unit {
            let mut unit_info = format!("**Unit:** `{}`", unit.unit);
            if !unit.dimension.is_empty() {
                unit_info.push_str(&format!(" (dimension: {})", unit.dimension));
            }
            parts.push(MarkedString::String(unit_info));
        }

        // Epistemic section
        if let Some(ref epistemic) = self.epistemic {
            let badge = confidence_badge(epistemic.confidence);
            parts.push(MarkedString::String(format!(
                "**Confidence:** {} {:.0}%",
                badge,
                epistemic.confidence * 100.0
            )));
            if !epistemic.source.is_empty() {
                parts.push(MarkedString::String(format!(
                    "**Source:** {}",
                    epistemic.source
                )));
            }
        }

        // Refinement section
        if let Some(ref refinement) = self.refinement {
            parts.push(MarkedString::String(format!(
                "**Refinement:** `{{ {} | {} }}`",
                refinement.variable, refinement.predicate
            )));
            if refinement.status == SmtStatus::Verified {
                parts.push(MarkedString::String("**SMT Status:** verified".to_string()));
            }
        }

        // Causal section
        if let Some(ref causal) = self.causal {
            let id_badge = if causal.identifiability.identifiable {
                "IDENTIFIABLE"
            } else {
                "NON-IDENTIFIABLE"
            };
            let method_str = match &causal.identifiability.method {
                CausalIdMethod::Backdoor => format!(
                    "backdoor on {{{}}}",
                    causal.identifiability.adjustment_set.join(", ")
                ),
                CausalIdMethod::Frontdoor => format!(
                    "frontdoor via {{{}}}",
                    causal.identifiability.adjustment_set.join(", ")
                ),
                CausalIdMethod::InstrumentalVariable => format!(
                    "IV: {{{}}}",
                    causal.identifiability.adjustment_set.join(", ")
                ),
                CausalIdMethod::Unconfounded => "unconfounded".to_string(),
                CausalIdMethod::NonIdentifiable => {
                    causal
                        .identifiability
                        .reason
                        .clone()
                        .unwrap_or_else(|| "confounded".to_string())
                }
                CausalIdMethod::Unknown => "unknown".to_string(),
            };
            parts.push(MarkedString::String(format!(
                "**Causal:** {} ({}) | {} nodes, {} edges",
                id_badge,
                method_str,
                causal.nodes.len(),
                causal.edges.len()
            )));
        }

        parts
    }

    /// Format as inlay hint label
    pub fn format_inlay(&self) -> String {
        let mut parts = Vec::new();

        // Type
        if let Some(ref ty) = self.base_type {
            parts.push(format!(": {}", ty.formatted));
        }

        // Effects (compact form)
        if let Some(ref effects) = self.effects {
            if !effects.effects.is_empty() {
                parts.push(format!("with {}", effects.effects.join(", ")));
            }
        }

        // Unit (compact form)
        if let Some(ref unit) = self.unit {
            parts.push(format!("@{}", unit.unit));
        }

        // Epistemic badge
        if let Some(ref epistemic) = self.epistemic {
            let badge = confidence_badge(epistemic.confidence);
            parts.push(format!("{} {:.0}%", badge, epistemic.confidence * 100.0));
        }

        // Causal badge
        if let Some(ref causal) = self.causal {
            if causal.identifiability.identifiable {
                parts.push("[ID]".to_string());
            } else {
                parts.push("[!ID]".to_string());
            }
        }

        parts.join(" ")
    }

    /// Format type signature
    fn format_signature(&self, name: &str) -> String {
        match self.kind {
            RichInfoKind::Function => {
                let ret_type = self
                    .base_type
                    .as_ref()
                    .map(|t| t.formatted.clone())
                    .unwrap_or_else(|| "()".to_string());

                let effects = self
                    .effects
                    .as_ref()
                    .filter(|e| !e.effects.is_empty())
                    .map(|e| format!(" with {}", e.effects.join(", ")))
                    .unwrap_or_default();

                format!("fn {}(...) -> {}{}", name, ret_type, effects)
            }
            RichInfoKind::Variable => {
                let ty = self
                    .base_type
                    .as_ref()
                    .map(|t| t.formatted.clone())
                    .unwrap_or_else(|| "_".to_string());

                format!("let {}: {}", name, ty)
            }
            RichInfoKind::Type | RichInfoKind::Struct => {
                format!("struct {}", name)
            }
            RichInfoKind::Enum => {
                format!("enum {}", name)
            }
            RichInfoKind::Effect => {
                format!("effect {}", name)
            }
            RichInfoKind::Constant => {
                let ty = self
                    .base_type
                    .as_ref()
                    .map(|t| t.formatted.clone())
                    .unwrap_or_else(|| "_".to_string());
                format!("const {}: {}", name, ty)
            }
            _ => name.to_string(),
        }
    }
}

impl Default for RichTypeInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeDescription {
    /// Create from AST type expression
    pub fn from_type_expr(ty: &TypeExpr) -> Self {
        let formatted = format_type_expr(ty);
        let is_tensor = matches!(
            ty,
            TypeExpr::Tensor { .. }
                | TypeExpr::ScientificArray { .. }
                | TypeExpr::ScientificMatrix { .. }
        );
        let is_epistemic = matches!(ty, TypeExpr::Knowledge { .. });
        let has_unit = matches!(ty, TypeExpr::Quantity { .. });
        let is_refined = matches!(ty, TypeExpr::Refinement { .. });

        let tensor_shape = match ty {
            TypeExpr::Tensor { shape, .. } => Some(shape.iter().map(format_tensor_dim).collect()),
            TypeExpr::ScientificArray { dim, .. } => Some(vec![format_tensor_dim(dim)]),
            TypeExpr::ScientificMatrix { rows, cols, .. } => {
                Some(vec![format_tensor_dim(rows), format_tensor_dim(cols)])
            }
            _ => None,
        };

        Self {
            formatted,
            is_tensor,
            tensor_shape,
            is_epistemic,
            has_unit,
            is_refined,
        }
    }
}

/// Get confidence badge emoji based on confidence level
pub fn confidence_badge(confidence: f64) -> &'static str {
    if confidence >= 0.95 {
        "\u{1F7E2}" // Green circle
    } else if confidence >= 0.8 {
        "\u{1F7E1}" // Yellow circle
    } else if confidence >= 0.5 {
        "\u{1F7E0}" // Orange circle
    } else if confidence >= 0.2 {
        "\u{1F534}" // Red circle
    } else {
        "\u{26AB}" // Black circle
    }
}

/// Format a type expression as a string
fn format_type_expr(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Unit => "()".to_string(),
        TypeExpr::SelfType => "Self".to_string(),
        TypeExpr::Never => "!".to_string(),
        TypeExpr::Infer => "_".to_string(),
        TypeExpr::Named { path, args, unit } => {
            let base = path.to_string();
            let type_args = if args.is_empty() {
                String::new()
            } else {
                let arg_strs: Vec<String> = args.iter().map(format_type_expr).collect();
                format!("<{}>", arg_strs.join(", "))
            };
            let unit_suffix = unit
                .as_ref()
                .map(|u| format!("<{}>", u))
                .unwrap_or_default();
            format!("{}{}{}", base, type_args, unit_suffix)
        }
        TypeExpr::Reference { mutable, inner } => {
            if *mutable {
                format!("&!{}", format_type_expr(inner))
            } else {
                format!("&{}", format_type_expr(inner))
            }
        }
        TypeExpr::Array { element, size } => {
            if size.is_some() {
                format!("[{}; N]", format_type_expr(element))
            } else {
                format!("[{}]", format_type_expr(element))
            }
        }
        TypeExpr::Tuple(types) => {
            let inner: Vec<String> = types.iter().map(format_type_expr).collect();
            format!("({})", inner.join(", "))
        }
        TypeExpr::Function {
            params,
            return_type,
            ..
        } => {
            let p: Vec<String> = params.iter().map(format_type_expr).collect();
            format!("fn({}) -> {}", p.join(", "), format_type_expr(return_type))
        }
        TypeExpr::Tensor {
            element_type,
            shape,
        } => {
            let dims: Vec<String> = shape.iter().map(format_tensor_dim).collect();
            format!(
                "Tensor[{}, ({})]",
                format_type_expr(element_type),
                dims.join(", ")
            )
        }
        TypeExpr::ScientificArray { element_type, dim } => {
            format!(
                "Array[{}, {}]",
                format_type_expr(element_type),
                format_tensor_dim(dim)
            )
        }
        TypeExpr::ScientificMatrix {
            element_type,
            rows,
            cols,
        } => {
            format!(
                "Matrix[{}, {}, {}]",
                format_type_expr(element_type),
                format_tensor_dim(rows),
                format_tensor_dim(cols)
            )
        }
        TypeExpr::Tile {
            element_type,
            tile_m,
            tile_n,
            layout,
        } => {
            let layout_str = layout
                .as_ref()
                .map(|l| format!(", {}", l))
                .unwrap_or_default();
            format!(
                "tile<{}, {}, {}{}>",
                format_type_expr(element_type),
                tile_m,
                tile_n,
                layout_str
            )
        }
        TypeExpr::Quantity { numeric_type, unit } => {
            let unit_str = unit
                .base_units
                .iter()
                .map(|(u, e)| {
                    if *e == 1 {
                        u.clone()
                    } else {
                        format!("{}^{}", u, e)
                    }
                })
                .collect::<Vec<_>>()
                .join("*");
            format!("{}@{}", format_type_expr(numeric_type), unit_str)
        }
        TypeExpr::Knowledge {
            value_type,
            epsilon,
            ..
        } => {
            let eps_str = epsilon
                .as_ref()
                .map(|_| ", ...".to_string())
                .unwrap_or_default();
            format!("Knowledge[{}{}]", format_type_expr(value_type), eps_str)
        }
        TypeExpr::Refinement {
            var,
            base_type,
            predicate: _,
        } => {
            format!("{{ {}: {} | ... }}", var, format_type_expr(base_type))
        }
        TypeExpr::Effected { inner, effects } => {
            let effects_str = effects.effects.join(", ");
            if effects_str.is_empty() {
                format_type_expr(inner)
            } else {
                format!("{} with {}", format_type_expr(inner), effects_str)
            }
        }
        TypeExpr::Linear { inner, linearity } => {
            let modifier = match linearity {
                crate::ast::LinearityKind::Linear => "linear ",
                crate::ast::LinearityKind::Affine => "affine ",
                crate::ast::LinearityKind::Relevant => "relevant ",
                crate::ast::LinearityKind::Unrestricted => "",
            };
            format!("{}{}", modifier, format_type_expr(inner))
        }
        TypeExpr::RawPointer { mutable, inner } => {
            if *mutable {
                format!("*!{}", format_type_expr(inner))
            } else {
                format!("*{}", format_type_expr(inner))
            }
        }
        TypeExpr::Ontology { ontology, term } => {
            if let Some(t) = term {
                format!("{}:{}", ontology, t)
            } else {
                ontology.clone()
            }
        }
        TypeExpr::Forall { vars, inner } => {
            let vars_str = vars
                .iter()
                .map(|v| v.name.clone())
                .collect::<Vec<_>>()
                .join(", ");
            format!("forall<{}> {}", vars_str, format_type_expr(inner))
        }
    }
}

/// Format a tensor dimension
fn format_tensor_dim(dim: &crate::ast::TensorDim) -> String {
    match dim {
        crate::ast::TensorDim::Named(name) => name.clone(),
        crate::ast::TensorDim::Fixed(size) => size.to_string(),
        crate::ast::TensorDim::Dynamic => "?".to_string(),
        crate::ast::TensorDim::Expr(_) => "_".to_string(),
    }
}

/// Extract rich type information from an AST for all entities
pub struct RichInfoExtractor;

impl RichInfoExtractor {
    /// Extract rich info for all functions in an AST
    pub fn extract_functions(ast: &Ast) -> Vec<RichTypeInfo> {
        ast.items
            .iter()
            .filter_map(|item| {
                if let Item::Function(f) = item {
                    Some(RichTypeInfo::for_function(f))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find rich info at a specific position
    pub fn info_at_position(
        ast: &Ast,
        symbols: Option<&SymbolTable>,
        line: u32,
        column: u32,
    ) -> Option<RichTypeInfo> {
        let offset = (line as usize) * 80 + (column as usize); // Rough estimate

        // Check functions
        for item in &ast.items {
            if let Item::Function(f) = item {
                if span_contains(&f.span, offset) {
                    return Some(RichTypeInfo::for_function(f));
                }
            }
        }

        None
    }
}

/// Check if a span contains an offset
fn span_contains(span: &Span, offset: usize) -> bool {
    offset >= span.start && offset < span.end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_badge() {
        assert_eq!(confidence_badge(0.99), "\u{1F7E2}"); // Green
        assert_eq!(confidence_badge(0.85), "\u{1F7E1}"); // Yellow
        assert_eq!(confidence_badge(0.60), "\u{1F7E0}"); // Orange
        assert_eq!(confidence_badge(0.30), "\u{1F534}"); // Red
        assert_eq!(confidence_badge(0.10), "\u{26AB}"); // Black
    }

    #[test]
    fn test_rich_info_function_kind() {
        let mut info = RichTypeInfo::new();
        info.name = Some("process".to_string());
        info.kind = RichInfoKind::Function;
        info.effects = Some(EffectInfo {
            effects: vec!["IO".to_string()],
            sources: vec![],
        });
        info.base_type = Some(TypeDescription {
            formatted: "i32".to_string(),
            is_tensor: false,
            tensor_shape: None,
            is_epistemic: false,
            has_unit: false,
            is_refined: false,
        });

        assert_eq!(info.name.as_deref(), Some("process"));
        assert!(matches!(info.kind, RichInfoKind::Function));
        assert!(info.effects.is_some());
        assert!(!info.effects.as_ref().unwrap().effects.is_empty());
        assert!(
            info.effects
                .as_ref()
                .unwrap()
                .effects
                .contains(&"IO".to_string())
        );

        // Test signature formatting
        let sig = info.format_signature("process");
        assert!(sig.contains("fn process"));
        assert!(sig.contains("i32"));
        assert!(sig.contains("with IO"));
    }

    #[test]
    fn test_format_hover() {
        let mut info = RichTypeInfo::new();
        info.name = Some("calculate".to_string());
        info.kind = RichInfoKind::Function;
        info.base_type = Some(TypeDescription {
            formatted: "f64".to_string(),
            is_tensor: false,
            tensor_shape: None,
            is_epistemic: false,
            has_unit: false,
            is_refined: false,
        });
        info.effects = Some(EffectInfo {
            effects: vec!["IO".to_string()],
            sources: vec![],
        });

        let hover_parts = info.format_hover();
        assert!(!hover_parts.is_empty());
    }

    #[test]
    fn test_format_inlay() {
        let mut info = RichTypeInfo::new();
        info.base_type = Some(TypeDescription {
            formatted: "f64".to_string(),
            is_tensor: false,
            tensor_shape: None,
            is_epistemic: false,
            has_unit: false,
            is_refined: false,
        });
        info.unit = Some(UnitInfo {
            unit: "m/s".to_string(),
            dimension: "L*T^-1".to_string(),
            is_dimensionless: false,
        });

        let inlay = info.format_inlay();
        assert!(inlay.contains(": f64"));
        assert!(inlay.contains("@m/s"));
    }
}
