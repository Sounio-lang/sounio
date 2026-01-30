//! Tests for epistemic type and expression parsing

#[cfg(test)]
mod tests {
    use crate::ast::{Expr, TypeExpr};
    use crate::lexer::lex;
    use crate::parser::Parser;

    fn parse_expr(input: &str) -> Result<Expr, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let mut parser = Parser::new(&tokens);
        parser.parse_expr().map_err(|e| format!("{:?}", e))
    }

    fn parse_type(input: &str) -> Result<TypeExpr, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let mut parser = Parser::new(&tokens);
        parser.parse_type().map_err(|e| format!("{:?}", e))
    }

    // ==================== KNOWLEDGE TYPE TESTS ====================

    #[test]
    fn test_knowledge_type_simple() {
        let result = parse_type("Knowledge[f64]");
        assert!(
            result.is_ok(),
            "Failed to parse Knowledge[f64]: {:?}",
            result
        );
        if let Ok(TypeExpr::Knowledge { value_type, .. }) = result {
            assert!(matches!(*value_type, TypeExpr::Named { .. }));
        } else {
            panic!("Expected Knowledge type");
        }
    }

    #[test]
    fn test_knowledge_type_with_epsilon() {
        let result = parse_type("Knowledge[f64, epsilon < 0.05]");
        assert!(
            result.is_ok(),
            "Failed to parse Knowledge with epsilon: {:?}",
            result
        );
        if let Ok(TypeExpr::Knowledge { epsilon, .. }) = result {
            assert!(epsilon.is_some(), "Expected epsilon bound");
        } else {
            panic!("Expected Knowledge type");
        }
    }

    #[test]
    fn test_knowledge_type_with_validity() {
        let result = parse_type("Knowledge[f64, Valid(24)]");
        assert!(
            result.is_ok(),
            "Failed to parse Knowledge with validity: {:?}",
            result
        );
        if let Ok(TypeExpr::Knowledge { validity, .. }) = result {
            assert!(validity.is_some(), "Expected validity condition");
        } else {
            panic!("Expected Knowledge type");
        }
    }

    #[test]
    fn test_knowledge_type_with_provenance() {
        let result = parse_type("Knowledge[f64, Derived]");
        assert!(
            result.is_ok(),
            "Failed to parse Knowledge with provenance: {:?}",
            result
        );
        if let Ok(TypeExpr::Knowledge { provenance, .. }) = result {
            assert!(provenance.is_some(), "Expected provenance marker");
        } else {
            panic!("Expected Knowledge type");
        }
    }

    // ==================== QUANTITY TYPE TESTS ====================

    #[test]
    fn test_quantity_type_simple() {
        let result = parse_type("Quantity[f64, meters]");
        assert!(
            result.is_ok(),
            "Failed to parse Quantity[f64, meters]: {:?}",
            result
        );
        if let Ok(TypeExpr::Quantity { unit, .. }) = result {
            assert_eq!(unit.base_units.len(), 1);
            assert_eq!(unit.base_units[0].0, "meters");
            assert_eq!(unit.base_units[0].1, 1);
        } else {
            panic!("Expected Quantity type");
        }
    }

    #[test]
    fn test_quantity_type_compound_unit() {
        let result = parse_type("Quantity[f64, kg*m/s^2]");
        assert!(
            result.is_ok(),
            "Failed to parse compound unit: {:?}",
            result
        );
        if let Ok(TypeExpr::Quantity { unit, .. }) = result {
            assert!(unit.base_units.len() >= 2);
        } else {
            panic!("Expected Quantity type");
        }
    }

    // ==================== TENSOR TYPE TESTS ====================

    #[test]
    fn test_tensor_type_named_dims() {
        let result = parse_type("Tensor[f32, (batch, channels, height, width)]");
        assert!(
            result.is_ok(),
            "Failed to parse Tensor with named dims: {:?}",
            result
        );
        if let Ok(TypeExpr::Tensor { shape, .. }) = result {
            assert_eq!(shape.len(), 4);
        } else {
            panic!("Expected Tensor type");
        }
    }

    #[test]
    fn test_tensor_type_fixed_dims() {
        let result = parse_type("Tensor[f32, (32, 64, 128)]");
        assert!(
            result.is_ok(),
            "Failed to parse Tensor with fixed dims: {:?}",
            result
        );
        if let Ok(TypeExpr::Tensor { shape, .. }) = result {
            assert_eq!(shape.len(), 3);
        } else {
            panic!("Expected Tensor type");
        }
    }

    // ==================== ONTOLOGY TYPE TESTS ====================

    #[test]
    fn test_ontology_type_simple() {
        let result = parse_type("OntologyTerm[SNOMED]");
        assert!(
            result.is_ok(),
            "Failed to parse OntologyTerm[SNOMED]: {:?}",
            result
        );
        if let Ok(TypeExpr::Ontology { ontology, term }) = result {
            assert_eq!(ontology, "SNOMED");
            assert!(term.is_none());
        } else {
            panic!("Expected Ontology type");
        }
    }

    #[test]
    fn test_ontology_type_with_term() {
        let result = parse_type("OntologyTerm[SNOMED:12345]");
        assert!(
            result.is_ok(),
            "Failed to parse OntologyTerm with term: {:?}",
            result
        );
        if let Ok(TypeExpr::Ontology { ontology, term }) = result {
            assert_eq!(ontology, "SNOMED");
            assert_eq!(term, Some("12345".to_string()));
        } else {
            panic!("Expected Ontology type");
        }
    }

    // ==================== DO EXPRESSION TESTS ====================

    #[test]
    fn test_do_expr_simple() {
        let result = parse_expr("do(X = 1)");
        assert!(result.is_ok(), "Failed to parse do(X = 1): {:?}", result);
        if let Ok(Expr::Do { interventions, .. }) = result {
            assert_eq!(interventions.len(), 1);
            assert_eq!(interventions[0].0, "X");
        } else {
            panic!("Expected Do expression");
        }
    }

    #[test]
    fn test_do_expr_multiple_interventions() {
        let result = parse_expr("do(X = 1, Y = 2)");
        assert!(
            result.is_ok(),
            "Failed to parse do with multiple interventions: {:?}",
            result
        );
        if let Ok(Expr::Do { interventions, .. }) = result {
            assert_eq!(interventions.len(), 2);
        } else {
            panic!("Expected Do expression");
        }
    }

    // ==================== COUNTERFACTUAL EXPRESSION TESTS ====================

    #[test]
    fn test_counterfactual_expr() {
        let result = parse_expr("counterfactual { observed; do(X = 1); outcome }");
        assert!(
            result.is_ok(),
            "Failed to parse counterfactual: {:?}",
            result
        );
        if let Ok(Expr::Counterfactual { .. }) = result {
            // Success
        } else {
            panic!("Expected Counterfactual expression");
        }
    }

    // ==================== OBSERVE EXPRESSION TESTS ====================

    #[test]
    fn test_observe_expr_tilde() {
        let result = parse_expr("observe(data ~ Normal)");
        assert!(
            result.is_ok(),
            "Failed to parse observe with ~: {:?}",
            result
        );
        if let Ok(Expr::Observe { .. }) = result {
            // Success
        } else {
            panic!("Expected Observe expression");
        }
    }

    #[test]
    fn test_observe_expr_comma() {
        let result = parse_expr("observe(data, Normal)");
        assert!(
            result.is_ok(),
            "Failed to parse observe with comma: {:?}",
            result
        );
        if let Ok(Expr::Observe { .. }) = result {
            // Success
        } else {
            panic!("Expected Observe expression");
        }
    }

    // ==================== KNOWLEDGE EXPRESSION TESTS ====================

    #[test]
    fn test_knowledge_expr_struct_syntax() {
        let result = parse_expr("Knowledge { value: 42 }");
        assert!(
            result.is_ok(),
            "Failed to parse Knowledge struct: {:?}",
            result
        );
        if let Ok(Expr::KnowledgeExpr { value, .. }) = result {
            // Success
        } else {
            panic!("Expected KnowledgeExpr expression");
        }
    }

    #[test]
    fn test_knowledge_expr_with_epsilon() {
        let result = parse_expr("Knowledge { value: x, epsilon: 0.05 }");
        assert!(
            result.is_ok(),
            "Failed to parse Knowledge with epsilon: {:?}",
            result
        );
        if let Ok(Expr::KnowledgeExpr { epsilon, .. }) = result {
            assert!(epsilon.is_some());
        } else {
            panic!("Expected KnowledgeExpr expression");
        }
    }

    #[test]
    fn test_knowledge_expr_constructor() {
        let result = parse_expr("Knowledge::new(42)");
        assert!(
            result.is_ok(),
            "Failed to parse Knowledge::new: {:?}",
            result
        );
        if let Ok(Expr::KnowledgeExpr { .. }) = result {
            // Success
        } else {
            panic!("Expected KnowledgeExpr expression");
        }
    }

    #[test]
    fn test_knowledge_expr_constructor_full() {
        let result = parse_expr("Knowledge::new(42, 0.05, valid, derived)");
        assert!(
            result.is_ok(),
            "Failed to parse Knowledge::new with all args: {:?}",
            result
        );
        if let Ok(Expr::KnowledgeExpr {
            epsilon,
            validity,
            provenance,
            ..
        }) = result
        {
            assert!(epsilon.is_some());
            assert!(validity.is_some());
            assert!(provenance.is_some());
        } else {
            panic!("Expected KnowledgeExpr expression");
        }
    }

    // ==================== QUERY EXPRESSION TESTS ====================

    #[test]
    fn test_query_expr_simple() {
        let result = parse_expr("query P(Y)");
        assert!(result.is_ok(), "Failed to parse query P(Y): {:?}", result);
        if let Ok(Expr::Query { .. }) = result {
            // Success
        } else {
            panic!("Expected Query expression");
        }
    }

    #[test]
    fn test_query_expr_conditional() {
        let result = parse_expr("query P(Y | X)");
        assert!(
            result.is_ok(),
            "Failed to parse query P(Y | X): {:?}",
            result
        );
        if let Ok(Expr::Query { given, .. }) = result {
            assert!(!given.is_empty());
        } else {
            panic!("Expected Query expression");
        }
    }

    #[test]
    fn test_query_expr_with_intervention() {
        let result = parse_expr("query P(Y | do(X = 1))");
        assert!(
            result.is_ok(),
            "Failed to parse query with intervention: {:?}",
            result
        );
        if let Ok(Expr::Query { interventions, .. }) = result {
            assert!(!interventions.is_empty());
        } else {
            panic!("Expected Query expression");
        }
    }
}
