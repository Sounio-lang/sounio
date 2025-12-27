//! Tests for refinement type parsing
//!
//! Refinement types have the syntax: `{ identifier: type | predicate }`
//! Examples:
//! - `{ x: i32 | x > 0 }` - positive integers
//! - `{ p: f64 | 0.0 <= p && p <= 1.0 }` - probabilities

#[cfg(test)]
mod tests {
    use crate::ast::{BinaryOp, Expr, TypeExpr};
    use crate::lexer::lex;
    use crate::parser::Parser;

    fn parse_type(input: &str) -> Result<TypeExpr, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let mut parser = Parser::new(&tokens);
        parser.parse_type().map_err(|e| format!("{:?}", e))
    }

    // ==================== BASIC REFINEMENT TYPE TESTS ====================

    #[test]
    fn test_refinement_type_positive_int() {
        let result = parse_type("{ x: i32 | x > 0 }");
        assert!(
            result.is_ok(),
            "Failed to parse {{ x: i32 | x > 0 }}: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement {
            var,
            base_type,
            predicate,
        }) = result
        {
            assert_eq!(var, "x");
            // Check base type is i32
            if let TypeExpr::Named { path, .. } = *base_type {
                assert_eq!(path.segments, vec!["i32"]);
            } else {
                panic!("Expected Named type for base_type");
            }
            // Check predicate is a binary comparison
            if let Expr::Binary { op, .. } = *predicate {
                assert_eq!(op, BinaryOp::Gt);
            } else {
                panic!("Expected Binary expression for predicate");
            }
        } else {
            panic!("Expected Refinement type, got: {:?}", result);
        }
    }

    #[test]
    fn test_refinement_type_non_negative() {
        let result = parse_type("{ n: i64 | n >= 0 }");
        assert!(
            result.is_ok(),
            "Failed to parse {{ n: i64 | n >= 0 }}: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { var, predicate, .. }) = result {
            assert_eq!(var, "n");
            if let Expr::Binary { op, .. } = *predicate {
                assert_eq!(op, BinaryOp::Ge);
            } else {
                panic!("Expected Binary expression for predicate");
            }
        } else {
            panic!("Expected Refinement type");
        }
    }

    #[test]
    fn test_refinement_type_probability() {
        let result = parse_type("{ p: f64 | 0.0 <= p && p <= 1.0 }");
        assert!(
            result.is_ok(),
            "Failed to parse probability refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { var, predicate, .. }) = result {
            assert_eq!(var, "p");
            // Should be a conjunction (And)
            if let Expr::Binary { op, .. } = *predicate {
                assert_eq!(op, BinaryOp::And);
            } else {
                panic!("Expected Binary And expression for predicate");
            }
        } else {
            panic!("Expected Refinement type");
        }
    }

    #[test]
    fn test_refinement_type_bounded_range() {
        let result = parse_type("{ val: i32 | val > 0 && val < 100 }");
        assert!(
            result.is_ok(),
            "Failed to parse bounded range refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement {
            var,
            base_type,
            predicate,
        }) = result
        {
            assert_eq!(var, "val");
            if let TypeExpr::Named { path, .. } = *base_type {
                assert_eq!(path.segments, vec!["i32"]);
            }
            if let Expr::Binary { op, .. } = *predicate {
                assert_eq!(op, BinaryOp::And);
            }
        } else {
            panic!("Expected Refinement type");
        }
    }

    // ==================== MEDICAL DOMAIN REFINEMENT TESTS ====================

    #[test]
    fn test_refinement_type_safe_dose() {
        // Test medical domain safe dose refinement
        let result = parse_type("{ dose: f64 | dose > 0.0 && dose <= 1000.0 }");
        assert!(
            result.is_ok(),
            "Failed to parse safe dose refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { var, .. }) = result {
            assert_eq!(var, "dose");
        } else {
            panic!("Expected Refinement type");
        }
    }

    #[test]
    fn test_refinement_type_valid_age() {
        let result = parse_type("{ age: i32 | age >= 0 && age <= 150 }");
        assert!(
            result.is_ok(),
            "Failed to parse valid age refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { var, .. }) = result {
            assert_eq!(var, "age");
        } else {
            panic!("Expected Refinement type");
        }
    }

    // ==================== COMPLEX PREDICATE TESTS ====================

    #[test]
    fn test_refinement_type_or_predicate() {
        let result = parse_type("{ x: i32 | x < 0 || x > 100 }");
        assert!(
            result.is_ok(),
            "Failed to parse or predicate refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { predicate, .. }) = result {
            if let Expr::Binary { op, .. } = *predicate {
                assert_eq!(op, BinaryOp::Or);
            } else {
                panic!("Expected Binary Or expression");
            }
        } else {
            panic!("Expected Refinement type");
        }
    }

    #[test]
    fn test_refinement_type_equality() {
        let result = parse_type("{ x: i32 | x == 42 }");
        assert!(
            result.is_ok(),
            "Failed to parse equality refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { predicate, .. }) = result {
            if let Expr::Binary { op, .. } = *predicate {
                assert_eq!(op, BinaryOp::Eq);
            } else {
                panic!("Expected Binary Eq expression");
            }
        } else {
            panic!("Expected Refinement type");
        }
    }

    #[test]
    fn test_refinement_type_not_equal() {
        let result = parse_type("{ x: i32 | x != 0 }");
        assert!(
            result.is_ok(),
            "Failed to parse not equal refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { predicate, .. }) = result {
            if let Expr::Binary { op, .. } = *predicate {
                assert_eq!(op, BinaryOp::Ne);
            } else {
                panic!("Expected Binary Ne expression");
            }
        } else {
            panic!("Expected Refinement type");
        }
    }

    // ==================== TYPE ALIAS INTEGRATION TESTS ====================

    #[test]
    fn test_refinement_in_type_alias_syntax() {
        // This tests that the syntax would work in a type alias context
        // (actual type alias parsing is tested elsewhere)
        let result = parse_type("{ x: i32 | x > 0 }");
        assert!(result.is_ok());
    }

    // ==================== FLOAT TYPE REFINEMENT TESTS ====================

    #[test]
    fn test_refinement_type_float_bounds() {
        let result = parse_type("{ x: f64 | x >= 0.0 && x <= 1.0 }");
        assert!(
            result.is_ok(),
            "Failed to parse float bounds refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { var, base_type, .. }) = result {
            assert_eq!(var, "x");
            if let TypeExpr::Named { path, .. } = *base_type {
                assert_eq!(path.segments, vec!["f64"]);
            }
        } else {
            panic!("Expected Refinement type");
        }
    }

    #[test]
    fn test_refinement_type_f32() {
        let result = parse_type("{ x: f32 | x > 0.0 }");
        assert!(
            result.is_ok(),
            "Failed to parse f32 refinement: {:?}",
            result
        );
        if let Ok(TypeExpr::Refinement { base_type, .. }) = result {
            if let TypeExpr::Named { path, .. } = *base_type {
                assert_eq!(path.segments, vec!["f32"]);
            }
        } else {
            panic!("Expected Refinement type");
        }
    }

    // ==================== ARITHMETIC IN PREDICATE TESTS ====================

    #[test]
    fn test_refinement_with_arithmetic() {
        let result = parse_type("{ x: i32 | x + 1 > 0 }");
        assert!(
            result.is_ok(),
            "Failed to parse refinement with arithmetic: {:?}",
            result
        );
    }

    #[test]
    fn test_refinement_with_multiplication() {
        let result = parse_type("{ x: i32 | x * 2 <= 100 }");
        assert!(
            result.is_ok(),
            "Failed to parse refinement with multiplication: {:?}",
            result
        );
    }

    // ==================== NESTED TYPE REFINEMENT TESTS ====================

    #[test]
    fn test_refinement_as_generic_arg() {
        // Test that refinement types can be used as generic arguments
        // This would be like Option<{ x: i32 | x > 0 }>
        let result = parse_type("{ x: i32 | x > 0 }");
        assert!(result.is_ok());
        // The refinement type is valid on its own - integration with
        // generics is tested in integration tests
    }

    // ==================== ERROR CASE TESTS ====================

    #[test]
    fn test_refinement_missing_pipe() {
        let result = parse_type("{ x: i32 x > 0 }");
        assert!(result.is_err(), "Should fail without pipe separator");
    }

    #[test]
    fn test_refinement_missing_colon() {
        let result = parse_type("{ x i32 | x > 0 }");
        assert!(result.is_err(), "Should fail without colon");
    }

    #[test]
    fn test_refinement_missing_predicate() {
        let result = parse_type("{ x: i32 | }");
        assert!(result.is_err(), "Should fail with empty predicate");
    }

    #[test]
    fn test_refinement_missing_closing_brace() {
        let result = parse_type("{ x: i32 | x > 0");
        assert!(result.is_err(), "Should fail without closing brace");
    }
}
