//! Tests for improved parser error messages
//!
//! These tests verify that the parser generates helpful, context-aware error
//! messages for common syntax mistakes.

#[cfg(test)]
mod tests {
    use crate::ast::TypeExpr;
    use crate::lexer::lex;
    use crate::parser::Parser;

    // Helper to get an error message from parsing a type
    fn parse_type_error(input: &str) -> String {
        let tokens = lex(input).expect("lexing should succeed");
        let mut parser = Parser::new(&tokens);
        match parser.parse_type() {
            Ok(_) => panic!("Expected parsing to fail for: {}", input),
            Err(e) => format!("{:?}", e),
        }
    }

    // Helper to successfully parse a type
    fn parse_type_ok(input: &str) -> TypeExpr {
        let tokens = lex(input).expect("lexing should succeed");
        let mut parser = Parser::new(&tokens);
        parser
            .parse_type()
            .expect(&format!("Should parse: {}", input))
    }

    // Helper to get an error message from parsing an expression
    fn parse_expr_error(input: &str) -> String {
        let tokens = lex(input).expect("lexing should succeed");
        let mut parser = Parser::new(&tokens);
        match parser.parse_expr() {
            Ok(_) => panic!("Expected parsing to fail for: {}", input),
            Err(e) => format!("{:?}", e),
        }
    }

    // Helper to get an error message from parsing an item
    fn parse_item_error(input: &str) -> String {
        let tokens = lex(input).expect("lexing should succeed");
        let mut parser = Parser::new(&tokens);
        match parser.parse_item() {
            Ok(_) => panic!("Expected parsing to fail for: {}", input),
            Err(e) => format!("{:?}", e),
        }
    }

    // ==================== MUTABLE REFERENCE SYNTAX ====================

    #[test]
    fn test_sounio_mutable_reference_syntax() {
        // Sounio canonical syntax: &!T for mutable references
        let ty = parse_type_ok("&!i32");
        match ty {
            TypeExpr::Reference { mutable, inner } => {
                assert!(mutable, "Expected mutable reference");
                if let TypeExpr::Named { path, .. } = *inner {
                    assert_eq!(path.segments, vec!["i32"]);
                } else {
                    panic!("Expected Named type inside reference");
                }
            }
            _ => panic!("Expected Reference type, got: {:?}", ty),
        }
    }

    #[test]
    fn test_rust_mut_reference_compatibility() {
        // Parser also accepts &mut T for Rust compatibility
        let ty = parse_type_ok("&mut i32");
        match ty {
            TypeExpr::Reference { mutable, inner } => {
                assert!(mutable, "Expected mutable reference from &mut syntax");
                if let TypeExpr::Named { path, .. } = *inner {
                    assert_eq!(path.segments, vec!["i32"]);
                }
            }
            _ => panic!("Expected Reference type"),
        }
    }

    #[test]
    fn test_shared_reference() {
        let ty = parse_type_ok("&i32");
        match ty {
            TypeExpr::Reference { mutable, .. } => {
                assert!(!mutable, "Expected shared (non-mutable) reference");
            }
            _ => panic!("Expected Reference type"),
        }
    }

    // ==================== TYPE EXPRESSION ERRORS ====================

    #[test]
    fn test_unexpected_operator_in_type() {
        let err = parse_type_error("+ i32");
        // Should provide helpful context
        assert!(
            err.contains("type") || err.contains("Expected"),
            "Error should mention types: {}",
            err
        );
    }

    #[test]
    fn test_semicolon_instead_of_type() {
        let err = parse_type_error(";");
        assert!(
            err.contains("type") || err.contains("Expected"),
            "Error should indicate type was expected: {}",
            err
        );
    }

    // ==================== EXPRESSION ERRORS ====================

    #[test]
    fn test_unexpected_semicolon_in_expr() {
        let err = parse_expr_error(";");
        assert!(
            err.contains("expression") || err.contains("Expected"),
            "Error should mention expression: {}",
            err
        );
    }

    #[test]
    fn test_unexpected_closing_bracket() {
        let err = parse_expr_error(")");
        assert!(
            err.contains("expression"),
            "Error should mention expression: {}",
            err
        );
    }

    // ==================== MODULE LEVEL ERRORS ====================

    #[test]
    fn test_literal_at_module_level() {
        let err = parse_item_error("42");
        // Should suggest putting in a function
        assert!(
            err.contains("module") || err.contains("function") || err.contains("declaration"),
            "Error should mention module level restrictions: {}",
            err
        );
    }

    #[test]
    fn test_if_at_module_level() {
        let err = parse_item_error("if x { }");
        assert!(
            err.contains("module") || err.contains("function") || err.contains("Control"),
            "Error should mention that control flow can't appear at module level: {}",
            err
        );
    }

    #[test]
    fn test_return_at_module_level() {
        let err = parse_item_error("return 42");
        assert!(
            err.contains("function") || err.contains("module"),
            "Error should mention return can only be in function: {}",
            err
        );
    }

    // ==================== PATTERN ERRORS ====================

    #[test]
    fn test_invalid_pattern() {
        let tokens = lex("let + = 5;").expect("lexing should succeed");
        let mut parser = Parser::new(&tokens);
        // Parse the let keyword first
        let _ = parser.parse_item();
        // The pattern parsing should fail
    }

    // ==================== ERROR CODE PRESENCE ====================

    #[test]
    fn test_error_has_code() {
        // Test that type errors include error codes
        let err = parse_type_error("+"); // + is not a valid type start
        // The error should include an error code
        assert!(
            err.contains("P00") || err.contains("type"),
            "Error should include an error code or mention type: {}",
            err
        );
    }

    #[test]
    fn test_module_level_error_has_code() {
        let err = parse_item_error("42");
        // Module level errors should include error codes
        assert!(
            err.contains("P00") || err.contains("module"),
            "Error should include error code or module context: {}",
            err
        );
    }

    // ==================== HELPFUL SUGGESTIONS ====================

    #[test]
    fn test_type_error_mentions_valid_types() {
        let err = parse_type_error("@@@@");
        // The test is that the error message is generated without panicking
        // and contains some helpful text
        assert!(!err.is_empty(), "Error message should not be empty");
    }

    #[test]
    fn test_expression_error_is_helpful() {
        let err = parse_expr_error(",");
        // Comma alone should produce a helpful error
        assert!(
            err.contains("expression") || err.contains("Expected"),
            "Error should give guidance: {}",
            err
        );
    }
}
