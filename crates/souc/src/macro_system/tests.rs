//! Comprehensive tests for the macro system

#[cfg(test)]
mod tests {
    use crate::macro_system::*;
    use crate::lexer::{Token, TokenKind};
    use crate::Span;

    fn make_token(kind: TokenKind, text: &str) -> Token {
        Token {
            kind,
            text: text.to_string(),
            span: Span::default(),
        }
    }

    fn make_tree(kind: TokenKind, text: &str) -> TokenTree {
        TokenTree::Token(TokenWithCtx::new(make_token(kind, text)))
    }

    // ========================================================================
    // Token Tree Tests
    // ========================================================================

    #[test]
    fn test_token_tree_creation() {
        let token = make_token(TokenKind::Ident, "foo");
        let tree = TokenTree::Token(TokenWithCtx::new(token));
        
        assert!(tree.is_token(TokenKind::Ident));
        assert!(!tree.is_token(TokenKind::Int));
    }

    #[test]
    fn test_syntax_context_fresh() {
        let ctx1 = SyntaxContext::fresh();
        let ctx2 = SyntaxContext::fresh();
        
        assert_ne!(ctx1, ctx2);
        assert_eq!(ctx1, ctx1);
    }

    #[test]
    fn test_delimiter_types() {
        let delims = vec![
            Delimiter::Parenthesis,
            Delimiter::Bracket,
            Delimiter::Brace,
            Delimiter::None,
        ];
        
        assert_eq!(delims.len(), 4);
    }

    // ========================================================================
    // Pattern Matching Tests
    // ========================================================================

    #[test]
    fn test_pattern_match_token() {
        let mut matcher = PatternMatcher::new();
        let pattern = Pattern::Token(TokenKind::Ident);
        let input = vec![make_tree(TokenKind::Ident, "foo")];
        
        let result = matcher.match_pattern(&pattern, &input);
        assert!(result.is_ok());
        
        let (_, consumed) = result.unwrap();
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_pattern_match_literal() {
        let mut matcher = PatternMatcher::new();
        let pattern = Pattern::Literal("foo".to_string());
        let input = vec![make_tree(TokenKind::Ident, "foo")];
        
        let result = matcher.match_pattern(&pattern, &input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pattern_mismatch() {
        let mut matcher = PatternMatcher::new();
        let pattern = Pattern::Token(TokenKind::Int);
        let input = vec![make_tree(TokenKind::Ident, "foo")];
        
        let result = matcher.match_pattern(&pattern, &input);
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_metavar() {
        let mut matcher = PatternMatcher::new();
        let pattern = Pattern::MetaVar {
            name: "x".to_string(),
            fragment: FragmentSpecifier::Ident,
        };
        let input = vec![make_tree(TokenKind::Ident, "foo")];
        
        let result = matcher.match_pattern(&pattern, &input);
        assert!(result.is_ok());
        
        let (bindings, _) = result.unwrap();
        assert!(bindings.get_single("x").is_some());
    }

    #[test]
    fn test_pattern_repetition_zero_or_more() {
        let mut matcher = PatternMatcher::new();
        let pattern = Pattern::Repeat {
            patterns: vec![Pattern::Token(TokenKind::Ident)],
            separator: None,
            kind: RepeatKind::ZeroOrMore,
        };
        
        // Empty input should match
        let result = matcher.match_pattern(&pattern, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pattern_repetition_one_or_more() {
        let mut matcher = PatternMatcher::new();
        let pattern = Pattern::Repeat {
            patterns: vec![Pattern::Token(TokenKind::Ident)],
            separator: None,
            kind: RepeatKind::OneOrMore,
        };
        
        // Empty input should fail
        let result = matcher.match_pattern(&pattern, &[]);
        assert!(result.is_err());
    }

    // ========================================================================
    // Declarative Macro Tests
    // ========================================================================

    #[test]
    fn test_macro_definition() {
        let mut expander = MacroExpander::new();
        
        let macro_def = MacroDef {
            name: "test_macro".to_string(),
            arms: vec![],
            is_pub: false,
            is_exported: false,
            doc: None,
            span: Span::default(),
        };
        
        expander.define(macro_def);
        assert!(expander.macros.contains_key("test_macro"));
    }

    #[test]
    fn test_macro_expansion_simple() {
        let mut expander = MacroExpander::new();
        
        let arm = MacroArm {
            pattern: vec![],
            template: vec![
                TemplateTree::Token(TokenKind::Ident, "result".to_string()),
            ],
            guard: None,
        };
        
        let macro_def = MacroDef {
            name: "simple".to_string(),
            arms: vec![arm],
            is_pub: false,
            is_exported: false,
            doc: None,
            span: Span::default(),
        };
        
        expander.define(macro_def);
        
        let result = expander.expand("simple", vec![]);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Procedural Macro Tests
    // ========================================================================

    #[test]
    fn test_token_stream_creation() {
        let stream = TokenStream::new();
        assert!(stream.is_empty());
    }

    #[test]
    fn test_token_stream_push() {
        let mut stream = TokenStream::new();
        let tree = make_tree(TokenKind::Ident, "foo");
        
        stream.push(tree);
        assert!(!stream.is_empty());
    }

    #[test]
    fn test_proc_macro_registry() {
        let registry = ProcMacroRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    // ========================================================================
    // CTFE Tests
    // ========================================================================

    #[test]
    fn test_ctfe_context_creation() {
        let ctx = CtfeContext::new();
        assert_eq!(ctx.fuel, 1_000_000);
    }

    #[test]
    fn test_ctfe_const_value_display() {
        let values = vec![
            ConstValue::Unit,
            ConstValue::Bool(true),
            ConstValue::Int(42),
            ConstValue::Float(3.14),
            ConstValue::String("hello".to_string()),
        ];
        
        for val in values {
            let s = format!("{}", val);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_ctfe_binary_add() {
        let ctx = CtfeContext::new();
        let left = ConstValue::Int(5);
        let right = ConstValue::Int(3);
        
        let result = ctx.eval_binary_op("+", &left, &right);
        assert!(result.is_ok());
        
        if let Ok(ConstValue::Int(n)) = result {
            assert_eq!(n, 8);
        } else {
            panic!("Expected Int(8)");
        }
    }

    #[test]
    fn test_ctfe_binary_mul() {
        let ctx = CtfeContext::new();
        let left = ConstValue::Int(6);
        let right = ConstValue::Int(7);
        
        let result = ctx.eval_binary_op("*", &left, &right);
        assert!(result.is_ok());
        
        if let Ok(ConstValue::Int(n)) = result {
            assert_eq!(n, 42);
        } else {
            panic!("Expected Int(42)");
        }
    }

    #[test]
    fn test_ctfe_unary_neg() {
        let ctx = CtfeContext::new();
        let val = ConstValue::Int(5);
        
        let result = ctx.eval_unary_op("-", &val);
        assert!(result.is_ok());
        
        if let Ok(ConstValue::Int(n)) = result {
            assert_eq!(n, -5);
        } else {
            panic!("Expected Int(-5)");
        }
    }

    #[test]
    fn test_ctfe_comparison() {
        let ctx = CtfeContext::new();
        let left = ConstValue::Int(5);
        let right = ConstValue::Int(3);
        
        let result = ctx.eval_binary_op("<", &left, &right);
        assert!(result.is_ok());
        
        if let Ok(ConstValue::Bool(b)) = result {
            assert!(!b);  // 5 < 3 is false
        } else {
            panic!("Expected Bool");
        }
    }

    // ========================================================================
    // Scientific Macro Tests
    // ========================================================================

    #[test]
    fn test_dimension_length() {
        let dim = Dimension::length();
        assert_eq!(dim.length, 1);
        assert_eq!(dim.mass, 0);
    }

    #[test]
    fn test_dimension_mul() {
        let length = Dimension::length();
        let time = Dimension::time();
        
        let velocity = length.div(&time);
        assert_eq!(velocity.length, 1);
        assert_eq!(velocity.time, -1);
    }

    #[test]
    fn test_dimension_pow() {
        let length = Dimension::length();
        let area = length.pow(2);
        
        assert_eq!(area.length, 2);
    }

    #[test]
    fn test_parse_unit_meter() {
        let dim = parse_unit("m");
        assert!(dim.is_some());
        assert_eq!(dim.unwrap(), Dimension::length());
    }

    #[test]
    fn test_parse_unit_kilogram() {
        let dim = parse_unit("kg");
        assert!(dim.is_some());
        assert_eq!(dim.unwrap(), Dimension::mass());
    }

    #[test]
    fn test_parse_unit_unknown() {
        let dim = parse_unit("unknown_unit");
        assert!(dim.is_none());
    }

    #[test]
    fn test_sym_expr_diff_const() {
        let expr = SymExpr::Const(5.0);
        let deriv = expr.diff("x");
        
        if let SymExpr::Const(c) = deriv {
            assert_eq!(c, 0.0);
        } else {
            panic!("Expected Const(0.0)");
        }
    }

    #[test]
    fn test_sym_expr_diff_var() {
        let expr = SymExpr::Var("x".to_string());
        let deriv = expr.diff("x");
        
        if let SymExpr::Const(c) = deriv {
            assert_eq!(c, 1.0);
        } else {
            panic!("Expected Const(1.0)");
        }
    }

    #[test]
    fn test_sym_expr_simplify() {
        let expr = SymExpr::Binary(
            Box::new(SymExpr::Const(2.0)),
            BinOp::Add,
            Box::new(SymExpr::Const(3.0)),
        );
        
        let simplified = expr.simplify();
        
        if let SymExpr::Const(c) = simplified {
            assert_eq!(c, 5.0);
        } else {
            panic!("Expected Const(5.0)");
        }
    }
}
