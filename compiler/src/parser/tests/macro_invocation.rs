//! Tests for macro invocation parsing

#[cfg(test)]
mod tests {
    use crate::ast::{Expr, Item, MacroInvocation, Stmt};
    use crate::lexer::lex;
    use crate::parser::Parser;

    fn parse_expr(input: &str) -> Result<Expr, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let mut parser = Parser::new(&tokens);
        parser.parse_expr().map_err(|e| format!("{:?}", e))
    }

    fn parse_stmt(input: &str) -> Result<Stmt, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let mut parser = Parser::new(&tokens);
        parser.parse_stmt().map_err(|e| format!("{:?}", e))
    }

    fn parse_item(input: &str) -> Result<Item, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let mut parser = Parser::new(&tokens);
        parser.parse_item().map_err(|e| format!("{:?}", e))
    }

    #[test]
    fn test_simple_macro_invocation_expr() {
        let result = parse_expr("vec![]");
        assert!(result.is_ok());
        if let Ok(Expr::MacroInvocation(m)) = result {
            assert_eq!(m.name, "vec");
        } else {
            panic!("Expected MacroInvocation");
        }
    }

    #[test]
    fn test_macro_with_parentheses() {
        let result = parse_expr("assert!(x > 0)");
        assert!(result.is_ok());
        if let Ok(Expr::MacroInvocation(m)) = result {
            assert_eq!(m.name, "assert");
            assert!(!m.args.is_empty());
        } else {
            panic!("Expected MacroInvocation");
        }
    }

    #[test]
    fn test_macro_with_braces() {
        let result = parse_expr("map!{ x => y }");
        assert!(result.is_ok());
        if let Ok(Expr::MacroInvocation(m)) = result {
            assert_eq!(m.name, "map");
        } else {
            panic!("Expected MacroInvocation");
        }
    }

    #[test]
    fn test_macro_with_multiple_args() {
        let result = parse_expr("println!(\"hello\", x, y)");
        assert!(result.is_ok());
        if let Ok(Expr::MacroInvocation(m)) = result {
            assert_eq!(m.name, "println");
            assert!(!m.args.is_empty());
        } else {
            panic!("Expected MacroInvocation");
        }
    }

    #[test]
    fn test_nested_macro_invocation() {
        let result = parse_expr("vec![vec![1, 2], vec![3, 4]]");
        assert!(result.is_ok());
        if let Ok(Expr::MacroInvocation(m)) = result {
            assert_eq!(m.name, "vec");
        } else {
            panic!("Expected MacroInvocation");
        }
    }

    #[test]
    fn test_macro_in_statement() {
        let result = parse_stmt("println!(\"test\");");
        assert!(result.is_ok());
    }

    #[test]
    fn test_macro_in_item() {
        let result = parse_item("vec![];");
        assert!(result.is_ok());
        if let Ok(Item::MacroInvocation(m)) = result {
            assert_eq!(m.name, "vec");
        } else {
            panic!("Expected MacroInvocation");
        }
    }

    #[test]
    fn test_macro_with_empty_args() {
        let result = parse_expr("vec!()");
        assert!(result.is_ok());
        if let Ok(Expr::MacroInvocation(m)) = result {
            assert_eq!(m.name, "vec");
        } else {
            panic!("Expected MacroInvocation");
        }
    }

    #[test]
    fn test_macro_name_with_underscore() {
        let result = parse_expr("my_macro![]");
        assert!(result.is_ok());
        if let Ok(Expr::MacroInvocation(m)) = result {
            assert_eq!(m.name, "my_macro");
        } else {
            panic!("Expected MacroInvocation");
        }
    }

    #[test]
    fn test_macro_in_binary_expr() {
        let result = parse_expr("vec![1] + vec![2]");
        assert!(result.is_ok());
    }

    #[test]
    fn test_macro_in_function_call() {
        let result = parse_expr("foo(vec![1, 2, 3])");
        assert!(result.is_ok());
    }

    #[test]
    fn test_macro_with_complex_args() {
        let result = parse_expr("matrix![[1, 2], [3, 4]]");
        assert!(result.is_ok());
        if let Ok(Expr::MacroInvocation(m)) = result {
            assert_eq!(m.name, "matrix");
        } else {
            panic!("Expected MacroInvocation");
        }
    }
}
