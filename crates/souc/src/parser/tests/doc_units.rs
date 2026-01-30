//! Tests for doc comments and unit definitions

#[cfg(test)]
mod tests {
    use crate::ast::{Item, UnitDefExpr};
    use crate::lexer::lex;
    use crate::parser::Parser;

    #[test]
    fn test_inner_doc_and_fn_doc() {
        let source = r#"
//! Inner module docs

/// Adds two numbers
fn add(a: i64, b: i64) -> i64 {
    a + b
}
"#;
        let tokens = lex(source).expect("lexing should succeed");
        let mut parser = Parser::new(&tokens);
        let ast = parser.parse_program().expect("parse program");

        assert_eq!(ast.inner_doc.as_deref(), Some("Inner module docs"));

        let fn_doc = ast.items.iter().find_map(|item| {
            if let Item::Function(f) = item {
                f.doc.as_deref()
            } else {
                None
            }
        });
        assert_eq!(fn_doc, Some("Adds two numbers"));
    }

    #[test]
    fn test_unit_definitions_parse() {
        let source = r#"
unit kg;
unit mg = 0.001 * kg;
unit velocity = m / s^2;
"#;
        let tokens = lex(source).expect("lexing should succeed");
        let mut parser = Parser::new(&tokens);
        let ast = parser.parse_program().expect("parse program");

        let units: Vec<_> = ast
            .items
            .iter()
            .filter_map(|item| {
                if let Item::Unit(u) = item {
                    Some(u)
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(units.len(), 3);
        assert_eq!(units[0].name, "kg");
        assert!(units[0].definition.is_none());

        assert_eq!(units[1].name, "mg");
        match units[1].definition.as_ref().expect("mg definition") {
            UnitDefExpr::Scale(scale, inner) => {
                assert!((*scale - 0.001).abs() < 1e-9);
                assert!(matches!(**inner, UnitDefExpr::Named(ref n) if n == "kg"));
            }
            other => panic!("unexpected mg definition: {:?}", other),
        }

        assert_eq!(units[2].name, "velocity");
        match units[2].definition.as_ref().expect("velocity definition") {
            UnitDefExpr::Quotient(lhs, rhs) => {
                assert!(matches!(**lhs, UnitDefExpr::Named(ref n) if n == "m"));
                match **rhs {
                    UnitDefExpr::Power(ref base, exp) => {
                        assert_eq!(exp, 2);
                        assert!(matches!(**base, UnitDefExpr::Named(ref n) if n == "s"));
                    }
                    _ => panic!("unexpected velocity rhs: {:?}", rhs),
                }
            }
            other => panic!("unexpected velocity definition: {:?}", other),
        }
    }
}
