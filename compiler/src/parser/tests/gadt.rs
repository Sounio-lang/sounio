//! Tests for GADT (Generalized Algebraic Data Types) parsing
//!
//! GADTs extend regular ADTs by allowing constructors to have explicit return types
//! that can specialize the type parameters differently for each constructor.
//!
//! Syntax examples:
//! ```sio
//! // Regular ADT (no GADT)
//! enum Option<T> {
//!     None,
//!     Some(T)
//! }
//!
//! // GADT with type-indexed constructors
//! enum Vec<T, N: Nat> {
//!     Nil: Vec<T, Zero>,                    // Unit GADT variant
//!     Cons(T, Vec<T, M>): Vec<T, Succ<M>>   // Tuple GADT variant
//! }
//!
//! // GADT for type-safe expression representation
//! enum Expr<T> {
//!     IntLit(i32): Expr<i32>,
//!     BoolLit(bool): Expr<bool>,
//!     Add(Expr<i32>, Expr<i32>): Expr<i32>,
//!     If(Expr<bool>, Expr<T>, Expr<T>): Expr<T>
//! }
//! ```

#[cfg(test)]
mod tests {
    use crate::ast::{Item, TypeExpr, VariantData};
    use crate::lexer::lex;
    use crate::parser::parse;

    fn parse_program(input: &str) -> Result<crate::ast::Ast, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        parse(&tokens, input).map_err(|e| format!("{:?}", e))
    }

    // ==================== BASIC GADT SYNTAX TESTS ====================

    #[test]
    fn test_gadt_unit_variant() {
        let input = r#"
            enum Vec<T, N> {
                Nil: Vec<T, Zero>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse unit GADT variant: {:?}", result);

        let ast = result.unwrap();
        assert_eq!(ast.items.len(), 1);

        if let Item::Enum(enum_def) = &ast.items[0] {
            assert_eq!(enum_def.name, "Vec");
            assert_eq!(enum_def.variants.len(), 1);

            let variant = &enum_def.variants[0];
            assert_eq!(variant.name, "Nil");
            assert!(matches!(variant.data, VariantData::Unit));
            assert!(variant.gadt_return_type.is_some());

            let gadt_type = variant.gadt_return_type.as_ref().unwrap();
            if let TypeExpr::Named { path, args, .. } = &gadt_type.return_type {
                assert_eq!(path.segments, vec!["Vec"]);
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected Named type for GADT return type");
            }
        } else {
            panic!("Expected Enum item");
        }
    }

    #[test]
    fn test_gadt_tuple_variant() {
        let input = r#"
            enum Vec<T, N> {
                Cons(T, Vec<T, M>): Vec<T, Succ<M>>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse tuple GADT variant: {:?}", result);

        let ast = result.unwrap();
        if let Item::Enum(enum_def) = &ast.items[0] {
            let variant = &enum_def.variants[0];
            assert_eq!(variant.name, "Cons");

            // Check tuple data
            if let VariantData::Tuple(types) = &variant.data {
                assert_eq!(types.len(), 2);
            } else {
                panic!("Expected Tuple variant data");
            }

            // Check GADT return type
            assert!(variant.gadt_return_type.is_some());
            let gadt_type = variant.gadt_return_type.as_ref().unwrap();
            if let TypeExpr::Named { path, args, .. } = &gadt_type.return_type {
                assert_eq!(path.segments, vec!["Vec"]);
                assert_eq!(args.len(), 2);
            }
        }
    }

    #[test]
    fn test_gadt_multiple_variants() {
        let input = r#"
            enum Vec<T, N> {
                Nil: Vec<T, Zero>,
                Cons(T, Vec<T, M>): Vec<T, Succ<M>>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse multi-variant GADT: {:?}", result);

        let ast = result.unwrap();
        if let Item::Enum(enum_def) = &ast.items[0] {
            assert_eq!(enum_def.variants.len(), 2);

            // First variant: Nil
            assert_eq!(enum_def.variants[0].name, "Nil");
            assert!(enum_def.variants[0].gadt_return_type.is_some());

            // Second variant: Cons
            assert_eq!(enum_def.variants[1].name, "Cons");
            assert!(enum_def.variants[1].gadt_return_type.is_some());
        }
    }

    // ==================== TYPE-SAFE EXPRESSION GADT TESTS ====================

    #[test]
    fn test_gadt_expr_type() {
        let input = r#"
            enum Expr<T> {
                IntLit(i32): Expr<i32>,
                BoolLit(bool): Expr<bool>,
                Add(Expr<i32>, Expr<i32>): Expr<i32>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse Expr GADT: {:?}", result);

        let ast = result.unwrap();
        if let Item::Enum(enum_def) = &ast.items[0] {
            assert_eq!(enum_def.name, "Expr");
            assert_eq!(enum_def.variants.len(), 3);

            // IntLit variant returns Expr<i32>
            let int_lit = &enum_def.variants[0];
            assert_eq!(int_lit.name, "IntLit");
            assert!(int_lit.gadt_return_type.is_some());

            // BoolLit variant returns Expr<bool>
            let bool_lit = &enum_def.variants[1];
            assert_eq!(bool_lit.name, "BoolLit");
            assert!(bool_lit.gadt_return_type.is_some());
        }
    }

    // ==================== MIXED ADT AND GADT TESTS ====================

    #[test]
    fn test_mixed_adt_and_gadt_variants() {
        let input = r#"
            enum Tree<T, H> {
                Empty,
                Leaf(T): Tree<T, Zero>,
                Node(Tree<T, N>, T, Tree<T, N>): Tree<T, Succ<N>>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse mixed ADT/GADT: {:?}", result);

        let ast = result.unwrap();
        if let Item::Enum(enum_def) = &ast.items[0] {
            // Empty is regular ADT variant (no GADT return type)
            assert!(enum_def.variants[0].gadt_return_type.is_none());

            // Leaf is GADT variant
            assert!(enum_def.variants[1].gadt_return_type.is_some());

            // Node is GADT variant
            assert!(enum_def.variants[2].gadt_return_type.is_some());
        }
    }

    // ==================== REGULAR ADT (NO GADT) TESTS ====================

    #[test]
    fn test_regular_adt_unchanged() {
        let input = r#"
            enum Option<T> {
                None,
                Some(T)
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse regular ADT: {:?}", result);

        let ast = result.unwrap();
        if let Item::Enum(enum_def) = &ast.items[0] {
            // Neither variant should have GADT return type
            assert!(enum_def.variants[0].gadt_return_type.is_none());
            assert!(enum_def.variants[1].gadt_return_type.is_none());
        }
    }

    #[test]
    fn test_regular_struct_variant() {
        let input = r#"
            enum Shape {
                Circle { radius: f64 },
                Rectangle { width: f64, height: f64 }
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse struct variant ADT: {:?}", result);

        let ast = result.unwrap();
        if let Item::Enum(enum_def) = &ast.items[0] {
            // Check struct variants work without GADT
            if let VariantData::Struct(fields) = &enum_def.variants[0].data {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].name, "radius");
            }
        }
    }

    // ==================== GADT WITH STRUCT VARIANT TESTS ====================

    #[test]
    fn test_gadt_struct_variant() {
        let input = r#"
            enum Node<T, Color> {
                Red { value: T, left: Node<T, Black>, right: Node<T, Black> }: Node<T, Red>,
                Black { value: T, left: Node<T, C1>, right: Node<T, C2> }: Node<T, Black>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse struct GADT variant: {:?}", result);

        let ast = result.unwrap();
        if let Item::Enum(enum_def) = &ast.items[0] {
            // Both variants are GADT with struct data
            for variant in &enum_def.variants {
                assert!(matches!(variant.data, VariantData::Struct(_)));
                assert!(variant.gadt_return_type.is_some());
            }
        }
    }

    // ==================== COMPLEX TYPE ARGUMENT TESTS ====================

    #[test]
    fn test_gadt_nested_type_args() {
        let input = r#"
            enum HList<T> {
                HNil: HList<Unit>,
                HCons(A, HList<B>): HList<Pair<A, B>>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse HList GADT: {:?}", result);
    }

    #[test]
    fn test_gadt_with_const_generic() {
        // Note: Const generic arguments as literals (e.g., SizedVec<T, 0>) are not yet supported.
        // Use type-level naturals (Zero, Succ) instead for now.
        let input = r#"
            enum SizedVec<T, const N: usize> {
                Empty: SizedVec<T, Zero>,
                Elem(T, SizedVec<T, M>): SizedVec<T, M>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse const generic GADT: {:?}", result);
    }

    // ==================== EQUALITY TYPE GADT TESTS ====================

    #[test]
    fn test_gadt_type_equality() {
        let input = r#"
            enum Equal<A, B> {
                Refl: Equal<T, T>
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse type equality GADT: {:?}", result);
    }

    // ==================== FORMATTING AND WHITESPACE TESTS ====================

    #[test]
    fn test_gadt_no_whitespace() {
        let input = r#"enum Vec<T,N>{Nil:Vec<T,Zero>,Cons(T,Vec<T,M>):Vec<T,Succ<M>>}"#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse compact GADT: {:?}", result);
    }

    #[test]
    fn test_gadt_trailing_comma() {
        let input = r#"
            enum Vec<T, N> {
                Nil: Vec<T, Zero>,
                Cons(T, Vec<T, M>): Vec<T, Succ<M>>,
            }
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse GADT with trailing comma: {:?}", result);
    }
}
