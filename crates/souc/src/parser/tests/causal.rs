//! Tests for causal syntax parsing
//!
//! Tests the enhanced causal model definitions, do-calculus expressions,
//! and counterfactual expressions.

#[cfg(test)]
mod tests {
    use crate::ast::{CausalNodeType, Expr, Item};
    use crate::lexer::lex;
    use crate::parser::{parse, Parser};

    fn parse_expr(input: &str) -> Result<Expr, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let mut parser = Parser::new(&tokens);
        parser.parse_expr().map_err(|e| format!("{:?}", e))
    }

    fn parse_item(input: &str) -> Result<Item, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let mut parser = Parser::new(&tokens);
        parser.parse_item().map_err(|e| format!("{:?}", e))
    }

    fn parse_program_items(input: &str) -> Result<Vec<Item>, String> {
        let tokens = lex(input).map_err(|e| format!("{:?}", e))?;
        let ast = parse(&tokens, input).map_err(|e| format!("{:?}", e))?;
        Ok(ast.items)
    }

    // ==================== CAUSAL MODEL DEFINITION TESTS ====================

    #[test]
    fn test_parse_empty_causal_model() {
        let result = parse_item("causal model Empty {}");
        assert!(
            result.is_ok(),
            "Failed to parse empty causal model: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.name, "Empty");
            assert!(def.nodes.is_empty());
            assert!(def.edges.is_empty());
            assert!(def.latent_edges.is_empty());
            assert!(def.equations.is_empty());
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_causal_model_with_typed_nodes() {
        let input = r#"causal model DrugEffect {
            node Age: Confounder
            node Drug: Treatment
            node Recovery: Outcome
            node Biomarker: Mediator
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse causal model with typed nodes: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.name, "DrugEffect");
            assert_eq!(def.nodes.len(), 4);

            assert_eq!(def.nodes[0].name, "Age");
            assert_eq!(def.nodes[0].node_type, Some(CausalNodeType::Confounder));

            assert_eq!(def.nodes[1].name, "Drug");
            assert_eq!(def.nodes[1].node_type, Some(CausalNodeType::Treatment));

            assert_eq!(def.nodes[2].name, "Recovery");
            assert_eq!(def.nodes[2].node_type, Some(CausalNodeType::Outcome));

            assert_eq!(def.nodes[3].name, "Biomarker");
            assert_eq!(def.nodes[3].node_type, Some(CausalNodeType::Mediator));
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_causal_model_with_untyped_nodes() {
        let input = r#"causal model G {
            node X
            node Y
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse causal model with untyped nodes: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.name, "G");
            assert_eq!(def.nodes.len(), 2);
            assert_eq!(def.nodes[0].name, "X");
            assert!(def.nodes[0].node_type.is_none());
            assert_eq!(def.nodes[1].name, "Y");
            assert!(def.nodes[1].node_type.is_none());
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_causal_model_with_edges() {
        let input = r#"causal model G {
            node X
            node Y
            edge X -> Y
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse causal model with edges: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.nodes.len(), 2);
            assert_eq!(def.edges.len(), 1);
            assert_eq!(def.edges[0].from, "X");
            assert_eq!(def.edges[0].to, "Y");
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_causal_model_with_latent_edges() {
        let input = r#"causal model G {
            node X
            node Y
            latent X <-> Y
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse causal model with latent edges: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.nodes.len(), 2);
            assert_eq!(def.latent_edges.len(), 1);
            assert_eq!(def.latent_edges[0].node_a, "X");
            assert_eq!(def.latent_edges[0].node_b, "Y");
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_full_causal_model() {
        let input = r#"causal model DrugEffect {
            node Age: Confounder
            node Drug: Treatment
            node Recovery: Outcome
            node Biomarker: Mediator

            edge Age -> Drug
            edge Age -> Recovery
            edge Drug -> Biomarker
            edge Biomarker -> Recovery

            latent Drug <-> Recovery
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse full causal model: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.name, "DrugEffect");
            assert_eq!(def.nodes.len(), 4);
            assert_eq!(def.edges.len(), 4);
            assert_eq!(def.latent_edges.len(), 1);

            // Verify edge directions
            assert_eq!(def.edges[0].from, "Age");
            assert_eq!(def.edges[0].to, "Drug");

            assert_eq!(def.edges[1].from, "Age");
            assert_eq!(def.edges[1].to, "Recovery");

            assert_eq!(def.edges[2].from, "Drug");
            assert_eq!(def.edges[2].to, "Biomarker");

            assert_eq!(def.edges[3].from, "Biomarker");
            assert_eq!(def.edges[3].to, "Recovery");

            // Verify latent edge
            assert_eq!(def.latent_edges[0].node_a, "Drug");
            assert_eq!(def.latent_edges[0].node_b, "Recovery");
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_causal_model_without_model_keyword() {
        // The 'model' keyword is optional: `causal G { ... }` also works
        let input = r#"causal G {
            node X: Treatment
            node Y: Outcome
            edge X -> Y
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse causal model without 'model' keyword: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.name, "G");
            assert_eq!(def.nodes.len(), 2);
            assert_eq!(def.edges.len(), 1);
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_causal_model_instrument_node() {
        let input = r#"causal model IV {
            node Z: Instrument
            node X: Treatment
            node Y: Outcome
            edge Z -> X
            edge X -> Y
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse causal model with instrument: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.nodes[0].node_type, Some(CausalNodeType::Instrument));
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_causal_model_mixed_syntax() {
        // Mix old-style nodes: list and new-style node declarations
        let input = r#"causal model Mixed {
            nodes: [A, B]
            node C: Outcome
            A -> B
            edge B -> C
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse mixed causal model: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            // nodes: list gives 2 nodes, node declaration gives 1 more
            assert_eq!(def.nodes.len(), 3);
            assert_eq!(def.edges.len(), 2);
        } else {
            panic!("Expected CausalModel item");
        }
    }

    // ==================== DO EXPRESSION TESTS ====================

    #[test]
    fn test_parse_do_simple() {
        let result = parse_expr("do(X = 1.0)");
        assert!(
            result.is_ok(),
            "Failed to parse do(X = 1.0): {:?}",
            result
        );
        if let Ok(Expr::Do {
            interventions,
            model,
            query,
            ..
        }) = result
        {
            assert_eq!(interventions.len(), 1);
            assert_eq!(interventions[0].0, "X");
            assert!(model.is_none());
            assert!(query.is_none());
        } else {
            panic!("Expected Do expression");
        }
    }

    #[test]
    fn test_parse_do_with_model_and_query() {
        let result = parse_expr("do(Drug = 100.0, model: DrugEffect, query: Recovery)");
        assert!(
            result.is_ok(),
            "Failed to parse do with model and query: {:?}",
            result
        );
        if let Ok(Expr::Do {
            interventions,
            model,
            query,
            ..
        }) = result
        {
            assert_eq!(interventions.len(), 1);
            assert_eq!(interventions[0].0, "Drug");
            assert_eq!(model, Some("DrugEffect".to_string()));
            assert_eq!(query, Some("Recovery".to_string()));
        } else {
            panic!("Expected Do expression");
        }
    }

    #[test]
    fn test_parse_do_multiple_interventions() {
        let result = parse_expr("do(X = 1, Y = 2)");
        assert!(
            result.is_ok(),
            "Failed to parse do with multiple interventions: {:?}",
            result
        );
        if let Ok(Expr::Do {
            interventions, ..
        }) = result
        {
            assert_eq!(interventions.len(), 2);
            assert_eq!(interventions[0].0, "X");
            assert_eq!(interventions[1].0, "Y");
        } else {
            panic!("Expected Do expression");
        }
    }

    // ==================== COUNTERFACTUAL EXPRESSION TESTS ====================

    #[test]
    fn test_parse_counterfactual_block_syntax() {
        let result = parse_expr("counterfactual { observed; do(X = 1); outcome }");
        assert!(
            result.is_ok(),
            "Failed to parse counterfactual block syntax: {:?}",
            result
        );
        if let Ok(Expr::Counterfactual {
            factual,
            model,
            evidence,
            ..
        }) = result
        {
            assert!(factual.is_some());
            assert!(model.is_none());
            assert!(evidence.is_empty());
        } else {
            panic!("Expected Counterfactual expression");
        }
    }

    #[test]
    fn test_parse_counterfactual_call_syntax() {
        let input = r#"counterfactual(
            model: DrugEffect,
            intervention: do(Drug = 50.0),
            query: Recovery,
            evidence: { Age = 65, Drug = 100.0, Recovery = 0.8 }
        )"#;
        let result = parse_expr(input);
        assert!(
            result.is_ok(),
            "Failed to parse counterfactual call syntax: {:?}",
            result
        );
        if let Ok(Expr::Counterfactual {
            factual,
            model,
            evidence,
            ..
        }) = result
        {
            assert!(factual.is_none());
            assert_eq!(model, Some("DrugEffect".to_string()));
            assert_eq!(evidence.len(), 3);
            assert_eq!(evidence[0].0, "Age");
            assert_eq!(evidence[1].0, "Drug");
            assert_eq!(evidence[2].0, "Recovery");
        } else {
            panic!("Expected Counterfactual expression");
        }
    }

    #[test]
    fn test_parse_counterfactual_call_minimal() {
        let input = "counterfactual(intervention: do(X = 1), query: Y)";
        let result = parse_expr(input);
        assert!(
            result.is_ok(),
            "Failed to parse minimal counterfactual call: {:?}",
            result
        );
        if let Ok(Expr::Counterfactual {
            factual,
            model,
            evidence,
            ..
        }) = result
        {
            assert!(factual.is_none());
            assert!(model.is_none());
            assert!(evidence.is_empty());
        } else {
            panic!("Expected Counterfactual expression");
        }
    }

    // ==================== LEXER TOKEN TESTS ====================

    #[test]
    fn test_lex_biarrow() {
        use crate::lexer::tokens::TokenKind;
        let tokens = lex("<->").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::BiArrow);
        assert_eq!(tokens[0].text, "<->");
    }

    #[test]
    fn test_lex_node_edge_latent_keywords() {
        use crate::lexer::tokens::TokenKind;
        let tokens = lex("node edge latent").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Node);
        assert_eq!(tokens[1].kind, TokenKind::Edge);
        assert_eq!(tokens[2].kind, TokenKind::Latent);
    }

    #[test]
    fn test_lex_biarrow_vs_left_arrow() {
        use crate::lexer::tokens::TokenKind;
        // <-> should be BiArrow, not LeftArrow + Gt
        let tokens = lex("<->").unwrap();
        assert_eq!(tokens.len(), 2); // BiArrow + Eof
        assert_eq!(tokens[0].kind, TokenKind::BiArrow);

        // <- should still be LeftArrow
        let tokens = lex("<-").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::LeftArrow);
    }

    // ==================== INTEGRATION TESTS ====================

    #[test]
    fn test_parse_causal_model_in_program() {
        let input = r#"
            causal model SmokingCancer {
                node Smoking: Treatment
                node Tar: Mediator
                node Cancer: Outcome
                node Genetics: Confounder

                edge Genetics -> Smoking
                edge Genetics -> Cancer
                edge Smoking -> Tar
                edge Tar -> Cancer
            }
        "#;
        let result = parse_program_items(input);
        assert!(
            result.is_ok(),
            "Failed to parse causal model in program: {:?}",
            result
        );
        let items = result.unwrap();
        assert_eq!(items.len(), 1);
        if let Item::CausalModel(def) = &items[0] {
            assert_eq!(def.name, "SmokingCancer");
            assert_eq!(def.nodes.len(), 4);
            assert_eq!(def.edges.len(), 4);
        } else {
            panic!("Expected CausalModel item");
        }
    }

    #[test]
    fn test_parse_causal_model_multiple_latent_edges() {
        let input = r#"causal model Complex {
            node A
            node B
            node C
            node D

            edge A -> B
            edge C -> D

            latent A <-> C
            latent B <-> D
        }"#;
        let result = parse_item(input);
        assert!(
            result.is_ok(),
            "Failed to parse model with multiple latent edges: {:?}",
            result
        );
        if let Ok(Item::CausalModel(def)) = result {
            assert_eq!(def.latent_edges.len(), 2);
            assert_eq!(def.latent_edges[0].node_a, "A");
            assert_eq!(def.latent_edges[0].node_b, "C");
            assert_eq!(def.latent_edges[1].node_a, "B");
            assert_eq!(def.latent_edges[1].node_b, "D");
        } else {
            panic!("Expected CausalModel item");
        }
    }
}
