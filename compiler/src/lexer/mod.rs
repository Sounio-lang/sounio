//! Lexer for the Sounio language
//!
//! Tokenizes source code into a stream of tokens using the Logos library.

pub mod tokens;

pub use tokens::{Token, TokenKind};

use crate::common::Span;
use logos::Logos;
use miette::Result;

/// Lex source code into tokens
pub fn lex(source: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut lexer = TokenKind::lexer(source);

    while let Some(result) = lexer.next() {
        let span = lexer.span();
        let kind = match result {
            Ok(kind) => kind,
            Err(_) => {
                return Err(miette::miette!(
                    "Unexpected character at position {}: {:?}",
                    span.start,
                    &source[span.clone()]
                ));
            }
        };

        tokens.push(Token {
            kind,
            span: Span::new(span.start, span.end),
            text: source[span].to_string(),
        });
    }

    // Add EOF token
    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(source.len(), source.len()),
        text: String::new(),
    });

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_simple() {
        let tokens = lex("let x = 42").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[1].kind, TokenKind::Ident);
        assert_eq!(tokens[1].text, "x");
        assert_eq!(tokens[2].kind, TokenKind::Eq);
        assert_eq!(tokens[3].kind, TokenKind::IntLit);
        assert_eq!(tokens[3].text, "42");
        assert_eq!(tokens[4].kind, TokenKind::Eof);
    }

    #[test]
    fn test_lex_keywords() {
        let tokens = lex("fn let mut if else match").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Fn);
        assert_eq!(tokens[1].kind, TokenKind::Let);
        assert_eq!(tokens[2].kind, TokenKind::Mut);
        assert_eq!(tokens[3].kind, TokenKind::If);
        assert_eq!(tokens[4].kind, TokenKind::Else);
        assert_eq!(tokens[5].kind, TokenKind::Match);
    }

    #[test]
    fn test_lex_operators() {
        let tokens = lex("+ - * / == != <= >= -> =>").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Plus);
        assert_eq!(tokens[1].kind, TokenKind::Minus);
        assert_eq!(tokens[2].kind, TokenKind::Star);
        assert_eq!(tokens[3].kind, TokenKind::Slash);
        assert_eq!(tokens[4].kind, TokenKind::EqEq);
        assert_eq!(tokens[5].kind, TokenKind::Ne);
        assert_eq!(tokens[6].kind, TokenKind::Le);
        assert_eq!(tokens[7].kind, TokenKind::Ge);
        assert_eq!(tokens[8].kind, TokenKind::Arrow);
        assert_eq!(tokens[9].kind, TokenKind::FatArrow);
    }

    #[test]
    fn test_lex_literals() {
        let tokens = lex(r#"42 3.14 "hello" true false"#).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::IntLit);
        assert_eq!(tokens[1].kind, TokenKind::FloatLit);
        assert_eq!(tokens[2].kind, TokenKind::StringLit);
        assert_eq!(tokens[3].kind, TokenKind::True);
        assert_eq!(tokens[4].kind, TokenKind::False);
    }

    #[test]
    fn test_lex_units() {
        let tokens = lex("500.0<mg> 10.0<mL>").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::FloatLit);
        assert_eq!(tokens[1].kind, TokenKind::Lt);
        assert_eq!(tokens[2].kind, TokenKind::Ident);
        assert_eq!(tokens[2].text, "mg");
        assert_eq!(tokens[3].kind, TokenKind::Gt);
    }

    #[test]
    fn test_lex_effects() {
        let tokens = lex("fn foo() -> i32 with IO, Mut").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Fn);
        assert_eq!(tokens[6].kind, TokenKind::With);
    }

    #[test]
    fn test_lex_comments() {
        let tokens = lex("let x = 1 // comment\nlet y = 2").unwrap();
        // Comments should be skipped
        assert_eq!(
            tokens.iter().filter(|t| t.kind == TokenKind::Let).count(),
            2
        );
    }

    #[test]
    fn test_lex_block_comments() {
        let tokens = lex("let /* comment */ x = 1").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Let);
        assert_eq!(tokens[1].kind, TokenKind::Ident);
        assert_eq!(tokens[1].text, "x");
    }

    #[test]
    fn test_lex_unit_literals() {
        // Integer with unit
        let tokens = lex("500_mg").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::IntUnitLit);
        assert_eq!(tokens[0].text, "500_mg");

        // Float with unit
        let tokens = lex("10.5_mL").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::FloatUnitLit);
        assert_eq!(tokens[0].text, "10.5_mL");

        // Complex unit
        let tokens = lex("5.0_mg/mL").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::FloatUnitLit);
        assert_eq!(tokens[0].text, "5.0_mg/mL");

        // Multiple unit literals
        let tokens = lex("let dose = 500_mg + 200_mg").unwrap();
        assert_eq!(tokens[3].kind, TokenKind::IntUnitLit);
        assert_eq!(tokens[3].text, "500_mg");
        assert_eq!(tokens[5].kind, TokenKind::IntUnitLit);
        assert_eq!(tokens[5].text, "200_mg");
    }

    #[test]
    fn test_lex_doc_comments() {
        // Outer doc comment
        let tokens = lex("/// This is a doc comment\nfn foo() {}").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::DocCommentOuter);
        assert!(tokens[0].text.contains("This is a doc comment"));
        assert_eq!(tokens[1].kind, TokenKind::Fn);

        // Inner doc comment
        let tokens = lex("//! Module-level doc\nfn bar() {}").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::DocCommentInner);
        assert!(tokens[0].text.contains("Module-level doc"));
        assert_eq!(tokens[1].kind, TokenKind::Fn);

        // Multiple doc comments
        let tokens = lex("/// First line\n/// Second line\nfn baz() {}").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::DocCommentOuter);
        assert_eq!(tokens[1].kind, TokenKind::DocCommentOuter);
        assert_eq!(tokens[2].kind, TokenKind::Fn);
    }

    #[test]
    fn test_lex_doc_block_comments() {
        // Outer block doc comment
        let tokens = lex("/** Block doc */\nfn foo() {}").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::DocBlockOuter);
        assert!(tokens[0].text.contains("Block doc"));
        assert_eq!(tokens[1].kind, TokenKind::Fn);

        // Inner block doc comment
        let tokens = lex("/*! Inner block doc */\nfn bar() {}").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::DocBlockInner);
        assert!(tokens[0].text.contains("Inner block doc"));
        assert_eq!(tokens[1].kind, TokenKind::Fn);
    }
}
