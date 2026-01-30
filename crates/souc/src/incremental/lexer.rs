use std::collections::BTreeMap;
use std::ops::Range;

use super::edits::{EditSequence, TextEdit};
use crate::Span;
use crate::lexer::{Lexer, Token, TokenKind};

/// A cached token with its byte range
#[derive(Debug, Clone)]
pub struct CachedToken {
    pub token: Token,
    pub range: Range<usize>,
}

/// Token cache for incremental lexing
#[derive(Debug, Clone)]
pub struct TokenCache {
    /// Tokens indexed by start position
    tokens: BTreeMap<usize, CachedToken>,

    /// Source text hash for validation
    source_hash: u64,
}

impl TokenCache {
    pub fn new() -> Self {
        TokenCache {
            tokens: BTreeMap::new(),
            source_hash: 0,
        }
    }

    /// Build cache from full lex
    pub fn from_tokens(tokens: Vec<Token>, source: &str) -> Self {
        let mut cache = TokenCache::new();
        cache.source_hash = hash_source(source);

        for token in tokens {
            let range = token.span.start..token.span.end;
            cache
                .tokens
                .insert(token.span.start, CachedToken { token, range });
        }

        cache
    }

    /// Get token at position
    pub fn get(&self, position: usize) -> Option<&CachedToken> {
        self.tokens.get(&position)
    }

    /// Get tokens in range
    pub fn tokens_in_range(&self, range: Range<usize>) -> Vec<&CachedToken> {
        self.tokens
            .range(range.start..range.end)
            .map(|(_, t)| t)
            .collect()
    }

    /// Invalidate tokens affected by edit
    pub fn invalidate(&mut self, edit: &TextEdit) {
        // Find tokens that overlap with the edit range
        let affected: Vec<usize> = self
            .tokens
            .iter()
            .filter(|(_, t)| ranges_overlap(&t.range, &edit.range))
            .map(|(k, _)| *k)
            .collect();

        for key in affected {
            self.tokens.remove(&key);
        }

        // Adjust positions of tokens after the edit
        let delta = edit.length_delta();
        if delta != 0 {
            let after_edit: Vec<(usize, CachedToken)> = self
                .tokens
                .range(edit.range.end..)
                .map(|(k, v)| (*k, v.clone()))
                .collect();

            for (old_pos, _) in &after_edit {
                self.tokens.remove(old_pos);
            }

            for (old_pos, mut cached) in after_edit {
                let new_pos = (old_pos as isize + delta) as usize;
                cached.range.start = (cached.range.start as isize + delta) as usize;
                cached.range.end = (cached.range.end as isize + delta) as usize;
                cached.token.span.start = new_pos;
                cached.token.span.end = cached.range.end;
                self.tokens.insert(new_pos, cached);
            }
        }
    }
}

/// Incremental lexer
pub struct IncrementalLexer {
    cache: TokenCache,
}

impl IncrementalLexer {
    pub fn new() -> Self {
        IncrementalLexer {
            cache: TokenCache::new(),
        }
    }

    /// Full lex (first time or cache miss)
    pub fn lex_full(&mut self, source: &str) -> Vec<Token> {
        let lexer = Lexer::new(source);
        let tokens: Vec<Token> = lexer.collect();
        self.cache = TokenCache::from_tokens(tokens.clone(), source);
        tokens
    }

    /// Incremental lex after edit
    pub fn lex_incremental(&mut self, new_source: &str, edit: &TextEdit) -> Vec<Token> {
        // Invalidate affected tokens
        self.cache.invalidate(edit);

        // Find the range that needs re-lexing
        let relex_start = self.find_relex_start(edit.range.start);
        let relex_end = self.find_relex_end(edit.range.start + edit.new_text.len(), new_source);

        // Re-lex the affected region
        let region = &new_source[relex_start..relex_end];
        let mut lexer = Lexer::new(region);
        lexer.set_offset(relex_start);

        let new_tokens: Vec<Token> = lexer.collect();

        // Insert new tokens into cache
        for token in &new_tokens {
            let range = token.span.start..token.span.end;
            self.cache.tokens.insert(
                token.span.start,
                CachedToken {
                    token: token.clone(),
                    range,
                },
            );
        }

        // Rebuild full token list
        self.cache
            .tokens
            .values()
            .map(|ct| ct.token.clone())
            .collect()
    }

    /// Find safe position to start re-lexing
    fn find_relex_start(&self, edit_start: usize) -> usize {
        // Find the token containing or just before the edit
        if let Some((&pos, _)) = self.cache.tokens.range(..=edit_start).next_back() {
            pos
        } else {
            0
        }
    }

    /// Find safe position to end re-lexing
    fn find_relex_end(&self, edit_end: usize, source: &str) -> usize {
        // Continue until we reach a token boundary that matches the cache
        // For simplicity, re-lex to end of line or next cached token
        let line_end = source[edit_end..]
            .find('\n')
            .map(|i| edit_end + i + 1)
            .unwrap_or(source.len());

        if let Some((&pos, _)) = self.cache.tokens.range(line_end..).next() {
            pos
        } else {
            source.len()
        }
    }
}

fn ranges_overlap(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

fn hash_source(source: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_cache_invalidate() {
        let mut cache = TokenCache::new();
        // Assume some tokens for testing
        let tokens = vec![
            Token {
                span: Span { start: 0, end: 3 },
                kind: TokenKind::Ident,
            },
            Token {
                span: Span { start: 4, end: 8 },
                kind: TokenKind::Keyword,
            },
            Token {
                span: Span { start: 9, end: 12 },
                kind: TokenKind::Punct,
            },
        ];
        cache = TokenCache::from_tokens(tokens, "let x = 42;");

        let edit = TextEdit::new(4..8, "var ");
        cache.invalidate(&edit);

        assert!(cache.get(0).is_some()); // Before edit
        assert!(cache.get(9).is_none()); // Should be adjusted
        assert!(cache.tokens.len() == 2); // One invalidated, one adjusted
    }
}
