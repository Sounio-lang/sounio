use std::collections::BTreeMap;
use std::ops::Range;

use crate::lexer::{Token, TokenKind, Lexer};
use crate::Span;
use super::edits::{TextEdit, EditSequence};

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
            cache.tokens.insert(token.span.start, CachedToken {
                token,
                range,
            });
        }

        cache
    }

    /// Get token at position
    pub fn get(&self, position: usize) -> Option<&CachedToken> {
        self.tokens.get(&position)
    }

    /// Get tokens in range
    pub fn tokens_in_range(&self, range: Range<usize>) -> Vec<&CachedToken> {
        self.tokens.range(range.start..range.end)
            .map(|(_, t)| t)
            .collect()
    }

    /// Invalidate tokens affected by edit
    pub fn invalidate(&mut self, edit: &TextEdit) {
        // Find tokens that overlap with the edit range
        let affected: Vec<usize> = self.tokens.iter()
            .filter(|(_, t)| ranges_overlap(&t.range, &edit.range))
            .map(|(k, _)| *k)
            .collect();

        for key in affected {
            self.tokens.remove(&key);
        }

        // Adjust positions of tokens after the edit
        let delta = edit.length_delta();
        if delta != 0 {
            let after_edit: Vec<(usize, CachedToken)> = self.tokens
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
        let lexer
