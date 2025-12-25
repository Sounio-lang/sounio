// medlang::lexer — Lexical Analyzer for MedLang eDSL
//
// Tokenizes MedLang source text for parsing.
// Handles:
// - Keywords (model, param, compartment, flow, dose, observe, fit, simulate)
// - Identifiers and literals
// - Operators and delimiters
// - Units (mg, L, h, /h, L/h, mg/L)
// - Distribution syntax (LogNormal, Normal, ~)

// ============================================================================
// TOKEN TYPES (as constants instead of enum)
// ============================================================================

fn TT_EOF() -> i32 { 0 }
fn TT_ERROR() -> i32 { 1 }

// Model structure keywords
fn TT_KW_MODEL() -> i32 { 10 }
fn TT_KW_PARAM() -> i32 { 11 }
fn TT_KW_COMPARTMENT() -> i32 { 12 }
fn TT_KW_FLOW() -> i32 { 13 }
fn TT_KW_DOSE() -> i32 { 14 }
fn TT_KW_OBSERVE() -> i32 { 15 }
fn TT_KW_FIT() -> i32 { 16 }
fn TT_KW_SIMULATE() -> i32 { 17 }
fn TT_KW_TO() -> i32 { 18 }
fn TT_KW_WITH() -> i32 { 19 }

// Type keywords
fn TT_KW_CLEARANCE() -> i32 { 20 }
fn TT_KW_VOLUME() -> i32 { 21 }
fn TT_KW_RATE() -> i32 { 22 }
fn TT_KW_AMOUNT() -> i32 { 23 }

// Distribution keywords
fn TT_KW_LOGNORMAL() -> i32 { 30 }
fn TT_KW_NORMAL() -> i32 { 31 }
fn TT_KW_UNIFORM() -> i32 { 32 }
fn TT_KW_FIXED() -> i32 { 33 }
fn TT_KW_OMEGA() -> i32 { 34 }

// Route keywords
fn TT_KW_IV() -> i32 { 40 }
fn TT_KW_ORAL() -> i32 { 41 }
fn TT_KW_INFUSION() -> i32 { 42 }

// Error model keywords
fn TT_KW_ADDITIVE() -> i32 { 50 }
fn TT_KW_PROPORTIONAL() -> i32 { 51 }
fn TT_KW_COMBINED() -> i32 { 52 }

// Fit method keywords
fn TT_KW_METHOD() -> i32 { 60 }
fn TT_KW_UNCERTAINTY() -> i32 { 61 }
fn TT_KW_LM() -> i32 { 62 }
fn TT_KW_GUM() -> i32 { 63 }
fn TT_KW_BOOTSTRAP() -> i32 { 64 }

// Simulate keywords
fn TT_KW_TIME() -> i32 { 70 }
fn TT_KW_SUBJECTS() -> i32 { 71 }

// Literals
fn TT_IDENTIFIER() -> i32 { 80 }
fn TT_INTEGER() -> i32 { 81 }
fn TT_FLOAT() -> i32 { 82 }
fn TT_STRING() -> i32 { 83 }

// Units
fn TT_UNIT_MG() -> i32 { 90 }
fn TT_UNIT_L() -> i32 { 91 }
fn TT_UNIT_H() -> i32 { 92 }
fn TT_UNIT_L_PER_H() -> i32 { 93 }
fn TT_UNIT_MG_PER_L() -> i32 { 94 }
fn TT_UNIT_PER_H() -> i32 { 95 }

// Operators
fn TT_OP_PLUS() -> i32 { 100 }
fn TT_OP_MINUS() -> i32 { 101 }
fn TT_OP_STAR() -> i32 { 102 }
fn TT_OP_SLASH() -> i32 { 103 }
fn TT_OP_CARET() -> i32 { 104 }
fn TT_OP_EQ() -> i32 { 105 }
fn TT_OP_TILDE() -> i32 { 106 }
fn TT_OP_ARROW() -> i32 { 107 }
fn TT_OP_DOT() -> i32 { 108 }
fn TT_OP_DOTDOT() -> i32 { 109 }
fn TT_OP_LT() -> i32 { 110 }
fn TT_OP_GT() -> i32 { 111 }

// Delimiters
fn TT_LPAREN() -> i32 { 120 }
fn TT_RPAREN() -> i32 { 121 }
fn TT_LBRACE() -> i32 { 122 }
fn TT_RBRACE() -> i32 { 123 }
fn TT_LBRACKET() -> i32 { 124 }
fn TT_RBRACKET() -> i32 { 125 }
fn TT_COMMA() -> i32 { 126 }
fn TT_COLON() -> i32 { 127 }
fn TT_SEMICOLON() -> i32 { 128 }
fn TT_NEWLINE() -> i32 { 129 }

// ============================================================================
// TOKEN STRUCTURE
// ============================================================================

/// Token with value and position
struct Token {
    token_type: i32,
    text: [i8; 64],
    text_len: i64,
    int_value: i64,
    float_value: f64,
    line: i64,
    column: i64,
}

fn token_new() -> Token {
    Token {
        token_type: TT_EOF(),
        text: [0; 64],
        text_len: 0,
        int_value: 0,
        float_value: 0.0,
        line: 1,
        column: 1,
    }
}

fn token_eof(line: i64, col: i64) -> Token {
    var tok = token_new();
    tok.line = line;
    tok.column = col;
    tok
}

fn token_simple(tt: i32, line: i64, col: i64) -> Token {
    var tok = token_new();
    tok.token_type = tt;
    tok.line = line;
    tok.column = col;
    tok
}

// ============================================================================
// LEXER STATE
// ============================================================================

/// Lexer state
struct Lexer {
    source: [i8; 8192],
    source_len: i64,
    pos: i64,
    line: i64,
    column: i64,
    current_char: i8,
}

fn lexer_new() -> Lexer {
    Lexer {
        source: [0; 8192],
        source_len: 0,
        pos: 0,
        line: 1,
        column: 1,
        current_char: 0,
    }
}

fn lexer_init(source: [i8; 8192], len: i64) -> Lexer {
    var lex = lexer_new();
    lex.source = source;
    lex.source_len = len;
    if len > 0 {
        lex.current_char = lex.source[0];
    }
    lex
}

fn lexer_at_end(lex: Lexer) -> bool {
    lex.pos >= lex.source_len
}

fn lexer_peek(lex: Lexer) -> i8 {
    if lex.pos + 1 < lex.source_len {
        lex.source[(lex.pos + 1) as usize]
    } else {
        0
    }
}

/// Advance result - returns new lexer state
fn lexer_advance(lex: Lexer) -> Lexer {
    var new_lex = lex;
    if new_lex.pos < new_lex.source_len {
        if new_lex.current_char == 10 {  // '\n'
            new_lex.line = new_lex.line + 1;
            new_lex.column = 1;
        } else {
            new_lex.column = new_lex.column + 1;
        }
        new_lex.pos = new_lex.pos + 1;
        if new_lex.pos < new_lex.source_len {
            new_lex.current_char = new_lex.source[new_lex.pos as usize];
        } else {
            new_lex.current_char = 0;
        }
    }
    new_lex
}

// ============================================================================
// CHARACTER CLASSIFICATION
// ============================================================================

fn is_whitespace(c: i8) -> bool {
    c == 32 || c == 9 || c == 13  // space, tab, CR
}

fn is_digit(c: i8) -> bool {
    c >= 48 && c <= 57  // '0'-'9'
}

fn is_alpha(c: i8) -> bool {
    (c >= 65 && c <= 90) ||   // 'A'-'Z'
    (c >= 97 && c <= 122) ||  // 'a'-'z'
    c == 95                    // '_'
}

fn is_alnum(c: i8) -> bool {
    is_alpha(c) || is_digit(c)
}

// ============================================================================
// KEYWORD LOOKUP
// ============================================================================

/// Compare two byte arrays
fn bytes_eq(a: [i8; 64], a_len: i64, b0: i8, b1: i8, b2: i8, b3: i8, b4: i8, b_len: i64) -> bool {
    if a_len != b_len {
        return false
    }
    if b_len >= 1 && a[0] != b0 { return false }
    if b_len >= 2 && a[1] != b1 { return false }
    if b_len >= 3 && a[2] != b2 { return false }
    if b_len >= 4 && a[3] != b3 { return false }
    if b_len >= 5 && a[4] != b4 { return false }
    true
}

fn bytes_eq_long(a: [i8; 64], a_len: i64, expected: [i8; 16], e_len: i64) -> bool {
    if a_len != e_len {
        return false
    }
    var i: i64 = 0;
    while i < e_len {
        if a[i as usize] != expected[i as usize] {
            return false
        }
        i = i + 1;
    }
    true
}

/// Lookup keyword from identifier text
fn lookup_keyword(text: [i8; 64], len: i64) -> i32 {
    // 2-character keywords
    if len == 2 {
        if text[0] == 116 && text[1] == 111 { return TT_KW_TO() }  // "to"
        if text[0] == 73 && text[1] == 86 { return TT_KW_IV() }    // "IV"
    }

    // 3-character keywords
    if len == 3 {
        if text[0] == 102 && text[1] == 105 && text[2] == 116 { return TT_KW_FIT() }  // "fit"
        if text[0] == 71 && text[1] == 85 && text[2] == 77 { return TT_KW_GUM() }     // "GUM"
    }

    // 4-character keywords
    if len == 4 {
        if text[0] == 119 && text[1] == 105 && text[2] == 116 && text[3] == 104 { return TT_KW_WITH() }  // "with"
        if text[0] == 100 && text[1] == 111 && text[2] == 115 && text[3] == 101 { return TT_KW_DOSE() }  // "dose"
        if text[0] == 102 && text[1] == 108 && text[2] == 111 && text[3] == 119 { return TT_KW_FLOW() }  // "flow"
        if text[0] == 79 && text[1] == 114 && text[2] == 97 && text[3] == 108 { return TT_KW_ORAL() }    // "Oral"
        if text[0] == 82 && text[1] == 97 && text[2] == 116 && text[3] == 101 { return TT_KW_RATE() }    // "Rate"
        if text[0] == 116 && text[1] == 105 && text[2] == 109 && text[3] == 101 { return TT_KW_TIME() }  // "time"
    }

    // 5-character keywords
    if len == 5 {
        if text[0] == 109 && text[1] == 111 && text[2] == 100 && text[3] == 101 && text[4] == 108 { return TT_KW_MODEL() }  // "model"
        if text[0] == 112 && text[1] == 97 && text[2] == 114 && text[3] == 97 && text[4] == 109 { return TT_KW_PARAM() }    // "param"
        if text[0] == 70 && text[1] == 105 && text[2] == 120 && text[3] == 101 && text[4] == 100 { return TT_KW_FIXED() }   // "Fixed"
        if text[0] == 111 && text[1] == 109 && text[2] == 101 && text[3] == 103 && text[4] == 97 { return TT_KW_OMEGA() }   // "omega"
    }

    // 6-character keywords
    if len == 6 {
        if text[0] == 78 && text[1] == 111 && text[2] == 114 && text[3] == 109 && text[4] == 97 && text[5] == 108 { return TT_KW_NORMAL() }  // "Normal"
        if text[0] == 86 && text[1] == 111 && text[2] == 108 && text[3] == 117 && text[4] == 109 && text[5] == 101 { return TT_KW_VOLUME() }  // "Volume"
        if text[0] == 65 && text[1] == 109 && text[2] == 111 && text[3] == 117 && text[4] == 110 && text[5] == 116 { return TT_KW_AMOUNT() }  // "Amount"
        if text[0] == 109 && text[1] == 101 && text[2] == 116 && text[3] == 104 && text[4] == 111 && text[5] == 100 { return TT_KW_METHOD() }  // "method"
    }

    // 7-character keywords
    if len == 7 {
        if text[0] == 111 && text[1] == 98 && text[2] == 115 && text[3] == 101 && text[4] == 114 && text[5] == 118 && text[6] == 101 { return TT_KW_OBSERVE() }  // "observe"
        if text[0] == 85 && text[1] == 110 && text[2] == 105 && text[3] == 102 && text[4] == 111 && text[5] == 114 && text[6] == 109 { return TT_KW_UNIFORM() }  // "Uniform"
    }

    // 8-character keywords
    if len == 8 {
        if text[0] == 115 && text[1] == 105 && text[2] == 109 && text[3] == 117 &&
           text[4] == 108 && text[5] == 97 && text[6] == 116 && text[7] == 101 { return TT_KW_SIMULATE() }  // "simulate"
        if text[0] == 73 && text[1] == 110 && text[2] == 102 && text[3] == 117 &&
           text[4] == 115 && text[5] == 105 && text[6] == 111 && text[7] == 110 { return TT_KW_INFUSION() }  // "Infusion"
        if text[0] == 65 && text[1] == 100 && text[2] == 100 && text[3] == 105 &&
           text[4] == 116 && text[5] == 105 && text[6] == 118 && text[7] == 101 { return TT_KW_ADDITIVE() }  // "Additive"
        if text[0] == 67 && text[1] == 111 && text[2] == 109 && text[3] == 98 &&
           text[4] == 105 && text[5] == 110 && text[6] == 101 && text[7] == 100 { return TT_KW_COMBINED() }  // "Combined"
        if text[0] == 115 && text[1] == 117 && text[2] == 98 && text[3] == 106 &&
           text[4] == 101 && text[5] == 99 && text[6] == 116 && text[7] == 115 { return TT_KW_SUBJECTS() }  // "subjects"
    }

    // 9-character keywords
    if len == 9 {
        if text[0] == 76 && text[1] == 111 && text[2] == 103 && text[3] == 78 &&
           text[4] == 111 && text[5] == 114 && text[6] == 109 && text[7] == 97 && text[8] == 108 { return TT_KW_LOGNORMAL() }  // "LogNormal"
        if text[0] == 67 && text[1] == 108 && text[2] == 101 && text[3] == 97 &&
           text[4] == 114 && text[5] == 97 && text[6] == 110 && text[7] == 99 && text[8] == 101 { return TT_KW_CLEARANCE() }  // "Clearance"
        if text[0] == 66 && text[1] == 111 && text[2] == 111 && text[3] == 116 &&
           text[4] == 115 && text[5] == 116 && text[6] == 114 && text[7] == 97 && text[8] == 112 { return TT_KW_BOOTSTRAP() }  // "Bootstrap"
    }

    // 11-character keywords
    if len == 11 {
        if text[0] == 99 && text[1] == 111 && text[2] == 109 && text[3] == 112 &&
           text[4] == 97 && text[5] == 114 && text[6] == 116 && text[7] == 109 &&
           text[8] == 101 && text[9] == 110 && text[10] == 116 { return TT_KW_COMPARTMENT() }  // "compartment"
        if text[0] == 117 && text[1] == 110 && text[2] == 99 && text[3] == 101 &&
           text[4] == 114 && text[5] == 116 && text[6] == 97 && text[7] == 105 &&
           text[8] == 110 && text[9] == 116 && text[10] == 121 { return TT_KW_UNCERTAINTY() }  // "uncertainty"
    }

    // 12-character keywords
    if len == 12 {
        if text[0] == 80 && text[1] == 114 && text[2] == 111 && text[3] == 112 &&
           text[4] == 111 && text[5] == 114 && text[6] == 116 && text[7] == 105 &&
           text[8] == 111 && text[9] == 110 && text[10] == 97 && text[11] == 108 { return TT_KW_PROPORTIONAL() }  // "Proportional"
    }

    TT_IDENTIFIER()  // Not a keyword
}

/// Lookup unit from identifier
fn lookup_unit(text: [i8; 64], len: i64) -> i32 {
    // Single character units
    if len == 1 {
        if text[0] == 76 { return TT_UNIT_L() }   // "L"
        if text[0] == 104 { return TT_UNIT_H() }  // "h"
    }

    // Two character units
    if len == 2 {
        if text[0] == 109 && text[1] == 103 { return TT_UNIT_MG() }  // "mg"
    }

    // Composite units like "L/h" are handled separately
    TT_IDENTIFIER()
}

// ============================================================================
// SCAN RESULT (replaces tuple return)
// ============================================================================

struct ScanResult {
    lexer: Lexer,
    token: Token,
}

fn scan_result_new(lex: Lexer, tok: Token) -> ScanResult {
    ScanResult { lexer: lex, token: tok }
}

// ============================================================================
// SCANNING FUNCTIONS
// ============================================================================

/// Skip whitespace (not newlines)
fn skip_whitespace(lex: Lexer) -> Lexer {
    var l = lex;
    while !lexer_at_end(l) && is_whitespace(l.current_char) {
        l = lexer_advance(l);
    }
    l
}

/// Skip line comment
fn skip_line_comment(lex: Lexer) -> Lexer {
    var l = lex;
    while !lexer_at_end(l) && l.current_char != 10 {
        l = lexer_advance(l);
    }
    l
}

/// Scan identifier or keyword
fn scan_identifier(lex: Lexer) -> ScanResult {
    let start_line = lex.line;
    let start_col = lex.column;
    var l = lex;
    var tok = token_new();
    tok.line = start_line;
    tok.column = start_col;

    var text_len: i64 = 0;
    while !lexer_at_end(l) && is_alnum(l.current_char) && text_len < 63 {
        tok.text[text_len as usize] = l.current_char;
        text_len = text_len + 1;
        l = lexer_advance(l);
    }
    tok.text_len = text_len;

    // Check for keyword or unit
    tok.token_type = lookup_keyword(tok.text, text_len);
    if tok.token_type == TT_IDENTIFIER() {
        let unit_type = lookup_unit(tok.text, text_len);
        if unit_type != TT_IDENTIFIER() {
            tok.token_type = unit_type;
        }
    }

    scan_result_new(l, tok)
}

/// Scan integer or float
fn scan_number(lex: Lexer) -> ScanResult {
    let start_line = lex.line;
    let start_col = lex.column;
    var l = lex;
    var tok = token_new();
    tok.line = start_line;
    tok.column = start_col;

    var text_len: i64 = 0;
    var is_float = false;

    // Integer part
    while !lexer_at_end(l) && is_digit(l.current_char) && text_len < 63 {
        tok.text[text_len as usize] = l.current_char;
        text_len = text_len + 1;
        l = lexer_advance(l);
    }

    // Decimal point
    if l.current_char == 46 && is_digit(lexer_peek(l)) {
        is_float = true;
        tok.text[text_len as usize] = 46;
        text_len = text_len + 1;
        l = lexer_advance(l);

        // Fractional part
        while !lexer_at_end(l) && is_digit(l.current_char) && text_len < 63 {
            tok.text[text_len as usize] = l.current_char;
            text_len = text_len + 1;
            l = lexer_advance(l);
        }
    }

    // Exponent
    if l.current_char == 101 || l.current_char == 69 {  // 'e' or 'E'
        is_float = true;
        tok.text[text_len as usize] = l.current_char;
        text_len = text_len + 1;
        l = lexer_advance(l);

        // Optional sign
        if l.current_char == 43 || l.current_char == 45 {
            tok.text[text_len as usize] = l.current_char;
            text_len = text_len + 1;
            l = lexer_advance(l);
        }

        // Exponent digits
        while !lexer_at_end(l) && is_digit(l.current_char) && text_len < 63 {
            tok.text[text_len as usize] = l.current_char;
            text_len = text_len + 1;
            l = lexer_advance(l);
        }
    }

    tok.text_len = text_len;

    if is_float {
        tok.token_type = TT_FLOAT();
        tok.float_value = parse_float_simple(tok.text, text_len);
    } else {
        tok.token_type = TT_INTEGER();
        tok.int_value = parse_int_simple(tok.text, text_len);
    }

    scan_result_new(l, tok)
}

/// Parse integer from text
fn parse_int_simple(s: [i8; 64], len: i64) -> i64 {
    var result: i64 = 0;
    var i: i64 = 0;
    while i < len {
        let c = s[i as usize];
        if is_digit(c) {
            result = result * 10 + (c - 48) as i64;
        }
        i = i + 1;
    }
    result
}

/// Parse float from text (simplified)
fn parse_float_simple(s: [i8; 64], len: i64) -> f64 {
    var result: f64 = 0.0;
    var divisor: f64 = 1.0;
    var in_fraction = false;
    var in_exponent = false;
    var exponent: i64 = 0;
    var exp_sign: i64 = 1;

    var i: i64 = 0;
    while i < len {
        let c = s[i as usize];

        if c == 46 {  // '.'
            in_fraction = true;
        } else if c == 101 || c == 69 {  // 'e' or 'E'
            in_exponent = true;
        } else if in_exponent && c == 45 {  // '-'
            exp_sign = -1;
        } else if in_exponent && c == 43 {  // '+'
            exp_sign = 1;
        } else if is_digit(c) {
            let digit = (c - 48) as f64;
            if in_exponent {
                exponent = exponent * 10 + (c - 48) as i64;
            } else if in_fraction {
                divisor = divisor * 10.0;
                result = result + digit / divisor;
            } else {
                result = result * 10.0 + digit;
            }
        }
        i = i + 1;
    }

    // Apply exponent
    if exponent > 0 {
        var e: i64 = 0;
        while e < exponent {
            if exp_sign > 0 {
                result = result * 10.0;
            } else {
                result = result / 10.0;
            }
            e = e + 1;
        }
    }

    result
}

/// Scan string literal
fn scan_string(lex: Lexer) -> ScanResult {
    let start_line = lex.line;
    let start_col = lex.column;
    let quote = lex.current_char;
    var l = lexer_advance(lex);  // Skip opening quote
    var tok = token_new();
    tok.line = start_line;
    tok.column = start_col;
    tok.token_type = TT_STRING();

    var text_len: i64 = 0;
    while !lexer_at_end(l) && l.current_char != quote && text_len < 63 {
        tok.text[text_len as usize] = l.current_char;
        text_len = text_len + 1;
        l = lexer_advance(l);
    }

    if l.current_char == quote {
        l = lexer_advance(l);  // Skip closing quote
    }

    tok.text_len = text_len;
    scan_result_new(l, tok)
}

// ============================================================================
// NEXT TOKEN
// ============================================================================

/// Get next token from lexer
fn lexer_next_token(lex: Lexer) -> ScanResult {
    var l = skip_whitespace(lex);

    if lexer_at_end(l) {
        return scan_result_new(l, token_eof(l.line, l.column))
    }

    let c = l.current_char;
    let start_line = l.line;
    let start_col = l.column;

    // Single-character tokens
    if c == 40 {  // '('
        return scan_result_new(lexer_advance(l), token_simple(TT_LPAREN(), start_line, start_col))
    }
    if c == 41 {  // ')'
        return scan_result_new(lexer_advance(l), token_simple(TT_RPAREN(), start_line, start_col))
    }
    if c == 123 {  // '{'
        return scan_result_new(lexer_advance(l), token_simple(TT_LBRACE(), start_line, start_col))
    }
    if c == 125 {  // '}'
        return scan_result_new(lexer_advance(l), token_simple(TT_RBRACE(), start_line, start_col))
    }
    if c == 91 {  // '['
        return scan_result_new(lexer_advance(l), token_simple(TT_LBRACKET(), start_line, start_col))
    }
    if c == 93 {  // ']'
        return scan_result_new(lexer_advance(l), token_simple(TT_RBRACKET(), start_line, start_col))
    }
    if c == 44 {  // ','
        return scan_result_new(lexer_advance(l), token_simple(TT_COMMA(), start_line, start_col))
    }
    if c == 58 {  // ':'
        return scan_result_new(lexer_advance(l), token_simple(TT_COLON(), start_line, start_col))
    }
    if c == 59 {  // ';'
        return scan_result_new(lexer_advance(l), token_simple(TT_SEMICOLON(), start_line, start_col))
    }
    if c == 43 {  // '+'
        return scan_result_new(lexer_advance(l), token_simple(TT_OP_PLUS(), start_line, start_col))
    }
    if c == 42 {  // '*'
        return scan_result_new(lexer_advance(l), token_simple(TT_OP_STAR(), start_line, start_col))
    }
    if c == 94 {  // '^'
        return scan_result_new(lexer_advance(l), token_simple(TT_OP_CARET(), start_line, start_col))
    }
    if c == 126 {  // '~'
        return scan_result_new(lexer_advance(l), token_simple(TT_OP_TILDE(), start_line, start_col))
    }
    if c == 61 {  // '='
        return scan_result_new(lexer_advance(l), token_simple(TT_OP_EQ(), start_line, start_col))
    }
    if c == 60 {  // '<'
        return scan_result_new(lexer_advance(l), token_simple(TT_OP_LT(), start_line, start_col))
    }
    if c == 62 {  // '>'
        return scan_result_new(lexer_advance(l), token_simple(TT_OP_GT(), start_line, start_col))
    }
    if c == 10 {  // '\n'
        return scan_result_new(lexer_advance(l), token_simple(TT_NEWLINE(), start_line, start_col))
    }

    // Multi-character tokens
    if c == 45 {  // '-' or '->'
        l = lexer_advance(l);
        if l.current_char == 62 {  // '>'
            return scan_result_new(lexer_advance(l), token_simple(TT_OP_ARROW(), start_line, start_col))
        }
        return scan_result_new(l, token_simple(TT_OP_MINUS(), start_line, start_col))
    }

    if c == 46 {  // '.' or '..'
        l = lexer_advance(l);
        if l.current_char == 46 {  // '..'
            return scan_result_new(lexer_advance(l), token_simple(TT_OP_DOTDOT(), start_line, start_col))
        }
        return scan_result_new(l, token_simple(TT_OP_DOT(), start_line, start_col))
    }

    if c == 47 {  // '/' or '//' comment or '/h'
        l = lexer_advance(l);
        if l.current_char == 47 {  // '//' comment
            l = skip_line_comment(l);
            return lexer_next_token(l)  // Recursive call for next real token
        }
        if l.current_char == 104 {  // '/h' unit
            return scan_result_new(lexer_advance(l), token_simple(TT_UNIT_PER_H(), start_line, start_col))
        }
        return scan_result_new(l, token_simple(TT_OP_SLASH(), start_line, start_col))
    }

    // String literal
    if c == 34 || c == 39 {  // '"' or '\''
        return scan_string(l)
    }

    // Number
    if is_digit(c) {
        return scan_number(l)
    }

    // Identifier or keyword
    if is_alpha(c) {
        return scan_identifier(l)
    }

    // Unknown character - error
    var err_tok = token_simple(TT_ERROR(), start_line, start_col);
    err_tok.text[0] = c;
    err_tok.text_len = 1;
    scan_result_new(lexer_advance(l), err_tok)
}

// ============================================================================
// TOKEN STREAM
// ============================================================================

/// Token stream for parser
struct TokenStream {
    tokens: [Token; 1024],
    n_tokens: i64,
    pos: i64,
}

fn token_stream_new() -> TokenStream {
    TokenStream {
        tokens: [token_new(); 1024],
        n_tokens: 0,
        pos: 0,
    }
}

/// Tokenize source into stream
fn tokenize(source: [i8; 8192], source_len: i64) -> TokenStream {
    var stream = token_stream_new();
    var lex = lexer_init(source, source_len);

    while stream.n_tokens < 1024 {
        let result = lexer_next_token(lex);
        lex = result.lexer;
        let tok = result.token;

        // Skip newlines for simpler parsing
        if tok.token_type != TT_NEWLINE() {
            stream.tokens[stream.n_tokens as usize] = tok;
            stream.n_tokens = stream.n_tokens + 1;
        }

        if tok.token_type == TT_EOF() || tok.token_type == TT_ERROR() {
            break
        }
    }

    stream
}

/// Get current token
fn stream_current(stream: TokenStream) -> Token {
    if stream.pos < stream.n_tokens {
        stream.tokens[stream.pos as usize]
    } else {
        token_eof(0, 0)
    }
}

/// Advance stream
fn stream_advance(stream: TokenStream) -> TokenStream {
    var s = stream;
    if s.pos < s.n_tokens {
        s.pos = s.pos + 1;
    }
    s
}

/// Check current token type
fn stream_check(stream: TokenStream, tt: i32) -> bool {
    stream.pos < stream.n_tokens && stream.tokens[stream.pos as usize].token_type == tt
}

// ============================================================================
// TESTS
// ============================================================================

fn test_lexer_simple() -> bool {
    var source: [i8; 8192] = [0; 8192];
    // "model X { }"
    source[0] = 109; source[1] = 111; source[2] = 100; source[3] = 101; source[4] = 108;  // "model"
    source[5] = 32;   // space
    source[6] = 88;   // "X"
    source[7] = 32;   // space
    source[8] = 123;  // "{"
    source[9] = 32;   // space
    source[10] = 125; // "}"

    let stream = tokenize(source, 11);

    // Should have: model, X, {, }, EOF
    if stream.n_tokens != 5 { return false }
    if stream.tokens[0].token_type != TT_KW_MODEL() { return false }
    if stream.tokens[1].token_type != TT_IDENTIFIER() { return false }
    if stream.tokens[2].token_type != TT_LBRACE() { return false }
    if stream.tokens[3].token_type != TT_RBRACE() { return false }
    if stream.tokens[4].token_type != TT_EOF() { return false }

    true
}

fn test_lexer_numbers() -> bool {
    var source: [i8; 8192] = [0; 8192];
    // "100 3.14"
    source[0] = 49; source[1] = 48; source[2] = 48;  // "100"
    source[3] = 32;   // space
    source[4] = 51; source[5] = 46; source[6] = 49; source[7] = 52;  // "3.14"

    let stream = tokenize(source, 8);

    if stream.n_tokens < 3 { return false }
    if stream.tokens[0].token_type != TT_INTEGER() { return false }
    if stream.tokens[0].int_value != 100 { return false }
    if stream.tokens[1].token_type != TT_FLOAT() { return false }

    true
}

fn test_lexer_operators() -> bool {
    var source: [i8; 8192] = [0; 8192];
    // "-> ~ ="
    source[0] = 45; source[1] = 62;  // "->"
    source[2] = 32;  // space
    source[3] = 126; // "~"
    source[4] = 32;  // space
    source[5] = 61;  // "="

    let stream = tokenize(source, 6);

    if stream.n_tokens < 4 { return false }
    if stream.tokens[0].token_type != TT_OP_ARROW() { return false }
    if stream.tokens[1].token_type != TT_OP_TILDE() { return false }
    if stream.tokens[2].token_type != TT_OP_EQ() { return false }

    true
}

fn main() -> i32 {
    print("Testing medlang::lexer module...\n");

    if !test_lexer_simple() {
        print("FAIL: lexer_simple\n");
        return 1
    }
    print("PASS: lexer_simple\n");

    if !test_lexer_numbers() {
        print("FAIL: lexer_numbers\n");
        return 2
    }
    print("PASS: lexer_numbers\n");

    if !test_lexer_operators() {
        print("FAIL: lexer_operators\n");
        return 3
    }
    print("PASS: lexer_operators\n");

    print("All medlang::lexer tests PASSED\n");
    0
}
