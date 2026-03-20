/*
 * stage0.c — Minimal Sounio-to-x86-64 ELF compiler (bootstrap stage 0)
 *
 * Usage: ./stage0 input.sio output.elf
 *
 * This is a ~2000-line C program that reads Sounio source and emits a
 * static x86-64 Linux ELF binary. No dependencies beyond libc.
 * Memory usage: <100 MB for any reasonable input.
 *
 * Supports: fn, let, var, if/else, while, break, return, struct,
 *           arrays, match (simple), impl, &/&!, print(), print_int(),
 *           i64, f64, bool, as casts, &&, ||, bitwise ops.
 *
 * Based on bootstrap_v0.sio's AST-direct codegen engine.
 *
 * Build: cc -O2 -o stage0 stage0.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ================================================================
 * LIMITS
 * ================================================================ */
#define MAX_CODE      (1024*1024)  /* 1 MB code output */
#define MAX_FUNCS     1024
#define MAX_STRUCTS   512
#define MAX_FIELDS    4096
#define MAX_LOCALS    1024
#define MAX_TOKENS    (2*1024*1024)
#define MAX_SOURCE    (4*1024*1024)  /* 4 MB source (multi-module) */
#define MAX_NAME      128
#define MAX_PARAMS    16
#define MAX_FWD       256
#define MAX_BREAKS    64
#define MAX_MATCH_ARMS 256
#define MAX_CALL_ARGS 24

/* Global data offsets (at code offset 0) */
#define GLOBAL_SAVED_RSP 0   /* offset 0: saved initial rsp (8 bytes) */
#define GLOBAL_HEAP_PTR  8   /* offset 8: heap bump pointer (8 bytes) */
#define GLOBAL_FIXED_DATA 16 /* first 16 bytes are fixed */
#define MAX_GLOBALS      256
static int GLOBAL_DATA_SIZE = 16; /* extended by scan_globals */

/* ================================================================
 * NAME (fixed-size identifier)
 * ================================================================ */
typedef struct { char buf[MAX_NAME]; int len; } Name;

static Name name_from(const char *s) {
    Name n = {0};
    n.len = (int)strlen(s);
    if (n.len > MAX_NAME-1) n.len = MAX_NAME-1;
    memcpy(n.buf, s, n.len);
    return n;
}

static bool name_eq(Name a, Name b) {
    if (a.len != b.len) return false;
    return memcmp(a.buf, b.buf, a.len) == 0;
}

static Name empty_name(void) { Name n = {0}; return n; }

static bool name_is(Name n, const char *s) {
    return name_eq(n, name_from(s));
}

/* ================================================================
 * TOKEN
 * ================================================================ */
enum TokenKind {
    TK_EOF=0, TK_IDENT, TK_INT, TK_FLOAT, TK_STRING, TK_BOOL_TRUE, TK_BOOL_FALSE,
    /* keywords */
    TK_FN, TK_LET, TK_VAR, TK_IF, TK_ELSE, TK_WHILE, TK_RETURN, TK_BREAK,
    TK_STRUCT, TK_ENUM, TK_IMPL, TK_MATCH, TK_WITH, TK_AS, TK_IN, TK_FOR,
    TK_PUB, TK_MUT, TK_CONST, TK_TYPE, TK_TRAIT, TK_LOOP, TK_CONTINUE,
    TK_MODULE, TK_USE, TK_SELF_LOWER, TK_SELF_UPPER, TK_WHERE,
    TK_EFFECT, TK_HANDLER, TK_HANDLE, TK_PERFORM, TK_RESUME,
    TK_LINEAR, TK_AFFINE, TK_MOVE, TK_COPY, TK_DROP, TK_REF, TK_DYN, TK_BOX,
    TK_ASYNC, TK_AWAIT, TK_SPAWN, TK_UNSAFE, TK_EXTERN, TK_STATIC,
    /* punctuation */
    TK_LPAREN, TK_RPAREN, TK_LBRACE, TK_RBRACE, TK_LBRACKET, TK_RBRACKET,
    TK_COMMA, TK_SEMICOLON, TK_COLON, TK_COLONCOLON, TK_DOT, TK_DOTDOT,
    TK_ARROW, TK_FAT_ARROW, TK_UNDERSCORE, TK_HASH, TK_AT,
    /* operators */
    TK_PLUS, TK_MINUS, TK_STAR, TK_SLASH, TK_PERCENT,
    TK_AMP, TK_PIPE, TK_CARET, TK_TILDE, TK_BANG,
    TK_SHL, TK_SHR, TK_AMPAMP, TK_PIPEPIPE,
    TK_EQ, TK_NE, TK_LT, TK_LE, TK_GT, TK_GE,
    TK_ASSIGN, TK_PLUS_ASSIGN, TK_MINUS_ASSIGN,
    TK_STAR_ASSIGN, TK_SLASH_ASSIGN,
    TK_NEWLINE, TK_ERROR,
    TK_KEYWORD_OTHER, /* catch-all for keywords we don't need */
};

typedef struct {
    int kind;
    Name name;       /* for TK_IDENT */
    int64_t int_val; /* for TK_INT */
    double float_val;/* for TK_FLOAT */
    char str_buf[256]; int str_len; /* for TK_STRING */
    int line, col;
} Token;

/* ================================================================
 * LEXER
 * ================================================================ */
static char g_src[MAX_SOURCE];
static int g_src_len;
static int g_pos, g_line, g_col;

static int ch(void) { return g_pos < g_src_len ? (unsigned char)g_src[g_pos] : 0; }
static int peek(int off) { int p = g_pos+off; return p < g_src_len ? (unsigned char)g_src[p] : 0; }
static void adv(void) { if (g_pos < g_src_len) { if (g_src[g_pos]=='\n') { g_line++; g_col=1; } else g_col++; g_pos++; } }

static bool is_alpha(int c) { return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_'; }
static bool is_digit(int c) { return c>='0'&&c<='9'; }
static bool is_alnum(int c) { return is_alpha(c)||is_digit(c); }

struct { const char *word; int kind; } g_keywords[] = {
    {"fn",TK_FN},{"let",TK_LET},{"var",TK_VAR},{"if",TK_IF},{"else",TK_ELSE},
    {"while",TK_WHILE},{"return",TK_RETURN},{"break",TK_BREAK},
    {"struct",TK_STRUCT},{"enum",TK_ENUM},{"impl",TK_IMPL},{"match",TK_MATCH},
    {"with",TK_WITH},{"as",TK_AS},{"in",TK_IN},{"for",TK_FOR},
    {"pub",TK_PUB},{"mut",TK_MUT},{"const",TK_CONST},{"type",TK_TYPE},
    {"trait",TK_TRAIT},{"loop",TK_LOOP},{"continue",TK_CONTINUE},
    {"module",TK_MODULE},{"use",TK_USE},{"self",TK_SELF_LOWER},{"Self",TK_SELF_UPPER},
    {"where",TK_WHERE},{"true",TK_BOOL_TRUE},{"false",TK_BOOL_FALSE},
    {"effect",TK_EFFECT},{"handler",TK_HANDLER},{"handle",TK_HANDLE},
    {"perform",TK_PERFORM},{"resume",TK_RESUME},
    {"linear",TK_LINEAR},{"affine",TK_AFFINE},{"move",TK_MOVE},{"copy",TK_COPY},
    {"drop",TK_DROP},{"ref",TK_REF},{"dyn",TK_DYN},{"box",TK_BOX},
    {"async",TK_ASYNC},{"await",TK_AWAIT},{"spawn",TK_SPAWN},
    {"unsafe",TK_UNSAFE},{"extern",TK_EXTERN},{"static",TK_STATIC},
    {NULL,0}
};

static int classify_keyword(const char *s, int len) {
    for (int i = 0; g_keywords[i].word; i++) {
        if ((int)strlen(g_keywords[i].word) == len && memcmp(g_keywords[i].word, s, len) == 0)
            return g_keywords[i].kind;
    }
    /* Any uppercase-starting identifier that isn't matched is still TK_IDENT */
    return TK_IDENT;
}

static Token g_tokens[MAX_TOKENS];
static int g_token_count;

static void skip_ws_and_comments(void) {
    for (;;) {
        while (ch()==' '||ch()=='\t'||ch()=='\r'||ch()=='\n') adv();
        if (ch()=='/' && peek(1)=='/') {
            while (ch() && ch()!='\n') adv();
            continue;
        }
        if (ch()=='/' && peek(1)=='*') {
            adv(); adv();
            while (g_pos < g_src_len-1) {
                if (ch()=='*' && peek(1)=='/') { adv(); adv(); break; }
                adv();
            }
            continue;
        }
        break;
    }
}

static void lex_all(void) {
    g_pos = 0; g_line = 1; g_col = 1; g_token_count = 0;
    for (;;) {
        skip_ws_and_comments();
        if (g_pos >= g_src_len) break;
        Token t = {0}; t.line = g_line; t.col = g_col;
        int c = ch();
        /* string */
        if (c == '"') {
            adv(); t.kind = TK_STRING; t.str_len = 0;
            while (ch() && ch()!='"') {
                if (ch()=='\\') {
                    adv();
                    if (ch()=='n') { t.str_buf[t.str_len++]='\n'; adv(); }
                    else if (ch()=='t') { t.str_buf[t.str_len++]='\t'; adv(); }
                    else if (ch()=='\\') { t.str_buf[t.str_len++]='\\'; adv(); }
                    else if (ch()=='"') { t.str_buf[t.str_len++]='"'; adv(); }
                    else if (ch()=='0') { t.str_buf[t.str_len++]='\0'; adv(); }
                    else { t.str_buf[t.str_len++]=(char)ch(); adv(); }
                } else { t.str_buf[t.str_len++]=(char)ch(); adv(); }
            }
            if (ch()=='"') adv();
            goto emit;
        }
        /* number */
        if (is_digit(c) || (c=='.' && is_digit(peek(1)))) {
            int64_t iv = 0; bool is_float = false; double fv = 0;
            if (c=='0' && (peek(1)=='x'||peek(1)=='X')) {
                adv(); adv();
                while (is_digit(ch())||(ch()>='a'&&ch()<='f')||(ch()>='A'&&ch()<='F')) {
                    int d = is_digit(ch()) ? ch()-'0' : (ch()>='a'?ch()-'a'+10:ch()-'A'+10);
                    iv = iv*16+d; adv();
                }
            } else if (c=='0' && (peek(1)=='b'||peek(1)=='B')) {
                adv(); adv();
                while (ch()=='0'||ch()=='1') { iv = iv*2+(ch()-'0'); adv(); }
            } else {
                while (is_digit(ch())) { iv = iv*10+(ch()-'0'); adv(); }
                if (ch()=='.' && is_digit(peek(1))) {
                    is_float = true; fv = (double)iv;
                    adv(); double frac = 0.1;
                    while (is_digit(ch())) { fv += (ch()-'0')*frac; frac *= 0.1; adv(); }
                }
            }
            /* skip type suffix */
            while (is_alpha(ch())) adv();
            if (is_float) { t.kind = TK_FLOAT; t.float_val = fv; }
            else { t.kind = TK_INT; t.int_val = iv; }
            goto emit;
        }
        /* identifier/keyword */
        if (is_alpha(c)) {
            char buf[MAX_NAME]; int len = 0;
            while (is_alnum(ch()) && len < MAX_NAME-1) { buf[len++] = (char)ch(); adv(); }
            buf[len] = 0;
            t.kind = classify_keyword(buf, len);
            t.name.len = len; memcpy(t.name.buf, buf, len);
            goto emit;
        }
        /* operators & punctuation */
        if (c=='(') { t.kind=TK_LPAREN; adv(); goto emit; }
        if (c==')') { t.kind=TK_RPAREN; adv(); goto emit; }
        if (c=='{') { t.kind=TK_LBRACE; adv(); goto emit; }
        if (c=='}') { t.kind=TK_RBRACE; adv(); goto emit; }
        if (c=='[') { t.kind=TK_LBRACKET; adv(); goto emit; }
        if (c==']') { t.kind=TK_RBRACKET; adv(); goto emit; }
        if (c==',') { t.kind=TK_COMMA; adv(); goto emit; }
        if (c==';') { t.kind=TK_SEMICOLON; adv(); goto emit; }
        if (c=='_' && !is_alnum(peek(1))) { t.kind=TK_UNDERSCORE; adv(); goto emit; }
        if (c=='#') { t.kind=TK_HASH; adv(); goto emit; }
        if (c=='@') { t.kind=TK_AT; adv(); goto emit; }
        if (c=='~') { t.kind=TK_TILDE; adv(); goto emit; }
        if (c==':') {
            if (peek(1)==':') { t.kind=TK_COLONCOLON; adv(); adv(); goto emit; }
            t.kind=TK_COLON; adv(); goto emit;
        }
        if (c=='.') {
            if (peek(1)=='.') { t.kind=TK_DOTDOT; adv(); adv(); goto emit; }
            t.kind=TK_DOT; adv(); goto emit;
        }
        if (c=='+') {
            if (peek(1)=='=') { t.kind=TK_PLUS_ASSIGN; adv(); adv(); goto emit; }
            t.kind=TK_PLUS; adv(); goto emit;
        }
        if (c=='-') {
            if (peek(1)=='>') { t.kind=TK_ARROW; adv(); adv(); goto emit; }
            if (peek(1)=='=') { t.kind=TK_MINUS_ASSIGN; adv(); adv(); goto emit; }
            t.kind=TK_MINUS; adv(); goto emit;
        }
        if (c=='*') {
            if (peek(1)=='=') { t.kind=TK_STAR_ASSIGN; adv(); adv(); goto emit; }
            t.kind=TK_STAR; adv(); goto emit;
        }
        if (c=='/') {
            if (peek(1)=='=') { t.kind=TK_SLASH_ASSIGN; adv(); adv(); goto emit; }
            t.kind=TK_SLASH; adv(); goto emit;
        }
        if (c=='%') { t.kind=TK_PERCENT; adv(); goto emit; }
        if (c=='&') {
            if (peek(1)=='&') { t.kind=TK_AMPAMP; adv(); adv(); goto emit; }
            t.kind=TK_AMP; adv(); goto emit;
        }
        if (c=='|') {
            if (peek(1)=='|') { t.kind=TK_PIPEPIPE; adv(); adv(); goto emit; }
            t.kind=TK_PIPE; adv(); goto emit;
        }
        if (c=='^') { t.kind=TK_CARET; adv(); goto emit; }
        if (c=='!') {
            if (peek(1)=='=') { t.kind=TK_NE; adv(); adv(); goto emit; }
            t.kind=TK_BANG; adv(); goto emit;
        }
        if (c=='=') {
            if (peek(1)=='=') { t.kind=TK_EQ; adv(); adv(); goto emit; }
            if (peek(1)=='>') { t.kind=TK_FAT_ARROW; adv(); adv(); goto emit; }
            t.kind=TK_ASSIGN; adv(); goto emit;
        }
        if (c=='<') {
            if (peek(1)=='<') { t.kind=TK_SHL; adv(); adv(); goto emit; }
            if (peek(1)=='=') { t.kind=TK_LE; adv(); adv(); goto emit; }
            t.kind=TK_LT; adv(); goto emit;
        }
        if (c=='>') {
            if (peek(1)=='>') { t.kind=TK_SHR; adv(); adv(); goto emit; }
            if (peek(1)=='=') { t.kind=TK_GE; adv(); adv(); goto emit; }
            t.kind=TK_GT; adv(); goto emit;
        }
        /* skip unknown */
        adv(); continue;
        emit:
        if (g_token_count < MAX_TOKENS) g_tokens[g_token_count++] = t;
    }
    /* EOF token */
    Token eof = {0}; eof.kind = TK_EOF; eof.line = g_line; eof.col = g_col;
    if (g_token_count < MAX_TOKENS) g_tokens[g_token_count++] = eof;
}

/* ================================================================
 * PARSER — lightweight top-down
 * ================================================================ */
static int g_tp; /* token position */

static Token *cur(void) { return &g_tokens[g_tp < g_token_count ? g_tp : g_token_count-1]; }
static int tk(void) { return cur()->kind; }
static void next(void) { if (g_tp < g_token_count-1) g_tp++; }
static bool eat(int k) { if (tk()==k) { next(); return true; } return false; }
static int g_error_count = 0;
static void expect(int k) {
    if (tk()!=k) {
        if (g_error_count < 10)
            fprintf(stderr, "stage0: expect %d got %d at line %d col %d\n",
                    k, tk(), cur()->line, cur()->col);
        g_error_count++;
    }
    next();
}
static bool is_keyword(int k) {
    return k >= TK_FN && k <= TK_STATIC;
}

/* Forward declarations */
typedef struct Expr Expr;
typedef struct Stmt Stmt;
typedef struct Block Block;
typedef struct FnDef FnDef;
typedef struct StructDef StructDef;
typedef struct Item Item;

/* ================================================================
 * AST NODES (heap-allocated, simple linked lists)
 * ================================================================ */
enum ExprKind {
    EX_INT, EX_FLOAT, EX_BOOL, EX_STRING, EX_IDENT, EX_BINARY, EX_UNARY,
    EX_CALL, EX_METHOD_CALL, EX_FIELD, EX_INDEX, EX_IF, EX_WHILE, EX_BLOCK,
    EX_RETURN, EX_BREAK, EX_CONTINUE, EX_ASSIGN, EX_STRUCT_LIT, EX_ARRAY_LIT, EX_ARRAY_REPEAT,
    EX_CAST, EX_REF, EX_DEREF, EX_MATCH, EX_TUPLE, EX_CLOSURE, EX_RANGE,
};

enum BinOp {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_REM,
    OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE,
    OP_AND, OP_OR, OP_BIT_AND, OP_BIT_OR, OP_BIT_XOR, OP_SHL, OP_SHR,
};

typedef struct ExprList { Expr *expr; struct ExprList *next; } ExprList;
typedef struct FieldInit { Name name; Expr *value; struct FieldInit *next; } FieldInit;
typedef struct MatchArm {
    int pat_kind; /* 0=wildcard, 1=ident binding, 2=enum path, 3=int lit */
    Name pat_name; /* binding or variant name */
    Name pat_path_prefix; /* e.g. "TokenKind" in TokenKind::Foo */
    int64_t pat_int;
    Expr *body;
    struct MatchArm *next;
} MatchArm;

struct Expr {
    int kind;
    int64_t int_val;
    double float_val;
    bool bool_val;
    char str_buf[256]; int str_len;
    Name name;
    int op; /* BinOp for EX_BINARY, unary type for EX_UNARY */
    Expr *left, *right; /* binary operands, or condition+body */
    Expr *cond;
    Block *block, *else_block;
    ExprList *args;
    FieldInit *fields;
    MatchArm *arms;
    Name cast_type;
    int line;
};

struct Stmt {
    int kind; /* 0=expr, 1=let, 2=var, 3=assign */
    Name name;
    Expr *expr;
    Expr *right; /* RHS for assignments */
    Expr *type_expr; /* for let/var type annotation */
    struct Stmt *next;
};

struct Block {
    Stmt *stmts;
};

typedef struct Param { Name name; Name type_name; bool is_ref; bool is_mut_ref; bool is_byte_array; struct Param *next; } Param;
typedef struct FieldDef { Name name; Name type_name; bool is_byte_array; struct FieldDef *next; } FieldDef;

struct FnDef {
    Name name;
    Param *params;
    int param_count;
    Name return_type;
    bool returns_ref;
    Block *body;
};

struct StructDef {
    Name name;
    FieldDef *fields;
    int field_count;
};

typedef struct EnumVariant { Name name; struct EnumVariant *next; } EnumVariant;
typedef struct EnumDef { Name name; EnumVariant *variants; } EnumDef;

typedef struct Method { FnDef *fn; struct Method *next; } Method;
typedef struct ImplDef { Name type_name; Method *methods; } ImplDef;

enum ItemKind { ITEM_FN, ITEM_STRUCT, ITEM_ENUM, ITEM_IMPL, ITEM_USE, ITEM_TYPE, ITEM_MODULE };
struct Item { int kind; FnDef *fn; StructDef *st; EnumDef *en; ImplDef *im; struct Item *next; };

/* Simple arena allocator */
static char g_arena[256*1024*1024]; /* 256 MB */
static int g_arena_pos = 0;

static void *arena_alloc(int size) {
    size = (size + 7) & ~7; /* align to 8 */
    if (g_arena_pos + size > (int)sizeof(g_arena)) {
        fprintf(stderr, "stage0: arena overflow (%d bytes)\n", g_arena_pos + size);
        exit(1);
    }
    void *p = g_arena + g_arena_pos;
    memset(p, 0, size);
    g_arena_pos += size;
    return p;
}

#define NEW(T) ((T*)arena_alloc(sizeof(T)))

/* ================================================================
 * PARSER IMPLEMENTATION
 * ================================================================ */

/* Skip type annotations (we don't need them for codegen) */
static void skip_type(void);
static Expr *parse_expr(void);
static Block *parse_block(void);
static Expr *parse_postfix(Expr *e);

static int g_gt_pending = 0; /* for >> handling in nested generics */
static int g_no_struct_lit = 0; /* suppress struct-lit parsing in if/while cond */

static void skip_type(void) {
    /* Handle &, &!, Box<T>, [T; N], Option<T>, etc. */
    if (g_gt_pending > 0) { g_gt_pending--; return; } /* consume pending > from >> */
    if (eat(TK_AMP)) { eat(TK_BANG); skip_type(); return; }
    if (eat(TK_STAR)) { skip_type(); return; } /* *T pointer type */
    if (eat(TK_LBRACKET)) {
        skip_type();
        if (eat(TK_SEMICOLON)) parse_expr(); /* array size */
        expect(TK_RBRACKET); return;
    }
    if (tk()==TK_IDENT || is_keyword(tk())) {
        next();
        while (eat(TK_COLONCOLON)) { if (tk()==TK_IDENT||is_keyword(tk())) next(); }
        if (eat(TK_LT)) {
            skip_type();
            while (1) {
                if (g_gt_pending > 0) { g_gt_pending--; break; }
                if (tk()==TK_SHR) { next(); g_gt_pending++; break; } /* >> = two >s */
                if (tk()==TK_GT) { next(); break; }
                if (!eat(TK_COMMA)) break; /* error recovery */
                skip_type();
            }
        }
        return;
    }
    if (eat(TK_LPAREN)) {
        if (tk()!=TK_RPAREN) { skip_type(); while (eat(TK_COMMA)) skip_type(); }
        expect(TK_RPAREN); return;
    }
    next(); /* skip unknown */
}

static void skip_effects(void) {
    /* with Effect1, Effect2, ... */
    if (eat(TK_WITH)) {
        if (tk()==TK_IDENT||is_keyword(tk())) next();
        while (eat(TK_COMMA)) { if (tk()==TK_IDENT||is_keyword(tk())) next(); }
    }
}

static Param *parse_params(int *count) {
    *count = 0;
    Param *head = NULL, **tail = &head;
    if (!eat(TK_LPAREN)) return NULL;
    while (tk() != TK_RPAREN && tk() != TK_EOF) {
        Param *p = NEW(Param);
        /* handle self/&self/&!self */
        if (tk()==TK_AMP) { next(); p->is_ref = true; if (eat(TK_BANG)) p->is_mut_ref = true; }
        if (tk()==TK_SELF_LOWER) { p->name = name_from("self"); next(); }
        else { p->name = cur()->name; next(); }
        if (eat(TK_COLON)) {
            if (tk()==TK_AMP) { p->is_ref = true; next(); if (eat(TK_BANG)) p->is_mut_ref = true; }
            /* Detect [i8; N] byte array type */
            if (tk()==TK_LBRACKET) {
                int save = g_tp;
                next(); /* [ */
                if (tk()==TK_IDENT && (name_is(cur()->name,"i8")||name_is(cur()->name,"u8"))) {
                    p->is_byte_array = true;
                }
                g_tp = save; /* restore — let skip_type handle it */
            }
            p->type_name = cur()->name;
            skip_type();
        }
        *tail = p; tail = &p->next; (*count)++;
        if (!eat(TK_COMMA)) break;
    }
    expect(TK_RPAREN);
    return head;
}

static Name parse_return_type(void) {
    Name ret = name_from("i64"); /* default */
    if (eat(TK_ARROW)) {
        if (tk()==TK_AMP) { next(); eat(TK_BANG); }
        if (tk()==TK_LBRACKET) { ret = name_from("array"); skip_type(); return ret; }
        ret = cur()->name; skip_type();
    }
    skip_effects();
    return ret;
}

/* Expression parser (Pratt-style precedence climbing) */
static Expr *parse_primary(void);
static Expr *parse_binop(int min_prec);

static int prec_of(int op) {
    switch (op) {
        case TK_PIPEPIPE: return 1;
        case TK_AMPAMP: return 2;
        case TK_EQ: case TK_NE: return 3;
        case TK_LT: case TK_LE: case TK_GT: case TK_GE: return 4;
        case TK_PIPE: return 5;
        case TK_CARET: return 6;
        case TK_AMP: return 7;
        case TK_SHL: case TK_SHR: return 8;
        case TK_PLUS: case TK_MINUS: return 9;
        case TK_STAR: case TK_SLASH: case TK_PERCENT: return 10;
        default: return -1;
    }
}

static int tk_to_binop(int t) {
    switch (t) {
        case TK_PLUS: return OP_ADD; case TK_MINUS: return OP_SUB;
        case TK_STAR: return OP_MUL; case TK_SLASH: return OP_DIV; case TK_PERCENT: return OP_REM;
        case TK_EQ: return OP_EQ; case TK_NE: return OP_NE;
        case TK_LT: return OP_LT; case TK_LE: return OP_LE;
        case TK_GT: return OP_GT; case TK_GE: return OP_GE;
        case TK_AMPAMP: return OP_AND; case TK_PIPEPIPE: return OP_OR;
        case TK_AMP: return OP_BIT_AND; case TK_PIPE: return OP_BIT_OR;
        case TK_CARET: return OP_BIT_XOR;
        case TK_SHL: return OP_SHL; case TK_SHR: return OP_SHR;
        default: return -1;
    }
}

static ExprList *parse_arglist(void) {
    ExprList *head = NULL, **tail = &head;
    expect(TK_LPAREN);
    while (tk() != TK_RPAREN && tk() != TK_EOF) {
        ExprList *n = NEW(ExprList);
        /* skip named args like "uncertainty:" */
        if (tk()==TK_IDENT && g_tokens[g_tp+1].kind==TK_COLON && g_tokens[g_tp+2].kind!=TK_COLON) {
            next(); next(); /* skip name: */
        }
        n->expr = parse_expr();
        *tail = n; tail = &n->next;
        if (!eat(TK_COMMA)) break;
    }
    expect(TK_RPAREN);
    return head;
}

static Expr *parse_primary(void) {
    Expr *e = NEW(Expr);
    e->line = cur()->line;

    if (tk()==TK_INT) { e->kind=EX_INT; e->int_val=cur()->int_val; next(); return e; }
    if (tk()==TK_FLOAT) { e->kind=EX_FLOAT; e->float_val=cur()->float_val; next(); return e; }
    if (tk()==TK_BOOL_TRUE) { e->kind=EX_BOOL; e->bool_val=true; next(); return e; }
    if (tk()==TK_BOOL_FALSE) { e->kind=EX_BOOL; e->bool_val=false; next(); return e; }
    if (tk()==TK_STRING) { e->kind=EX_STRING; e->str_len=cur()->str_len; memcpy(e->str_buf,cur()->str_buf,e->str_len); next(); return e; }

    if (tk()==TK_RETURN) { e->kind=EX_RETURN; next(); if (tk()!=TK_RBRACE&&tk()!=TK_EOF&&tk()!=TK_SEMICOLON) e->left=parse_expr(); return e; }
    if (tk()==TK_BREAK) { e->kind=EX_BREAK; next(); return e; }
    if (tk()==TK_CONTINUE) { e->kind=EX_CONTINUE; next(); return e; }

    if (tk()==TK_IF) {
        e->kind = EX_IF; next();
        g_no_struct_lit++; e->cond = parse_expr(); g_no_struct_lit--;
        e->block = parse_block();
        if (eat(TK_ELSE)) {
            if (tk()==TK_IF) { Block *b = NEW(Block); Stmt *s = NEW(Stmt); s->expr = parse_primary(); b->stmts = s; e->else_block = b; }
            else e->else_block = parse_block();
        }
        return e;
    }

    if (tk()==TK_WHILE) {
        e->kind = EX_WHILE; next();
        g_no_struct_lit++; e->cond = parse_expr(); g_no_struct_lit--;
        e->block = parse_block();
        return e;
    }

    if (tk()==TK_MATCH) {
        e->kind = EX_MATCH; next();
        g_no_struct_lit++; e->left = parse_expr(); g_no_struct_lit--; /* scrutinee */
        expect(TK_LBRACE);
        MatchArm **arm_tail = &e->arms;
        while (tk()!=TK_RBRACE && tk()!=TK_EOF) {
            MatchArm *arm = NEW(MatchArm);
            if (tk()==TK_UNDERSCORE) { arm->pat_kind = 0; next(); }
            else if (tk()==TK_INT) { arm->pat_kind = 3; arm->pat_int = cur()->int_val; next(); }
            else if (tk()==TK_MINUS && g_tokens[g_tp+1].kind==TK_INT) { next(); arm->pat_kind=3; arm->pat_int = -cur()->int_val; next(); }
            else if (tk()==TK_IDENT || is_keyword(tk())) {
                arm->pat_name = cur()->name; next();
                if (tk()==TK_COLONCOLON) {
                    arm->pat_path_prefix = arm->pat_name;
                    next(); /* skip :: */
                    arm->pat_name = cur()->name; next();
                    arm->pat_kind = 2;
                    if (eat(TK_LPAREN)) {
                        /* Some(x) pattern — parse binding */
                        if (tk()==TK_IDENT) { arm->pat_name = cur()->name; next(); }
                        else { parse_expr(); /* skip complex sub-pattern */ }
                        expect(TK_RPAREN);
                    }
                } else if (eat(TK_LPAREN)) {
                    /* Some(x) without path prefix */
                    arm->pat_kind = 2;
                    arm->pat_path_prefix = arm->pat_name;
                    if (tk()==TK_IDENT) { arm->pat_name = cur()->name; next(); }
                    expect(TK_RPAREN);
                } else {
                    arm->pat_kind = 1; /* simple binding */
                }
            }
            expect(TK_FAT_ARROW);
            if (tk()==TK_LBRACE) {
                Block *b = parse_block();
                Expr *be = NEW(Expr); be->kind = EX_BLOCK; be->block = b;
                arm->body = be;
            } else {
                arm->body = parse_expr();
            }
            eat(TK_COMMA);
            *arm_tail = arm; arm_tail = &arm->next;
        }
        expect(TK_RBRACE);
        return e;
    }

    if (tk()==TK_LBRACE) { e->kind = EX_BLOCK; e->block = parse_block(); return e; }

    if (tk()==TK_LPAREN) {
        next();
        if (tk()==TK_RPAREN) { e->kind = EX_TUPLE; next(); return e; } /* unit */
        Expr *inner = parse_expr();
        if (eat(TK_COMMA)) {
            /* tuple */
            e->kind = EX_TUPLE;
            ExprList *list = NEW(ExprList); list->expr = inner;
            ExprList **tail = &list->next;
            if (tk()!=TK_RPAREN) {
                ExprList *n = NEW(ExprList); n->expr = parse_expr();
                *tail = n; tail = &n->next;
                while (eat(TK_COMMA)) { ExprList *nn = NEW(ExprList); nn->expr = parse_expr(); *tail = nn; tail = &nn->next; }
            }
            e->args = list;
            expect(TK_RPAREN);
            return e;
        }
        expect(TK_RPAREN);
        /* parenthesized expr */
        return inner;
    }

    /* unary operators — capture postfix (so !f(x) = !(f(x)), -x.y = -(x.y)) */
    if (tk()==TK_MINUS) { e->kind=EX_UNARY; e->op=0; next(); e->left=parse_postfix(parse_primary()); return e; }
    if (tk()==TK_BANG) { e->kind=EX_UNARY; e->op=1; next(); e->left=parse_postfix(parse_primary()); return e; }
    if (tk()==TK_AMP) {
        e->kind=EX_REF; next();
        if (eat(TK_BANG)) e->op = 1; /* &! */
        e->left=parse_postfix(parse_primary()); return e;
    }
    if (tk()==TK_STAR) { e->kind=EX_DEREF; next(); e->left=parse_postfix(parse_primary()); return e; }

    /* array literal [a, b, c] or [val; count] */
    if (tk()==TK_LBRACKET) {
        next();
        Expr *first = parse_expr();
        if (eat(TK_SEMICOLON)) {
            e->kind = EX_ARRAY_REPEAT; e->left = first; e->right = parse_expr();
            expect(TK_RBRACKET); return e;
        }
        e->kind = EX_ARRAY_LIT;
        ExprList *list = NEW(ExprList); list->expr = first;
        ExprList **tail = &list->next;
        while (eat(TK_COMMA)) {
            if (tk()==TK_RBRACKET) break;
            ExprList *n = NEW(ExprList); n->expr = parse_expr(); *tail = n; tail = &n->next;
        }
        e->args = list; expect(TK_RBRACKET); return e;
    }

    /* identifier — possibly path, struct lit, function call, or variable */
    if (tk()==TK_IDENT || is_keyword(tk())) {
        e->name = cur()->name; next(); e->kind = EX_IDENT;
        /* Path expression: Foo::Bar or Foo::Bar::Baz */
        while (tk()==TK_COLONCOLON && (g_tokens[g_tp+1].kind==TK_IDENT || is_keyword(g_tokens[g_tp+1].kind))) {
            e->cast_type = e->name; /* save prefix (enum name) */
            next(); /* skip :: */
            e->name = cur()->name; next(); /* take last segment as the name */
        }
        /* Enum variant with payload: Some(x) or Foo::Bar(args) */
        if (tk()==TK_LPAREN && e->name.buf[0]>='A' && e->name.buf[0]<='Z') {
            /* Could be an enum constructor or a function call — treat as call */
            e->kind = EX_CALL; e->args = parse_arglist();
            return e;
        }
        /* Check for struct literal: Name { field: val, ... } */
        if (tk()==TK_LBRACE && e->name.buf[0]>='A' && e->name.buf[0]<='Z' && !g_no_struct_lit) {
            /* Struct literal */
            e->kind = EX_STRUCT_LIT; next();
            FieldInit **ftail = &e->fields;
            while (tk()!=TK_RBRACE && tk()!=TK_EOF) {
                FieldInit *fi = NEW(FieldInit);
                fi->name = cur()->name; next();
                if (eat(TK_COLON)) {
                    fi->value = parse_expr();
                } else {
                    /* shorthand: Point { x, y } — value is same as field name */
                    Expr *ve = NEW(Expr); ve->kind = EX_IDENT; ve->name = fi->name;
                    fi->value = ve;
                }
                *ftail = fi; ftail = &fi->next;
                if (!eat(TK_COMMA)) break;
            }
            expect(TK_RBRACE); return e;
        }
        return e;
    }

    /* fallback: skip token */
    next();
    return e;
}

static Expr *parse_postfix(Expr *e) {
    for (;;) {
        /* field access: .name */
        if (tk()==TK_DOT) {
            next();
            /* method call: expr.method(args) */
            if ((tk()==TK_IDENT||is_keyword(tk())) && g_tokens[g_tp+1].kind==TK_LPAREN) {
                Expr *mc = NEW(Expr); mc->kind = EX_METHOD_CALL; mc->line = cur()->line;
                mc->name = cur()->name; next();
                mc->left = e; mc->args = parse_arglist();
                e = mc; continue;
            }
            Expr *fa = NEW(Expr); fa->kind = EX_FIELD; fa->line = cur()->line;
            fa->name = cur()->name; next();
            fa->left = e; e = fa; continue;
        }
        /* index: [expr] */
        if (tk()==TK_LBRACKET) {
            Expr *idx = NEW(Expr); idx->kind = EX_INDEX; idx->line = cur()->line;
            next(); idx->right = parse_expr(); expect(TK_RBRACKET);
            idx->left = e; e = idx; continue;
        }
        /* function call: name(args) */
        if (tk()==TK_LPAREN && e->kind==EX_IDENT) {
            Expr *call = NEW(Expr); call->kind = EX_CALL; call->line = e->line;
            call->name = e->name; call->args = parse_arglist();
            e = call; continue;
        }
        /* cast: expr as Type */
        if (tk()==TK_AS) {
            next();
            Expr *cast = NEW(Expr); cast->kind = EX_CAST; cast->line = cur()->line;
            cast->left = e; cast->cast_type = cur()->name; skip_type();
            e = cast; continue;
        }
        break;
    }
    return e;
}

static Expr *parse_binop(int min_prec) {
    Expr *left = parse_postfix(parse_primary());
    for (;;) {
        int p = prec_of(tk());
        if (p < min_prec) break;
        int op_tk = tk(); next();
        /* Parse right operand with higher precedence (left-associative) */
        Expr *right = parse_binop(p + 1);
        Expr *bin = NEW(Expr); bin->kind = EX_BINARY; bin->line = left->line;
        bin->op = tk_to_binop(op_tk); bin->left = left; bin->right = right;
        left = bin;
    }
    return left;
}

static Expr *parse_expr(void) { return parse_binop(0); }

static Block *parse_block(void) {
    Block *b = NEW(Block);
    expect(TK_LBRACE);
    Stmt **tail = &b->stmts;
    while (tk()!=TK_RBRACE && tk()!=TK_EOF) {
        Stmt *s = NEW(Stmt);
        if (tk()==TK_LET || tk()==TK_VAR) {
            s->kind = (tk()==TK_LET) ? 1 : 2; next();
            /* optional mut */
            eat(TK_MUT);
            s->name = cur()->name; next();
            if (eat(TK_COLON)) skip_type();
            expect(TK_ASSIGN);
            s->expr = parse_expr();
        } else {
            s->kind = 0;
            s->expr = parse_expr();
            /* check for assignment: expr = rhs */
            if (tk()==TK_ASSIGN || tk()==TK_PLUS_ASSIGN || tk()==TK_MINUS_ASSIGN) {
                int assign_op = tk(); next();
                Expr *rhs = parse_expr();
                if (assign_op != TK_ASSIGN) {
                    Expr *bin = NEW(Expr); bin->kind = EX_BINARY; bin->line = s->expr->line;
                    bin->op = (assign_op==TK_PLUS_ASSIGN)?OP_ADD:(assign_op==TK_MINUS_ASSIGN)?OP_SUB:OP_MUL;
                    bin->left = s->expr; bin->right = rhs;
                    rhs = bin;
                }
                s->kind = 3; s->right = rhs; /* use s->right for RHS, s->expr for LHS */
            }
        }
        eat(TK_SEMICOLON); /* optional */
        *tail = s; tail = &s->next;
    }
    expect(TK_RBRACE);
    return b;
}

/* Skip balanced braces — for error recovery */
static void skip_balanced_braces(void) {
    int depth = 1;
    next(); /* skip opening { */
    while (depth > 0 && tk() != TK_EOF) {
        if (tk() == TK_LBRACE) depth++;
        else if (tk() == TK_RBRACE) depth--;
        next();
    }
}

static FnDef *parse_fn(void) {
    FnDef *f = NEW(FnDef);
    expect(TK_FN);
    f->name = cur()->name; next();
    f->params = parse_params(&f->param_count);
    f->return_type = parse_return_type();
    if (tk() == TK_LBRACE) {
        f->body = parse_block();
    } else {
        fprintf(stderr, "stage0: SKIP fn %.*s (no brace at line %d, tk=%d)\n",
                f->name.len, f->name.buf, cur()->line, tk());
        /* Skip to next top-level item */
        while (tk() != TK_LBRACE && tk() != TK_FN && tk() != TK_STRUCT &&
               tk() != TK_ENUM && tk() != TK_IMPL && tk() != TK_EOF) next();
        if (tk() == TK_LBRACE) skip_balanced_braces();
        f->body = NEW(Block);
    }
    return f;
}

static StructDef *parse_struct(void) {
    StructDef *s = NEW(StructDef);
    expect(TK_STRUCT);
    s->name = cur()->name; next();
    expect(TK_LBRACE);
    FieldDef **tail = &s->fields;
    while (tk()!=TK_RBRACE && tk()!=TK_EOF) {
        FieldDef *f = NEW(FieldDef);
        f->name = cur()->name; next();
        expect(TK_COLON);
        /* Detect [i8; N] or [u8; N] byte array field type */
        if (tk()==TK_LBRACKET) {
            int save = g_tp;
            next(); /* [ */
            if (tk()==TK_IDENT && (name_is(cur()->name,"i8")||name_is(cur()->name,"u8"))) {
                f->is_byte_array = true;
            }
            g_tp = save;
        }
        /* Also detect &[i8; N] or &![i8; N] */
        if (tk()==TK_AMP) {
            int save = g_tp;
            next(); /* & */
            if (tk()==TK_BANG) next(); /* ! */
            if (tk()==TK_LBRACKET) {
                next(); /* [ */
                if (tk()==TK_IDENT && (name_is(cur()->name,"i8")||name_is(cur()->name,"u8"))) {
                    f->is_byte_array = true;
                }
            }
            g_tp = save;
        }
        f->type_name = cur()->name;
        skip_type();
        eat(TK_COMMA);
        *tail = f; tail = &f->next; s->field_count++;
    }
    expect(TK_RBRACE);
    return s;
}

/* Global variable registry — forward declarations (used by scan_globals) */
static Name g_global_names[MAX_GLOBALS];
static int g_global_offsets[MAX_GLOBALS];
static int g_global_is_array[MAX_GLOBALS];
static int g_global_array_size[MAX_GLOBALS];
static int g_global_count;

static int find_global(Name n) {
    for (int i = 0; i < g_global_count; i++)
        if (name_eq(g_global_names[i], n)) return i;
    return -1;
}

/* Prepass: scan for module-level var declarations and register globals */
static void scan_globals(void) {
    int saved_tp = g_tp;
    g_tp = 0;
    g_global_count = 0;
    int data_off = GLOBAL_FIXED_DATA;
    int brace_depth = 0;

    while (tk() != TK_EOF) {
        if (tk() == TK_LBRACE) { brace_depth++; next(); continue; }
        if (tk() == TK_RBRACE) { if (brace_depth > 0) brace_depth--; next(); continue; }
        if (tk() == TK_VAR && brace_depth == 0) {
            next(); /* eat var */
            if (tk() == TK_IDENT) {
                Name gname = cur()->name;
                next(); /* eat name */
                int is_arr = 0, elem_sz = 8, arr_cnt = 0;
                if (eat(TK_COLON)) {
                    if (tk() == TK_LBRACKET) {
                        next(); /* eat [ */
                        if (tk() == TK_IDENT) {
                            if (name_is(cur()->name, "i8") || name_is(cur()->name, "u8"))
                                elem_sz = 1;
                            else
                                elem_sz = 8;
                            next(); /* eat type */
                        }
                        if (eat(TK_SEMICOLON)) {
                            if (tk() == TK_INT) {
                                arr_cnt = (int)cur()->int_val;
                                next(); /* eat count */
                            }
                        }
                        eat(TK_RBRACKET); /* eat ] */
                        is_arr = 1;
                    }
                    /* Skip remaining type tokens and initializer */
                }
                if (g_global_count < MAX_GLOBALS) {
                    g_global_names[g_global_count] = gname;
                    g_global_offsets[g_global_count] = data_off;
                    if (is_arr) {
                        g_global_is_array[g_global_count] = (elem_sz == 1) ? 1 : 2;
                        g_global_array_size[g_global_count] = arr_cnt * elem_sz;
                    } else {
                        g_global_is_array[g_global_count] = 0;
                        g_global_array_size[g_global_count] = 0;
                    }
                    data_off += 8; /* 8 bytes per global (pointer or scalar) */
                    g_global_count++;
                    fprintf(stderr, "stage0: global=%.*s off=%d is_arr=%d sz=%d\n",
                            gname.len, gname.buf, g_global_offsets[g_global_count-1],
                            g_global_is_array[g_global_count-1],
                            g_global_array_size[g_global_count-1]);
                }
            }
            /* Skip to next top-level item */
            while (tk() != TK_EOF && tk() != TK_FN && tk() != TK_VAR &&
                   tk() != TK_STRUCT && tk() != TK_ENUM && tk() != TK_IMPL)
                next();
        } else {
            next();
        }
    }

    GLOBAL_DATA_SIZE = data_off;
    g_tp = saved_tp;
}

static Item *parse_program(void) {
    Item *head = NULL, **tail = &head;
    g_tp = 0;
    while (tk() != TK_EOF) {
        Item *item = NEW(Item);
        eat(TK_PUB); /* skip pub */
        if (tk()==TK_FN) { item->kind = ITEM_FN; item->fn = parse_fn(); }
        else if (tk()==TK_STRUCT) { item->kind = ITEM_STRUCT; item->st = parse_struct(); }
        else if (tk()==TK_ENUM) {
            item->kind = ITEM_ENUM; item->en = NEW(EnumDef);
            next(); item->en->name = cur()->name; next();
            expect(TK_LBRACE);
            EnumVariant **vtail = &item->en->variants;
            while (tk()!=TK_RBRACE && tk()!=TK_EOF) {
                EnumVariant *v = NEW(EnumVariant);
                v->name = cur()->name; next();
                if (eat(TK_LPAREN)) { skip_type(); while (eat(TK_COMMA)) skip_type(); expect(TK_RPAREN); }
                eat(TK_COMMA);
                *vtail = v; vtail = &v->next;
            }
            expect(TK_RBRACE);
        }
        else if (tk()==TK_IMPL) {
            item->kind = ITEM_IMPL; item->im = NEW(ImplDef);
            next(); item->im->type_name = cur()->name; next();
            expect(TK_LBRACE);
            Method **mtail = &item->im->methods;
            while (tk()!=TK_RBRACE && tk()!=TK_EOF) {
                eat(TK_PUB);
                if (tk()==TK_FN) {
                    Method *m = NEW(Method); m->fn = parse_fn();
                    *mtail = m; mtail = &m->next;
                } else { next(); }
            }
            expect(TK_RBRACE);
        }
        else if (tk()==TK_MODULE) { item->kind = ITEM_MODULE; next(); while (tk()!=TK_EOF && tk()!=TK_FN && tk()!=TK_STRUCT && tk()!=TK_ENUM && tk()!=TK_IMPL && tk()!=TK_USE && tk()!=TK_PUB && tk()!=TK_TYPE) next(); continue; }
        else if (tk()==TK_USE) { item->kind = ITEM_USE; next(); while (tk()!=TK_EOF && tk()!=TK_FN && tk()!=TK_STRUCT && tk()!=TK_ENUM && tk()!=TK_IMPL && tk()!=TK_USE && tk()!=TK_PUB && tk()!=TK_TYPE) next(); continue; }
        else if (tk()==TK_TYPE) { item->kind = ITEM_TYPE; while (tk()!=TK_LBRACE && tk()!=TK_EOF) next(); if (tk()==TK_LBRACE) { int depth=1; next(); while(depth>0&&tk()!=TK_EOF){if(tk()==TK_LBRACE)depth++;if(tk()==TK_RBRACE)depth--;next();} } continue; }
        else { next(); continue; }
        *tail = item; tail = &item->next;
    }
    return head;
}

/* ================================================================
 * CODE EMITTER
 * ================================================================ */
static uint8_t g_code[MAX_CODE];
static int g_code_len;

static void emit(uint8_t b) { if (g_code_len < MAX_CODE) g_code[g_code_len++] = b; }
static void emit32(int32_t v) { emit(v&0xFF); emit((v>>8)&0xFF); emit((v>>16)&0xFF); emit((v>>24)&0xFF); }
static void emit64(int64_t v) { for(int i=0;i<8;i++) emit((v>>(i*8))&0xFF); }

static void patch32(int off, int32_t v) {
    g_code[off] = v&0xFF; g_code[off+1] = (v>>8)&0xFF;
    g_code[off+2] = (v>>16)&0xFF; g_code[off+3] = (v>>24)&0xFF;
}

/* x86-64 helpers */
static void emit_push_rbp(void) { emit(0x55); }
static void emit_pop_rbp(void) { emit(0x5D); }
static void emit_mov_rbp_rsp(void) { emit(0x48); emit(0x89); emit(0xE5); }
static void emit_leave(void) { emit(0xC9); }
static void emit_ret(void) { emit(0xC3); }
static void emit_push_rax(void) { emit(0x50); }
static void emit_pop_rcx(void) { emit(0x59); }

static void emit_sub_rsp(int32_t n) {
    if (n == 0) return;
    emit(0x48); emit(0x81); emit(0xEC); emit32(n);
}

/* store rax to [rbp + disp32] */
static void emit_store_rax(int vreg) {
    int32_t disp = -(vreg+1)*8;
    emit(0x48); emit(0x89); emit(0x85); emit32(disp);
}

/* load [rbp + disp32] to rax */
static void emit_load_rax(int vreg) {
    int32_t disp = -(vreg+1)*8;
    emit(0x48); emit(0x8B); emit(0x85); emit32(disp);
}

/* load [rbp + disp32] to rcx */
static void emit_load_rcx(int vreg) {
    int32_t disp = -(vreg+1)*8;
    emit(0x48); emit(0x8B); emit(0x8D); emit32(disp);
}

/* mov rax, imm64 */
static void emit_mov_rax_imm64(int64_t v) {
    emit(0x48); emit(0xB8); emit64(v);
}

/* ================================================================
 * STRUCT & FUNCTION REGISTRY
 * ================================================================ */
static Name g_struct_names[MAX_STRUCTS];
static int g_struct_field_count[MAX_STRUCTS];
static Name g_struct_field_names[MAX_FIELDS];
static bool g_struct_field_is_byte[MAX_FIELDS]; /* true if field is [i8;N] or [u8;N] */
static int g_struct_field_base[MAX_STRUCTS]; /* index into g_struct_field_names */
static int g_struct_count;

static Name g_fn_names[MAX_FUNCS];
static int g_fn_offsets[MAX_FUNCS]; /* code offset */
static int g_fn_ret_type[MAX_FUNCS]; /* 0=i64,1=f64,10+=struct */
static int g_fn_count;
static int g_main_fn_idx = -1;

/* Enum variant lookup */
#define MAX_ENUMS 256
#define MAX_ENUM_VARIANTS 8192
static Name g_enum_names[MAX_ENUMS];
static int g_enum_variant_base[MAX_ENUMS]; /* index into variant arrays */
static int g_enum_variant_count[MAX_ENUMS];
static Name g_enum_variant_names[MAX_ENUM_VARIANTS];
static int g_enum_count;

static int find_struct(Name n) {
    for (int i = 0; i < g_struct_count; i++)
        if (name_eq(g_struct_names[i], n)) return i;
    return -1;
}

/* Look up Enum::Variant and return variant index, or -1 if not found */
static int find_enum_variant(Name enum_name, Name variant_name) {
    for (int i = 0; i < g_enum_count; i++) {
        if (name_eq(g_enum_names[i], enum_name)) {
            int base = g_enum_variant_base[i];
            for (int j = 0; j < g_enum_variant_count[i]; j++) {
                if (name_eq(g_enum_variant_names[base + j], variant_name))
                    return j;
            }
            return -1;
        }
    }
    return -1;
}

/* Look up a variant name across all enums */
static int find_variant_any_enum(Name variant_name) {
    for (int i = 0; i < g_enum_count; i++) {
        int base = g_enum_variant_base[i];
        for (int j = 0; j < g_enum_variant_count[i]; j++) {
            if (name_eq(g_enum_variant_names[base + j], variant_name))
                return j;
        }
    }
    return -1;
}

static int find_fn(Name n) {
    for (int i = 0; i < g_fn_count; i++)
        if (name_eq(g_fn_names[i], n)) return i;
    return -1;
}

/* ================================================================
 * LOCAL VARIABLE TRACKING
 * ================================================================ */
typedef struct {
    Name names[MAX_LOCALS];
    int slots[MAX_LOCALS];
    int types[MAX_LOCALS]; /* 0=i64,1=f64,2=array,10+=struct */
    int count;
    int next_slot;
} Locals;

/* ================================================================
 * EMIT PRINT_INT BUILTIN
 * ================================================================ */
static void emit_print_int_builtin(void) {
    /*
     * print_int(n: i64) — itoa + sys_write to stdout
     * Proven working machine code from test harness.
     * Handles positive integers; zero prints nothing (simplified).
     */
    static const uint8_t pi_code[] = {
        0x55, 0x48, 0x89, 0xe5, 0x48, 0x81, 0xec, 0x30, 0x00, 0x00, 0x00,
        /* mov rax, rdi */
        0x48, 0x89, 0xf8,
        /* r11 = sign */
        0x4d, 0x31, 0xdb,
        /* test rax,rax; jns pos (skip neg+mov_r11 = 3+7 = 10 bytes) */
        0x48, 0x85, 0xc0,
        0x79, 0x0a,
        /* neg rax; mov r11,1 */
        0x48, 0xf7, 0xd8,
        0x49, 0xc7, 0xc3, 0x01, 0x00, 0x00, 0x00,
        /* lea r10, [rbp-9] */
        0x4c, 0x8d, 0x55, 0xf7,
        /* test rax,rax; jnz digit_loop */
        0x48, 0x85, 0xc0,
        0x75, 0x09,
        /* zero: dec r10; mov byte [r10], '0'; jmp after */
        0x49, 0xff, 0xca,
        0x41, 0xc6, 0x02, 0x30,
        0xeb, 0x1a,
        /* digit_loop: */
        0x31, 0xd2,                         /* xor edx,edx */
        0x48, 0xc7, 0xc1, 0x0a, 0x00, 0x00, 0x00, /* mov rcx,10 */
        0x48, 0xf7, 0xf1,                   /* div rcx */
        0x80, 0xc2, 0x30,                   /* add dl,'0' */
        0x49, 0xff, 0xca,                   /* dec r10 */
        0x41, 0x88, 0x12,                   /* mov [r10],dl */
        0x48, 0x85, 0xc0,                   /* test rax,rax */
        0x75, 0xe6,                         /* jnz digit_loop */
        /* after_loop: check sign */
        0x4d, 0x85, 0xdb,                   /* test r11,r11 */
        0x74, 0x07,                         /* je skip */
        0x49, 0xff, 0xca,                   /* dec r10 */
        0x41, 0xc6, 0x02, 0x2d,             /* mov byte [r10], '-' */
        /* skip: sys_write */
        0x4c, 0x89, 0xd6,                   /* mov rsi, r10 */
        0x48, 0x8d, 0x55, 0xf7,             /* lea rdx, [rbp-9] */
        0x4c, 0x29, 0xd2,                   /* sub rdx, r10 */
        0xbf, 0x01, 0x00, 0x00, 0x00,       /* mov edi, 1 */
        0xb8, 0x01, 0x00, 0x00, 0x00,       /* mov eax, 1 */
        0x0f, 0x05,                         /* syscall */
        0xc9, 0xc3,                         /* leave; ret */
    };
    for (size_t i = 0; i < sizeof(pi_code); i++) emit(pi_code[i]);
}

/* ================================================================
 * EMIT PRINT_F64 BUILTIN
 * ================================================================ */
static void emit_print_f64_builtin(void) {
    /*
     * print_f64(x: f64) — outputs sign + integer + '.' + 6 fractional digits + '\n'
     * f64 arrives as bit pattern in rdi (Sounio passes f64 as i64).
     * Uses: movq xmm0, rdi to recover f64; cvttsd2si for integer extraction;
     *       multiply fractional by 1e6 for 6-digit itoa. No libc, bare sys_write.
     * Proven working: 1.5→"1.500000", sqrt(2)→"1.414213" (native ELF gate).
     */
    static const uint8_t pf_code[] = {
        /* prologue */
        0x55, 0x48, 0x89, 0xe5,                         /* push rbp; mov rbp, rsp */
        0x48, 0x83, 0xec, 0x50,                         /* sub rsp, 80 */
        /* movq xmm0, rdi — recover f64 from integer register */
        0x66, 0x48, 0x0f, 0x6e, 0xc7,
        /* movq rax, xmm0 — get bits for sign check */
        0x66, 0x48, 0x0f, 0x7e, 0xc0,
        /* test rax, rax — check sign bit */
        0x48, 0x85, 0xc0,
        /* jns skip_neg (patched below) */
        0x79, 0x00, /* placeholder */
    };
    for (size_t i = 0; i < sizeof(pf_code); i++) emit(pf_code[i]);
    int jns_patch = g_code_len - 1; /* location of jns displacement byte */

    /* Negative path: print '-', clear sign bit, reload xmm0 */
    static const uint8_t neg_path[] = {
        0xc6, 0x45, 0xff, 0x2d,                         /* mov byte [rbp-1], '-' */
        0x48, 0x8d, 0x75, 0xff,                         /* lea rsi, [rbp-1] */
        0xba, 0x01, 0x00, 0x00, 0x00,                   /* mov edx, 1 */
        0xbf, 0x01, 0x00, 0x00, 0x00,                   /* mov edi, 1 */
        0xb8, 0x01, 0x00, 0x00, 0x00,                   /* mov eax, 1 */
        0x0f, 0x05,                                     /* syscall (write '-') */
        /* and rax, 0x7FFFFFFFFFFFFFFF — clear sign bit */
        0x48, 0x25, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0x7f,
        /* movq xmm0, rax — reload absolute value */
        0x66, 0x48, 0x0f, 0x6e, 0xc0,
    };
    for (size_t i = 0; i < sizeof(neg_path); i++) emit(neg_path[i]);

    /* Patch jns displacement */
    g_code[jns_patch] = (uint8_t)(g_code_len - (jns_patch + 1));

    /* Integer part: cvttsd2si rax, xmm0 */
    emit(0xf2); emit(0x48); emit(0x0f); emit(0x2c); emit(0xc0);
    /* Save xmm0 to [rbp-48] for fractional computation */
    emit(0xf2); emit(0x0f); emit(0x11); emit(0x45); emit(0xd0);
    /* Save integer part to [rbp-56] */
    emit(0x48); emit(0x89); emit(0x45); emit(0xc8);

    /* itoa loop (right-to-left into [rbp-19..rbp-10]) */
    emit(0x4c); emit(0x8d); emit(0x55); emit(0xf7); /* lea r10, [rbp-9] */
    emit(0x48); emit(0x85); emit(0xc0);             /* test rax, rax */
    emit(0x75); emit(0x09);                          /* jnz digit_loop */
    /* zero: dec r10; mov byte [r10], '0'; jmp do_write */
    emit(0x49); emit(0xff); emit(0xca);
    emit(0x41); emit(0xc6); emit(0x02); emit(0x30);
    int jz_int = g_code_len;
    emit(0xeb); emit(0x00);

    int dl_int = g_code_len;
    emit(0x31); emit(0xd2);                          /* xor edx, edx */
    emit(0x48); emit(0xc7); emit(0xc1);             /* mov rcx, 10 */
    emit(0x0a); emit(0x00); emit(0x00); emit(0x00);
    emit(0x48); emit(0xf7); emit(0xf1);             /* div rcx */
    emit(0x80); emit(0xc2); emit(0x30);             /* add dl, '0' */
    emit(0x49); emit(0xff); emit(0xca);             /* dec r10 */
    emit(0x41); emit(0x88); emit(0x12);             /* mov [r10], dl */
    emit(0x48); emit(0x85); emit(0xc0);             /* test rax, rax */
    { int8_t b = (int8_t)(dl_int - (g_code_len + 2));
      emit(0x75); emit((uint8_t)b); }

    g_code[jz_int + 1] = (uint8_t)(g_code_len - (jz_int + 2));

    /* sys_write(1, r10, (rbp-9)-r10) */
    emit(0x4c); emit(0x89); emit(0xd6);             /* mov rsi, r10 */
    emit(0x48); emit(0x8d); emit(0x55); emit(0xf7); /* lea rdx, [rbp-9] */
    emit(0x4c); emit(0x29); emit(0xd2);             /* sub rdx, r10 */
    emit(0xbf); emit32(1); emit(0xb8); emit32(1);
    emit(0x0f); emit(0x05);

    /* Print decimal point */
    emit(0xc6); emit(0x45); emit(0xfe); emit(0x2e);
    emit(0x48); emit(0x8d); emit(0x75); emit(0xfe);
    emit(0xba); emit32(1); emit(0xbf); emit32(1);
    emit(0xb8); emit32(1); emit(0x0f); emit(0x05);

    /* Restore for fractional part */
    emit(0x48); emit(0x8b); emit(0x45); emit(0xc8);             /* mov rax, [rbp-56] */
    emit(0xf2); emit(0x0f); emit(0x10); emit(0x45); emit(0xd0); /* movsd xmm0, [rbp-48] */
    emit(0xf2); emit(0x48); emit(0x0f); emit(0x2a); emit(0xc8); /* cvtsi2sd xmm1, rax */
    emit(0xf2); emit(0x0f); emit(0x5c); emit(0xc1);             /* subsd xmm0, xmm1 */
    /* mov rax, 1e6 (IEEE 754 LE) */
    emit(0x48); emit(0xb8);
    emit(0x00); emit(0x00); emit(0x00); emit(0x00);
    emit(0x80); emit(0x84); emit(0x2e); emit(0x41);
    emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xd0); /* movq xmm2, rax */
    emit(0xf2); emit(0x0f); emit(0x59); emit(0xc2);             /* mulsd xmm0, xmm2 */
    emit(0xf2); emit(0x48); emit(0x0f); emit(0x2c); emit(0xc8); /* cvttsd2si rcx, xmm0 */

    /* Extract 6 fractional digits from rcx via itoa (unrolled) */
    static const uint8_t frac_digits[] = {
        /* digit 0: rcx / 100000 */
        0x48, 0x89, 0xc8,                               /* mov rax, rcx */
        0x31, 0xd2,                                     /* xor edx, edx */
        0xb9, 0xa0, 0x86, 0x01, 0x00,                   /* mov ecx, 100000 */
        0x48, 0xf7, 0xf1,                               /* div rcx */
        0x04, 0x30,                                     /* add al, '0' */
        0x88, 0x45, 0xf7,                               /* mov [rbp-9], al */
        /* digit 1: rdx / 10000 */
        0x48, 0x89, 0xd0,                               /* mov rax, rdx */
        0x31, 0xd2,
        0xb9, 0x10, 0x27, 0x00, 0x00,                   /* mov ecx, 10000 */
        0x48, 0xf7, 0xf1,
        0x04, 0x30,
        0x88, 0x45, 0xf8,                               /* mov [rbp-8], al */
        /* digit 2: rdx / 1000 */
        0x48, 0x89, 0xd0,
        0x31, 0xd2,
        0xb9, 0xe8, 0x03, 0x00, 0x00,                   /* mov ecx, 1000 */
        0x48, 0xf7, 0xf1,
        0x04, 0x30,
        0x88, 0x45, 0xf9,                               /* mov [rbp-7], al */
        /* digit 3: rdx / 100 */
        0x48, 0x89, 0xd0,
        0x31, 0xd2,
        0xb9, 0x64, 0x00, 0x00, 0x00,                   /* mov ecx, 100 */
        0x48, 0xf7, 0xf1,
        0x04, 0x30,
        0x88, 0x45, 0xfa,                               /* mov [rbp-6], al */
        /* digit 4: rdx / 10 */
        0x48, 0x89, 0xd0,
        0x31, 0xd2,
        0xb9, 0x0a, 0x00, 0x00, 0x00,                   /* mov ecx, 10 */
        0x48, 0xf7, 0xf1,
        0x04, 0x30,
        0x88, 0x45, 0xfb,                               /* mov [rbp-5], al */
        /* digit 5: rdx is last digit */
        0x80, 0xc2, 0x30,                               /* add dl, '0' */
        0x88, 0x55, 0xfc,                               /* mov [rbp-4], dl */
        /* sys_write(1, [rbp-9], 6) */
        0x48, 0x8d, 0x75, 0xf7,                         /* lea rsi, [rbp-9] */
        0xba, 0x06, 0x00, 0x00, 0x00,                   /* mov edx, 6 */
        0xbf, 0x01, 0x00, 0x00, 0x00,
        0xb8, 0x01, 0x00, 0x00, 0x00,
        0x0f, 0x05,
        /* Print newline */
        0xc6, 0x45, 0xfd, 0x0a,                         /* mov byte [rbp-3], '\n' */
        0x48, 0x8d, 0x75, 0xfd,                         /* lea rsi, [rbp-3] */
        0xba, 0x01, 0x00, 0x00, 0x00,
        0xbf, 0x01, 0x00, 0x00, 0x00,
        0xb8, 0x01, 0x00, 0x00, 0x00,
        0x0f, 0x05,
        /* epilogue */
        0xc9, 0xc3,                                     /* leave; ret */
    };
    for (size_t i = 0; i < sizeof(frac_digits); i++) emit(frac_digits[i]);
}

/* ================================================================
 * EMIT INLINE STRING PRINT
 * ================================================================ */
static void emit_inline_print(const char *str, int len) {
    /* jmp over string */
    emit(0xEB); emit((uint8_t)len); /* JMP rel8 */
    int str_start = g_code_len;
    for (int i = 0; i < len; i++) emit((uint8_t)str[i]);
    /* lea rsi, [rip - offset] */
    int32_t rip_off = -(g_code_len + 7 - str_start);
    emit(0x48); emit(0x8D); emit(0x35); emit32(rip_off);
    /* mov edx, len */
    emit(0xBA); emit32(len);
    /* mov edi, 1 */
    emit(0xBF); emit32(1);
    /* mov eax, 1 */
    emit(0xB8); emit32(1);
    /* syscall */
    emit(0x0F); emit(0x05);
}

/* ================================================================
 * AST-DIRECT CODE GENERATION
 * ================================================================ */
static void compile_expr(Expr *e, Locals *loc);
static void compile_block(Block *b, Locals *loc);

/* Check if an index base expression should use byte (1-byte) indexing.
 * Heuristics:
 *  1. Base is a local variable with type==3 (byte array)
 *  2. Base is a field access on a struct field that is [i8;N] or [u8;N]
 *  3. Index expression is an `as usize` cast (common pattern in bootstrap_v0)
 */
static int is_byte_index(Expr *base, Expr *index, Locals *loc) {
    /* Check local variable type */
    if (base && base->kind == EX_IDENT) {
        for (int i = loc->count-1; i >= 0; i--) {
            if (name_eq(loc->names[i], base->name)) {
                if (loc->types[i] == 3) return 1;
                break;
            }
        }
    }
    /* Check struct field: base is EX_FIELD with a known byte-array field */
    if (base && base->kind == EX_FIELD) {
        /* Search all structs for this field name and check is_byte */
        for (int si = 0; si < g_struct_count; si++) {
            int fb = g_struct_field_base[si];
            for (int fi = 0; fi < g_struct_field_count[si]; fi++) {
                if (name_eq(g_struct_field_names[fb + fi], base->name)) {
                    if (g_struct_field_is_byte[fb + fi]) return 1;
                }
            }
        }
    }
    /* Check deref-then-field: base is EX_FIELD on EX_DEREF */
    if (base && base->kind == EX_FIELD && base->left && base->left->kind == EX_DEREF) {
        /* Already handled above by field name check */
    }
    /* Check global variable type */
    if (base && base->kind == EX_IDENT) {
        int gi = find_global(base->name);
        if (gi >= 0) {
            return (g_global_is_array[gi] == 1) ? 1 : 0; /* 1=i8, 2=i64 */
        }
    }
    /* Heuristic: if index is `expr as usize`, assume byte indexing (only if no global match) */
    if (index && index->kind == EX_CAST && name_is(index->cast_type, "usize")) {
        return 1;
    }
    return 0;
}

/* Loop context for break */
static int g_loop_top[16];
static int g_break_patches[16][MAX_BREAKS];
static int g_break_count[16];
static int g_loop_depth = -1;

/* Determine if an expression produces an f64 value */
static int expr_is_float(Expr *e, Locals *loc) {
    if (!e) return 0;
    if (e->kind == EX_FLOAT) return 1;
    if (e->kind == EX_IDENT) {
        for (int i = loc->count-1; i >= 0; i--) {
            if (name_eq(loc->names[i], e->name)) return loc->types[i] == 1;
        }
    }
    if (e->kind == EX_BINARY) return expr_is_float(e->left, loc) || expr_is_float(e->right, loc);
    if (e->kind == EX_UNARY) return expr_is_float(e->left, loc);
    if (e->kind == EX_CALL) {
        if (name_is(e->name, "sqrt") || name_is(e->name, "sin") ||
            name_is(e->name, "cos") || name_is(e->name, "exp") ||
            name_is(e->name, "log") || name_is(e->name, "fabs")) return 1;
        /* Check user function return type */
        int fi = find_fn(e->name);
        if (fi >= 0 && g_fn_ret_type[fi] == 1) return 1;
    }
    if (e->kind == EX_CAST && name_is(e->cast_type, "f64")) return 1;
    if (e->kind == EX_CAST && name_is(e->cast_type, "i64")) return 0;
    if (e->kind == EX_IF) return expr_is_float(e->left, loc);
    return 0;
}

/* Emit SSE2 binary op: rax=left, rcx=right → result in rax */
static void emit_f64_binop(int op) {
    /* movq xmm0, rax */
    emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc0);
    /* movq xmm1, rcx */
    emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc9);
    switch (op) {
    case OP_ADD: emit(0xf2); emit(0x0f); emit(0x58); emit(0xc1); break; /* addsd xmm0, xmm1 */
    case OP_SUB: emit(0xf2); emit(0x0f); emit(0x5c); emit(0xc1); break; /* subsd xmm0, xmm1 */
    case OP_MUL: emit(0xf2); emit(0x0f); emit(0x59); emit(0xc1); break; /* mulsd xmm0, xmm1 */
    case OP_DIV: emit(0xf2); emit(0x0f); emit(0x5e); emit(0xc1); break; /* divsd xmm0, xmm1 */
    }
    /* movq rax, xmm0 */
    emit(0x66); emit(0x48); emit(0x0f); emit(0x7e); emit(0xc0);
}

/* Emit SSE2 comparison: rax=left, rcx=right → result (0 or 1) in rax */
static void emit_f64_cmp(int op) {
    emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc0); /* movq xmm0, rax */
    emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc9); /* movq xmm1, rcx */
    emit(0x66); emit(0x0f); emit(0x2f); emit(0xc1);             /* comisd xmm0, xmm1 */
    emit(0xb8); emit(0x00); emit(0x00); emit(0x00); emit(0x00); /* mov eax, 0 (preserves flags) */
    switch (op) {
    case OP_EQ: emit(0x0f); emit(0x94); emit(0xc0); break;     /* sete al */
    case OP_NE: emit(0x0f); emit(0x95); emit(0xc0); break;     /* setne al */
    case OP_LT: emit(0x0f); emit(0x92); emit(0xc0); break;     /* setb al (below for unsigned/float) */
    case OP_LE: emit(0x0f); emit(0x96); emit(0xc0); break;     /* setbe al */
    case OP_GT: emit(0x0f); emit(0x97); emit(0xc0); break;     /* seta al */
    case OP_GE: emit(0x0f); emit(0x93); emit(0xc0); break;     /* setae al */
    }
}

static void compile_expr(Expr *e, Locals *loc) {
    if (!e) return;
    switch (e->kind) {
    case EX_INT:
        emit_mov_rax_imm64(e->int_val);
        break;
    case EX_BOOL:
        emit_mov_rax_imm64(e->bool_val ? 1 : 0);
        break;
    case EX_FLOAT: {
        union { double d; int64_t i; } u; u.d = e->float_val;
        emit_mov_rax_imm64(u.i);
        break;
    }
    case EX_STRING:
        emit_inline_print(e->str_buf, e->str_len);
        emit_mov_rax_imm64(0);
        break;
    case EX_IDENT: {
        for (int i = loc->count-1; i >= 0; i--) {
            if (name_eq(loc->names[i], e->name)) {
                emit_load_rax(loc->slots[i]);
                return;
            }
        }
        /* Check globals */
        { int gi = find_global(e->name);
          if (gi >= 0) {
            /* Load from data section via RIP-relative */
            emit(0x48); emit(0x8B); emit(0x05); /* mov rax, [rip+disp32] */
            emit32(g_global_offsets[gi] - (g_code_len + 4));
            return;
          }
        }
        /* Check enum variant: Path::Variant stored as cast_type=Path, name=Variant */
        if (e->cast_type.len > 0) {
            int val = find_enum_variant(e->cast_type, e->name);
            if (val >= 0) { emit_mov_rax_imm64(val); break; }
        }
        /* Try variant name across all enums (no path prefix) */
        { int val = find_variant_any_enum(e->name);
          if (val >= 0) { emit_mov_rax_imm64(val); break; }
        }
        /* might be a function name — emit 0 */
        emit_mov_rax_imm64(0);
        break;
    }
    case EX_BINARY: {
        if (e->op == OP_AND) {
            /* short-circuit && */
            compile_expr(e->left, loc);
            emit(0x48); emit(0x85); emit(0xC0); /* test rax,rax */
            emit(0x0F); emit(0x84); int patch = g_code_len; emit32(0); /* JZ */
            compile_expr(e->right, loc);
            emit(0xEB); emit(0x05); /* JMP +5 */
            patch32(patch, g_code_len - (patch+4));
            emit(0x48); emit(0x31); emit(0xC0); /* XOR rax,rax */
            emit(0x90); emit(0x90); /* pad to 5 bytes */
            break;
        }
        if (e->op == OP_OR) {
            /* short-circuit || */
            compile_expr(e->left, loc);
            emit(0x48); emit(0x85); emit(0xC0); /* test rax,rax */
            emit(0x0F); emit(0x85); int patch = g_code_len; emit32(0); /* JNZ */
            compile_expr(e->right, loc);
            emit(0xEB); emit(0x0A); /* JMP +10 */
            patch32(patch, g_code_len - (patch+4));
            emit_mov_rax_imm64(1);
            break;
        }
        {
        int is_float = expr_is_float(e->left, loc) || expr_is_float(e->right, loc);
        compile_expr(e->left, loc);
        emit_push_rax();
        compile_expr(e->right, loc);
        emit(0x48); emit(0x89); emit(0xC1); /* mov rcx, rax */
        emit(0x58); /* pop rax (left) */
        /* now rax=left, rcx=right */
        if (is_float && (e->op == OP_ADD || e->op == OP_SUB || e->op == OP_MUL || e->op == OP_DIV)) {
            emit_f64_binop(e->op);
        } else if (is_float && (e->op >= OP_EQ && e->op <= OP_GE)) {
            emit_f64_cmp(e->op);
        } else
        switch (e->op) {
        case OP_ADD: emit(0x48); emit(0x01); emit(0xC8); break; /* add rax,rcx */
        case OP_SUB: emit(0x48); emit(0x29); emit(0xC8); break; /* sub rax,rcx */
        case OP_MUL: emit(0x48); emit(0x0F); emit(0xAF); emit(0xC1); break; /* imul rax,rcx */
        case OP_DIV: emit(0x48); emit(0x99); emit(0x48); emit(0xF7); emit(0xF9); break; /* cqo; idiv rcx */
        case OP_REM: emit(0x48); emit(0x99); emit(0x48); emit(0xF7); emit(0xF9); /* cqo; idiv rcx */
                     emit(0x48); emit(0x89); emit(0xD0); break; /* mov rax,rdx */
        case OP_BIT_AND: emit(0x48); emit(0x21); emit(0xC8); break;
        case OP_BIT_OR:  emit(0x48); emit(0x09); emit(0xC8); break;
        case OP_BIT_XOR: emit(0x48); emit(0x31); emit(0xC8); break;
        case OP_SHL: emit(0x48); emit(0xD3); emit(0xE0); break; /* shl rax,cl */
        case OP_SHR: emit(0x48); emit(0xD3); emit(0xF8); break; /* sar rax,cl */
        case OP_EQ: emit(0x48); emit(0x39); emit(0xC8); emit(0x0F); emit(0x94); emit(0xC0);
                    emit(0x48); emit(0x0F); emit(0xB6); emit(0xC0); break;
        case OP_NE: emit(0x48); emit(0x39); emit(0xC8); emit(0x0F); emit(0x95); emit(0xC0);
                    emit(0x48); emit(0x0F); emit(0xB6); emit(0xC0); break;
        case OP_LT: emit(0x48); emit(0x39); emit(0xC8); emit(0x0F); emit(0x9C); emit(0xC0);
                    emit(0x48); emit(0x0F); emit(0xB6); emit(0xC0); break;
        case OP_LE: emit(0x48); emit(0x39); emit(0xC8); emit(0x0F); emit(0x9E); emit(0xC0);
                    emit(0x48); emit(0x0F); emit(0xB6); emit(0xC0); break;
        case OP_GT: emit(0x48); emit(0x39); emit(0xC8); emit(0x0F); emit(0x9F); emit(0xC0);
                    emit(0x48); emit(0x0F); emit(0xB6); emit(0xC0); break;
        case OP_GE: emit(0x48); emit(0x39); emit(0xC8); emit(0x0F); emit(0x9D); emit(0xC0);
                    emit(0x48); emit(0x0F); emit(0xB6); emit(0xC0); break;
        }
        }
        break;
    }
    case EX_UNARY:
        compile_expr(e->left, loc);
        if (e->op == 0) { emit(0x48); emit(0xF7); emit(0xD8); } /* neg rax */
        if (e->op == 1) { emit(0x48); emit(0x83); emit(0xF0); emit(0x01); } /* xor rax,1 */
        break;
    case EX_CALL: {
        /* Special case: print("literal") */
        if (name_is(e->name, "print") && e->args && e->args->expr->kind == EX_STRING) {
            emit_inline_print(e->args->expr->str_buf, e->args->expr->str_len);
            emit_mov_rax_imm64(0);
            break;
        }
        /* General call */
        int fn_idx = find_fn(e->name);
        /* Evaluate args */
        int arg_slots[MAX_CALL_ARGS]; int argc = 0;
        for (ExprList *a = e->args; a && argc < MAX_CALL_ARGS; a = a->next) {
            compile_expr(a->expr, loc);
            int slot = loc->next_slot++;
            emit_store_rax(slot);
            arg_slots[argc++] = slot;
        }
        /* Load args into registers */
        for (int i = argc-1; i >= 0; i--) {
            emit_load_rax(arg_slots[i]);
            switch (i) {
                case 0: emit(0x48); emit(0x89); emit(0xC7); break; /* mov rdi,rax */
                case 1: emit(0x48); emit(0x89); emit(0xC6); break; /* mov rsi,rax */
                case 2: emit(0x48); emit(0x89); emit(0xC2); break; /* mov rdx,rax */
                case 3: emit(0x48); emit(0x89); emit(0xC1); break; /* mov rcx,rax */
                case 4: emit(0x49); emit(0x89); emit(0xC0); break; /* mov r8,rax */
                case 5: emit(0x49); emit(0x89); emit(0xC1); break; /* mov r9,rax */
                default: emit_push_rax(); break;
            }
        }
        /* call rel32 with CAFE magic */
        if (fn_idx >= 0) {
            emit(0xE8); emit(fn_idx & 0xFF); emit((fn_idx >> 8) & 0xFF);
            emit(0xCA); emit(0xFE);
        } else {
            /* unknown function — emit nop call */
            emit(0xE8); emit32(0);
        }
        /* Clean up stack args */
        if (argc > 6) {
            int stack_size = (argc - 6) * 8;
            emit(0x48); emit(0x81); emit(0xC4); emit32(stack_size); /* add rsp, N */
        }
        loc->next_slot -= argc;
        break;
    }
    case EX_METHOD_CALL: {
        /* receiver.method(args) → method(receiver, args) */
        int fn_idx = find_fn(e->name);
        /* Evaluate receiver */
        compile_expr(e->left, loc);
        int recv_slot = loc->next_slot++;
        emit_store_rax(recv_slot);
        /* Evaluate args */
        int arg_slots[MAX_CALL_ARGS]; int argc = 0;
        arg_slots[argc++] = recv_slot;
        for (ExprList *a = e->args; a && argc < MAX_CALL_ARGS; a = a->next) {
            compile_expr(a->expr, loc);
            int slot = loc->next_slot++;
            emit_store_rax(slot);
            arg_slots[argc++] = slot;
        }
        /* Load args */
        for (int i = argc-1; i >= 0; i--) {
            emit_load_rax(arg_slots[i]);
            switch (i) {
                case 0: emit(0x48); emit(0x89); emit(0xC7); break;
                case 1: emit(0x48); emit(0x89); emit(0xC6); break;
                case 2: emit(0x48); emit(0x89); emit(0xC2); break;
                case 3: emit(0x48); emit(0x89); emit(0xC1); break;
                case 4: emit(0x49); emit(0x89); emit(0xC0); break;
                case 5: emit(0x49); emit(0x89); emit(0xC1); break;
                default: emit_push_rax(); break;
            }
        }
        if (fn_idx >= 0) {
            emit(0xE8); emit(fn_idx & 0xFF); emit((fn_idx >> 8) & 0xFF);
            emit(0xCA); emit(0xFE);
        } else { emit(0xE8); emit32(0); }
        if (argc > 6) { emit(0x48); emit(0x81); emit(0xC4); emit32((argc-6)*8); }
        loc->next_slot -= argc;
        break;
    }
    case EX_FIELD: {
        compile_expr(e->left, loc);
        /* rax = address of struct on stack; load field at offset */
        /* Find struct type from left expr to get field index */
        int field_idx = -1;
        /* Try to find the struct type from the left expression's variable */
        int struct_idx = -1;
        if (e->left && e->left->kind == EX_IDENT) {
            for (int i = loc->count-1; i >= 0; i--) {
                if (name_eq(loc->names[i], e->left->name) && loc->types[i] >= 10) {
                    struct_idx = loc->types[i] - 10;
                    break;
                }
            }
        }
        if (struct_idx >= 0) {
            int base = g_struct_field_base[struct_idx];
            for (int i = 0; i < g_struct_field_count[struct_idx]; i++) {
                if (name_eq(g_struct_field_names[base + i], e->name)) {
                    field_idx = i; break;
                }
            }
        }
        if (field_idx < 0) {
            /* Fallback: search all structs for this field name */
            for (int si = 0; si < g_struct_count && field_idx < 0; si++) {
                int base = g_struct_field_base[si];
                for (int fi = 0; fi < g_struct_field_count[si]; fi++) {
                    if (name_eq(g_struct_field_names[base + fi], e->name)) {
                        field_idx = fi; break;
                    }
                }
            }
        }
        if (field_idx >= 0) {
            /* rax = base address of struct; field at [rax - 8*field_idx] */
            /* Actually struct fields are stored at increasing negative offsets from base */
            /* base slot is LEA'd as [rbp - 8*(base_slot+1)] */
            /* field i is at [rax + 8*i] (positive offset from base address) */
            /* But our struct lit stores fields at base_slot, base_slot+1, etc */
            /* LEA returns base_slot address = rbp - 8*(base_slot+1) */
            /* field 0 = [rbp - 8*(base_slot+1)] = [rax] */
            /* field 1 = [rbp - 8*(base_slot+2)] = [rax - 8] */
            if (field_idx == 0) {
                /* mov rax, [rax] */
                emit(0x48); emit(0x8B); emit(0x00);
            } else {
                /* mov rax, [rax - 8*field_idx] */
                int32_t off = -field_idx * 8;
                emit(0x48); emit(0x8B); emit(0x80); emit32(off);
            }
        } else {
            /* Unknown field — load [rax] as fallback */
            emit(0x48); emit(0x8B); emit(0x00);
        }
        break;
    }
    case EX_INDEX: {
        compile_expr(e->left, loc);
        emit_push_rax();
        compile_expr(e->right, loc);
        emit_pop_rcx();
        /* Check if base is a byte array (type == 3, struct field, or as usize heuristic) */
        int is_byte = is_byte_index(e->left, e->right, loc);
        if (is_byte) {
            /* movsx rax, byte [rcx + rax] — byte-indexed */
            emit(0x48); emit(0x0F); emit(0xBE); emit(0x04); emit(0x01);
        } else {
            /* mov rax, [rcx + rax*8] — qword-indexed */
            emit(0x48); emit(0x8B); emit(0x04); emit(0xC1);
        }
        break;
    }
    case EX_IF: {
        compile_expr(e->cond, loc);
        emit(0x48); emit(0x85); emit(0xC0); /* test rax,rax */
        emit(0x0F); emit(0x84); int else_patch = g_code_len; emit32(0); /* JZ else */
        if (e->block) compile_block(e->block, loc);
        if (e->else_block) {
            emit(0xE9); int end_patch = g_code_len; emit32(0); /* JMP end */
            patch32(else_patch, g_code_len - (else_patch+4));
            compile_block(e->else_block, loc);
            patch32(end_patch, g_code_len - (end_patch+4));
        } else {
            patch32(else_patch, g_code_len - (else_patch+4));
        }
        break;
    }
    case EX_WHILE: {
        g_loop_depth++;
        g_loop_top[g_loop_depth] = g_code_len;
        g_break_count[g_loop_depth] = 0;
        int loop_top = g_code_len;
        compile_expr(e->cond, loc);
        emit(0x48); emit(0x85); emit(0xC0); /* test rax,rax */
        emit(0x0F); emit(0x84); int exit_patch = g_code_len; emit32(0); /* JZ exit */
        if (e->block) compile_block(e->block, loc);
        emit(0xE9); int32_t back = loop_top - (g_code_len+4); emit32(back); /* JMP top */
        patch32(exit_patch, g_code_len - (exit_patch+4));
        /* Patch breaks */
        for (int i = 0; i < g_break_count[g_loop_depth]; i++)
            patch32(g_break_patches[g_loop_depth][i], g_code_len - (g_break_patches[g_loop_depth][i]+4));
        g_loop_depth--;
        break;
    }
    case EX_RETURN:
        if (e->left) compile_expr(e->left, loc);
        emit_leave(); emit_ret();
        break;
    case EX_BREAK:
        if (g_loop_depth >= 0) {
            emit(0xE9); /* JMP rel32 */
            g_break_patches[g_loop_depth][g_break_count[g_loop_depth]++] = g_code_len;
            emit32(0);
        }
        break;
    case EX_CONTINUE:
        if (g_loop_depth >= 0) {
            emit(0xE9); /* JMP rel32 */
            int32_t back_cont = g_loop_top[g_loop_depth] - (g_code_len+4);
            emit32(back_cont);
        }
        break;
    case EX_BLOCK:
        if (e->block) compile_block(e->block, loc);
        break;
    case EX_MATCH: {
        compile_expr(e->left, loc);
        int scrut_slot = loc->next_slot++;
        emit_store_rax(scrut_slot);
        static int end_patches[MAX_MATCH_ARMS]; int ep_count = 0;
        for (MatchArm *arm = e->arms; arm; arm = arm->next) {
            if (arm->pat_kind == 0) {
                /* wildcard — always matches */
                compile_expr(arm->body, loc);
            } else if (arm->pat_kind == 1) {
                /* binding — bind scrutinee to name */
                loc->names[loc->count] = arm->pat_name;
                loc->slots[loc->count] = scrut_slot;
                loc->types[loc->count] = 0;
                loc->count++;
                compile_expr(arm->body, loc);
                loc->count--;
            } else if (arm->pat_kind == 2) {
                /* enum variant — compare scrutinee tag against variant index */
                int variant_val = -1;
                if (arm->pat_path_prefix.len > 0)
                    variant_val = find_enum_variant(arm->pat_path_prefix, arm->pat_name);
                if (variant_val < 0)
                    variant_val = find_variant_any_enum(arm->pat_name);
                emit_load_rax(scrut_slot);
                emit(0x48); emit(0x3D); emit32(variant_val >= 0 ? variant_val : 0); /* cmp rax, imm32 */
                emit(0x0F); emit(0x85); int skip_patch = g_code_len; emit32(0); /* JNE skip */
                compile_expr(arm->body, loc);
                patch32(skip_patch, g_code_len - (skip_patch+4));
            } else if (arm->pat_kind == 3) {
                /* integer literal pattern */
                emit_load_rax(scrut_slot);
                emit(0x48); emit(0x3D); emit32((int32_t)arm->pat_int); /* cmp rax, imm32 */
                emit(0x0F); emit(0x85); int skip_patch = g_code_len; emit32(0); /* JNE skip */
                compile_expr(arm->body, loc);
                patch32(skip_patch, g_code_len - (skip_patch+4));
            } else {
                /* unknown pattern — fall through */
                compile_expr(arm->body, loc);
            }
            if (arm->next) {
                emit(0xE9); end_patches[ep_count] = g_code_len; emit32(0); ep_count++;
            }
        }
        for (int i = 0; i < ep_count; i++)
            patch32(end_patches[i], g_code_len - (end_patches[i]+4));
        loc->next_slot--;
        break;
    }
    case EX_STRUCT_LIT: {
        /* Allocate space for struct fields on stack */
        int base_slot = loc->next_slot;
        int si = find_struct(e->name);
        int nfields = (si >= 0) ? g_struct_field_count[si] : 0;
        loc->next_slot += (nfields > 0 ? nfields : 1);
        /* Store each field value */
        int fi = 0;
        for (FieldInit *f = e->fields; f; f = f->next, fi++) {
            compile_expr(f->value, loc);
            emit_store_rax(base_slot + fi);
        }
        /* LEA rax, [rbp - 8*(base_slot+1)] — return address of struct */
        int32_t disp = -(base_slot+1)*8;
        emit(0x48); emit(0x8D); emit(0x85); emit32(disp);
        break;
    }
    case EX_ARRAY_REPEAT: {
        /* [value; count] — allocate count slots, fill with value */
        int64_t count = 0;
        if (e->right && e->right->kind == EX_INT) count = e->right->int_val;
        if (count > 4096) {
            /* Large array: use mmap(NULL, count, PROT_R|W=3, MAP_PRIVATE|ANON=0x22, -1, 0)
             * Returns pointer in rax — already zero-filled by kernel */
            emit(0x48); emit(0x31); emit(0xFF); /* xor rdi, rdi */
            emit(0x48); emit(0xBE); emit64(count * 8); /* mov rsi, count*8 (bytes, not elements) */
            emit(0xBA); emit32(3); /* mov edx, PROT_R|W */
            emit(0x41); emit(0xBA); emit32(0x22); /* mov r10d, MAP_PRIVATE|ANON */
            emit(0x49); emit(0xC7); emit(0xC0); emit32(-1); /* mov r8, -1 */
            emit(0x4D); emit(0x31); emit(0xC9); /* xor r9, r9 */
            emit(0xB8); emit32(9); /* sys_mmap */
            emit(0x0F); emit(0x05); /* syscall */
            /* rax = pointer to zero-filled buffer */
        } else {
            if (count > 512) count = 512; /* cap for stack arrays */
            int base_slot = loc->next_slot;
            loc->next_slot += (int)count;
            compile_expr(e->left, loc);
            for (int i = 0; i < (int)count; i++) emit_store_rax(base_slot + i);
            /* LEA rax, base */
            int32_t disp = -(base_slot+1)*8;
            emit(0x48); emit(0x8D); emit(0x85); emit32(disp);
        }
        break;
    }
    case EX_CAST:
        compile_expr(e->left, loc);
        if (name_is(e->cast_type, "i8") || name_is(e->cast_type, "u8")) {
            emit(0x48); emit(0x25); emit32(0xFF); /* and rax, 0xFF */
        } else if (name_is(e->cast_type, "f64")) {
            /* i64 → f64: cvtsi2sd xmm0, rax; movq rax, xmm0 */
            emit(0xf2); emit(0x48); emit(0x0f); emit(0x2a); emit(0xc0);
            emit(0x66); emit(0x48); emit(0x0f); emit(0x7e); emit(0xc0);
        } else if (name_is(e->cast_type, "i64") && expr_is_float(e->left, loc)) {
            /* f64 → i64: movq xmm0, rax; cvttsd2si rax, xmm0 */
            emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc0);
            emit(0xf2); emit(0x48); emit(0x0f); emit(0x2c); emit(0xc0);
        }
        break;
    case EX_REF:
        compile_expr(e->left, loc);
        /* For simple ident refs, compute address */
        /* This is a simplification */
        break;
    case EX_DEREF:
        compile_expr(e->left, loc);
        /* Load value at address: mov rax, [rax] */
        emit(0x48); emit(0x8B); emit(0x00);
        break;
    default:
        emit_mov_rax_imm64(0);
        break;
    }
}

static void compile_block(Block *b, Locals *loc) {
    if (!b) return;
    int saved_count = loc->count;
    for (Stmt *s = b->stmts; s; s = s->next) {
        if (s->kind == 1 || s->kind == 2) {
            /* let/var binding */
            compile_expr(s->expr, loc);
            int slot = loc->next_slot++;
            emit_store_rax(slot);
            loc->names[loc->count] = s->name;
            loc->slots[loc->count] = slot;
            /* Detect type from RHS expression or type annotation */
            int vtype = 0;
            if (s->expr && s->expr->kind == EX_STRUCT_LIT) {
                int si = find_struct(s->expr->name);
                if (si >= 0) vtype = 10 + si;
            }
            if (s->expr && s->expr->kind == EX_ARRAY_REPEAT) {
                vtype = 3; /* byte array */
            }
            /* read_file returns a byte buffer pointer */
            if (s->expr && s->expr->kind == EX_CALL && name_is(s->expr->name, "read_file")) {
                vtype = 3;
            }
            /* f64 type inference */
            if (s->expr && expr_is_float(s->expr, loc)) {
                vtype = 1;
            }
            loc->types[loc->count] = vtype;
            loc->count++;
        } else if (s->kind == 3) {
            /* assignment */
            if (s->expr && s->expr->kind == EX_IDENT && s->right) {
                /* Simple variable assignment: x = rhs */
                compile_expr(s->right, loc);
                int found_local = 0;
                for (int i = loc->count-1; i >= 0; i--) {
                    if (name_eq(loc->names[i], s->expr->name)) {
                        emit_store_rax(loc->slots[i]);
                        found_local = 1;
                        break;
                    }
                }
                if (!found_local) {
                    /* Check globals */
                    int gi = find_global(s->expr->name);
                    if (gi >= 0) {
                        /* Store to data section via RIP-relative */
                        emit(0x48); emit(0x89); emit(0x05); /* mov [rip+disp32], rax */
                        emit32(g_global_offsets[gi] - (g_code_len + 4));
                    }
                }
            } else if (s->expr && s->expr->kind == EX_INDEX && s->right) {
                /* Indexed assignment: arr[idx] = val */
                /* Guard: skip struct-array field sub-indexing (e.g., arr[i].text[j] = val) */
                if (s->expr->left && s->expr->left->kind == EX_FIELD &&
                    s->expr->left->left && s->expr->left->left->kind == EX_INDEX) {
                    compile_expr(s->right, loc); /* RHS for side effects only */
                    break;
                }
                /* Compile base address */
                compile_expr(s->expr->left, loc);
                int base_slot = loc->next_slot++;
                emit_store_rax(base_slot);
                /* Compile index */
                compile_expr(s->expr->right, loc);
                int idx_slot = loc->next_slot++;
                emit_store_rax(idx_slot);
                /* Compile value */
                compile_expr(s->right, loc);
                /* rax = value, load base → rcx, idx → rdx */
                emit_push_rax(); /* save value */
                emit_load_rax(idx_slot);
                emit(0x48); emit(0x89); emit(0xC2); /* mov rdx, rax (index) */
                emit_load_rax(base_slot);
                emit(0x48); emit(0x89); emit(0xC1); /* mov rcx, rax (base) */
                emit(0x58); /* pop rax (value) */
                /* Check if base is a byte array (local, struct field, or as usize) */
                int is_byte = is_byte_index(s->expr->left, s->expr->right, loc);
                if (is_byte) {
                    /* mov byte [rcx + rdx], al */
                    emit(0x88); emit(0x04); emit(0x11);
                } else {
                    /* mov [rcx + rdx*8], rax */
                    emit(0x48); emit(0x89); emit(0x04); emit(0xD1);
                }
                loc->next_slot -= 2;
            } else if (s->expr && s->expr->kind == EX_FIELD && s->right) {
                /* Field assignment: obj.field = val */
                if (s->expr->left && s->expr->left->kind == EX_INDEX) {
                    /* arr[i].field = val — struct element field write not supported,
                     * compile RHS for side effects only (scalar parallel arrays handle this) */
                    compile_expr(s->right, loc);
                } else if (s->expr->left && s->expr->left->kind == EX_IDENT) {
                    /* Simple field write: obj.field = val */
                    /* Find the local's struct type to get field offset */
                    int struct_idx = -1, field_idx = -1;
                    for (int i = loc->count-1; i >= 0; i--) {
                        if (name_eq(loc->names[i], s->expr->left->name) && loc->types[i] >= 10) {
                            struct_idx = loc->types[i] - 10; break;
                        }
                    }
                    if (struct_idx >= 0) {
                        int base = g_struct_field_base[struct_idx];
                        for (int fi = 0; fi < g_struct_field_count[struct_idx]; fi++) {
                            if (name_eq(g_struct_field_names[base + fi], s->expr->name)) {
                                field_idx = fi; break;
                            }
                        }
                    }
                    if (field_idx < 0) {
                        /* Fallback: search all structs */
                        for (int si = 0; si < g_struct_count && field_idx < 0; si++) {
                            int base = g_struct_field_base[si];
                            for (int fi = 0; fi < g_struct_field_count[si]; fi++) {
                                if (name_eq(g_struct_field_names[base + fi], s->expr->name)) {
                                    field_idx = fi; break;
                                }
                            }
                        }
                    }
                    compile_expr(s->right, loc); /* value in rax */
                    /* Find base address of the struct local */
                    int base_found = 0;
                    for (int i = loc->count-1; i >= 0; i--) {
                        if (name_eq(loc->names[i], s->expr->left->name)) {
                            /* LEA base slot address */
                            int disp = -8 * (loc->slots[i] + 1);
                            emit(0x48); emit(0x8D); emit(0x8D); emit32(disp); /* lea rcx, [rbp+disp] */
                            if (field_idx > 0) {
                                /* sub rcx, 8*field_idx */
                                emit(0x48); emit(0x81); emit(0xE9); emit32(field_idx * 8);
                            }
                            emit(0x48); emit(0x89); emit(0x01); /* mov [rcx], rax */
                            base_found = 1;
                            break;
                        }
                    }
                    if (!base_found) {
                        /* global field write — best-effort no-op */
                    }
                } else {
                    /* Other patterns (deref.field, etc.) — compile RHS only */
                    compile_expr(s->right, loc);
                }
            } else if (s->right) {
                compile_expr(s->right, loc);
            } else {
                compile_expr(s->expr, loc);
            }
        } else {
            compile_expr(s->expr, loc);
        }
    }
    loc->count = saved_count;
}

/* ================================================================
 * COMPILE FUNCTION
 * ================================================================ */
static void compile_function(FnDef *fn) {
    static Locals loc;
    memset(&loc, 0, sizeof(loc));
    /* Prologue */
    emit_push_rbp(); emit_mov_rbp_rsp();
    int frame_raw = (fn->param_count + 2048 + 2) * 8; /* generous frame for large local arrays */
    int frame_size = (frame_raw + 15) & ~15;
    emit_sub_rsp(frame_size);
    /* Spill params */
    int pi = 0;
    for (Param *p = fn->params; p; p = p->next, pi++) {
        switch (pi) {
            case 0: emit(0x48); emit(0x89); emit(0xF8); break; /* mov rax,rdi */
            case 1: emit(0x48); emit(0x89); emit(0xF0); break; /* mov rax,rsi */
            case 2: emit(0x48); emit(0x89); emit(0xD0); break; /* mov rax,rdx */
            case 3: emit(0x48); emit(0x89); emit(0xC8); break; /* mov rax,rcx */
            case 4: emit(0x4C); emit(0x89); emit(0xC0); break; /* mov rax,r8 */
            case 5: emit(0x4C); emit(0x89); emit(0xC8); break; /* mov rax,r9 */
            default: {
                int32_t disp = 16 + (pi-6)*8;
                emit(0x48); emit(0x8B); emit(0x85); emit32(disp);
                break;
            }
        }
        int slot = loc.next_slot++;
        emit_store_rax(slot);
        loc.names[loc.count] = p->name;
        loc.slots[loc.count] = slot;
        loc.types[loc.count] = p->is_byte_array ? 3 : name_is(p->type_name, "f64") ? 1 : 0;
        loc.count++;
    }
    /* Body */
    compile_block(fn->body, &loc);
    /* Epilogue (in case body doesn't have explicit return) */
    /* The last expression's value is already in rax — just leave+ret */
    emit_leave(); emit_ret();
}

/* ================================================================
 * ELF EMISSION
 * ================================================================ */
static void write_elf(const char *path) {
    int base_addr = 0x400000;
    int text_off = 4096;
    int total_code = g_code_len;

    /*
     * Trampoline entry: save rsp to GLOBAL_SAVED_RSP, call main, exit.
     */
    int tramp_off = g_code_len;

    /* mov [rip + disp32], rsp — save initial stack pointer to offset 0 */
    emit(0x48); emit(0x89); emit(0x25);
    int32_t save_disp = GLOBAL_SAVED_RSP - (g_code_len + 4);
    emit32(save_disp);

    /* Allocate global arrays via mmap, store pointers in data section */
    for (int gi = 0; gi < g_global_count; gi++) {
        if (g_global_is_array[gi] > 0 && g_global_array_size[gi] > 0) {
            /* mmap(NULL, size, PROT_RW=3, MAP_PRIVATE|ANON=0x22, -1, 0) */
            emit(0x48); emit(0x31); emit(0xFF); /* xor rdi, rdi */
            emit(0x48); emit(0xBE); emit64((int64_t)g_global_array_size[gi]); /* mov rsi, size */
            emit(0xBA); emit32(3); /* mov edx, 3 */
            emit(0x41); emit(0xBA); emit32(0x22); /* mov r10d, 0x22 */
            emit(0x49); emit(0xC7); emit(0xC0); emit32(-1); /* mov r8, -1 */
            emit(0x4D); emit(0x31); emit(0xC9); /* xor r9, r9 */
            emit(0xB8); emit32(9); /* mov eax, 9 (sys_mmap) */
            emit(0x0F); emit(0x05); /* syscall */
            /* Store mmap'd pointer in data section: mov [rip+disp32], rax */
            emit(0x48); emit(0x89); emit(0x05);
            emit32(g_global_offsets[gi] - (g_code_len + 4));
        }
    }

    /* call main */
    int moff = (g_main_fn_idx >= 0) ? g_fn_offsets[g_main_fn_idx] : 0;
    int32_t cd = moff - (g_code_len + 5);
    emit(0xE8); emit32(cd);
    /* mov edi, eax */
    emit(0x89); emit(0xC7);
    /* mov eax, 60 (sys_exit) */
    emit(0xB8); emit32(60);
    /* syscall */
    emit(0x0F); emit(0x05);

    total_code = g_code_len;

    int64_t entry_va = (int64_t)base_addr + text_off + tramp_off;
    int64_t fsz = text_off + total_code;

    uint8_t *out = calloc(1, (size_t)fsz + 4096);
    if (!out) { fprintf(stderr, "stage0: alloc failed\n"); exit(1); }

    /* Helper to write 64-bit LE value */
    #define W64(off, v) do { int64_t _v=(v); for(int _i=0;_i<8;_i++) out[(off)+_i]=(_v>>(_i*8))&0xFF; } while(0)
    #define W32(off, v) do { int32_t _v=(v); for(int _i=0;_i<4;_i++) out[(off)+_i]=(_v>>(_i*8))&0xFF; } while(0)
    #define W16(off, v) do { out[off]=(v)&0xFF; out[(off)+1]=((v)>>8)&0xFF; } while(0)

    /* ELF header (64 bytes) */
    out[0]=0x7F; out[1]='E'; out[2]='L'; out[3]='F';
    out[4]=2; out[5]=1; out[6]=1; /* 64-bit, LE, version 1 */
    W16(16, 2);    /* e_type = ET_EXEC */
    W16(18, 0x3E); /* e_machine = x86-64 */
    W32(20, 1);    /* e_version */
    W64(24, entry_va); /* e_entry */
    W64(32, 64);   /* e_phoff */
    W64(40, 0);    /* e_shoff */
    W32(48, 0);    /* e_flags */
    W16(52, 64);   /* e_ehsize */
    W16(54, 56);   /* e_phentsize */
    W16(56, 1);    /* e_phnum */

    /* Program header at offset 64 (56 bytes) */
    W32(64, 1);    /* p_type = PT_LOAD */
    W32(68, 7);    /* p_flags = R|W|X (needed for global data like saved_rsp) */
    W64(72, 0);    /* p_offset */
    W64(80, (int64_t)base_addr); /* p_vaddr */
    W64(88, (int64_t)base_addr); /* p_paddr */
    W64(96, fsz);  /* p_filesz */
    W64(104, fsz + 4*1024*1024); /* p_memsz = filesz + 4MB for heap/bss */
    W64(112, 0x1000); /* p_align */

    /* Copy code to text offset */
    memcpy(out + text_off, g_code, total_code);

    /* Write file */
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "stage0: cannot open %s\n", path); exit(1); }
    fwrite(out, 1, (size_t)fsz, f);
    fclose(f);
    free(out);
    /* chmod +x */
    char cmd[512]; snprintf(cmd, sizeof(cmd), "chmod +x %s", path);
    (void)system(cmd);

    fprintf(stderr, "stage0: ok written=%s size=%lld entry=0x%llx\n", path, (long long)fsz, (long long)entry_va);
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: stage0 input.sio output.elf\n"); return 1; }

    /* Read source */
    FILE *f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "stage0: cannot open %s\n", argv[1]); return 1; }
    g_src_len = (int)fread(g_src, 1, MAX_SOURCE-1, f);
    fclose(f);
    fprintf(stderr, "stage0: source=%s (%d bytes)\n", argv[1], g_src_len);

    /* Preprocess imports: scan for `use` lines, load referenced files */
    {
        /* Determine base directory from input file path */
        char basedir[512] = ".";
        { const char *sl = strrchr(argv[1], '/');
          if (sl) { int n = (int)(sl - argv[1]); if (n > 510) n = 510;
                     memcpy(basedir, argv[1], (size_t)n); basedir[n] = 0; } }

        /* Search paths: basedir, self-hosted/, stdlib/ */
        const char *search_paths[] = { basedir, "self-hosted", "stdlib", NULL };
        char imported[64][256]; int imported_count = 0;

        int pass = 0;
        while (pass < 5) { /* max 5 import passes to handle transitive deps */
            int found_new = 0;
            int p = 0;
            while (p < g_src_len) {
                /* Find lines starting with 'use ' */
                if (p == 0 || g_src[p-1] == '\n') {
                    if (g_src_len - p > 4 && strncmp(&g_src[p], "use ", 4) == 0) {
                        /* Extract module path: use foo::bar::* → foo/bar */
                        char modpath[256] = {0};
                        int mp = 0, sp = p + 4;
                        while (sp < g_src_len && g_src[sp] != '\n' && g_src[sp] != '{' &&
                               !(g_src[sp] == ':' && sp+1 < g_src_len && g_src[sp+1] == ':' &&
                                 sp+2 < g_src_len && g_src[sp+2] == '*') &&
                               !(g_src[sp] == ':' && sp+1 < g_src_len && g_src[sp+1] == ':' &&
                                 sp+2 < g_src_len && g_src[sp+2] == '{') &&
                               mp < 254) {
                            if (g_src[sp] == ':' && sp+1 < g_src_len && g_src[sp+1] == ':') {
                                modpath[mp++] = '/'; sp += 2;
                            } else {
                                modpath[mp++] = g_src[sp++];
                            }
                        }
                        modpath[mp] = 0;
                        /* Skip if empty or already imported */
                        if (mp > 0) {
                            int dup = 0;
                            for (int i = 0; i < imported_count; i++)
                                if (strcmp(imported[i], modpath) == 0) { dup = 1; break; }
                            if (!dup && imported_count < 64) {
                                strcpy(imported[imported_count++], modpath);
                                /* Try to find and load the file */
                                for (int si = 0; search_paths[si]; si++) {
                                    char filepath[768];
                                    snprintf(filepath, sizeof(filepath), "%s/%s.sio", search_paths[si], modpath);
                                    FILE *mf = fopen(filepath, "r");
                                    if (!mf) {
                                        /* Try mod.sio inside directory */
                                        snprintf(filepath, sizeof(filepath), "%s/%s/mod.sio", search_paths[si], modpath);
                                        mf = fopen(filepath, "r");
                                    }
                                    if (mf) {
                                        /* Append a newline separator */
                                        if (g_src_len < MAX_SOURCE-1) g_src[g_src_len++] = '\n';
                                        int nr = (int)fread(&g_src[g_src_len], 1, (size_t)(MAX_SOURCE - g_src_len - 1), mf);
                                        fclose(mf);
                                        g_src_len += nr;
                                        g_src[g_src_len] = 0;
                                        fprintf(stderr, "stage0: import %s (%s, %d bytes)\n", modpath, filepath, nr);
                                        found_new = 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                p++;
            }
            if (!found_new) break;
            pass++;
        }
    }

    /* Lex */
    lex_all();
    fprintf(stderr, "stage0: tokens=%d\n", g_token_count);

    /* Scan globals (prepass over tokens) */
    scan_globals();
    fprintf(stderr, "stage0: globals=%d data_size=%d\n", g_global_count, GLOBAL_DATA_SIZE);

    /* Parse */
    g_tp = 0;
    Item *program = parse_program();
    fprintf(stderr, "stage0: arena=%d bytes\n", g_arena_pos);

    /* Phase 0: Collect structs */
    int total_fields = 0;
    for (Item *it = program; it; it = it->next) {
        if (it->kind == ITEM_STRUCT && it->st && g_struct_count < MAX_STRUCTS) {
            int si = g_struct_count++;
            g_struct_names[si] = it->st->name;
            g_struct_field_base[si] = total_fields;
            int fc = 0;
            for (FieldDef *fd = it->st->fields; fd; fd = fd->next) {
                if (total_fields < MAX_FIELDS) {
                    g_struct_field_names[total_fields] = fd->name;
                    g_struct_field_is_byte[total_fields] = fd->is_byte_array;
                    total_fields++;
                }
                fc++;
            }
            g_struct_field_count[si] = fc;
        }
    }
    fprintf(stderr, "stage0: structs=%d\n", g_struct_count);

    /* Phase 0.5: Collect enums */
    int total_variants = 0;
    for (Item *it = program; it; it = it->next) {
        if (it->kind == ITEM_ENUM && it->en && g_enum_count < MAX_ENUMS) {
            int ei = g_enum_count++;
            g_enum_names[ei] = it->en->name;
            g_enum_variant_base[ei] = total_variants;
            int vc = 0;
            for (EnumVariant *v = it->en->variants; v; v = v->next) {
                if (total_variants < MAX_ENUM_VARIANTS) {
                    g_enum_variant_names[total_variants] = v->name;
                    total_variants++;
                }
                vc++;
            }
            g_enum_variant_count[ei] = vc;
        }
    }
    fprintf(stderr, "stage0: enums=%d variants=%d\n", g_enum_count, total_variants);

    /* Phase 1: Collect function signatures */
    /* Register builtins */
    g_fn_names[g_fn_count] = name_from("print_int");
    g_fn_ret_type[g_fn_count] = 0;
    int print_int_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("print_f64");
    g_fn_ret_type[g_fn_count] = 0;
    int print_f64_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("sqrt");
    g_fn_ret_type[g_fn_count] = 0;
    int sqrt_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("sin");
    g_fn_ret_type[g_fn_count] = 0;
    int sin_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("cos");
    g_fn_ret_type[g_fn_count] = 0;
    int cos_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("fabs");
    g_fn_ret_type[g_fn_count] = 0;
    int fabs_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("print");
    g_fn_ret_type[g_fn_count] = 0;
    int print_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("arg_count");
    g_fn_ret_type[g_fn_count] = 0;
    int arg_count_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("get_arg");
    g_fn_ret_type[g_fn_count] = 0;
    int get_arg_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("read_file");
    g_fn_ret_type[g_fn_count] = 0;
    int read_file_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("file_size");
    g_fn_ret_type[g_fn_count] = 0;
    int file_size_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("write_file");
    g_fn_ret_type[g_fn_count] = 0;
    int write_file_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("read_byte");
    g_fn_ret_type[g_fn_count] = 0;
    int read_byte_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("str_len");
    g_fn_ret_type[g_fn_count] = 0;
    int str_len_idx = g_fn_count++;

    g_fn_names[g_fn_count] = name_from("str_char_at");
    g_fn_ret_type[g_fn_count] = 0;
    int str_char_at_idx = g_fn_count++;

    /* Collect user functions */
    for (Item *it = program; it; it = it->next) {
        if (it->kind == ITEM_FN && it->fn && g_fn_count < MAX_FUNCS) {
            g_fn_names[g_fn_count] = it->fn->name;
            g_fn_ret_type[g_fn_count] = name_is(it->fn->return_type, "f64") ? 1 : 0;
            if (name_is(it->fn->name, "main")) g_main_fn_idx = g_fn_count;
            g_fn_count++;
        }
        if (it->kind == ITEM_IMPL && it->im) {
            for (Method *m = it->im->methods; m; m = m->next) {
                if (m->fn && g_fn_count < MAX_FUNCS) {
                    g_fn_names[g_fn_count] = m->fn->name;
                    g_fn_ret_type[g_fn_count] = 0;
                    g_fn_count++;
                }
            }
        }
    }
    fprintf(stderr, "stage0: fn_count=%d main_idx=%d\n", g_fn_count, g_main_fn_idx);

    /* Phase 2: Emit code */
    g_code_len = 0;

    /*
     * Global data at code offset 0 (before all functions).
     * Writable because PT_LOAD has RWX flags.
     */
    for (int i = 0; i < GLOBAL_DATA_SIZE; i++) emit(0);

    /* Emit builtin stubs — return 0 for unimplemented */
    #define EMIT_STUB(idx, name) do { \
        g_fn_offsets[idx] = g_code_len; \
        emit_mov_rax_imm64(0); emit_ret(); \
    } while(0)

    /*
     * print(s: string) — write null-terminated string to stdout
     * rdi = pointer to null-terminated string
     */
    g_fn_offsets[print_idx] = g_code_len;
    emit_push_rbp(); emit_mov_rbp_rsp(); emit_sub_rsp(16);
    /* Save string ptr */
    emit(0x48); emit(0x89); emit(0x7D); emit(0xF8); /* mov [rbp-8], rdi */
    /* Compute length: scan for null byte */
    emit(0x48); emit(0x89); emit(0xF8); /* mov rax, rdi */
    /* strlen loop: */
    int strlen_top = g_code_len;
    emit(0x80); emit(0x38); emit(0x00); /* cmp byte [rax], 0 */
    emit(0x74); emit(0x04); /* je done */
    emit(0x48); emit(0xFF); emit(0xC0); /* inc rax */
    int8_t sback = (int8_t)(strlen_top - (g_code_len + 2));
    emit(0xEB); emit((uint8_t)sback); /* jmp strlen_top */
    /* done: rax = pointer to null byte; length = rax - rdi */
    emit(0x48); emit(0x8B); emit(0x7D); emit(0xF8); /* mov rdi, [rbp-8] (restore ptr) */
    emit(0x48); emit(0x89); emit(0xC2); /* mov rdx, rax (end) */
    emit(0x48); emit(0x29); emit(0xFA); /* sub rdx, rdi (length) */
    emit(0x48); emit(0x89); emit(0xFE); /* mov rsi, rdi (buf) */
    emit(0xBF); emit32(1); /* mov edi, 1 (stdout) */
    emit(0xB8); emit32(1); /* mov eax, 1 (sys_write) */
    emit(0x0F); emit(0x05); /* syscall */
    emit_leave(); emit_ret();
    EMIT_STUB(read_byte_idx, "read_byte");
    EMIT_STUB(str_len_idx, "str_len");
    EMIT_STUB(str_char_at_idx, "str_char_at");

    /*
     * arg_count() -> i64: return [saved_rsp] (argc from Linux ELF entry)
     */
    g_fn_offsets[arg_count_idx] = g_code_len;
    /* mov rax, [rip + disp32] — load saved_rsp */
    emit(0x48); emit(0x8B); emit(0x05);
    emit32(GLOBAL_SAVED_RSP - (g_code_len + 4)); /* RIP-relative to saved_rsp */
    /* mov rax, [rax] — deref to get argc */
    emit(0x48); emit(0x8B); emit(0x00);
    /* sub rax, 1 — don't count program name */
    emit(0x48); emit(0xFF); emit(0xC8); /* dec rax */
    emit_ret();

    /*
     * get_arg(n: i64) -> i64: return argv[n] as pointer
     * argv[n] = [saved_rsp + 8 + 8*n]
     * Returns the char* pointer (Sounio treats strings as i64 pointers)
     */
    g_fn_offsets[get_arg_idx] = g_code_len;
    emit_push_rbp(); emit_mov_rbp_rsp();
    /* rdi = n */
    /* load saved_rsp into rax */
    emit(0x48); emit(0x8B); emit(0x05);
    emit32(GLOBAL_SAVED_RSP - (g_code_len + 4));
    /* rax = saved_rsp value; argv[n+1] = [rax + 16 + 8*rdi] (skip argc and argv[0]) */
    emit(0x48); emit(0x8B); emit(0x44); emit(0xF8); emit(0x10); /* mov rax, [rax + rdi*8 + 16] */
    emit_leave(); emit_ret();

    /*
     * file_size(path: i64) -> i64: use stat syscall
     * syscall 4 = stat; struct stat has st_size at offset 48
     */
    g_fn_offsets[file_size_idx] = g_code_len;
    emit_push_rbp(); emit_mov_rbp_rsp(); emit_sub_rsp(256);
    /* rdi = path (already set) */
    /* rsi = stat buffer at [rbp-256] */
    emit(0x48); emit(0x8D); emit(0xB5); emit32(-256); /* lea rsi, [rbp-256] */
    /* mov eax, 4 (sys_stat) */
    emit(0xB8); emit32(4);
    emit(0x0F); emit(0x05); /* syscall */
    /* mov rax, [rbp-256+48] = [rbp-208] — st_size */
    emit(0x48); emit(0x8B); emit(0x85); emit32(-208);
    emit_leave(); emit_ret();

    /*
     * read_file(path: i64) -> i64: open + read + close, return buffer address
     * For bootstrap: allocate from heap, read file contents
     * Returns pointer to buffer (caller provides buf via array)
     * Actually in bootstrap_v0: read_file is called with path, returns [i8; N] array
     * For simplicity: mmap a buffer, read into it, return pointer
     */
    g_fn_offsets[read_file_idx] = g_code_len;
    emit_push_rbp(); emit_mov_rbp_rsp(); emit_sub_rsp(48);
    /* Save path in [rbp-8] */
    emit(0x48); emit(0x89); emit(0x7D); emit(0xF8); /* mov [rbp-8], rdi */
    /* open(path, O_RDONLY=0) → fd in rax */
    /* rdi already = path */
    emit(0x48); emit(0x31); emit(0xF6); /* xor rsi, rsi (O_RDONLY) */
    emit(0xB8); emit32(2); /* sys_open */
    emit(0x0F); emit(0x05);
    emit(0x48); emit(0x89); emit(0x45); emit(0xF0); /* mov [rbp-16], rax (fd) */
    /* mmap(NULL, 2MB, PROT_READ|WRITE=3, MAP_PRIVATE|ANON=0x22, -1, 0) */
    emit(0x48); emit(0x31); emit(0xFF); /* xor rdi, rdi (addr=NULL) */
    emit(0x48); emit(0xBE); emit64(2*1024*1024); /* mov rsi, 2MB */
    emit(0xBA); emit32(3); /* mov edx, PROT_R|W */
    emit(0x41); emit(0xBA); emit32(0x22); /* mov r10d, MAP_PRIVATE|ANON */
    emit(0x49); emit(0xC7); emit(0xC0); emit32(-1); /* mov r8, -1 */
    emit(0x4D); emit(0x31); emit(0xC9); /* xor r9, r9 */
    emit(0xB8); emit32(9); /* sys_mmap */
    emit(0x0F); emit(0x05);
    emit(0x48); emit(0x89); emit(0x45); emit(0xE8); /* mov [rbp-24], rax (buf) */
    /* read(fd, buf, 2MB) */
    emit(0x48); emit(0x8B); emit(0x7D); emit(0xF0); /* mov rdi, [rbp-16] (fd) */
    emit(0x48); emit(0x8B); emit(0x75); emit(0xE8); /* mov rsi, [rbp-24] (buf) */
    emit(0x48); emit(0xBA); emit64(2*1024*1024); /* mov rdx, 2MB */
    emit(0xB8); emit32(0); /* sys_read */
    emit(0x0F); emit(0x05);
    /* close(fd) */
    emit(0x48); emit(0x8B); emit(0x7D); emit(0xF0); /* mov rdi, [rbp-16] */
    emit(0xB8); emit32(3); /* sys_close */
    emit(0x0F); emit(0x05);
    /* return buf pointer */
    emit(0x48); emit(0x8B); emit(0x45); emit(0xE8); /* mov rax, [rbp-24] */
    emit_leave(); emit_ret();

    /*
     * write_file(path: i64, data: i64, len: i64) -> i64
     * open(path, O_WRONLY|O_CREAT|O_TRUNC=0x241, 0755) → fd
     * write(fd, data, len)
     * close(fd)
     * return 0 on success
     */
    g_fn_offsets[write_file_idx] = g_code_len;
    emit_push_rbp(); emit_mov_rbp_rsp(); emit_sub_rsp(48);
    /* Save args: rdi=path, rsi=data, rdx=len */
    emit(0x48); emit(0x89); emit(0x7D); emit(0xF8); /* mov [rbp-8], rdi */
    emit(0x48); emit(0x89); emit(0x75); emit(0xF0); /* mov [rbp-16], rsi */
    emit(0x48); emit(0x89); emit(0x55); emit(0xE8); /* mov [rbp-24], rdx */
    /* open(path, O_WRONLY|O_CREAT|O_TRUNC=0x241, 0755) */
    /* rdi already = path */
    emit(0x48); emit(0xBE); emit64(0x241); /* mov rsi, O_WRONLY|O_CREAT|O_TRUNC */
    emit(0xBA); emit32(0x1ED); /* mov edx, 0755 */
    emit(0xB8); emit32(2); /* sys_open */
    emit(0x0F); emit(0x05);
    emit(0x48); emit(0x89); emit(0x45); emit(0xE0); /* mov [rbp-32], rax (fd) */
    /* write(fd, data, len) */
    emit(0x48); emit(0x8B); emit(0x7D); emit(0xE0); /* mov rdi, [rbp-32] */
    emit(0x48); emit(0x8B); emit(0x75); emit(0xF0); /* mov rsi, [rbp-16] */
    emit(0x48); emit(0x8B); emit(0x55); emit(0xE8); /* mov rdx, [rbp-24] */
    emit(0xB8); emit32(1); /* sys_write */
    emit(0x0F); emit(0x05);
    /* close(fd) */
    emit(0x48); emit(0x8B); emit(0x7D); emit(0xE0); /* mov rdi, [rbp-32] */
    emit(0xB8); emit32(3); /* sys_close */
    emit(0x0F); emit(0x05);
    /* return 0 */
    emit(0x48); emit(0x31); emit(0xC0); /* xor rax, rax */
    emit_leave(); emit_ret();

    /* Emit print_int builtin */
    g_fn_offsets[print_int_idx] = g_code_len;
    emit_print_int_builtin();
    fprintf(stderr, "stage0: builtin=print_int off=%d sz=%d\n",
            g_fn_offsets[print_int_idx], g_code_len - g_fn_offsets[print_int_idx]);

    /* Emit print_f64 builtin */
    g_fn_offsets[print_f64_idx] = g_code_len;
    emit_print_f64_builtin();
    fprintf(stderr, "stage0: builtin=print_f64 off=%d sz=%d\n",
            g_fn_offsets[print_f64_idx], g_code_len - g_fn_offsets[print_f64_idx]);

    /* Emit math builtins: sqrt, sin, cos, fabs
     * All follow same pattern: movq xmm0, rdi; <op> xmm0, xmm0; movq rax, xmm0; ret
     * sin/cos use x87 FPU (fsin/fcos via sub rsp,8; movsd [rsp], xmm0; fld qword [rsp]; fsin; fstp qword [rsp]; movsd xmm0, [rsp]; add rsp,8)
     */
    #define EMIT_SSE2_UNARY(idx, name_str, op1, op2, op3, op4) do { \
        g_fn_offsets[idx] = g_code_len; \
        /* movq xmm0, rdi */ \
        emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc7); \
        /* SSE2 op xmm0, xmm0 */ \
        emit(op1); emit(op2); emit(op3); emit(op4); \
        /* movq rax, xmm0 */ \
        emit(0x66); emit(0x48); emit(0x0f); emit(0x7e); emit(0xc0); \
        emit_ret(); \
        fprintf(stderr, "stage0: builtin=%s off=%d sz=%d\n", name_str, \
                g_fn_offsets[idx], g_code_len - g_fn_offsets[idx]); \
    } while(0)

    /* sqrtsd xmm0, xmm0 = F2 0F 51 C0 */
    EMIT_SSE2_UNARY(sqrt_idx, "sqrt", 0xf2, 0x0f, 0x51, 0xc0);

    /* fabs: andpd xmm0, [mask] — clear sign bit. Simpler: movq rax,xmm0; btr rax,63; movq xmm0,rax */
    g_fn_offsets[fabs_idx] = g_code_len;
    emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc7); /* movq xmm0, rdi */
    emit(0x66); emit(0x48); emit(0x0f); emit(0x7e); emit(0xc0); /* movq rax, xmm0 */
    emit(0x48); emit(0x0f); emit(0xba); emit(0xf0); emit(0x3f); /* btr rax, 63 */
    emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc0); /* movq xmm0, rax */
    emit(0x66); emit(0x48); emit(0x0f); emit(0x7e); emit(0xc0); /* movq rax, xmm0 */
    emit_ret();
    fprintf(stderr, "stage0: builtin=fabs off=%d sz=%d\n",
            g_fn_offsets[fabs_idx], g_code_len - g_fn_offsets[fabs_idx]);

    /* sin: use x87 FPU — sub rsp,8; movsd [rsp],xmm0; fld [rsp]; fsin; fstp [rsp]; movsd xmm0,[rsp]; add rsp,8 */
    #define EMIT_X87_TRIG(idx, name_str, fop1, fop2) do { \
        g_fn_offsets[idx] = g_code_len; \
        emit(0x66); emit(0x48); emit(0x0f); emit(0x6e); emit(0xc7); /* movq xmm0, rdi */ \
        emit(0x48); emit(0x83); emit(0xec); emit(0x08);             /* sub rsp, 8 */ \
        emit(0xf2); emit(0x0f); emit(0x11); emit(0x04); emit(0x24); /* movsd [rsp], xmm0 */ \
        emit(0xdd); emit(0x04); emit(0x24);                         /* fld qword [rsp] */ \
        emit(fop1); emit(fop2);                                     /* fsin/fcos */ \
        emit(0xdd); emit(0x1c); emit(0x24);                         /* fstp qword [rsp] */ \
        emit(0xf2); emit(0x0f); emit(0x10); emit(0x04); emit(0x24); /* movsd xmm0, [rsp] */ \
        emit(0x48); emit(0x83); emit(0xc4); emit(0x08);             /* add rsp, 8 */ \
        emit(0x66); emit(0x48); emit(0x0f); emit(0x7e); emit(0xc0); /* movq rax, xmm0 */ \
        emit_ret(); \
        fprintf(stderr, "stage0: builtin=%s off=%d sz=%d\n", name_str, \
                g_fn_offsets[idx], g_code_len - g_fn_offsets[idx]); \
    } while(0)

    /* fsin = D9 FE; fcos = D9 FF */
    EMIT_X87_TRIG(sin_idx, "sin", 0xd9, 0xfe);
    EMIT_X87_TRIG(cos_idx, "cos", 0xd9, 0xff);

    /* Compile user functions */
    int fi = 15; /* skip builtin slots (print_int, print_f64, sqrt, sin, cos, fabs, print, arg_count, get_arg, read_file, file_size, write_file, read_byte, str_len, str_char_at) */
    for (Item *it = program; it; it = it->next) {
        if (it->kind == ITEM_FN && it->fn && fi < MAX_FUNCS) {
            g_fn_offsets[fi] = g_code_len;
            compile_function(it->fn);
            fprintf(stderr, "stage0: fn=%.*s off=%d sz=%d\n",
                    it->fn->name.len, it->fn->name.buf,
                    g_fn_offsets[fi], g_code_len - g_fn_offsets[fi]);
            fi++;
        }
        if (it->kind == ITEM_IMPL && it->im) {
            for (Method *m = it->im->methods; m; m = m->next) {
                if (m->fn && fi < MAX_FUNCS) {
                    g_fn_offsets[fi] = g_code_len;
                    compile_function(m->fn);
                    fi++;
                }
            }
        }
    }

    /* Patch calls */
    for (int i = 0; i < g_code_len - 4; i++) {
        if (g_code[i] == 0xE8 && g_code[i+3] == 0xCA && g_code[i+4] == 0xFE) {
            int target_idx = (g_code[i+1] & 0xFF) | ((g_code[i+2] & 0xFF) << 8);
            if (target_idx >= 0 && target_idx < g_fn_count) {
                int target_off = g_fn_offsets[target_idx];
                int32_t disp = target_off - (i + 5);
                patch32(i+1, disp);
            }
        }
    }

    fprintf(stderr, "stage0: total_code=%d\n", g_code_len);

    /* Write ELF */
    write_elf(argv[2]);

    return 0;
}
