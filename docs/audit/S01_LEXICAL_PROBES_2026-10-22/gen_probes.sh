#!/usr/bin/env bash
# Generates S01 lexical probes. Each probe is a complete program; the expected
# reading of each is recorded in the S01 draft, not here.
set -u
D="$(cd "$(dirname "$0")" && pwd)/probes"
rm -rf "$D"; mkdir -p "$D"
w() { printf '%b' "$2" > "$D/$1.sio"; }
M='fn main() with IO {\n'
E='}\n'

# --- whitespace and line terminators
w ws_crlf        "fn main() with IO {\r\n    println(1)\r\n}\r\n"
w ws_formfeed    "${M}    let x = 1\f\n    println(x)\n${E}"
w ws_dollar      "${M}    let x = 1 \$ 2\n    println(x)\n${E}"
# --- comments
w cm_nested      "${M}    /* a /* b */ c */\n    println(7)\n${E}"
w cm_unterminated_tail "${M}    println(7)\n${E}/* never closed\n"
w cm_unterminated_body "${M}    println(7)\n    /* never closed\n    println(8)\n${E}"
w cm_doc_line    "/// outer doc\n//! inner doc\n${M}    println(7)\n${E}"
w cm_doc_block   "/** outer */\n/*! inner */\n${M}    println(7)\n${E}"
w cm_in_string   "${M}    println(\"a // b /* c */ d\")\n${E}"
# --- identifiers
LONG=$(printf 'a%.0s' $(seq 1 300))
w id_long300     "${M}    let ${LONG} = 5\n    println(${LONG})\n${E}"
LONG2=$(printf 'b%.0s' $(seq 1 256))
w id_long256     "${M}    let ${LONG2} = 5\n    println(${LONG2})\n${E}"
w id_nonascii    "${M}    let \xc3\xa9 = 5\n    println(1)\n${E}"
w id_underscore  "${M}    let _ = 3\n    let _x = 4\n    println(_x)\n${E}"
w id_digitstart  "${M}    let x1 = 4\n    println(x1)\n${E}"
# --- reserved words used as names (let binding, then use)
for kw in sample prior posterior entropy evidence belief confidence infer observe step level scope route objective strategy unit model kernel send recv close dual chan copy move drop box ref dyn require measured valid input policy models audit on is study effect handler super crate session spawn await async affine linear uncertain likelihood perform resume handle; do
  w "kw_let_${kw}" "${M}    let ${kw} = 1\n    println(${kw})\n${E}"
done
# --- integer literals
w int_sep        "${M}    println(1_000_000)\n${E}"
w int_sep_double "${M}    println(1__0)\n${E}"
w int_sep_trail  "${M}    println(10_)\n${E}"
w int_max        "${M}    println(9223372036854775807)\n${E}"
w int_max_plus1  "${M}    println(9223372036854775808)\n${E}"
w int_huge       "${M}    println(99999999999999999999)\n${E}"
w int_hex        "${M}    println(0xff)\n    println(0XFF)\n    println(0x1_0)\n${E}"
w int_hex_allf   "${M}    println(0xFFFFFFFFFFFFFFFF)\n${E}"
w int_hex_over   "${M}    println(0x1FFFFFFFFFFFFFFFF)\n${E}"
w int_bin        "${M}    println(0b101)\n${E}"
w int_oct        "${M}    println(0o17)\n${E}"
w int_leading0   "${M}    println(017)\n${E}"
w int_suffix_u8  "${M}    let x = 5u8\n    println(x)\n${E}"
w int_suffix_i32 "${M}    let x = 7i32\n    println(x)\n${E}"
w int_suffix_bad "${M}    let x = 7q\n    println(x)\n${E}"
# --- float literals
w flt_basic      "${M}    print_f64(1.5)\n    println(\"\")\n    print_f64(1.5e3)\n    println(\"\")\n    print_f64(2E-2)\n    println(\"\")\n${E}"
w flt_exp_noint  "${M}    print_f64(1e3)\n    println(\"\")\n${E}"
w flt_trailing_dot "${M}    let x = 1.\n    println(x)\n${E}"
w flt_leading_dot "${M}    let x = .5\n    print_f64(x)\n    println(\"\")\n${E}"
w flt_sep        "${M}    print_f64(1_000.000_5)\n    println(\"\")\n${E}"
w flt_suffix     "${M}    let x = 2.5f64\n    print_f64(x)\n    println(\"\")\n${E}"
w flt_hexfloat   "${M}    let x: f64 = 0x1.8p1\n    print_f64(x)\n    println(\"\")\n${E}"
w flt_range      "${M}    for i in 1..3 { println(i) }\n${E}"
w flt_method     "${M}    let x = 1.5\n    let y = 2\n    println(y)\n${E}"
# --- strings
w str_escapes    "${M}    println(\"[\\\\t]\")\n    println(\"a\\\\nb\")\n    println(\"q\\\\\"q\")\n    println(\"bs\\\\\\\\\")\n${E}"
w str_escape_unknown "${M}    println(\"x\\\\qy\")\n${E}"
w str_escape_hex "${M}    println(\"\\\\x41\")\n${E}"
w str_escape_unicode "${M}    println(\"\\\\u{41}\")\n${E}"
w str_multiline_lineno "${M}    println(\"line1\nline2\nline3\")\n    let z: i64 = \"oops\"\n${E}"
w str_unterminated "${M}    println(\"never closed)\n${E}"
w str_utf8       "${M}    println(\"\xc3\xa9\xe2\x82\xac\")\n${E}"
w str_len300     "${M}    println(\"$(printf 'x%.0s' $(seq 1 300))\")\n${E}"
# --- char literals
w chr_basic      "${M}    println('a' as i64)\n    println('\\\\n' as i64)\n    println('\\\\'' as i64)\n    println('\\\\\\\\' as i64)\n${E}"
w chr_unknown_escape "${M}    println('\\\\q' as i64)\n${E}"
w chr_utf8       "${M}    println('\xc3\xa9' as i64)\n${E}"
w chr_two        "${M}    println('ab' as i64)\n${E}"
w chr_unclosed   "${M}    let c = 'a\n    println(1)\n${E}"
# --- operators: maximal munch hazards
w op_lt_neg      "${M}    let x: i64 = -5\n    if x<-1 { println(1) } else { println(0) }\n${E}"
w op_lt_neg_sp   "${M}    let x: i64 = -5\n    if x < -1 { println(1) } else { println(0) }\n${E}"
w op_plus_neg    "${M}    let a: i64 = 5\n    println(a+-2)\n${E}"
w op_plus_neg_sp "${M}    let a: i64 = 5\n    println(a + -2)\n${E}"
w op_and_not     "${M}    let a: i64 = 6\n    let b: i64 = 3\n    println(a&!b)\n${E}"
w op_bool_and_not "${M}    let a = true\n    let b = false\n    if a&&!b { println(1) } else { println(0) }\n${E}"
w op_shift_assign "${M}    var x: i64 = 1\n    x <<= 3\n    println(x)\n${E}"
w op_pluseq       "${M}    var x: i64 = 1\n    x += 3\n    println(x)\n${E}"
w op_concat       "${M}    let a = [1, 2]\n    let b = [3]\n    let c = a ++ b\n    println(c[2])\n${E}"

# --- batch 2
w flt_round_midpoint "${M}    let x: f64 = 1.0000000000000001110223024625156541\n    if x > 1.0 { println(1) } else { println(0) }\n${E}"
w flt_frac_digits20  "${M}    let x: f64 = 0.12345678901234567890\n    let y: f64 = 0.1234567890123456789\n    if x == y { println(1) } else { println(0) }\n${E}"
w lit_unit_suffix    "${M}    let d = 500_mg\n    println(d)\n${E}"
w id_epsilon         "${M}    let \xce\xb5 = 3\n    println(\xce\xb5)\n${E}"
w top_junk           "fn helper() -> i64 { 1 }\ngarbage tokens here\n${M}    println(helper())\n${E}"
w top_junk_symbols   "fn helper() -> i64 { 1 }\n) ] } ;;\n${M}    println(helper())\n${E}"
w cm_block_lean      "${M}    /* plain block */\n    println(7)\n${E}"
# --- batch 3
w str_escape_nul       "${M}    println(\"a\\\\0b\")\n${E}"
w str_len400           "${M}    println(\"$(printf 'y%.0s' $(seq 1 400))\")\n${E}"
w str_multiline_parse_lineno "${M}    println(\"line1\nline2\nline3\")\n    let = 5\n${E}"
w str_singleline_parse_lineno "${M}    println(\"line1\")\n\n\n    let = 5\n${E}"
echo "$(ls "$D" | wc -l) probes in $D"
