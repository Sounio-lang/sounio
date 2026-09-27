#!/usr/bin/env bash
# Usage (repo root): classify.sh OUT_DIR SIG_TSV E035_PERFILE MANUAL_TSV > CLASS_TSV
# Output: file \t verdict \t signature \t class \t group.
# Order: manual assignments; mechanism rules (each established by reading and/or a two-engine experiment);
# then the test's declared contract (//@ run-pass | compile-fail | check-only | ignore, requires:, known-failure).
O="$1"; S="$2"; PF="$3"; MAN="$4"
modfile(){ local m=$1 main=$2 c
  if [ "$(basename "$(dirname "$main")")/$(basename "$main" .sio)" = "$m" ]; then echo "$main"; return; fi
  for c in "stdlib/$m.sio" "$(dirname "$main")/$(basename "$m").sio" "self-hosted/$m.sio"; do [ -f "$c" ] && { echo "$c"; return; }; done
  find stdlib self-hosted -path "*/$m.sio" 2>/dev/null | head -1; }
usefile(){ local p=$1 main=$2 c; p=${p//:://}
  for c in "stdlib/$p.sio" "$(dirname "$main")/$p.sio" "$(dirname "$main")/$(basename "$p").sio" "self-hosted/$p.sio"; do [ -f "$c" ] && { echo "$c"; return; }; done; }
dup_struct(){ [ "$(git grep -lE "^\s*(pub\s+)?struct $1\b" -- '*.sio' ':!archive' 2>/dev/null | wc -l)" -ge 2 ]; }
code_of(){ echo "$1" | grep -oE 'E[0-9]{3}|P[0-9]{4}' | head -1; }
while IFS=$'\t' read -r f v s; do
  c=PENDING; g="unclassified"
  has(){ grep -qE "$1" "$f" 2>/dev/null; }
  directive=$(grep -m1 -oE '^//@ *(current-source-run-pass|run-pass|typecheck-fail|compile-fail|check-only|ignore|build-pass)' "$f" 2>/dev/null | sed -E 's|//@ *||'); case "$directive" in current-source-run-pass) directive=run-pass;; typecheck-fail) directive=compile-fail;; esac
  requires=$(grep -m1 -hoE '^//@? *requires: *[A-Za-z_-]+' "$f" 2>/dev/null | sed -E 's#.*requires: *##')
  kf=$(grep -c '^//@ *known-failure' "$f" 2>/dev/null)
  man=$(awk -F'\t' -v f="$f" '$1==f{print $2"\t"$3; exit}' "$MAN")
  if [ -n "$man" ]; then c=${man%%$'\t'*}; g=${man#*$'\t'}
  elif [ "$v" = LEAN_ONLY ]; then
    llog="$O/lean_only.$(echo "$f" | tr / _).log"
    raw=$(grep -m1 -E '(^|[[:space:]])error(\[|:)' "$llog" 2>/dev/null)
    loc=$(echo "$raw" | grep -oE "(<main>|[A-Za-z0-9_/.-]+\.sio):[0-9]+|at line [0-9]+" | head -1)
    case "$loc" in "at line "*) lf=$f; n=${loc#at line };; "<main>:"*) lf=$f; n=${loc#<main>:};; "") lf=$f; n="";; *) lf=${loc%:*}; n=${loc##*:};; esac
    [ -f "$lf" ] || lf=$f
    id=$(echo "$raw" | grep -oE '`[^`]+`' | head -1 | tr -d '`')
    srcl=$([ -n "$n" ] && sed -n "${n}p" "$lf"); win=$([ -n "$n" ] && sed -n "${n},$((n+6))p" "$lf" | tr '\n' ' ')
    if has "Aspirational example preserved below"; then c=DEFECT-LEAN; g="lean analyses code inside /* */ (stub files)"
    elif [[ "$s" == *"E221]: no main"* ]]; then
      if has '^\s*(pub )?fn main\b'; then g="E221 but file has fn main"; else c=HARNESS; g="no main: lean check is a full compile"; fi
    elif [[ "$s" == *"Segmentation fault"* ]]; then c=DEFECT-LEAN; g="lean_single crashed (segmentation fault)"
    elif [[ "$s" == *"E035]: effect not declared in function signature at line"* ]]; then
      k=$(awk -F'\t' -v f="$f" '$1==f{print $2; exit}' "$PF"); loc2=0; obs=0
      [[ "$k" == *localvar=* || "$k" == *nonlocal_ident=* ]] && loc2=1
      [[ "$k" == *global=* || "$k" == *fieldstore=* || "$k" == *iocall=* || "$k" =~ (^|\;)call= ]] && obs=1
      if [ -z "$k" ]; then g="E035 without site table"
      elif [ $loc2 = 1 ] && [ $obs = 1 ]; then c=BOTH; g="E035 mixed: local-var sites (lean wrong) and caller-observable sites (Madaros Mut/IO gap)"
      elif [ $obs = 1 ]; then c=DEFECT-MADAROS; g="E035 caller-observable mutation or IO without the effect (documented Madaros gap)"
      elif [ $loc2 = 1 ]; then c=DEFECT-LEAN; g="E035 Mut demanded for function-local mutation"
      else c=UNRESOLVED; g="E035 sites not recognised"; fi
    elif [[ "$s" == *"E042]"* ]]; then c=SOURCE-BUG; g="Rust attribute #[...] outside comments"
    elif [[ "$s" == *"tuple index out of bounds"* ]]; then c=DEFECT-LEAN; g="tuple index > 1"
    elif [[ "$s" == *"E218]: tail type mismatch"* ]]; then c=DEFECT-LEAN; g="int-literal array as fn tail"
    elif [[ "$s" == *"duplicate parameter name"* ]] && has '\*(const|mut) '; then c=DEFECT-LEAN; g="pointer-typed params reported as duplicates"
    elif [[ "$s" == *"P0003"* ]]; then
      case "$f" in demo_portas_*) c=UNRESOLVED; g="Knowledge<T where {..}>: lean static check vs runtime-guard expansion";; *) c=INTENTIONAL; g="Knowledge annotation surface (audit 2026-08-19)";; esac
    elif [[ "$s" == *"private struct field access"* ]]; then c=UNRESOLVED; g="field privacy (lean enforces; Madaros has no check; docs silent)"
    elif [[ "$s" == *"cannot borrow as shared because it is also borrowed as mutable"* ]]; then c=UNRESOLVED; g="borrows of disjoint fields of one struct (lean tracks per variable; docs silent)"
    elif [[ "$s" == *"identifier"* && -n "$id" ]] && grep -qE "^\s*(pub\s+)?use\s[^;]*\{[^}]*\b$id\b" <(tr '\n' ' ' < "$f" | sed 's/}/}\n/g'); then
      up=$(tr '\n' ' ' < "$f" | sed 's/}/}\n/g' | grep -oE "use +[A-Za-z0-9_:]+::\{[^}]*\b$id\b" | head -1 | sed -E 's/use +//; s/::\{.*//'); uf=$(usefile "$up" "$f")
      if [ -n "$uf" ] && ! grep -qE "fn +$id *\(|(const|let|var|struct|enum|type) +$id\b" "$uf" && ! grep -qE '^\s*pub\s+use ' "$uf"; then c=DEFECT-MADAROS; g="named import of a function the module does not define (Madaros accepts)"
      elif [ -n "$uf" ] && grep -qE '^\s*pub\s+use ' "$uf"; then c=DEFECT-LEAN; g="re-export (pub use) not followed by lean"; fi
    elif [[ "$s" == *"identifier"* ]] && has '^\s*import [A-Za-z_0-9]+::\*'; then c=DEFECT-LEAN; g="import X::* not supported"
    fi
    if [ "$c" = PENDING ]; then
      if [[ "$s" == *"identifier"* ]]; then
        for up in $(grep -oE '^\s*use +[A-Za-z0-9_:]+::(\*|\{)' "$f" | sed -E 's/^\s*use +//; s/::(\*|\{)$//'); do uf=$(usefile "$up" "$f"); [ -n "$uf" ] && grep -qE '^\s*pub\s+use ' "$uf" && { c=DEFECT-LEAN; g="re-export (pub use) not followed by lean"; break; }; done
      fi
    fi
    if [ "$c" = PENDING ]; then
      if [[ "$s" == *"E200]: undefined identifier"* ]] && has '^\s*(pub )?enum ' && has '::[A-Z][A-Za-z0-9_]* \{'; then c=DEFECT-LEAN; g="data-carrying enum variant V::M { .. }"
      elif [[ "$s" == *"identifier"* && "$id" == loop ]]; then c=DEFECT-LEAN; g="loop { } not supported"
      elif [[ "$s" == *"identifier"* && ( "$id" == second_order_mean || "$id" == correlate ) ]]; then c=DEFECT-LEAN; g="Madaros builtin absent from lean_single (second_order_mean, correlate)"
      elif [[ "$s" == *"identifier"* ]] && [ -n "$id" ] && echo "$srcl" | grep -qE "&!?\s*$id\("; then c=DEFECT-LEAN; g="reference to a call result (&f())"
      elif [[ "$s" == *"identifier"* ]] && [ -n "$id" ] && grep -qE "^\s*var $id: \[[^]]+\]\s*$" "$lf"; then c=DEFECT-LEAN; g="array var declared without initializer"
      elif [[ "$s" == *"identifier"* || "$s" == *"array index must be integer"* ]] && echo "$srcl" | grep -qE '\[\[[A-Za-z0-9_]+; *[0-9]+\]; *[0-9]+\]'; then c=DEFECT-LEAN; g="nested array type on a local"
      elif [[ "$s" == *"identifier"* ]] && [ -n "$id" ] && grep -qE "\blet\s+$id\s*=\s*[A-Za-z_][A-Za-z0-9_]*\(" "$lf"; then c=DEFECT-LEAN; g="let bound to a call lean cannot type (nested or forward call)"
      elif [[ "$s" == *"array index must be integer"* ]] && [ -n "$srcl" ]; then
        hit=0; for x in $(echo "$srcl" | grep -oE '[A-Za-z_][A-Za-z0-9_]*\[' | tr -d '[' | sort -u); do grep -qE "\blet\s+$x\s*=\s*[A-Za-z_][A-Za-z0-9_]*\(" "$lf" && hit=1; done
        [ $hit = 1 ] && { c=DEFECT-LEAN; g="let bound to a call lean cannot type (nested or forward call)"; }
      elif [[ "$s" == *"E001]: Type mismatch in call argument"* ]] && echo "$win" | grep -qE '&!?\s*[A-Za-z_][A-Za-z0-9_.]*\[[^]]+\]'; then c=DEFECT-LEAN; g="reference to an indexed element as call argument"
      elif [[ "$s" == *"E001]: Type mismatch in call argument"* ]] && echo "$win" | grep -qE '[(,]\s*\[[0-9]+; *[0-9]+\]'; then c=DEFECT-LEAN; g="int-literal array as call argument"
      elif [[ "$s" == *"expected f64, got i64"* ]] && echo "$srcl" | grep -qE ":\s*char\s*=\s*'"; then c=DEFECT-LEAN; g="char literal typed as integer"
      elif [[ "$s" == *"expected f64, got i64"* || "$s" == *"shift requires integer operands"* ]] && grep -qE '\b(i128|i256|i512|u128|u256)\b' "$lf"; then c=DEFECT-LEAN; g="wide integers (i128/i256/i512) unsupported"
      elif [[ "$s" =~ expected\ \[i8\;\ [0-9]+\],\ got\ \[i64\; ]]; then c=DEFECT-LEAN; g="int-literal list into [i8; N]"
      elif [[ "$s" == *"E006]: arity mismatch"* ]]; then
        hit=0; for nm in $(grep -oE '^\s*(pub\s+)?fn +[a-z_][A-Za-z0-9_]*\(' "$f" | sed -E 's/.*fn +//; s/\($//'); do
          for up in $(grep -oE '^\s*use +[A-Za-z0-9_:]+::' "$f" | sed -E 's/^\s*use +//; s/::$//'); do uf=$(usefile "$up" "$f"); [ -n "$uf" ] && grep -qE "^\s*fn +$nm\(" "$uf" && hit=1; done; done
        [ $hit = 1 ] && { c=DEFECT-LEAN; g="private fn names collide across modules"; }
      elif [[ "$s" == *"unknown field access"* ]] && [ -n "$n" ]; then
        sig=$(awk -v n=$n 'NR>n{exit} /^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]/{s=$0} END{print s}' "$lf")
        hit=0; for t in $(echo "$sig" | grep -oE ':\s*&?!?[A-Z][A-Za-z0-9_]*' | grep -oE '[A-Z][A-Za-z0-9_]*' | sort -u); do dup_struct "$t" && hit=1; done
        [ $hit = 1 ] && { c=DEFECT-LEAN; g="struct name resolved by bare name across modules"; }
      elif [[ "$s" == *"return type does not match function signature"* ]] && [ -n "$n" ]; then
        sig=$(awk -v n=$n 'NR>n{exit} /^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]/{s=$0} END{print s}' "$lf")
        if ! echo "$sig" | grep -q -- '->'; then c=DEFECT-MADAROS; g="value returned from a unit fn (Madaros accepts)"
        elif echo "$sig" | grep -qE -- '-> *\('; then c=DEFECT-LEAN; g="type alias inside a tuple return type"; fi
      fi
    fi
    # declared contract
    code=$(code_of "$s"); short=${code:-$(echo "$s" | cut -c1-60)}
    if [ "$directive" = compile-fail ] && [[ "$c" == PENDING || "$c" == DEFECT-LEAN ]]; then
      [ "$c" = DEFECT-LEAN ] && { c=BOTH; g="compile-fail test accepted by Madaros; lean rejects via its own defect: $g"; } || { c=DEFECT-MADAROS; g="compile-fail test accepted by Madaros (lean: $short)"; }
    elif [[ "$directive" == run-pass || "$directive" == check-only ]] && [ "$requires" = madaros ] && [[ "$c" == PENDING || "$c" == DEFECT-LEAN || "$c" == BOTH ]]; then
      mech=$([ "$c" = PENDING ] && echo "$short" || echo "$g"); c=INTENTIONAL; g="test declares requires: madaros (lean: $mech)"
    elif [[ "$directive" == run-pass || "$directive" == check-only ]] && [ "$c" = PENDING ]; then c=DEFECT-LEAN; g="run-pass test rejected by lean ($short)"
    elif [ "$directive" = ignore ] && [ "$c" = PENDING ]; then c=UNRESOLVED; g="test marked //@ ignore"; fi
  else
    lg="$O/madaros_only.$(echo "$f" | tr / _).log"
    raw=$(grep -m1 -E '(^|[[:space:]])error(\[|:)|parse error|run_check_mode: (module failed|AST closure)' "$lg" 2>/dev/null)
    m=$(echo "$raw" | sed -nE 's/.* in ([^ ]+)::[A-Za-z0-9_]+ at .*/\1/p'); ab=$(echo "$raw" | grep -oE "at [0-9]+\.\.[0-9]+" | head -1 | grep -oE "[0-9]+" | tr "\n" " ")
    a=$(echo $ab | cut -d" " -f1); b=$(echo $ab | cut -d" " -f2); span=""; mf=$f
    if [ -n "$a" ]; then mf=$([ -n "$m" ] && modfile "$m" "$f" || echo "$f"); [ -f "$mf" ] && span=$(head -c "$b" "$mf" | tail -c $((b-a)) | tr '\n' ' ') || mf=$f; fi
    pf=$(grep -m1 -oE "module failed to parse: [^ ]+" "$lg" | sed 's/module failed to parse: //'); [ -n "$pf" ] && [ -f "$pf" ] || pf=$f
    lc=$(echo "$raw" | grep -oE "at line [0-9]+:[0-9]+" | grep -oE "[0-9]+:[0-9]+"); tokl=$([ -n "$lc" ] && sed -n "${lc%%:*}p" "$pf")
    mhas(){ grep -qE "$1" "$mf" 2>/dev/null || grep -qE "$1" "$f" 2>/dev/null; }
    if [[ "$s" == *"E019]"* ]]; then c=DEFECT-MADAROS; g="method calls on non-struct receivers (E019)"
    elif [[ "$s" == *"E175]"* ]]; then c=MADAROS-GUARANTEE; g="private fn used across modules (source bug exposed)"
    elif [[ "$s" == *"E241]"* || "$f" == *knowledge-annotation-parser-coverage* ]]; then c=INTENTIONAL; g="Knowledge annotation surface (audit 2026-08-19)"
    elif [[ "$s" == *"E245]"* ]]; then c=DEFECT-MADAROS; g="Knowledge op Knowledge declared unsupported (E245)"
    elif [[ "$s" == *"E230]"* ]]; then c=MADAROS-GUARANTEE; g="independence-assuming op over correlated uncertainty"
    elif [[ "$s" == *"E249]"* || "$s" == *"E250]"* ]]; then c=INTENTIONAL; g="documented reservations and fixtures (E249, E250)"
    elif [[ "$s" == *"E247]"* || "$s" == *"E012]"* ]] && has 'Negative control'; then c=MADAROS-GUARANTEE; g="negative controls"
    elif [[ "$f" == scripts/ci/fixtures/* && "$f" == *reject* ]]; then c=MADAROS-GUARANTEE; g="CI fixture built to be rejected by Madaros"
    elif [[ "$s" == *"module failed to parse"* ]]; then c=DEFECT-MADAROS; g="parse failure with no diagnostic printed"
    elif [[ "$s" == *"AST closure incomplete"* ]] && has '^\s*pub use '; then c=DEFECT-LEAN; g="unresolved pub use silently accepted by lean"
    elif [[ "$s" == *"AST closure incomplete"* ]] && has '^\s*use (native|check|ir|compiler|lexer|parser|hlir|gpu|types|sir)::'; then c=UNRESOLVED; g="module root differs (lean also searches self-hosted/)"
    elif [[ "$s" == *"E137]"* && "$span" =~ ^(read_line|print_i64|ln|append_file)$ ]]; then c=DEFECT-MADAROS; g="builtin used by tracked code missing ($span)"
    elif [[ "$s" == *"E137]"* && "$span" =~ ^_[a-z][a-z0-9]*$ ]]; then c=DEFECT-MADAROS; g="underscore literal suffix (documented 500_mg) not lexed"
    elif [[ "$s" == *"E011]"* ]] && has 'read_file\(' && has '\.as_bytes\(\)'; then c=DEFECT-MADAROS; g="read_file result untyped (.as_bytes)"
    elif [[ "$s" == *"E035] in M::F"* ]]; then c=MADAROS-GUARANTEE; g="missing effect on a caller of an effectful fn (Madaros E035; lean does not report)"
    elif [[ "$s" == *"E007]"* ]]; then c=DEFECT-MADAROS; g="if-branch join (literal adoption or statement position)"
    elif [[ "$s" == *"E016]"* ]]; then c=DEFECT-MADAROS; g="float-literal array into an f32 array"
    elif [[ "$s" == *"E012]"* && "$span" == *obligation.scaled_value* ]]; then c=MADAROS-GUARANTEE; g="error in an imported module that lean_single does not report"
    elif [[ "$s" == *"E056]"* || "$s" == *"E057]"* ]] && [ "$directive" != compile-fail ] && [ -n "$span" ] && grep -qE "while [A-Za-z_][A-Za-z0-9_]* *>= *1|\b[a-z_]+ = [a-z_]+ \* 10\b" "$mf"; then c=DEFECT-MADAROS; g="compile-time division-by-zero check ignores reassignment and loop guard"
    elif [[ "$s" == *"E004]"* ]] && mhas 'Knowledge<'; then c=DEFECT-MADAROS; g="Knowledge op Knowledge declared unsupported (E245)"
    elif [[ "$s" == *"E004]"* || "$s" == *"E009]"* ]] && mhas ':\s*&str\b'; then c=DEFECT-MADAROS; g="str alias missing (&str)"
    elif [[ "$s" == *"E009]"* && "$span" == *print_char\(* ]]; then c=UNRESOLVED; g="builtin print_char argument type (u8 vs i64), docs silent"
    elif [[ "$s" == *"E009]"* && "$f" == packages/epistemic-core/* ]]; then c=MADAROS-GUARANTEE; g="builtin Knowledge passed where KCoreKnowledge is declared (source bug)"
    elif [[ "$s" == *"E009]"* ]] && has ':\s*fn\('; then c=MADAROS-GUARANTEE; g="effectful fn ref passed as an effect-free fn type"
    elif [[ "$s" == *"E001]"* && "$span" == *"[f32;"* ]]; then c=DEFECT-MADAROS; g="float-literal array into an f32 array"
    elif [[ "$s" == *"E010]"* && "$span" == *"{:"* ]]; then c=SOURCE-BUG; g="Rust format arguments in print"
    elif [[ "$s" == *"E008]"* ]] && has '^\s*type [A-Z][A-Za-z0-9_]* = \{ *[a-z_]+: *[a-z0-9]+ *\|'; then c=DEFECT-MADAROS; g="refinement returned as its base type"
    elif [[ "$s" == *"parse error"* && -n "$tokl" ]]; then
      if echo "$tokl" | grep -qE "\[ *'.' *; *[0-9]+ *\]"; then c=DEFECT-MADAROS; g="char literal as array-repeat element does not parse"
      elif echo "$tokl" | grep -qE 'f64<[A-Za-z]+/[A-Za-z]+'; then c=DEFECT-MADAROS; g="compound unit f64<a/b> does not parse"
      elif echo "$tokl" | grep -qE '(^|[^A-Za-z_])loop\('; then c=SOURCE-BUG; g="loop used as an identifier (reserved keyword)"
      elif echo "$tokl" | grep -qE '\) -> .* where result\.'; then c=DEFECT-MADAROS; g="fn-return where clause does not parse"
      elif echo "$tokl" | grep -qE '\[[0-9]+(u|i)(8|16|32|64); *[0-9]+\]'; then c=DEFECT-MADAROS; g="typed int literal in array repeat"
      elif echo "$tokl" | grep -qE 'Validated\[[^]]*>'; then c=SOURCE-BUG; g="malformed probe"
      elif [ "$pf" != "$f" ]; then c=UNRESOLVED; g="imported module does not parse in Madaros (not isolated)"; fi
    fi
    # declared contract
    code=$(code_of "$s"); short=${code:-$(echo "$s" | cut -c1-60)}
    if [ "$c" = PENDING ]; then
      if [ "$directive" = compile-fail ]; then
        if [ "$requires" = madaros ]; then c=MADAROS-GUARANTEE; g="compile-fail test declares requires: madaros ($short)"; else c=DEFECT-LEAN; g="compile-fail test accepted by lean (Madaros: $short)"; fi
      elif [[ "$directive" == run-pass || "$directive" == check-only ]]; then
        if [ "$requires" = lean_single ] || [ "$requires" = lean ]; then c=INTENTIONAL; g="test declares requires: lean_single (Madaros: $short)"; else c=DEFECT-MADAROS; g="run-pass test rejected by Madaros ($short)"; fi
      elif [ "$directive" = ignore ]; then c=UNRESOLVED; g="test marked //@ ignore"; fi
    fi
  fi
  [ "$kf" -gt 0 ] && g="$g [documented known-failure]"
  printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$v" "$s" "$c" "$g"
done < "$S"
