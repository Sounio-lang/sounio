#!/usr/bin/env bash
# Usage (repo root): classify_post.sh OUT_DIR CLASS_TSV > CLASS_TSV_V5
# Second pass over classify.sh output: rules established after v4, cause labels for contract rows,
# and the declared-error-pattern status for compile-fail contract rows.
O="$1"; C="$2"
usefile(){ local p=$1 main=$2 c; p=${p//:://}
  for c in "stdlib/$p.sio" "$(dirname "$main")/$p.sio" "$(dirname "$main")/$(basename "$p").sio" "self-hosted/$p.sio"; do [ -f "$c" ] && { echo "$c"; return; }; done; }
dup_struct(){ [ "$(git grep -lE "^\s*(pub\s+)?struct $1\b" -- '*.sio' ':!archive' 2>/dev/null | wc -l)" -ge 2 ]; }
while IFS=$'\t' read -r f v s c g; do
  mc=""; mg=""
  k=$([ "$v" = LEAN_ONLY ] && echo lean_only || echo madaros_only); log="$O/$k.$(echo "$f" | tr / _).log"
  if [ "$v" = LEAN_ONLY ]; then
    raw=$(grep -m1 -E '(^|[[:space:]])error(\[|:)' "$log" 2>/dev/null)
    loc=$(echo "$raw" | grep -oE "(<main>|[A-Za-z0-9_/.-]+\.sio):[0-9]+|at line [0-9]+" | head -1)
    case "$loc" in "at line "*) lf=$f; n=${loc#at line };; "<main>:"*) lf=$f; n=${loc#<main>:};; "") lf=$f; n="";; *) lf=${loc%:*}; n=${loc##*:};; esac
    [ -f "$lf" ] || lf=$f
    id=$(echo "$raw" | grep -oE '`[^`]+`' | head -1 | tr -d '`'); srcl=$([ -n "$n" ] && sed -n "${n}p" "$lf")
    if [[ "$s" == *"E006]: arity mismatch"* ]] && grep -qE 'fn +[A-Za-z0-9_]+\([^)]*<[^>()]*,[^>()]*>' "$f"; then mc=DEFECT-LEAN; mg="comma inside generic type arguments counted as a parameter separator"
    elif [[ "$s" == *"identifier"* && -n "$id" ]] && echo "$srcl" | grep -qE "(^|[^A-Za-z0-9_])$id::[A-Za-z_]"; then mc=DEFECT-LEAN; mg="qualified path call a::b::f() not supported"
    elif echo "$srcl" | grep -qE 'contest +\['; then mc=DEFECT-LEAN; mg="contest expression not supported"
    elif [[ "$s" == *"E200]: undefined identifier"* ]] && echo "$srcl" | grep -qE '::[A-Z][A-Za-z0-9_]*\{'; then mc=DEFECT-LEAN; mg="data-carrying enum variant V::M { .. }"
    elif [[ "$s" == *"field initializer type does not match"* || "$s" == *"E001]"* ]] && echo "$srcl" | grep -qE '(:|=)\s*\[\]\s*,?\s*$'; then mc=DEFECT-LEAN; mg="empty array literal [] for an unsized field or local"
    elif [[ "$s" == *"E001]: Type mismatch in call argument"* ]] && echo "$srcl" | grep -qE '(^|[^A-Za-z0-9_.])sqrt\('; then mc=DEFECT-LEAN; mg="a sqrt method on an imported type shadows the free sqrt(f64)"
    elif [[ "$s" == *"E001]: Type mismatch in call argument"* ]] && [ -n "$srcl" ]; then
      callee=$(echo "$srcl" | grep -oE '[a-z_][A-Za-z0-9_]*\(' | head -1 | tr -d '(')
      if [ -n "$callee" ]; then def=$(git grep -hE "^\s*(pub\s+)?fn +$callee\(" -- '*.sio' ':!archive' 2>/dev/null | head -1)
        for t in $(echo "$def" | grep -oE '[A-Z][A-Za-z0-9_]*' | sort -u); do dup_struct "$t" && { mc=DEFECT-LEAN; mg="struct name resolved by bare name across modules"; break; }; done; fi
    elif [[ "$s" == *"E224]: unreadable import"* ]]; then mc=UNRESOLVED; mg="module root differs (lean resolves imports under self-hosted/ or the parent directory)"
    elif [[ "$f" == tests/known_failures/* ]] && echo "$srcl" | grep -qE '\bu(8|16|32)\b'; then mc=DEFECT-LEAN; mg="narrow unsigned integers (u8/u16/u32)"
    fi
  else
    raw=$(grep -m1 -E '(^|[[:space:]])error(\[|:)|parse error|run_check_mode: (module failed|AST closure)' "$log" 2>/dev/null)
    ab=$(echo "$raw" | grep -oE "at [0-9]+\.\.[0-9]+" | head -1 | grep -oE "[0-9]+" | tr "\n" " "); a=$(echo $ab | cut -d" " -f1); b=$(echo $ab | cut -d" " -f2)
    m=$(echo "$raw" | sed -nE 's/.* in ([^ ]+)::[A-Za-z0-9_]+ at .*/\1/p'); span=""
    if [ -n "$a" ]; then mf=$f; if [ -n "$m" ]; then cand=$(find stdlib self-hosted "$(dirname "$f")" -path "*/$(basename "$m").sio" 2>/dev/null | head -1); [ "$(basename "$(dirname "$f")")/$(basename "$f" .sio)" = "$m" ] || mf=${cand:-$f}; fi; span=$(head -c "$b" "$mf" | tail -c $((b-a)) | tr '\n' ' '); fi
    eff=$(echo "$raw" | grep -oE "unknown effect '[A-Za-z0-9_]+'" | grep -oE "'[^']+'" | tr -d "'")
    if [[ "$f" == */parser_stability/invalid/* || "$f" == tests/selfhost/native_typecheck/* || "$f" == *_must_reject.sio ]]; then mc=DEFECT-LEAN; mg="negative probe accepted by lean (expected rejection by path or name)"
    elif [ -n "$eff" ] && git grep -qE "^\s*(pub\s+)?effect +$eff\b" -- '*.sio' ':!archive' 2>/dev/null; then mc=DEFECT-MADAROS; mg="imported user-declared effect not resolved (E246)"
    elif [[ "$s" == *"E137]"* && "$span" == atan2 ]]; then mc=DEFECT-MADAROS; mg="builtin used by tracked code missing (atan2)"
    elif [[ "$s" == *"E001]"* && "$span" == *"[str;"* ]]; then mc=DEFECT-MADAROS; mg="str alias missing (&str)"
    elif [[ "$s" == *"AST closure incomplete"* ]] && grep -qE '^\s*(pub\s+)?use +[A-Za-z_][A-Za-z0-9_:]*;?\s*$' "$f"; then mc=UNRESOLVED; mg="whole-module use a::b (unresolved in Madaros; lean accepts)"
    fi
  fi
  if [ -n "$mc" ]; then
    if [ "$c" = PENDING ]; then c=$mc; g=$mg
    elif [[ "$g" == "run-pass test rejected by"* || "$g" == "compile-fail test accepted by"* ]]; then g="${g%% (*} -- cause: $mg"
    elif [[ "$g" == "test declares requires: madaros"* && "$g" == *"(lean: E"* ]]; then g="test declares requires: madaros (lean: $mg)"; fi
  fi
  if [[ "$g" == "compile-fail test"* ]]; then
    pats=$(grep -E '^//@ *error-pattern:' "$f" | sed -E 's#^//@ *error-pattern: *##')
    if [ -z "$pats" ]; then st="no declared error-pattern"; else st="declared error-pattern not in the rejecting engine's output"; while IFS= read -r p; do [ -n "$p" ] && grep -qF -- "$p" "$log" && { st="declared error-pattern matched"; break; }; done <<< "$pats"; fi
    g="$g [$st]"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$v" "$s" "$c" "$g"
done < "$C"
