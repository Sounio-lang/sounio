#!/usr/bin/env bash
# CI gate: forbid exact associativity/distributivity axioms over Float.
# These are FALSE for IEEE-754 (rounding/overflow) and silently weaken any
# Float-instantiated theorem. Float arithmetic must go through the RNE-bounded
# model in SounioIEEE754Spec / the BoundedOrderedCarrier (Higham 2002 §2.1).
#
# The match runs over NORMALISED axiom declarations: an `axiom` keyword plus
# every following continuation line is joined onto one record before matching.
# formal/ writes most of its longer axioms across two lines (`axiom name :`
# then an indented statement), and a line-anchored grep is blind to those.
set -euo pipefail
ROOT="${1:-formal}"

# One record per axiom declaration, shaped "<file>:<line>: <joined statement>".
# A declaration ends at a blank line, a column-0 line (next top-level item),
# a comment line, or another `axiom`.
normalised="$(find "$ROOT" -type f -name '*.lean' -print0 \
  | xargs -0 -r awk '
      function flush() {
        if (inax) {
          gsub(/[ \t]+/, " ", stmt)
          sub(/^ /, "", stmt)
          print axfile ":" axline ": " stmt
          inax = 0
          stmt = ""
        }
      }
      FNR == 1 { flush() }
      /^[ \t]*axiom([ \t]|$)/ {
        flush()
        inax = 1; axfile = FILENAME; axline = FNR; stmt = $0
        next
      }
      inax && (/^[ \t]*$/ || /^[^ \t]/ || /^[ \t]*--/) { flush(); next }
      inax { stmt = stmt " " $0; next }
      END { flush() }
    ')"

# Forbidden: an `axiom` over Float whose statement asserts exact
# (a*b)*c=a*(b*c), (a+b)+c=a+(b+c), or a*(b+c)=a*b+a*c.
hits=$(printf '%s\n' "$normalised" \
  | grep -E 'Float' \
  | grep -E '\(a \* b\) \* c = a \* \(b \* c\)|a \+ b \+ c = a \+ \(b \+ c\)|a \* \(b \+ c\) = a \* b \+ a \* c' || true)
if [ -n "$hits" ]; then
  echo "FAIL: false exact Float algebra axiom(s) detected:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "OK: no false exact Float algebra axioms."
