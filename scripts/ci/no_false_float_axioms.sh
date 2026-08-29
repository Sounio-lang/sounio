#!/usr/bin/env bash
# CI gate: forbid exact associativity/distributivity axioms over Float.
# These are FALSE for IEEE-754 (rounding/overflow) and silently weaken any
# Float-instantiated theorem. Float arithmetic must go through the RNE-bounded
# model in SounioIEEE754Spec / the BoundedOrderedCarrier (Higham 2002 §2.1).
set -euo pipefail
ROOT="${1:-formal}"
# Forbidden: an `axiom` whose statement asserts exact (a*b)*c=a*(b*c),
# (a+b)+c=a+(b+c), or a*(b+c)=a*b+a*c over Float.
hits=$(grep -rnE '^[[:space:]]*axiom[[:space:]]+[A-Za-z_.]*[[:space:]]*\([^)]*Float[^)]*\)[[:space:]]*:[[:space:]]*' "$ROOT" --include='*.lean' \
  | grep -E '\(a \* b\) \* c = a \* \(b \* c\)|a \+ b \+ c = a \+ \(b \+ c\)|a \* \(b \+ c\) = a \* b \+ a \* c' || true)
if [ -n "$hits" ]; then
  echo "FAIL: false exact Float algebra axiom(s) detected:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "OK: no false exact Float algebra axioms."
