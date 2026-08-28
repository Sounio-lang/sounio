#!/usr/bin/env python3
"""Exact oracle for stdlib/data/interval_rat.sio (certified rational interval arithmetic)."""
from fractions import Fraction as F
def emit(key, fr): print(f"{key}_num={fr.numerator}"); print(f"{key}_den={fr.denominator}")
a=(F(1,3),F(1,2)); b=(F(1,6),F(1,4))
add=(a[0]+b[0], a[1]+b[1]);            emit("add_lo",add[0]); emit("add_hi",add[1])
sub=(a[0]-b[1], a[1]-b[0]);            emit("sub_lo",sub[0]); emit("sub_hi",sub[1])
c=(F(-1,3),F(1,2)); d=(F(1,6),F(3,1))
ps=[c[0]*d[0],c[0]*d[1],c[1]*d[0],c[1]*d[1]]
emit("mul_lo",min(ps)); emit("mul_hi",max(ps))
emit("width",a[1]-a[0])                # 1/2 - 1/3 = 1/6
dec=(F("0.1"),F("0.2")); emit("dec_lo",dec[0]); emit("dec_hi",dec[1])
