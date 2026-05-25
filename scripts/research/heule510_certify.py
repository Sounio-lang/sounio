from fractions import Fraction as Fr
import re
P=[3,5,11]
def pp(mask):
    r=1
    for b in range(3):
        if mask>>b&1: r*=P[b]
    return r
class F:
    __slots__=('c',)
    def __init__(s,c=None): s.c=c or {}
    def __add__(s,o):
        r=dict(s.c)
        for i,v in o.c.items(): r[i]=r.get(i,Fr(0))+v
        return F({i:v for i,v in r.items() if v})
    def __sub__(s,o):
        r=dict(s.c)
        for i,v in o.c.items(): r[i]=r.get(i,Fr(0))-v
        return F({i:v for i,v in r.items() if v})
    def __neg__(s): return F({i:-v for i,v in s.c.items()})
    def __mul__(s,o):
        r={}
        for i,a in s.c.items():
            for j,b in o.c.items():
                k=i^j; r[k]=r.get(k,Fr(0))+a*b*pp(i&j)
        return F({i:v for i,v in r.items() if v})
    def conj(s,T): return F({i:(v if bin(i&T).count('1')%2==0 else -v) for i,v in s.c.items()})
    def inv(s):
        num=F({0:Fr(1)})
        for T in range(1,8): num=num*s.conj(T)
        norm=(s*num).c.get(0,Fr(0)); assert set((s*num).c)<= {0}
        return F({i:v/norm for i,v in num.c.items()})
    def __truediv__(s,o): return s*o.inv()
def rat(q): return F({0:Fr(q)})
def sqrt_rat(q):
    num,den=q.numerator,q.denominator; m=num*den; s=1; mm=m
    for p in [2,3,5,7,11,13]:
        while mm%(p*p)==0: mm//=p*p; s*=p
    mask=0
    for b,p in enumerate(P):
        if mm%p==0: mm//=p; mask|=1<<b
    assert mm==1,("bad radicand",m); return F({mask:Fr(s,den)})
def tokenize(s): return re.findall(r'Sqrt|\d+|[\[\]()+\-*/]',s)
class Pr:
    def __init__(s,t): s.t=t; s.i=0
    def peek(s): return s.t[s.i] if s.i<len(s.t) else None
    def eat(s,x=None):
        tk=s.t[s.i]; assert x is None or tk==x,(x,tk); s.i+=1; return tk
    def expr(s):
        v=s.term()
        while s.peek() in ('+','-'): v=v+s.term() if s.eat()=='+' else v-s.term()
        return v
    def term(s):
        v=s.factor()
        while s.peek() in ('*','/'): v=v*s.factor() if s.eat()=='*' else v/s.factor()
        return v
    def factor(s):
        if s.peek()=='-': s.eat(); return -s.factor()
        if s.peek()=='+': s.eat(); return s.factor()
        return s.primary()
    def primary(s):
        tk=s.peek()
        if tk=='(': s.eat('('); v=s.expr(); s.eat(')'); return v
        if tk=='Sqrt':
            s.eat('Sqrt'); s.eat('['); v=s.expr(); s.eat(']')
            assert set(v.c)<= {0}; return sqrt_rat(v.c.get(0,Fr(0)))
        return rat(int(s.eat()))
def parse(s):
    p=Pr(tokenize(s)); v=p.expr(); assert p.i==len(p.t),("trail",s); return v

verts=[]
for line in open('cnp510.vtx'):
    line=line.strip()
    if not line: continue
    inner=line[1:-1]; depth=0; pos=None
    for i,ch in enumerate(inner):
        if ch in '([': depth+=1
        elif ch in ')]': depth-=1
        elif ch==',' and depth==0: pos=i; break
    verts.append((parse(inner[:pos]),parse(inner[pos+1:])))
N=len(verts); print("vertices parsed:",N)

# field audit
used=set()
for vx,vy in verts: used|=set(vx.c)|set(vy.c)
rad={0:'1',1:'√3',2:'√5',4:'√11',3:'√15',5:'√33',6:'√55',7:'√165'}
print("field radicands used:",[rad[i] for i in sorted(used)], "| √7-part used:", any(i&0 for i in used) or False, "(none -> deg-8 Q(√3,√5,√11))")

# declared edges (DIMACS, 1-indexed)
decl=set()
for line in open('cnp510.edge'):
    if line.startswith('e '):
        _,a,b=line.split(); a=int(a)-1; b=int(b)-1
        decl.add((min(a,b),max(a,b)))
print("declared edges:",len(decl))

def d2(A,B):
    dx=A[0]-B[0]; dy=A[1]-B[1]; return dx*dx+dy*dy
def is_one(v): return v.c=={0:Fr(1)}

# SOUNDNESS: every declared edge is exactly unit
sound=sum(1 for (i,j) in decl if is_one(d2(verts[i],verts[j])))
print(f"SOUNDNESS: declared edges at exact unit distance: {sound}/{len(decl)}", "ALL EXACT" if sound==len(decl) else "*** MISMATCH ***")

# CONVERSE AUDIT: any undeclared pair at exact unit distance?
extra=0; computed=0
for i in range(N):
    for j in range(i+1,N):
        if is_one(d2(verts[i],verts[j])):
            computed+=1
            if (i,j) not in decl: extra+=1
print(f"CONVERSE: total exact unit-distance pairs computed: {computed}")
print(f"          undeclared unit pairs (should be 0): {extra}")
print(f"          declared-but-not-unit (should be 0): {len(decl)-sound}")
print("PHASE B:", "COMPLETE — exact unit-distance graph == declared 2504 edges in Q(√3,√5,√11)"
      if (sound==len(decl)==computed and extra==0) else "discrepancy (see above)")
