from fractions import Fraction as Fr
# Exact field Q(√3,√5,√11), 8-dim. basis mask: bit0=3,bit1=5,bit2=11. index m <-> √(prod primes).
P=[3,5,11]
def pp(mask):
    r=1
    for b in range(3):
        if mask>>b&1: r*=P[b]
    return r
class F:
    __slots__=('c',)
    def __init__(self,c=None): self.c=c or {}
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
    def conj(s,T):  # flip signs of generators in mask T
        return F({i:(v if bin(i&T).count('1')%2==0 else -v) for i,v in s.c.items()})
    def inv(s):
        num=F({0:Fr(1)}); 
        for T in range(1,8): num=num*s.conj(T)
        norm=(s*num).c.get(0,Fr(0))  # rational
        assert set((s*num).c.keys())<= {0}, "norm not rational"
        return F({i:v/norm for i,v in num.c.items()})
    def __truediv__(s,o): return s*o.inv()
def rat(q): return F({0:Fr(q)})
def sqrt_rat(q):  # field sqrt of a positive rational q whose squarefree part | 165 (over {3,5,11})
    num,den=q.numerator,q.denominator
    m=num*den  # sqrt(num/den)=sqrt(m)/den
    # factor square part
    s=1; mm=m
    for p in [2,3,5,7,11,13]:
        while mm%(p*p)==0: mm//=p*p; s*=p
    # mm now squarefree; must be product of subset of {3,5,11}
    mask=0
    for b,p in enumerate(P):
        if mm%p==0: mm//=p; mask|=1<<b
    assert mm==1, ("bad radicand",m)
    return F({mask:Fr(s,den)})

# ---- tokenizer + recursive descent parser over F ----
import re
def tokenize(s):
    return re.findall(r'Sqrt|\d+|[\[\]()+\-*/,]', s)
class Pr:
    def __init__(s,toks): s.t=toks; s.i=0
    def peek(s): return s.t[s.i] if s.i<len(s.t) else None
    def eat(s,x=None):
        tok=s.t[s.i]; 
        if x: assert tok==x,(x,tok)
        s.i+=1; return tok
    def expr(s):
        v=s.term()
        while s.peek() in ('+','-'):
            op=s.eat(); v=v+s.term() if op=='+' else v-s.term()
        return v
    def term(s):
        v=s.factor()
        while s.peek() in ('*','/'):
            op=s.eat(); v=v*s.factor() if op=='*' else v/s.factor()
        return v
    def factor(s):
        if s.peek()=='-': s.eat(); return -s.factor()
        if s.peek()=='+': s.eat(); return s.factor()
        return s.primary()
    def primary(s):
        tok=s.peek()
        if tok=='(':
            s.eat('('); v=s.expr(); s.eat(')'); return v
        if tok=='Sqrt':
            s.eat('Sqrt'); s.eat('['); v=s.expr(); s.eat(']')
            assert set(v.c.keys())<={0}, "sqrt of non-rational"
            return sqrt_rat(v.c.get(0,Fr(0)))
        # number (integer)
        return rat(int(s.eat()))
def parse(s): 
    p=Pr(tokenize(s)); v=p.expr(); assert p.i==len(p.t),("trailing",s); return v

verts=[]
for line in open('/tmp/heule510.vtx'):
    line=line.strip()
    if not line: continue
    inner=line[1:-1]
    depth=0; pos=None
    for i,ch in enumerate(inner):
        if ch in '([': depth+=1
        elif ch in ')]': depth-=1
        elif ch==',' and depth==0: pos=i; break
    verts.append((parse(inner[:pos]), parse(inner[pos+1:])))
print("parsed vertices:", len(verts))
used=set()
for vx,vy in verts: used|=set(vx.c)|set(vy.c)
rad={0:'1',1:'√3',2:'√5',4:'√11',3:'√15',5:'√33',6:'√55',7:'√165'}
print("basis indices used:", sorted(used),"->",[rad[i] for i in sorted(used)])

def d2(A,B):
    dx=A[0]-B[0]; dy=A[1]-B[1]; return dx*dx+dy*dy
def is_one(v): return v.c=={0:Fr(1)}
n=len(verts); E=0; deg=[0]*n
for i in range(n):
    for j in range(i+1,n):
        if is_one(d2(verts[i],verts[j])): E+=1; deg[i]+=1; deg[j]+=1
print("EXACT unit-distance edges among",n,"vertices:",E)
from collections import Counter
print("degree distribution:",dict(sorted(Counter(deg).items())))
# spot-check: vertices 1..6 (hexagon around 0) should each be unit from 0
print("edges from vertex 0:",deg[0])
