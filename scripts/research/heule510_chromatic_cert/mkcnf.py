import re
src=open('/workspace/.cnp-phaseB/formal/lean4/SounioHeule510.lean').read()
m=re.search(r'def E : List \(Nat × Nat\) := \[(.*?)\]', src, re.S)
pairs=re.findall(r'\((\d+),(\d+)\)', m.group(1))
E=[(int(a),int(b)) for a,b in pairs]
assert len(E)==2504, len(E)
N=510; k=4
def v(vv,c): return vv*k+c+1
cl=[]
for vv in range(N):
    cl.append([v(vv,c) for c in range(k)])
    for c in range(k):
        for d in range(c+1,k): cl.append([-v(vv,c),-v(vv,d)])
for a,b in E:
    for c in range(k): cl.append([-v(a,c),-v(b,c)])
with open('heule510_4col.cnf','w') as f:
    f.write(f"p cnf {N*k} {len(cl)}\n")
    for c in cl: f.write(" ".join(map(str,c))+" 0\n")
print("CNF from Lean E: vars",N*k,"clauses",len(cl),"edges",len(E))
