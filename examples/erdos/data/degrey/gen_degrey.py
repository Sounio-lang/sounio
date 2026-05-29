import math, itertools, json, csv
from collections import defaultdict, Counter

sqrt=math.sqrt
s3=sqrt(3); s7=sqrt(7); s11=sqrt(11); s12=sqrt(12); s15=sqrt(15); s33=sqrt(33)

def P(x,y): return (float(x), float(y))
S = [
P(0,0), P(1/3,0), P(1,0), P(2,0), P((s33-3)/6,0), P(1/2,1/s12),
P(1,1/s3), P(3/2,s3/2), P(7/6,s11/6), P(1/6,(s12-s11)/6), P(5/6,(s12-s11)/6),
P(2/3,(s11-s3)/6), P(2/3,(3*s3-s11)/6), P(s33/6,1/s12),
P((s33+3)/6,1/s3), P((s33+1)/6,(3*s3-s11)/6), P((s33-1)/6,(3*s3-s11)/6),
P((s33+1)/6,(s11-s3)/6), P((s33-1)/6,(s11-s3)/6),
P((s33-2)/6,(2*s3-s11)/6), P((s33-4)/6,(2*s3-s11)/6),
P((s33+13)/12,(s11-s3)/12), P((s33+11)/12,(s3+s11)/12),
P((s33+9)/12,(s11-s3)/4), P((s33+9)/12,(3*s3+s11)/12),
P((s33+7)/12,(s3+s11)/12), P((s33+7)/12,(3*s3-s11)/12),
P((s33+5)/12,(5*s3-s11)/12), P((s33+5)/12,(s11-s3)/12),
P((s33+3)/12,(3*s11-5*s3)/12), P((s33+3)/12,(s3+s11)/12),
P((s33+3)/12,(3*s3-s11)/12), P((s33+1)/12,(s11-s3)/12),
P((s33-1)/12,(3*s3-s11)/12), P((s33-3)/12,(s11-s3)/12),
P((15-s33)/12,(s11-s3)/4), P((15-s33)/12,(7*s3-3*s11)/12),
P((13-s33)/12,(3*s3-s11)/12), P((11-s33)/12,(s11-s3)/12)
]

def rot(p, c, s, center=(0.0,0.0)):
    x,y=p; cx,cy=center
    X=x-cx; Y=y-cy
    return (cx + c*X - s*Y, cy + s*X + c*Y)

def key(p, nd=12):
    return (round(p[0], nd), round(p[1], nd))

def unique(points, nd=12):
    d={}
    for p in points:
        d[key(p,nd)] = p
    return list(d.values())

# Sa: rotations by multiples of 60 and reflection y -> -y
Sa=[]
for x,y in S:
    for refl in [1,-1]:
        p=(x,refl*y)
        for k in range(6):
            theta=k*math.pi/3
            Sa.append(rot(p, math.cos(theta), math.sin(theta)))
Sa=unique(Sa)
print('S',len(S),'Sa',len(Sa))
# Sb: rotate Sa by 2 asin(1/4), anticlockwise
c_beta=7/8
s_beta=s15/8
Sb=[rot(p,c_beta,s_beta) for p in Sa]
# Y union delete +-1/3,0
Y=unique(Sa+Sb)
# delete vertices (1/3,0), (-1/3,0)
delkeys={key((1/3,0.0)), key((-1/3,0.0))}
Y=[p for p in Y if key(p) not in delkeys]
print('Sb',len(Sb),'union SaSb after delete Y',len(Y))
# Ya/Yb rotations about (-2,0) by pi/2 +/- asin(1/8)
# angle plus: cos=-1/8, sin=sqrt63/8 = 3sqrt7/8
c_plus=-1/8; s_plus=3*s7/8
c_minus=1/8; s_minus=3*s7/8
center=(-2.0,0.0)
Ya=[rot(p,c_plus,s_plus,center) for p in Y]
Yb=[rot(p,c_minus,s_minus,center) for p in Y]
G=unique(Ya+Yb)
print('Ya',len(Ya),'Yb',len(Yb),'G',len(G))
# sort vertices for deterministic ids by x then y, maybe also radial? Use x,y numeric
G_sorted=sorted(G, key=lambda p:(round(p[0],12), round(p[1],12)))
# compute unit edges: grid hashing cell size 1, check cells in neighborhood [-1,1]
vertices=G_sorted
n=len(vertices)
# use spatial grid cell size 1.0
cell=1.0
grid=defaultdict(list)
for i,p in enumerate(vertices, start=1):
    grid[(math.floor(p[0]/cell), math.floor(p[1]/cell))].append((i,p))
edges=[]
tol=1e-9
for i,(x,y) in enumerate(vertices, start=1):
    ci,cj=math.floor(x/cell), math.floor(y/cell)
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            for j,(x2,y2) in grid.get((ci+di,cj+dj),[]):
                if j<=i: continue
                d2=(x-x2)**2+(y-y2)**2
                if abs(d2-1.0) < 1e-8:
                    edges.append((i,j))
print('edges',len(edges))
# write files
base='/mnt/data/degrey_graph'
with open(base+'_vertices.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['id','x','y'])
    for i,(x,y) in enumerate(vertices, start=1):
        w.writerow([i, f'{x:.15g}', f'{y:.15g}'])
with open(base+'_edges.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['source','target'])
    w.writerows(edges)
with open(base+'_edges.dimacs','w') as f:
    f.write(f'p edge {n} {len(edges)}\n')
    for i,j in edges:
        f.write(f'e {i} {j}\n')
with open(base+'_graph.json','w') as f:
    json.dump({'name':'de Grey graph','vertices':[{'id':i,'x':x,'y':y} for i,(x,y) in enumerate(vertices, start=1)], 'edges':[{'source':i,'target':j} for i,j in edges]}, f, indent=2)
# adjacency list
adj=[[] for _ in range(n+1)]
for i,j in edges:
    adj[i].append(j); adj[j].append(i)
with open(base+'_adjacency.txt','w') as f:
    for i in range(1,n+1):
        f.write(str(i)+': '+' '.join(map(str, sorted(adj[i])))+'\n')
# stats degree
cnt=Counter(len(adj[i]) for i in range(1,n+1))
print('degree distribution', sorted(cnt.items()))
