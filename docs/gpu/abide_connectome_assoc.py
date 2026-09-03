#!/usr/bin/env python3
# PRE-REGISTERED (protocol fixed before results; run ONCE, no representation search).
# Genuinely untested door: the FULL 200x200 CC200 connectome (not the 8x8 summary that gave the null).
# Hypothesis: the octonion associator FIELD over connectome node-octonions carries ASD/TD signal beyond
# a standard associative connectome classifier.
# Protocol:
#   - connectome: per subject, Fisher-z 200x200 Pearson correlation from the raw CC200 .1D timeseries.
#   - node octonion: per LOSO fold, PCA-8 fit on TRAIN subjects' node connectivity-profiles (leakage-safe),
#     project every node -> o_i in R^8.
#   - associator field: a FIXED random sample of M=300 node-triples (i,j,k) (seed=20260719, same for all
#     subjects); feature = 7 summary stats [mean,std,p10,p25,p50,p75,p90] of ||[o_i,o_j,o_k]||^2.
#   - control: quaternion-truncated associator (comps 0..3) -> ~0 -> chance.
#   - associative baseline: PCA-50 of the connectome upper-triangle (fit on train), the standard feature.
#   - classifier: L2 logistic (GD), LEAVE-ONE-SITE-OUT CV, balanced accuracy, mean+-std over sites.
#   - decision: a model beats chance only if mean-1.96*se(over folds) > 50%. Report once whatever comes.
import numpy as np, glob, os, re
np.seterr(all='ignore')
AB='/workspace/sounio/artifacts/research/abide/'
MAN='/workspace/sounio/abide_roi_manifest.tsv'
# ---- Cayley-Dickson octonion multiply (bits=3) ----
def cds(a,b,bits=3):
    s=1
    while bits>0:
        if a==0 or b==0: return s
        if bits==1: return -s
        h=1<<(bits-1); ah=a>=h; bh=b>=h; al=a&(h-1); bl=b&(h-1)
        if not ah and not bh: a,b=al,bl
        elif not ah and bh: a,b=bl,al
        elif ah and not bh:
            if bl==0: a,b=al,0
            else: s=-s; a,b=al,bl
        else:
            if bl==0: s=-s; a,b=0,al
            else: a,b=bl,al
        bits-=1
    return s
SIG=np.zeros((8,8)); IDX=np.zeros((8,8),int)
for i in range(8):
    for j in range(8): SIG[i,j]=cds(i,j); IDX[i,j]=i^j
def omul(A,B):                 # A,B: (...,8) -> (...,8); each (i,j)->one m, so plain += (no add.at)
    out=np.zeros(A.shape,A.dtype)
    for i in range(8):
        for j in range(8):
            out[...,IDX[i,j]]+=SIG[i,j]*A[...,i]*B[...,j]
    return out
def assoc_norm2(a,b,c):        # ||(a*b)*c - a*(b*c)||^2 over last axis
    z=omul(omul(a,b),c)-omul(a,omul(b,c)); return np.sum(z*z,axis=-1)
# ---- load manifest ----
rows=[l.rstrip('\n').split('\t') for l in open(MAN) if not l.startswith('#') and not l.startswith('subject_id') and l.strip()]
files={}
for f in glob.glob(AB+'*.1D'):
    m=re.search(r'_(\d{7})_rois',os.path.basename(f))
    if m: files[str(int(m.group(1)))]=f
subj=[r for r in rows if r[0] in files][:500]
lab=np.array([1 if r[1][0]=='A' else 0 for r in subj])
sites=[r[2] for r in subj]; usite=sorted(set(sites)); site=np.array([usite.index(s) for s in subj and sites])
print(f"{len(subj)} subjects, {len(usite)} sites, ASD={lab.sum()}")
# ---- build connectomes (Fisher-z 200x200) + upper triangle ----
N=len(subj); R=200
iu=np.triu_indices(R,1)
CACHE='/tmp/claude-1000/-workspace-sounio/9c02eb82-481e-4e79-ab75-2c6b767f47fa/scratchpad/abide_conn_cache.npz'
CONN=np.zeros((N,R,R),np.float32)
def load_ts(path):
    rows=[]
    for ln in open(path):
        p=ln.split()
        try: vals=[float(x) for x in p]
        except ValueError: continue          # header / label row
        if len(vals)>=R: rows.append(vals[:R])
    return np.array(rows,np.float64)
if os.path.exists(CACHE):
    CONN=np.load(CACHE)['CONN']; print("loaded cached connectomes")
else:
    for n,r in enumerate(subj):
        ts=load_ts(files[r[0]])
        if ts.shape[1]!=R: ts=ts[:,:R]
        ts=(ts-ts.mean(0))/(ts.std(0)+1e-8)
        c=np.corrcoef(ts,rowvar=False); c=np.nan_to_num(c)
        c=np.arctanh(np.clip(c,-0.999,0.999)); np.fill_diagonal(c,0)
        CONN[n]=c
    np.savez_compressed(CACHE,CONN=CONN); print("built + cached connectomes")
UT=CONN[:,iu[0],iu[1]]          # (N,19900)
# ---- fixed triples ----
rng=np.random.default_rng(20260719); M=300
TI,TJ,TK=rng.integers(0,R,M),rng.integers(0,R,M),rng.integers(0,R,M)
def pca_fit(X,k):              # eigen-based (fast for tall OR wide X)
    mu=X.mean(0); Xc=X-mu; n,d=Xc.shape
    if d<=n:
        ev,V=np.linalg.eigh(Xc.T@Xc); return mu, V[:,::-1][:,:k].T
    G=Xc@Xc.T; ev,U=np.linalg.eigh(G); U=U[:,::-1][:,:k]; ev=np.maximum(ev[::-1][:k],1e-9)
    return mu, (Xc.T@U/np.sqrt(ev)).T
def logistic_bal(Xtr,ytr,Xte,yte):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    w=np.zeros(Xtr.shape[1]); b=0.0; lr=0.2; l2=0.02
    for _ in range(500):
        p=1/(1+np.exp(-(Xtr@w+b))); g=p-ytr
        w-=lr*(Xtr.T@g/len(ytr)+l2*w); b-=lr*g.mean()
    pr=(Xte@w+b)>0
    pos=yte==1; neg=yte==0
    sens=pr[pos].mean() if pos.any() else .5; spec=(~pr[neg]).mean() if neg.any() else .5
    return .5*(sens+spec)
def field_feats(o):            # o:(N,R,8) -> (N,7)
    a=o[:,TI]; b=o[:,TJ]; c=o[:,TK]; nn=assoc_norm2(a,b,c)   # (N,M)
    q=np.percentile(nn,[10,25,50,75,90],axis=1).T
    return np.column_stack([nn.mean(1),nn.std(1),q])
def field_feats_quat(o):
    oq=o.copy(); oq[...,4:]=0
    a=oq[:,TI]; b=oq[:,TJ]; c=oq[:,TK]; nn=assoc_norm2(a,b,c)
    q=np.percentile(nn,[10,25,50,75,90],axis=1).T
    return np.column_stack([nn.mean(1),nn.std(1),q])
# ---- LOSO ----
res={'RAW connectome PCA-50 (assoc.)':[], 'QUAT assoc field (control)':[], 'OCT assoc field':[], 'OCT field + RAW':[]}
for h in range(len(usite)):
    tr=site!=h; te=site==h
    # node PCA-8 on train nodes
    Xnodes=CONN[tr].reshape(-1,R); mu8,V8=pca_fit(Xnodes,8)
    o=((CONN.reshape(N*R,R)-mu8)@V8.T).reshape(N,R,8)
    # connectome PCA-50 baseline on train
    mu50,V50=pca_fit(UT[tr],50); Xb=(UT-mu50)@V50.T
    Fo=field_feats(o); Fq=field_feats_quat(o)
    res['RAW connectome PCA-50 (assoc.)'].append(logistic_bal(Xb[tr],lab[tr],Xb[te],lab[te]))
    res['QUAT assoc field (control)'].append(logistic_bal(Fq[tr],lab[tr],Fq[te],lab[te]))
    res['OCT assoc field'].append(logistic_bal(Fo[tr],lab[tr],Fo[te],lab[te]))
    res['OCT field + RAW'].append(logistic_bal(np.column_stack([Xb,Fo])[tr],lab[tr],np.column_stack([Xb,Fo])[te],lab[te]))
    print(f"  fold {h+1}/{len(usite)} {usite[h]:10s} done",flush=True)
print("ABIDE-I FULL 200x200 connectome, leave-one-site-out CV (chance=50%). PRE-REGISTERED, single run.")
for k,v in res.items():
    v=np.array(v)*100; se=v.std()/np.sqrt(len(v)); lo=v.mean()-1.96*se
    beats='  <-- beats chance (95% CI)' if lo>50 else ''
    print(f"  {k:34s} {v.mean():5.1f}% +- {v.std():4.1f}  (95%CI lo {lo:4.1f}){beats}")
