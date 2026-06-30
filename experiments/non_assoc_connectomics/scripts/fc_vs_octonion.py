#!/usr/bin/env python3
"""Does octonionic structure (O-SSM) add anything OVER a linear baseline,
given REAL functional-connectivity signal?

Fair comparison at MATCHED input:
  - linear on full FC (19900)         -> ceiling
  - linear on per-fold PCA-64 FC      -> what a linear model gets at the SSM's input size
  - O-SSM  on per-fold PCA-64 FC      -> repo's real octonion model, same input
  - H-SSM  on per-fold PCA-64 FC      -> repo's real quaternion model, same input
All LOSO over 20 sites. SSMs averaged over a few seeds.
"""
import os, glob, csv, sys
import numpy as np

sys.path.insert(0, "/workspace/sounio/scripts/research")
from brain_ossm_benchmark import run_fold_oct, run_fold_hssm, XorShift128, SEEDS_A, SEEDS_B

ABIDE = "/workspace/sounio/artifacts/research/abide"
CACHE = "/workspace/.tmp/claude-1000/-workspace-sounio/b70e058e-f1c5-424f-a527-da432d125564/scratchpad"
PC = 64           # PCA dims -> reshape to (8,8) for the SSMs
N_SSM_SEEDS = 3   # keep runtime sane

def build_fc():
    fx, fy, fs = os.path.join(CACHE,"X.npy"), os.path.join(CACHE,"y.npy"), os.path.join(CACHE,"sites.npy")
    if os.path.exists(fx):
        return np.load(fx), np.load(fy), np.load(fs, allow_pickle=True)
    pheno = {}
    with open(os.path.join(ABIDE,"phenotypic.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            fid=(row.get("FILE_ID") or "").strip(); dx=(row.get("DX_GROUP") or "").strip(); site=(row.get("SITE_ID") or "").strip()
            if fid and fid!="no_filename" and dx in ("1","2"):
                pheno[fid]=(1 if dx=="1" else 0, site)
    def fc_upper(ts):
        ts=ts-ts.mean(0,keepdims=True); sd=ts.std(0,keepdims=True); sd[sd<1e-8]=1.0; ts=ts/sd
        c=np.corrcoef(ts.T); c=np.clip(c,-0.999999,0.999999); c=np.arctanh(c)
        iu=np.triu_indices(c.shape[0],k=1); return c[iu]
    X,y,sites=[],[],[]
    for f in sorted(glob.glob(os.path.join(ABIDE,"*_rois_cc200.1D"))):
        fid=os.path.basename(f)[:-len("_rois_cc200.1D")]
        if fid not in pheno: continue
        try: ts=np.loadtxt(f,comments="#")
        except Exception: continue
        if ts.ndim!=2 or ts.shape[1]<50 or ts.shape[0]<30: continue
        feat=fc_upper(ts)
        if not np.all(np.isfinite(feat)): continue
        X.append(feat.astype(np.float32)); y.append(pheno[fid][0]); sites.append(pheno[fid][1])
    X=np.asarray(X,np.float64); y=np.asarray(y); sites=np.asarray(sites)
    np.save(fx,X); np.save(fy,y); np.save(fs,sites)
    return X,y,sites

def balacc(pred,yte):
    tpr=((pred==1)&(yte==1)).sum()/max(1,(yte==1).sum())
    tnr=((pred==0)&(yte==0)).sum()/max(1,(yte==0).sum())
    return 50.0*(tpr+tnr)

def linear_rls(Xtr,ytr,Xte,lam=1.0):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1.0
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    t=np.where(ytr==1,1.0,-1.0); K=Xtr@Xtr.T; n=K.shape[0]
    alpha=np.linalg.solve(K+lam*n*np.eye(n),t)
    return ((Xte@Xtr.T)@alpha >= 0).astype(int)

X,y,sites=build_fc()
print(f"subjects={len(y)} feats={X.shape[1]} ASD={int((y==1).sum())} ctrl={int((y==0).sum())}",flush=True)
usites=sorted(set(sites.tolist()))

res={"lin_full":[], "lin_pca":[], "ossm":[], "hssm":[]}
for s in usites:
    te=sites==s; tr=~te
    if te.sum()==0 or len(set(y[tr].tolist()))<2 or len(set(y[te].tolist()))<2: continue
    Xtr,Xte,ytr,yte=X[tr],X[te],y[tr],y[te]
    # full-FC linear ceiling
    res["lin_full"].append(balacc(linear_rls(Xtr,ytr,Xte),yte))
    # per-fold PCA-64 (fit on train only -> no leakage)
    mu=Xtr.mean(0,keepdims=True); Xc=Xtr-mu
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    B=Vt[:PC].T                       # (19900, 64)
    Ptr=Xc@B; Pte=(Xte-mu)@B          # (n,64)
    # standardize PCA comps on train
    pmu=Ptr.mean(0,keepdims=True); psd=Ptr.std(0,keepdims=True); psd[psd<1e-8]=1.0
    Ptr=(Ptr-pmu)/psd; Pte=(Pte-pmu)/psd
    res["lin_pca"].append(balacc(linear_rls(Ptr,ytr,Pte),yte))
    # SSMs need (N,8,8) over the WHOLE index space with masks
    feats=np.zeros((len(y),8,8))
    Pall=np.zeros((len(y),PC)); Pall[tr]=Ptr; Pall[te]=Pte
    feats=Pall.reshape(len(y),8,8)
    o_b=[]; h_b=[]
    for k in range(N_SSM_SEEDS):
        rng_o=XorShift128(SEEDS_A[k],SEEDS_B[k]); rng_h=XorShift128(SEEDS_A[k],SEEDS_B[k])
        bo=run_fold_oct(feats,y,tr,te,rng_o,len(y))
        bh=run_fold_hssm(feats,y,tr,te,rng_h,len(y))
        if bo is not None: o_b.append(bo)
        if bh is not None: h_b.append(bh)
    if o_b: res["ossm"].append(np.mean(o_b))
    if h_b: res["hssm"].append(np.mean(h_b))
    print(f"  {s:12s} n={int(te.sum()):3d}  linFull={res['lin_full'][-1]:5.1f}  linPCA64={res['lin_pca'][-1]:5.1f}  "
          f"O-SSM={np.mean(o_b) if o_b else float('nan'):5.1f}  H-SSM={np.mean(h_b) if h_b else float('nan'):5.1f}",flush=True)

print("\n==== LOSO mean balanced accuracy ====",flush=True)
for k,name in [("lin_full","linear on FULL FC (19900)"),("lin_pca",f"linear on PCA-{PC} FC"),
               ("ossm",f"O-SSM (octonion) on PCA-{PC} FC"),("hssm",f"H-SSM (quaternion) on PCA-{PC} FC")]:
    a=np.array(res[k]); print(f"  {name:38s}: {a.mean():5.2f}% +/- {a.std():4.2f}  (n_sites={len(a)})",flush=True)
print("\nChance=50. Manifest-8x8 era: all models ~50%.",flush=True)
