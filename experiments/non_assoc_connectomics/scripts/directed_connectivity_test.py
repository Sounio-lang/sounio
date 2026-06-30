#!/usr/bin/env python3
"""Track C: does DIRECTED (lead-lag, asymmetric) connectivity carry ASD signal that
symmetric Pearson FC misses? (The modality the brain-verdict flagged as untested.)

HONEST scope: composition of directed connectivity as matrices is ASSOCIATIVE
(non-associativity stays bounded out); what is open is whether the DIRECTED/asymmetric
*structure* carries classification signal beyond symmetric FC. This tests exactly that.

Per subject (ABIDE CC200 .1D):
  - z-score ROI time series
  - lag-1 cross-correlation M_ij = corr(x_i(t), x_j(t+1))  [asymmetric]
  - symmetric FC S = lag-0 Pearson (upper-tri = baseline, ~66% known)
  - directed/antisymmetric part K = (M - M^T)/2  (strict upper-tri = directed features)
LOSO (20 sites) linear classifier: FC alone vs K alone vs FC+K. Plus ‖K‖/‖M‖ asymmetry.
"""
import os, glob, csv
import numpy as np

ABIDE = "/workspace/sounio/artifacts/research/abide"
pheno = {}
with open(os.path.join(ABIDE, "phenotypic.csv"), newline="") as fh:
    for row in csv.DictReader(fh):
        fid=(row.get("FILE_ID") or "").strip(); dx=(row.get("DX_GROUP") or "").strip(); site=(row.get("SITE_ID") or "").strip()
        if fid and fid!="no_filename" and dx in ("1","2"):
            pheno[fid]=(1 if dx=="1" else 0, site)

def zscore(ts):
    ts=ts-ts.mean(0,keepdims=True); sd=ts.std(0,keepdims=True); sd[sd<1e-8]=1.0; return ts/sd

fc_list=[]; dir_list=[]; y=[]; sites=[]; asym=[]
iu=None
for f in sorted(glob.glob(os.path.join(ABIDE,"*_rois_cc200.1D"))):
    fid=os.path.basename(f)[:-len("_rois_cc200.1D")]
    if fid not in pheno: continue
    try: ts=np.loadtxt(f,comments="#")
    except Exception: continue
    if ts.ndim!=2 or ts.shape[1]<50 or ts.shape[0]<40: continue
    Z=zscore(ts); T,R=Z.shape
    if iu is None: iu=np.triu_indices(R,k=1)
    # lag-0 symmetric FC
    S=np.corrcoef(Z.T); S=np.clip(S,-0.999999,0.999999); S=np.arctanh(S)
    # lag-1 cross-correlation (asymmetric): M_ij = corr(x_i(t), x_j(t+1))
    A=Z[:-1]; B=Z[1:]                      # A=t, B=t+1
    M=(A.T@B)/(T-1)                        # ~corr since z-scored (approx)
    K=(M-M.T)/2.0                          # antisymmetric (directed) part
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(K))): continue
    fc_list.append(S[iu].astype(np.float32))
    dir_list.append(K[iu].astype(np.float32))   # strict upper-tri of antisymmetric part
    asym.append(np.linalg.norm(K)/max(1e-9,np.linalg.norm(M)))
    y.append(pheno[fid][0]); sites.append(pheno[fid][1])

FC=np.asarray(fc_list,np.float64); DIR=np.asarray(dir_list,np.float64)
y=np.asarray(y); sites=np.asarray(sites); asym=np.array(asym)
print(f"subjects={len(y)} feats(FC)={FC.shape[1]} feats(dir)={DIR.shape[1]}  ASD={int((y==1).sum())} ctrl={int((y==0).sum())}")
print(f"asymmetry ||K||/||M|| mean={asym.mean():.3f} (0=symmetric, directed part is real if >>0)")

def balacc(p,yt):
    tpr=((p==1)&(yt==1)).sum()/max(1,(yt==1).sum()); tnr=((p==0)&(yt==0)).sum()/max(1,(yt==0).sum()); return 50*(tpr+tnr)
def lin_rls(Xtr,ytr,Xte,lam=1.0):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True); sd[sd<1e-8]=1
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd; t=np.where(ytr==1,1.,-1.); n=Xtr.shape[0]
    a=np.linalg.solve(Xtr@Xtr.T+lam*n*np.eye(n),t); return ((Xte@Xtr.T)@a>=0).astype(int)

usites=sorted(set(sites.tolist()))
res={"FC (symmetric)":[], "directed (antisym lag-1)":[], "FC + directed":[]}
for s in usites:
    te=sites==s; tr=~te
    if te.sum()==0 or len(set(y[tr].tolist()))<2 or len(set(y[te].tolist()))<2: continue
    res["FC (symmetric)"].append(balacc(lin_rls(FC[tr],y[tr],FC[te]),y[te]))
    res["directed (antisym lag-1)"].append(balacc(lin_rls(DIR[tr],y[tr],DIR[te]),y[te]))
    comb=np.concatenate([FC,DIR],1)
    res["FC + directed"].append(balacc(lin_rls(comb[tr],y[tr],comb[te]),y[te]))

print("\nLOSO balanced accuracy:")
for k in ["FC (symmetric)","directed (antisym lag-1)","FC + directed"]:
    a=np.array(res[k]); print(f"  {k:26s} {a.mean():.2f} +/- {a.std():.2f}")
print("\nVerdict logic: directed carries signal if 'directed' > chance(50);")
print("it adds value beyond FC if 'FC + directed' > 'FC' meaningfully.")
