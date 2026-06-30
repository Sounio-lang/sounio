#!/usr/bin/env python3
"""LEARNED-embedding fair ablation on ABIDE FC (the octonion's best possible chance).
End-to-end train the FC->octonions projection + readout, toggling ONLY the algebra
product (octonion non-assoc vs H(+)H assoc). Identical architecture/capacity => the
oct-assoc DIFFERENCE controls for overfitting. LOSO, 3 seeds. PRE-COMMITTED:
  oct-assoc >= 10 -> non-assoc structure in brain ; 3..10 suggestive ; <3 none.
"""
import os, numpy as np, torch, torch.nn as nn
torch.set_num_threads(16)
CACHE="/workspace/.tmp/claude-1000/-workspace-sounio/b70e058e-f1c5-424f-a527-da432d125564/scratchpad"
X=np.load(os.path.join(CACHE,"X.npy")); y=np.load(os.path.join(CACHE,"y.npy"))
sites=np.load(os.path.join(CACHE,"sites.npy"),allow_pickle=True)

def oct_mul(a,b):
    r=np.empty(8)
    r[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4]-a[5]*b[5]-a[6]*b[6]-a[7]*b[7]
    r[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]+a[4]*b[5]-a[5]*b[4]-a[6]*b[7]+a[7]*b[6]
    r[2]=a[0]*b[2]+a[2]*b[0]-a[1]*b[3]+a[3]*b[1]+a[4]*b[6]-a[6]*b[4]+a[5]*b[7]-a[7]*b[5]
    r[3]=a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1]+a[4]*b[7]-a[7]*b[4]-a[5]*b[6]+a[6]*b[5]
    r[4]=a[0]*b[4]+a[4]*b[0]-a[1]*b[5]+a[5]*b[1]-a[2]*b[6]+a[6]*b[2]-a[3]*b[7]+a[7]*b[3]
    r[5]=a[0]*b[5]+a[5]*b[0]+a[1]*b[4]-a[4]*b[1]-a[2]*b[7]+a[7]*b[2]+a[3]*b[6]-a[6]*b[3]
    r[6]=a[0]*b[6]+a[6]*b[0]+a[1]*b[7]-a[7]*b[1]+a[2]*b[4]-a[4]*b[2]-a[3]*b[5]+a[5]*b[3]
    r[7]=a[0]*b[7]+a[7]*b[0]-a[1]*b[6]+a[6]*b[1]-a[2]*b[5]+a[5]*b[2]+a[3]*b[4]-a[4]*b[3]
    return r
def quat_mul(a,b):
    return np.array([a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                     a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])
def hh_mul(a,b): return np.concatenate([quat_mul(a[:4],b[:4]),quat_mul(a[4:],b[4:])])
def struct_tensor(mul):
    C=np.zeros((8,8,8))
    for i in range(8):
        for j in range(8):
            ei=np.zeros(8); ei[i]=1; ej=np.zeros(8); ej[j]=1
            C[:,i,j]=mul(ei,ej)
    return torch.tensor(C,dtype=torch.float32)
C_OCT=struct_tensor(oct_mul); C_HH=struct_tensor(hh_mul)
TRIPLES=[(0,1,2),(2,3,4),(4,5,6),(6,7,0),(1,3,5),(2,4,6)]

def amul(C,a,b):  # batched algebra product: a,b (n,8) -> (n,8)
    return torch.einsum('kij,ni,nj->nk',C,a,b)

class AlgNet(nn.Module):
    def __init__(self,din,C,M=8,H=128):
        super().__init__(); self.C=C; self.M=M
        self.proj=nn.Linear(din,M*8)
        feat=M*8 + len(TRIPLES)*16
        self.readout=nn.Sequential(nn.Linear(feat,H),nn.ReLU(),nn.Linear(H,1))
    def forward(self,x):
        P=self.proj(x); O=P.view(-1,self.M,8)
        parts=[P]
        for (i,j,k) in TRIPLES:
            L=amul(self.C,amul(self.C,O[:,i],O[:,j]),O[:,k])
            R=amul(self.C,O[:,i],amul(self.C,O[:,j],O[:,k]))
            parts.append(L); parts.append(R)
        return self.readout(torch.cat(parts,1)).squeeze(1)

def bal(pred,yt):
    pred=pred.numpy(); yt=yt.numpy()
    tpr=((pred==1)&(yt==1)).sum()/max(1,(yt==1).sum()); tnr=((pred==0)&(yt==0)).sum()/max(1,(yt==0).sum())
    return 50*(tpr+tnr)

def train_eval(Xtr,ytr,Xte,yte,C,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n=len(Xtr); vi=np.random.permutation(n); nv=int(.2*n); val=vi[:nv]; trn=vi[nv:]
    Xt=torch.tensor(Xtr,dtype=torch.float32); Yt=torch.tensor(ytr,dtype=torch.float32)
    Xe=torch.tensor(Xte,dtype=torch.float32)
    net=AlgNet(Xtr.shape[1],C); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-3)
    lossf=nn.BCEWithLogitsLoss(); best=(1e9,None)
    for ep in range(250):
        net.train(); opt.zero_grad()
        out=net(Xt[trn]); loss=lossf(out,Yt[trn]); loss.backward(); opt.step()
        if ep%10==0:
            net.eval()
            with torch.no_grad():
                vl=lossf(net(Xt[val]),Yt[val]).item()
            if vl<best[0]: best=(vl,{k:v.clone() for k,v in net.state_dict().items()})
    if best[1]: net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        pe=(torch.sigmoid(net(Xe))>=0.5).int()
    return bal(pe,torch.tensor(yte))

usites=sorted(set(sites.tolist())); SEEDS=3
res={"octonion":[], "H(+)H assoc":[]}
for s in usites:
    te=sites==s; tr=~te
    if te.sum()==0 or len(set(y[tr].tolist()))<2 or len(set(y[te].tolist()))<2: continue
    Xtr,Xte=X[tr],X[te]
    mu=Xtr.mean(0,keepdims=True); Xc=Xtr-mu
    _,_,Vt=np.linalg.svd(Xc,full_matrices=False); B=Vt[:128].T
    Ptr=Xc@B; Pte=(Xte-mu)@B
    pmu=Ptr.mean(0,keepdims=True); psd=Ptr.std(0,keepdims=True); psd[psd<1e-8]=1
    Ptr=(Ptr-pmu)/psd; Pte=(Pte-pmu)/psd
    for name,C in [("octonion",C_OCT),("H(+)H assoc",C_HH)]:
        accs=[train_eval(Ptr,y[tr],Pte,y[te],C,seed=sd) for sd in range(SEEDS)]
        res[name].append(np.mean(accs))
    print(f"site {s:12s} oct={res['octonion'][-1]:.1f} assoc={res['H(+)H assoc'][-1]:.1f}",flush=True)

print("\n==== LEARNED-embedding LOSO (ABIDE FC), 3 seeds/fold ====")
for k in res: a=np.array(res[k]); print(f"  {k:16s} {a.mean():.2f} +/- {a.std():.2f}")
o=np.array(res["octonion"]).mean(); h=np.array(res["H(+)H assoc"]).mean()
print(f"\n  oct - assoc = {o-h:+.2f}")
print("  PRE-COMMITTED: >=10 non-assoc structure | 3..10 suggestive | <3 none")
