import numpy as np
sig=lambda z:1/(1+np.exp(-z))
def lstm_init(H, seed=0):
    rng=np.random.default_rng(seed); k=1/np.sqrt(H)
    return dict(Wih=rng.uniform(-k,k,(4*H,2)), Whh=rng.uniform(-k,k,(4*H,H)),
                bih=rng.uniform(-k,k,4*H), bhh=rng.uniform(-k,k,4*H), H=H)
def Vts(W, X):                       # right singular vectors (Vt) of each per-step Jacobian, computed ONCE
    H=W['H']; Whi,Whf,Whg,Who=W['Whh'][:H],W['Whh'][H:2*H],W['Whh'][2*H:3*H],W['Whh'][3*H:4*H]
    h=np.zeros(H); c=np.zeros(H); VH=[]; VC=[]
    for t in range(X.shape[0]):
        pre=W['Wih']@X[t]+W['Whh']@h+W['bih']+W['bhh']
        i=sig(pre[:H]); f=sig(pre[H:2*H]); g=np.tanh(pre[2*H:3*H]); o=sig(pre[3*H:4*H])
        cp=f*c+i*g; tc=np.tanh(cp); dtc=1-tc*tc
        di=(i*(1-i))[:,None]*Whi; df=(f*(1-f))[:,None]*Whf; dg=(1-g*g)[:,None]*Whg; do=(o*(1-o))[:,None]*Who
        dcp_dh=c[:,None]*df+g[:,None]*di+i[:,None]*dg
        dhp_dh=tc[:,None]*do+(o*dtc)[:,None]*dcp_dh
        VH.append(np.linalg.svd(dhp_dh)[2]); VC.append(np.linalg.svd(np.diag(f))[2]); h,c=o*tc,cp
    return VH,VC                       # h→h block SVD, c→c block (diagonal)
def align_from_Vt(Vt_list, ks):
    return np.array([np.mean([np.linalg.svd(Vt_list[l][-k:]@Vt_list[l+1][-k:].T,compute_uv=False).mean()
                    for l in range(len(Vt_list)-1)]) for k in ks])
def gen_adding(T,rng):
    v=rng.random(T).astype('float32'); mk=np.zeros(T,'float32'); mk[rng.choice(T,2,replace=False)]=1
    return np.stack([v,mk],-1)
H=256;T=200;ks=list(range(1,64)); HH=[];CC=[]
for s in range(6):
    W=lstm_init(H,seed=100+s); X=gen_adding(T,np.random.default_rng(500+s))
    vh,vc=Vts(W,X); HH.append(align_from_Vt(vh,ks)); CC.append(align_from_Vt(vc,[k for k in ks if k<H]))
    print(f"  seq {s+1}/6 done",flush=True)
hh=np.array(HH).mean(0); cc=np.array(CC).mean(0); base=np.array([np.sqrt(k/(2*H)) for k in ks])
show=[1,2,4,8,16,32,63]
print(f"\nUNTRAINED H=256, T=200 — align(k) (init/architectural control):")
print(f"  k          "+" ".join(f"{k:>5}" for k in show))
print(f"  baseline   "+" ".join(f"{np.sqrt(k/(2*H)):5.2f}" for k in show))
print(f"  INIT h→h   "+" ".join(f"{hh[ks.index(k)]:5.2f}" for k in show))
print(f"  INIT c→c   "+" ".join(f"{cc[ks.index(k)]:5.2f}" for k in show if k<H))
sh=ks[int(np.argmax((hh-base)[:-1]-(hh-base)[1:]))]
print(f"\n  shoulder k={sh}; align@4={hh[3]:.2f} @32={hh[31]:.2f} @63={hh[62]:.2f} (base@63={np.sqrt(63/512):.2f})")
print(f"  → {'LOW-RANK / architectural (high to large k, no small-k shoulder)' if hh[31]>0.55 else 'annihilation shape (falls to base by k=32)'}")
print("DONE",flush=True)
