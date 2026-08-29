#!/usr/bin/env python3
"""Piloto 1 v2 on BEAGLE node: 3 fields (PMI gpt2, jump trivial, Ollivier-Ricci curvature), corrected
permutation p-values (c+k sweep), AND the positive control / sensitivity floor delta*. Writes results+curves
to PILOT_BASE/results. Faithful to PREREG addendum 1 items 2 & 3.1 & 3.3."""
import os, sys, re, json, base64, numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
BASE=os.environ.get("PILOT_BASE","/tmp/pres"); os.makedirs(f"{BASE}/results",exist_ok=True)
os.environ["HF_HOME"]=os.environ.get("HF_HOME",f"{BASE}/hf")
import certifi, ssl, urllib.request
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where(); CTX=ssl.create_default_context(cafile=certifi.where())
import torch
torch.set_num_threads(int(os.environ.get("NTHREADS","16")))
DEV="cuda" if torch.cuda.is_available() else "cpu"
if DEV=="cuda":
    free,_=torch.cuda.mem_get_info()
    if free<2e9: DEV="cpu"; print(f"GPU busy -> CPU",flush=True)
def log(*a): print(*a,flush=True)
log("device",DEV)
def fetch(series):
    url=f"https://www.dreambank.net/random_sample.cgi?series={series}&min=1&max=100000&n=20000"
    raw=urllib.request.urlopen(url,context=CTX,timeout=120).read().decode('latin-1')
    import html as H
    out=[]
    for s in re.findall(r'<span>\s*(#\d+.*?)</span>', raw, re.S):
        m=re.match(r'#(\d+)\s*(?:\(([^)]*)\))?', s)
        if not m: continue
        txt=re.sub(r'^#\d+\s*(?:\([^)]*\))?','',s,count=1); txt=re.sub(r'<[^>]+>',' ',txt)
        txt=re.sub(r'\(\d+\s+words?\)\s*$','',txt).strip(); txt=H.unescape(re.sub(r'\s+',' ',txt))
        if txt: out.append((int(m.group(1)),txt))
    out.sort(key=lambda x:x[0]); return [t for _,t in out]
def sents(texts):
    S=[]
    for t in texts: S+=[x.strip() for x in re.split(r'(?<=[.!?])\s+',t) if len(x.split())>=3]
    return S

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok=GPT2TokenizerFast.from_pretrained("gpt2"); model=GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEV)
BOS=tok.bos_token_id; EOS=tok.eos_token_id
@torch.no_grad()
def mean_logp(seqs,tl,bs=12):
    res=np.zeros(len(seqs))
    for b in range(0,len(seqs),bs):
        ch=seqs[b:b+bs]; t=tl[b:b+bs]; mx=max(len(s) for s in ch)
        ids=torch.full((len(ch),mx),EOS,dtype=torch.long); msk=torch.zeros((len(ch),mx),dtype=torch.long)
        for i,s in enumerate(ch): ids[i,:len(s)]=torch.tensor(s); msk[i,:len(s)]=1
        ids=ids.to(DEV); msk=msk.to(DEV)
        lp=torch.log_softmax(model(ids,attention_mask=msk).logits,-1)
        for i,s in enumerate(ch):
            L=len(s); tt=t[i]; idx=torch.arange(L-tt,L,device=DEV)
            res[b+i]=lp[i,idx-1,ids[i,idx]].mean().item()
    return res
def pmi_field(S):
    enc=[tok.encode(s) for s in S]; encsp=[tok.encode(" "+s) for s in S]; N=len(S)
    us=[[BOS]+enc[i] for i in range(N)]; ut=[len(enc[i]) for i in range(N)]
    cs=[[BOS]+enc[0]]; ct=[len(enc[0])]
    for i in range(1,N):
        cs.append(([BOS]+enc[i-1]+encsp[i])[-1024:]); ct.append(len(encsp[i]))
    s=(mean_logp(us,ut)-mean_logp(cs,ct)); s[0]=np.nan; return s


from sklearn.neighbors import NearestNeighbors
def mutual_knn(E,k):
    _,idx=NearestNeighbors(n_neighbors=k+1,metric='cosine').fit(E).kneighbors(E)
    nbr=[set(r[1:]) for r in idx]; adj=[[] for _ in range(len(E))]
    for i in range(len(E)):
        for j in nbr[i]:
            if i in nbr[j]: adj[i].append(j)
    return adj
def sweep(order,adj):
    n=len(order); par=list(range(n)); sz=[1]*n; pres=np.zeros(n,bool)
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    largest=0; rho=np.empty(n)
    for m,v in enumerate(order,1):
        pres[v]=True
        for u in adj[v]:
            if pres[u]:
                a,b=find(u),find(v)
                if a!=b:
                    if sz[a]<sz[b]: a,b=b,a
                    par[b]=a; sz[a]+=sz[b]
        largest=max(largest,sz[find(v)]); rho[m-1]=largest/m
    return rho
def _perm_bank(adjv,n,rng,B):
    P=np.empty((B,n))
    for b in range(B): P[b]=sweep(rng.permutation(n),adjv)
    return P
def analyze_field(s,adj,rng,B=200):
    """Returns dict incl empirical p (min over c of frac perms as-or-more disconnected), longest below-band run."""
    vi=np.where(~np.isnan(s))[0]; remap={v:i for i,v in enumerate(vi)}
    adjv=[[remap[u] for u in adj[v] if u in remap] for v in vi]; sv=s[vi]; n=len(vi)
    rho=sweep(np.argsort(sv,kind='stable'),adjv)
    P=_perm_bank(adjv,n,rng,B); lo=np.percentile(P,2.5,0); med=np.percentile(P,50,0)
    below=rho<lo; start=max(1,int(0.05*n)); best=cur=0
    for m in range(start,n):
        cur=cur+1 if below[m] else 0
        if cur>best: best=cur
    T=lambda r: float(np.maximum(0.0, med[start:]-r[start:]).sum())          # integrated deficit below null median
    Tobs=T(rho); Tp=np.array([T(P[b]) for b in range(B)])
    p_corr=float((1+np.sum(Tp>=Tobs))/(B+1))                                  # c-sweep-corrected permutation p
    return dict(n=int(n),delta_min=float((rho-med)[start:].min()),longest_below_frac=float(best/max(1,n-start)),
                p_corrected=p_corr, rho=rho, null_lo=lo)
def fiedler_separator(adj,n):
    """Spectral bisection -> vertex separator (R-side boundary): removing it disconnects the graph into 2 large comps."""
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    rows=[];cols=[]
    for i in range(n):
        for j in adj[i]:
            rows.append(i);cols.append(j)
    if not rows: return None,None,None
    A=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); A=((A+A.T)>0).astype(float)
    deg=np.asarray(A.sum(1)).ravel(); L=sp.diags(deg)-A
    try:
        vals,vecs=spla.eigsh(L,k=2,which='SM',maxiter=5000)
        f=vecs[:,1]
    except Exception:
        return None,None,None
    Lset=set(np.where(f<=np.median(f))[0].tolist()); Rset=set(range(n))-Lset
    sep=[i for i in Rset if any(j in Lset for j in adj[i])]     # R nodes adjacent to L = vertex cut
    Lsize=len(Lset); Rint=len(Rset)-len(sep)
    return sep, Lsize, Rint
def positive_control(s_base,adj,rng,deltas=(4,3,2,1.5,1.0,0.75,0.5,0.35,0.25,0.15),B=200):
    """Inject barrier: raise s by delta*std on a Fiedler vertex separator -> {s<=c} splits in 2. Sweep delta -> delta*.
    Detection = longest below-null run >= 0.10 (an order above the trivial ~0.03 floor)."""
    vi=np.where(~np.isnan(s_base))[0]; remap={v:i for i,v in enumerate(vi)}
    adjv=[[remap[u] for u in adj[v] if u in remap] for v in vi]; n=len(vi); sb=s_base[vi].astype(float)
    sep,Lsz,Rint=fiedler_separator(adjv,n)
    if sep is None or Lsz<0.15*n or Rint<0.15*n or len(sep)==0:
        return dict(ok=False,reason=f"no balanced vertex separator (L={Lsz},Rint={Rint},|sep|={len(sep) if sep else 0})")
    sd=sb.std()+1e-9; out=[]; dstar=None
    for d in deltas:
        s2=sb.copy(); s2[sep]+=d*sd
        r=analyze_field_arr(s2,adjv,rng,B)
        det=r['longest_below_frac']>=0.10
        out.append((float(d),r['longest_below_frac'],float(r['p_min']),det))
        if det: dstar=float(d)                      # smallest detected so far (list descends)
    return dict(ok=True, sep_frac=len(sep)/n, L_frac=Lsz/n, Rint_frac=Rint/n, sweep=out, delta_star=dstar)
def analyze_field_arr(s,adjv,rng,B):
    n=len(s); rho=sweep(np.argsort(s,kind='stable'),adjv)
    P=_perm_bank(adjv,n,rng,B); lo=np.percentile(P,2.5,0); med=np.percentile(P,50,0)
    below=rho<lo; start=max(1,int(0.05*n)); best=cur=0
    for m in range(start,n):
        cur=cur+1 if below[m] else 0; best=max(best,cur)
    T=lambda r: float(np.maximum(0.0, med[start:]-r[start:]).sum())
    Tobs=T(rho); Tp=np.array([T(P[b]) for b in range(B)]); p=float((1+np.sum(Tp>=Tobs))/(B+1))
    return dict(longest_below_frac=float(best/max(1,n-start)), p_min=p)


# ---- data ----
log("fetching DreamBank…")
A=sents(fetch("b")); B=sents(fetch("norms-f")+fetch("norms-m"))
log(f"A={len(A)} B={len(B)}")
from sentence_transformers import SentenceTransformer
EMB=SentenceTransformer("all-MiniLM-L6-v2",device=DEV)
import ot
def ollivier_field(adj,E,alpha=0.5):
    n=len(E); kap=np.zeros(n); cnt=np.zeros(n)
    for i in range(n):
        Ni=adj[i]
        if not Ni: continue
        for j in Ni:
            if j<i: continue
            Nx=[i]+list(adj[i]); Ny=[j]+list(adj[j])
            mx=np.full(len(Nx),(1-alpha)/max(1,len(adj[i]))); mx[0]=alpha
            my=np.full(len(Ny),(1-alpha)/max(1,len(adj[j]))); my[0]=alpha
            mx/=mx.sum(); my/=my.sum()
            C=1-E[Nx]@E[Ny].T
            w=ot.emd2(mx,my,np.ascontiguousarray(C))
            d=1-float(E[i]@E[j]); k=1-w/d if d>1e-9 else 0.0
            kap[i]+=k; kap[j]+=k; cnt[i]+=1; cnt[j]+=1
    kap=np.where(cnt>0,kap/np.maximum(cnt,1),0.0)
    s=-kap; s[0]=np.nan; return s   # high s = negative curvature (bridge/bottleneck)
def run(S,name):
    log(f"[{name}] PMI…"); s_pmi=pmi_field(S)
    log(f"[{name}] embed…"); E=np.asarray(EMB.encode(S,batch_size=256,show_progress_bar=False,normalize_embeddings=True),dtype=np.float32)
    s_jump=np.empty(len(S)); s_jump[0]=np.nan; s_jump[1:]=1.0-np.sum(E[1:]*E[:-1],1)
    rng=np.random.default_rng(0); res={"sample":name,"N":len(S),"device":DEV,"LM":"gpt2","fields":{}}
    adjs={k:mutual_knn(E,k) for k in [5,10,20]}
    log(f"[{name}] curvature…"); s_curv=ollivier_field(adjs[10],E)
    for fn,s in [("PMI",s_pmi),("jump",s_jump),("curv",s_curv)]:
        res["fields"][fn]={}
        for k in [5,10,20]:
            r=analyze_field(s,adjs[k],rng,B=200)
            res["fields"][fn][f"k{k}"]={"n":r["n"],"delta_min":r["delta_min"],
                "longest_below_frac":r["longest_below_frac"],"p_corrected":r["p_corrected"]}
            log(f"  {name} {fn} k={k}: longest_below={r['longest_below_frac']:.3f} p_corr={r['p_corrected']:.3f}")
    # positive control on the PMI graph (k=10): sensitivity floor delta*
    log(f"[{name}] positive control…")
    pcres=positive_control(s_pmi,adjs[10],rng,B=200)
    res["positive_control"]=pcres
    if pcres.get("ok"):
        log(f"  {name} PC: delta*={pcres['delta_star']} sep_frac={pcres['sep_frac']:.2f}")
        for d,lbf,p,det in pcres["sweep"]: log(f"    delta={d}: longest_below={lbf:.3f} p={p:.3f} {'DET' if det else '-'}")
    else: log(f"  {name} PC failed: {pcres.get('reason')}")
    json.dump(res,open(f"{BASE}/results/{name}_v2.json","w"))
    log(f"RESULTJSON {name} "+base64.b64encode(json.dumps(res).encode()).decode())
run(B,"B_norms"); run(A,"A_barb"); log("ALL_DONE")
