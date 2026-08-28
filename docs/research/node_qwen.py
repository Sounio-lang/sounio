#!/usr/bin/env python3
"""Item 3.2 — re-run the PMI field with a MODERN LM (Qwen2.5-Coder-1.5B) on GPU, same analysis
(corrected c+k permutation p). Only the PMI field depends on the LM; jump/curv/positive-control from v2
are LM-independent and stand. Loads Qwen from the BEAGLE shared HF cache."""
import os, re, json, base64, numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
BASE=os.environ.get("PILOT_BASE","/orangefs/training/sounio/pilot1"); os.makedirs(f"{BASE}/results",exist_ok=True)
import certifi, ssl, urllib.request
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where(); CTX=ssl.create_default_context(cafile=certifi.where())
import torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
def log(*a): print(*a,flush=True)
free,total=(torch.cuda.mem_get_info() if DEV=="cuda" else (0,0))
log(f"device {DEV} free={free/1e9:.1f}GB")
assert DEV=="cuda" and free>4.5e9, f"need GPU with >4.5GB free, have {free/1e9:.1f}"
LM=os.environ.get("LM_MODEL","Qwen/Qwen2.5-Coder-1.5B")
# ---- data ----
def fetch(series):
    url=f"https://www.dreambank.net/random_sample.cgi?series={series}&min=1&max=100000&n=20000"
    raw=urllib.request.urlopen(url,context=CTX,timeout=120).read().decode('latin-1'); import html as H
    out=[]
    for s in re.findall(r'<span>\s*(#\d+.*?)</span>', raw, re.S):
        m=re.match(r'#(\d+)\s*(?:\(([^)]*)\))?', s)
        if not m: continue
        t=re.sub(r'^#\d+\s*(?:\([^)]*\))?','',s,count=1); t=re.sub(r'<[^>]+>',' ',t)
        t=re.sub(r'\(\d+\s+words?\)\s*$','',t).strip(); t=H.unescape(re.sub(r'\s+',' ',t))
        if t: out.append((int(m.group(1)),t))
    out.sort(key=lambda x:x[0]); return [t for _,t in out]
def sents(texts):
    S=[]
    for t in texts: S+=[x.strip() for x in re.split(r'(?<=[.!?])\s+',t) if len(x.split())>=3]
    return S
log("fetch…"); A=sents(fetch("b")); B=sents(fetch("norms-f")+fetch("norms-m")); log(f"A={len(A)} B={len(B)}")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok=AutoTokenizer.from_pretrained(LM); model=AutoModelForCausalLM.from_pretrained(LM,torch_dtype=torch.float16).eval().to(DEV)
PRE=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
log(f"LM {LM} loaded, vocab {model.config.vocab_size}")
@torch.no_grad()
def mean_logp(seqs,tl,bs=16):
    res=np.zeros(len(seqs))
    for b in range(0,len(seqs),bs):
        ch=seqs[b:b+bs]; t=tl[b:b+bs]; mx=max(len(s) for s in ch)
        ids=torch.full((len(ch),mx),PRE,dtype=torch.long); msk=torch.zeros((len(ch),mx),dtype=torch.long)
        for i,s in enumerate(ch): ids[i,:len(s)]=torch.tensor(s); msk[i,:len(s)]=1
        ids=ids.to(DEV); msk=msk.to(DEV)
        logits=model(ids,attention_mask=msk).logits            # (B,T,V) fp16
        lse=torch.logsumexp(logits[:,:-1,:].float(),-1)        # (B,T-1)
        gath=logits[:,:-1,:].gather(-1,ids[:,1:].unsqueeze(-1)).squeeze(-1).float()  # (B,T-1)
        toklp=(gath-lse).cpu().numpy()                         # logp of token at pos j+1
        del logits,lse,gath
        for i,s in enumerate(ch):
            L=len(s); tt=t[i]; res[b+i]=toklp[i, L-1-tt:L-1].mean()   # last tt target tokens
    return res
def pmi_field(S):
    enc=[tok.encode(s,add_special_tokens=False) for s in S]; N=len(S)
    us=[[PRE]+enc[i] for i in range(N)]; ut=[len(enc[i]) for i in range(N)]
    cs=[[PRE]+enc[0]]; ct=[len(enc[0])]
    for i in range(1,N):
        cs.append(([PRE]+enc[i-1]+enc[i])[-1024:]); ct.append(len(enc[i]))
    s=(mean_logp(us,ut)-mean_logp(cs,ct)); s[0]=np.nan; return s
# ---- graph + analysis (MiniLM graph, LM-independent; corrected-p sweep) ----
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
EMB=SentenceTransformer("all-MiniLM-L6-v2",device=DEV)
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
def analyze(s,adj,rng,Bp=200):
    vi=np.where(~np.isnan(s))[0]; remap={v:i for i,v in enumerate(vi)}
    adjv=[[remap[u] for u in adj[v] if u in remap] for v in vi]; sv=s[vi]; n=len(vi)
    rho=sweep(np.argsort(sv,kind='stable'),adjv)
    P=np.array([sweep(rng.permutation(n),adjv) for _ in range(Bp)])
    lo=np.percentile(P,2.5,0); med=np.percentile(P,50,0); start=max(1,int(0.05*n))
    below=rho<lo; best=cur=0
    for m in range(start,n):
        cur=cur+1 if below[m] else 0; best=max(best,cur)
    T=lambda r: float(np.maximum(0.0,med[start:]-r[start:]).sum())
    Tobs=T(rho); Tp=np.array([T(P[b]) for b in range(Bp)]); p=float((1+np.sum(Tp>=Tobs))/(Bp+1))
    return float(best/max(1,n-start)), p
def run(S,name):
    log(f"[{name}] PMI(Qwen)…"); s=pmi_field(S)
    E=np.asarray(EMB.encode(S,batch_size=256,show_progress_bar=False,normalize_embeddings=True),dtype=np.float32)
    rng=np.random.default_rng(0); res={"sample":name,"N":len(S),"LM":LM,"PMI":{}}
    for k in [5,10,20]:
        lbf,p=analyze(s,mutual_knn(E,k),rng); res["PMI"][f"k{k}"]={"longest_below_frac":lbf,"p_corrected":p}
        log(f"  {name} PMI(Qwen) k={k}: longest_below={lbf:.3f} p_corr={p:.3f}")
    json.dump(res,open(f"{BASE}/results/{name}_qwen.json","w"))
    log(f"RESULTJSON {name} "+base64.b64encode(json.dumps(res).encode()).decode())
run(B,"B_norms"); run(A,"A_barb"); log("ALL_DONE")
