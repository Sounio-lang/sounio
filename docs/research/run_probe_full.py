# Stress-test the d=56 positive: the full align(k) curve + the c→c control + the untrained-init control.
# A genuine LEARNED subspace-death claim needs ALL of: (i) h→h shoulder at SMALL k with a healthy bulk above
# (annihilation shape, not the large-k rise of low rank); (ii) h→h alignment >> the c→c diagonal block
# (architectural free control); (iii) trained h→h >> untrained-init h→h (learning, not architecture).
import importlib.util as u, numpy as np, torch
spec=u.spec_from_file_location("m","train_and_probe_lstm.py"); m=u.module_from_spec(spec); spec.loader.exec_module(m)
dev='cpu'
def curves(net, n_seq=40, T=30, seed=1):
    H=net.H; ks=list(range(1,H)); base=np.sqrt(np.array(ks)/H)
    HH=[]; CC=[]
    for s in range(n_seq):
        X,_=m.gen_adding(1,T,np.random.default_rng(30000+s)); Xt=torch.tensor(X[0],device=dev)
        J=m.state_jacobians(net,Xt,dev)
        hh=[j[:H,:H] for j in J]; cc=[j[H:,H:] for j in J]
        ah,_=m.align_curve(hh,ks); ac,_=m.align_curve(cc,ks); HH.append(ah); CC.append(ac)
    return ks,base,np.array(HH).mean(0),np.array(CC).mean(0)
print("training…",flush=True); net=m.train(H=40,T=30,steps=2500,dev=dev)
print("untrained control net…",flush=True); net0=m.Net(40).to(dev)
ks,base,hh,cc=curves(net); _,_,hh0,cc0=curves(net0)
sh=lambda a: ks[int(np.argmax((a-base)[:-1]-(a-base)[1:]))]
print("\nalign(k), k=1..12   (baseline √(k/H)):")
print("  k        " + " ".join(f"{k:>5}" for k in ks[:12]))
print("  baseline " + " ".join(f"{v:5.2f}" for v in base[:12]))
print("  TRAINED h→h " + " ".join(f"{v:5.2f}" for v in hh[:12]) + f"   shoulder k={sh(hh)}")
print("  TRAINED c→c " + " ".join(f"{v:5.2f}" for v in cc[:12]) + f"   (diagonal architectural control)")
print("  INIT   h→h  " + " ".join(f"{v:5.2f}" for v in hh0[:12]) + f"   shoulder k={sh(hh0)}")
print("  INIT   c→c  " + " ".join(f"{v:5.2f}" for v in cc0[:12]))
print(f"\nverdict checks:")
print(f"  (i)  trained h→h shoulder at small k with healthy bulk?  shoulder k={sh(hh)}, align@4={hh[3]:.2f}, align@12={hh[11]:.2f} (bulk should fall toward baseline {base[11]:.2f})")
print(f"  (ii) trained h→h >> c→c at k=4?   h→h {hh[3]:.2f}  vs  c→c {cc[3]:.2f}   {'PASS' if hh[3]-cc[3]>0.1 else 'FAIL — c→c is also aligned (architectural)'}")
print(f"  (iii) trained h→h >> untrained-init h→h at k=4?   trained {hh[3]:.2f}  vs  init {hh0[3]:.2f}   {'PASS (learned)' if hh[3]-hh0[3]>0.1 else 'FAIL — already present at init (architecture, not learning)'}")
print("DONE",flush=True)
