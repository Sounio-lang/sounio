#!/usr/bin/env python3
# PROBE — does a real trained checkpoint carry the sedenion signature, or is its vanishing-gradient just
# magnitude? Test (OPUS-4.8-EXTRA): compute the spectrum of the composed Jacobian (J = J_L…J_1, or ∂h_T/∂h_0
# for a recurrence) along a few input paths and classify:
#   • LOW-MULT-GAP  — a few singular values collapse behind a large gap while the BULK stays healthy
#                     → structural annihilation; residual/norm tricks preserve norm and do nothing for it;
#                       INVISIBLE to dynamical isometry (which asks if the WHOLE spectrum is near 1).
#   • UNIFORM-SLIDE — the whole spectrum decays smoothly, no dominant gap → magnitude (classical vanishing
#                     gradient); dynamical isometry (Saxe/Pennington) is the right frame; residuals help.
#   • RANK-COLLAPSE — nearly all modes dead, 1–2 survive → whole representation degenerates (Dong et al.).
# No training required. Core classifier is pure numpy (validated below); the Jacobian is via torch autograd.
import numpy as np
np.seterr(all='ignore')

# ============================ the transferable core: spectral classifier ============================
def classify_spectrum(sv, dead=1e-6):
    """sv: singular values of the composed Jacobian. Returns verdict + metrics. No torch."""
    s=np.sort(np.asarray(sv,float))[::-1]; s=s/max(s[0],1e-30); n=len(s)
    logs=np.log10(s+1e-30)
    g=logs[:-1]-logs[1:]                      # consecutive log-drops (>=0 for a descending spectrum)
    gi=int(np.argmax(g)); max_gap=float(g[gi])         # the dominant gap: between mode gi and gi+1
    n_above=gi+1; n_below=n-n_above                     # modes above / below the dominant gap
    bulk_span=float(logs[0]-logs[gi])                   # log-range of the surviving bulk (above the gap)
    n_deadmodes=int((s<dead).sum())
    gap_dominance=max_gap/(bulk_span+1e-9)              # gap vs the spread of the bulk it sits above
    typ_gap=float(np.median(g))                         # typical per-mode decay
    # verdict
    has_gap = (max_gap>2.0) and (gap_dominance>1.0)     # a clean cliff, not part of a continuous slide
    if has_gap and n_above>=2 and n_below<=n/2:
        verdict='LOW-MULT-GAP (structural annihilation)'
    elif n_above<=2 and n_deadmodes>=n/2:
        verdict='RANK-COLLAPSE (whole rep → low rank)'
    else:
        verdict='UNIFORM-SLIDE (magnitude / dynamical isometry)'
    return dict(verdict=verdict, n=n, n_healthy_bulk=n_above, n_dead_behind_gap=n_below,
                n_dead_modes=n_deadmodes, max_gap_decades=round(max_gap,2),
                bulk_span_decades=round(bulk_span,2), gap_dominance=round(gap_dominance,2),
                typical_gap_decades=round(typ_gap,3), gap_at_rank=n_above)

def report(sv,label):
    r=classify_spectrum(sv)
    print(f"  {label}")
    print(f"    log10(σ/σmax): {np.round(np.log10(np.sort(np.asarray(sv,float))[::-1]/max(sv)+1e-30),1)}")
    print(f"    → {r['verdict']}")
    print(f"      healthy bulk {r['n_healthy_bulk']}/{r['n']} | dead-behind-gap {r['n_dead_behind_gap']} | "
          f"dominant gap {r['max_gap_decades']} dec (rank {r['gap_at_rank']}→{r['gap_at_rank']+1}) | "
          f"gap/bulk-spread {r['gap_dominance']}\n")
    return r

# ============================ Jacobian of a real checkpoint (torch) ============================
def io_jacobian(forward_fn, x):
    """J = ∂ forward_fn(x) / ∂ x, flattened to a 2-D matrix. forward_fn returns a tensor; x requires_grad.
    For a feedforward net use forward_fn = model (input→output). For a recurrence, make forward_fn map the
    initial state h0 → final state hT (its Jacobian is ∏_t ∂h_{t+1}/∂h_t). For a big model, first restrict
    to a submodule / a tractable input & output block so the Jacobian is square-ish and affordable."""
    import torch
    x=x.detach().clone().requires_grad_(True)
    J=torch.autograd.functional.jacobian(forward_fn, x, vectorize=True)
    J=J.reshape(-1, x.numel()).cpu().numpy()
    return np.linalg.svd(J, compute_uv=False)

def probe_checkpoint(build_model, ckpt_path, make_forward, sample_inputs, n_paths=8):
    """build_model()->nn.Module ; ckpt_path loaded into it ; make_forward(model)->forward_fn ;
       sample_inputs(k)->list of input tensors. Aggregates the verdict over n_paths inputs."""
    import torch
    model=build_model(); sd=torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(sd.get('state_dict', sd), strict=False); model.eval()
    fwd=make_forward(model); verdicts={}
    for x in sample_inputs(n_paths):
        sv=io_jacobian(fwd, x); v=classify_spectrum(sv)['verdict']; verdicts[v]=verdicts.get(v,0)+1
    print(f"checkpoint {ckpt_path}: verdicts over {n_paths} paths → {verdicts}")
    return verdicts


# ============================ the MECHANISM: subspace alignment (the valid discriminant) ============================
# The product-spectrum gap is NOT sufficient: a stack of matrices each with a low-mult tail but with the
# dead directions ROTATING between factors produces a large product gap WITHOUT any composing structure
# (validated: the rotating control's gap_dominance exceeds genuine structure). The gap only means
# "composing annihilation" if consecutive dead subspaces are ALIGNED. Measure that directly.
def dying_subspace(J, k=4):
    import numpy as np
    U,sv,Vt=np.linalg.svd(J); return Vt[-k:]                     # bottom-k right singular vectors
def subspace_alignment(J_list, k=4):
    """mean cos(principal angle) between the dying k-subspaces of consecutive Jacobians.
    ~1 = aligned (dead directions compose → genuine structural annihilation);
    ~sqrt(k/dim) = baseline (dead directions rotate → gap is an artifact, NOT structure)."""
    import numpy as np
    cs=[np.linalg.svd(dying_subspace(J_list[l],k)@dying_subspace(J_list[l+1],k).T,compute_uv=False).mean()
        for l in range(len(J_list)-1)]
    return float(np.mean(cs)), np.sqrt(k/J_list[0].shape[1])     # (measured, baseline)
def gap_vs_T(J_list, Ts=(1,2,4,8,16,32)):
    """gap_dominance of the product of the first T factors — report the CURVE, not a point."""
    import numpy as np
    def gd(A):
        sv=np.linalg.svd(A,compute_uv=False); s=np.sort(sv)[::-1]; s/=s[0]; lg=np.log10(s+1e-30)
        g=lg[:-1]-lg[1:]; gi=int(np.argmax(g)); return float(g[gi])/(float(lg[0]-lg[gi])+1e-9)
    P=np.eye(J_list[0].shape[0]); out={}
    for t in range(1,max(Ts)+1):
        P=J_list[t-1]@P
        if t in Ts: out[t]=gd(np.linalg.svd(P,compute_uv=False)) if False else gd(P)
    return out

# ============================ self-contained validation (numpy, runs now) ============================
if __name__=='__main__':
    import sys
    if '--checkpoint' in sys.argv:
        print("To probe a real checkpoint, import probe_checkpoint(...) and pass your model builder,\n"
              "the .pt path, a make_forward that returns the composition you care about (input→output, or\n"
              "h0→hT for a recurrence), and a sample_inputs generator. See docstrings.")
        sys.exit(0)
    print("Validating the classifier on three known spectra (the probe's transferable core):\n")
    rng=np.random.default_rng(0)
    # (1) sedenion-structured composed Jacobian (from spectral_signature.py Test B): 4/8/4 tiers, cliff
    sed=np.array([1,1,1,1]+[10**-1.8]*8+[10**-14.3]*4)
    report(sed, "(1) SEDENION-structured composed Jacobian (few die behind a cliff, bulk lives)")
    # (2) real Gaussian deep product: smooth continuous decay
    real=10**np.array([0,-0.3,-0.6,-0.7,-1.1,-1.4,-1.9,-2.2,-2.5,-2.6,-3.2,-3.6,-5.1,-5.6,-9.3,-10.6])
    report(real, "(2) REAL Gaussian deep product (continuous slide, no clean cliff)")
    # (3) rank collapse to ~1 (attention-style)
    rc=10**np.array([0]+[-6.0-0.3*i for i in range(15)])
    report(rc, "(3) RANK-COLLAPSE toward rank 1 (whole representation degenerates)")
    print("If the classifier calls (1) LOW-MULT-GAP, (2) UNIFORM-SLIDE, (3) RANK-COLLAPSE, it is calibrated;")
    print("then point probe_checkpoint(...) at a real .pt and read the verdict over a few input paths.")
