#!/usr/bin/env python3
"""Independent check of Singh (2508.10131) Table I: parameter-free sqrt-mass-ratio
formulas (delta^2=3/8) vs experiment. Values are sqrt(mass ratios). Paper compares
at M_Z; lepton ratios barely run so we also check those at pole scale independently.
"""
import numpy as np
d = np.sqrt(3/8)                      # delta = 0.6123724
A = (1+d)/(1-d)                       # recurring left-edge factor

# Paper's closed forms (sqrt mass ratios)
theory = {
 "sqrt(m_tau/m_mu)": A,
 "sqrt(m_mu/m_e)":   A*(d+1/3)/(d-1/3),
 "sqrt(m_s/m_d)":    A,
 "sqrt(m_b/m_s)":    A*(1+d),
 "sqrt(m_c/m_u)":    (2/3+d)/(2/3-d),
 "sqrt(m_t/m_c)":    (2/3)/(2/3-d),
 "sqrt(m_u/m_e) [gen1]": 2.0,
 "sqrt(m_d/m_e) [gen1]": 3.0,
}
# Experimental @ M_Z as quoted in the paper's Table I (col EXPERIMENTAL @MZ)
exp_MZ = {
 "sqrt(m_tau/m_mu)": (4.11930, 0.00006),
 "sqrt(m_mu/m_e)":   (14.543, 0.001),
 "sqrt(m_s/m_d)":    (4.46, 0.25),
 "sqrt(m_b/m_s)":    (7.37, 0.34),
 "sqrt(m_c/m_u)":    (22.43, 2.45),
 "sqrt(m_t/m_c)":    (16.65, 1.14),     # RG range 15.96-17.19
 "sqrt(m_u/m_e) [gen1]": (1.59, 0.14),
 "sqrt(m_d/m_e) [gen1]": (3.02, 0.10),  # ~low-scale; at MZ ~2.34
}

print(f"delta = sqrt(3/8) = {d:.6f}   A=(1+d)/(1-d) = {A:.5f}\n")
print(f"{'ratio':24s} {'theory':>9s} {'exp@MZ':>9s} {'dev%':>7s} {'within err?':>11s}")
print("-"*64)
for k in theory:
    t=theory[k]; e,se=exp_MZ[k]
    dev=100*(t-e)/e
    nsig=abs(t-e)/se if se>0 else float('inf')
    ok = "yes" if nsig<=2 else f"NO ({nsig:.0f}sig)"
    print(f"{k:24s} {t:9.4f} {e:9.4f} {dev:+7.1f} {ok:>11s}")

# Independent lepton check at pole (lepton masses ~scale-independent in ratio)
me,mmu,mtau=0.51099895,105.6583755,1776.86
print("\nIndependent pole-scale lepton checks (no RG needed):")
print(f"  sqrt(m_tau/m_mu): theory {A:.4f}  pole-exp {np.sqrt(mtau/mmu):.4f}  dev {100*(A/np.sqrt(mtau/mmu)-1):+.1f}%")
print(f"  sqrt(m_mu/m_e):   theory {theory['sqrt(m_mu/m_e)']:.4f}  pole-exp {np.sqrt(mmu/me):.4f}  dev {100*(theory['sqrt(m_mu/m_e)']/np.sqrt(mmu/me)-1):+.1f}%")

# Koide: paper predicts K_th=0.66916 post-breaking; observed leptons:
s=np.sqrt([me,mmu,mtau]); Kobs=(me+mmu+mtau)/s.sum()**2
print(f"\nKoide(leptons): observed {Kobs:.5f} | exact 2/3 {2/3:.5f} | Singh K_th 0.66916")
print(f"  -> observed is closer to plain 2/3 (|d|={abs(Kobs-2/3):.5f}) than to Singh's offset (|d|={abs(Kobs-0.66916):.5f})")
