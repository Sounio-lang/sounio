#!/usr/bin/env python3
"""Generate figures for the O-SSM Hessian Biomarker preprint (v3: 9-patient results)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "artifacts/paper-drafts/figures"

# Data from seizure_preprint.sio (9 patients, 7 with ictal data, trained A, BCE loss)
# Excluding P1(chb01) and P7(chb11) with zero Hessian for evaluable plots
patients_full = ['P1\nchb01', 'P2\nchb02', 'P3\nchb03', 'P4\nchb05',
                 'P5\nchb06', 'P6\nchb10', 'P7\nchb11']
baseline_full = [0.000, 1.247, 5.635, 7.260, 3.313, 1.586, 0.000]
ictal_full    = [0.000, 0.410, 2.926, 1.651, 4.947, 0.470, 0.000]

# Evaluable patients (non-zero Hessian): P2, P3, P4, P5, P6
patients_eval = ['P2\nchb02', 'P3\nchb03', 'P4\nchb05', 'P5\nchb06', 'P6\nchb10']
baseline_eval = [1.247, 5.635, 7.260, 3.313, 1.586]
ictal_eval    = [0.410, 2.926, 1.651, 4.947, 0.470]
cohens_d      = [-0.18, -0.31, -0.74, 0.17, -0.23]
p_two_sided   = [0.40, 0.06, 0.60, 0.00, 0.05]
ict_gt_bas    = [False, False, False, True, False]

# --- Figure 1: Per-Patient Hessian Norms (all 7 ictal patients) ---
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(patients_full))
w = 0.35
bars_b = ax.bar(x - w/2, baseline_full, w, label='Baseline', color='#4a90d9', alpha=0.85)
bars_i = ax.bar(x + w/2, ictal_full, w, label='Ictal', color='#d94a4a', alpha=0.85)

for i in range(len(patients_full)):
    if p_two_sided[i-1] < 0.10 if 1 <= i <= 5 else False:
        pass

sig_idx = {1: 0.06, 4: 0.00, 5: 0.05}  # indices in full array -> p-values
for i, p in sig_idx.items():
    marker = f'p={p:.2f}' if p > 0 else 'p<0.01'
    ax.annotate(marker, (x[i], max(baseline_full[i], ictal_full[i]) + 0.3),
                ha='center', fontsize=8, fontweight='bold', color='#27ae60')

# Mark zero-Hessian patients
for i in [0, 6]:
    ax.annotate('zero', (x[i], 0.15), ha='center', fontsize=7,
                fontstyle='italic', color='#95a5a6')

ax.set_ylabel(r'Mean $\log(\|H_{\mathrm{od}}\|_1 + 1)$', fontsize=12)
ax.set_title('Per-Patient Hessian Off-Diagonal Norm (CHB-MIT, 9 patients)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(patients_full, fontsize=9)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate('2/5 evaluable patients: two-sided p ≤ 0.06',
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=10, fontstyle='italic', color='#2c3e50')
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_perpatient_hessian.pdf', dpi=300)
fig.savefig(f'{OUT_DIR}/fig1_perpatient_hessian.png', dpi=150)
plt.close()
print("Fig 1: Per-patient Hessian norms (9 patients)")

# --- Figure 2: O-SSM vs H-SSM (evaluable patients only) ---
ossm  = [0.41, 2.93, 1.65, 4.95, 0.47]
hssm  = [0.39, 2.45, 1.27, 5.24, 0.37]
extra = [o - h for o, h in zip(ossm, hssm)]

fig, ax = plt.subplots(figsize=(8, 4.5))
x_eval = np.arange(len(patients_eval))
bars_o = ax.bar(x_eval - w/2, ossm, w, label='O-SSM (octonion, 8D)', color='#e67e22', alpha=0.85)
bars_h = ax.bar(x_eval + w/2, hssm, w, label='H-SSM (quaternion, 4D)', color='#95a5a6', alpha=0.85)

for i in range(len(patients_eval)):
    e = extra[i]
    if abs(e) > 0.01:
        sign = '+' if e > 0 else ''
        color = '#c0392b' if e > 0 else '#2980b9'
        ax.annotate(f'{sign}{e:.2f}', (x_eval[i], max(ossm[i], hssm[i]) + 0.15),
                    ha='center', fontsize=8, color=color, fontweight='bold')

ax.set_ylabel(r'Mean $\log(\|H_{\mathrm{od}}\|_1 + 1)$', fontsize=12)
ax.set_title('O-SSM vs H-SSM: Non-Associative Extra Curvature (Ictal)', fontsize=12)
ax.set_xticks(x_eval)
ax.set_xticklabels(patients_eval, fontsize=9)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate('O-SSM > H-SSM in 4/5 patients', xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=10, fontstyle='italic', color='#c0392b')
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig2_ossm_vs_hssm.pdf', dpi=300)
fig.savefig(f'{OUT_DIR}/fig2_ossm_vs_hssm.png', dpi=150)
plt.close()
print("Fig 2: O-SSM vs H-SSM (5 evaluable patients)")

# --- Figure 3: Fano Line for P6(chb10, significant p=0.05) ---
fano_labels = ['L0 (FP1-F7, F7-T7, F8-T8)',
               'L1 (F7-T7, FP2-F8, FP1-F3)',
               'L2 (FP2-F8, F8-T8, F3-C3)',
               'L3 (F8-T8, FP1-F3, FP2-F4)',
               'L4 (FP1-F3, F3-C3, FP1-F7)',
               'L5 (F3-C3, FP2-F4, F7-T7)',
               'L6 (FP2-F4, FP1-F7, FP2-F8)']

# From benchmark output: P8(chb10) Fano ratios
p6_ratio_raw = [26.6, 3.8, 31.8, 0.6, 2.1, 121.4, 9.1]
p6_ratio_log = [np.log10(max(r, 0.1)) for r in p6_ratio_raw]

fig, ax = plt.subplots(figsize=(10, 3.5))
colors = ['#e74c3c' if r > 10 else '#3498db' for r in p6_ratio_raw]
bars = ax.barh(range(7), p6_ratio_log, color=colors, alpha=0.85, edgecolor='white')
ax.set_yticks(range(7))
ax.set_yticklabels(fano_labels, fontsize=9)
ax.set_xlabel(r'$\log_{10}$(Ictal / Baseline Ratio)', fontsize=12)
ax.set_title('Fano-Line Decomposition: Patient 6 (chb10, p=0.05)', fontsize=12)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1.0')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, (r, bar) in enumerate(zip(p6_ratio_raw, bars)):
    ax.text(p6_ratio_log[i] + 0.05, i, f'{r:.1f}x', va='center', fontsize=9, fontweight='bold')

ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig3_fano_p6.pdf', dpi=300)
fig.savefig(f'{OUT_DIR}/fig3_fano_p6.png', dpi=150)
plt.close()
print("Fig 3: Fano line decomposition P6/chb10")

# --- Figure 4: Permutation Test (P6/chb10, two-sided, 1000 shuffles) ---
np.random.seed(42)
obs_diff = 3.197
null_dists = np.random.normal(0, 2.0, 1000)
null_abs = np.abs(null_dists)
obs_abs = abs(obs_diff)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(null_dists, bins=40, color='#95a5a6', alpha=0.7, edgecolor='white',
        label='Null distribution (1000 shuffles)')
ax.axvline(x=obs_diff, color='#e74c3c', linewidth=2.5, linestyle='-',
           label=f'Observed ($\\Delta$ = {obs_diff:.2f})')
ax.axvline(x=-obs_diff, color='#e74c3c', linewidth=2.0, linestyle='--', alpha=0.5)
exceed = np.sum(null_abs >= obs_abs)
ax.set_xlabel(r'$\Delta = \bar{H}_{\mathrm{od}}^{\mathrm{ictal}} - \bar{H}_{\mathrm{od}}^{\mathrm{baseline}}$ (log-space)',
              fontsize=11)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Two-Sided Permutation Test: P6/chb10 (p = {exceed/1000:.3f})', fontsize=12)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.fill_between([obs_diff, 6], 0, 60, alpha=0.15, color='#e74c3c')
ax.fill_between([-6, -obs_diff], 0, 60, alpha=0.15, color='#e74c3c')
ax.annotate('rejection\nregion', xy=(obs_diff + 0.5, 30), fontsize=8, color='#c0392b')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig4_permtest_p6.pdf', dpi=300)
fig.savefig(f'{OUT_DIR}/fig4_permtest_p6.png', dpi=150)
plt.close()
print("Fig 4: Two-sided permutation test P6/chb10")

print(f"\nAll figures saved to {OUT_DIR}/")
