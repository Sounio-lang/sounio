#!/usr/bin/env python3
# demos/hydrogen/tools/render_site_figures.py
#
# Renders the publication-quality figures for the epistemic UHS
# site-screening brief from the DETERMINISTIC stdout of
# demos/hydrogen/site_screening.sio (lean_single engine). The demo emits
# machine-readable FIGDATA lines; this script parses them and writes:
#
#   fig_a_loss_pbox_fan.png   per-site 30-yr H2 loss p-box fan (BANDS)
#   fig_b_pgate_vs_baselines.png  P(<6 EUR/kg) per site vs the two
#                             valley baselines, corner p-box whiskers
#   fig_c_chain_waterfall.png composed-chain waterfall per site
#                             (nominal -> +subsurface -> +compressor)
#
# Usage:
#   python3 render_site_figures.py <demo_stdout.txt> <outdir>
#
# Reproducibility: Agg backend, no timestamps in PNG metadata, fixed
# figure geometry — same FIGDATA in, byte-identical PNG out (verified by
# the repo's regenerate-and-diff step; see SITE_SCREENING_BRIEF.md).
#
# Data provenance (stated in every caption): site geology from
# demos/hydrogen/site_screening_data.md (HRADF 2020; Koukouzas et al.
# 2021, Energies 14:3321; Hystories D1.4; Dotsika et al. 2021, Sci. Rep.
# 11:16291; HyUSPRe D1.3); kinetic network after Ghaedi et al. 2025 (DOI
# 10.1002/ghg.2368) abstract-level slots + public PHREEQC/PWP anchors;
# economics from trieres_chain.sio / valley_chain_epistemic.sio.

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SITES = {
    1: ("S1 South Kavala", "depleted gas field, T = 95 °C measured"),
    2: ("S2 Pentalofos Fm", "saline aquifer, 1500 m, T 52.5–69 °C"),
    3: ("S3 Eptachori Fm", "saline aquifer, 2000 m, T 65–87 °C"),
}
COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}

PROV = ("Provenance: site geology — site_screening_data.md (HRADF 2020; Koukouzas 2021 Energies 14:3321;\n"
        "Hystories D1.4; Dotsika 2021 Sci.Rep. 11:16291; HyUSPRe D1.3). Kinetics — Ghaedi 2025 (10.1002/ghg.2368)\n"
        "abstract-level slots + public PHREEQC/PWP rates. Economics — trieres_chain/valley_chain_epistemic.\n"
        "Rendered deterministically from site_screening.sio stdout (lean_single, seeded MC n = 20000).")


def parse(text):
    fans = {1: [], 2: [], 3: []}
    pbox, pgate, chain, base = {}, {}, {}, None
    for line in text.splitlines():
        tok = line.strip().split()
        if len(tok) < 3 or tok[0] != "FIGDATA":
            continue
        kind = tok[1]
        if kind == "FAN":
            s = int(tok[2][1])
            fans[s].append((float(tok[3]), float(tok[4]), float(tok[5])))
        elif kind == "PBOX":
            pbox[int(tok[2][1])] = (float(tok[3]), float(tok[4]))
        elif kind == "PGATE":
            pgate[int(tok[2][1])] = tuple(float(x) for x in tok[3:7])
        elif kind == "CHAIN":
            chain[int(tok[2][1])] = tuple(float(x) for x in tok[3:6])
        elif kind == "BASE":
            base = (float(tok[2]), float(tok[3]))
    for s in fans:
        fans[s].sort()
    if base is None or len(pbox) != 3 or len(pgate) != 3 or len(chain) != 3:
        raise SystemExit("FIGDATA parse incomplete — is this the full demo stdout?")
    return fans, pbox, pgate, chain, base


def save(fig, path):
    fig.savefig(path, dpi=200, metadata={"Software": "render_site_figures.py", "CreationDate": None})
    plt.close(fig)
    print(f"wrote {path}")


def fig_a(fans, pbox, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, s in zip(axes, (1, 2, 3)):
        pts = fans[s]
        ts = [p[0] for p in pts]
        lo = [p[1] for p in pts]
        hi = [p[2] for p in pts]
        name, sub = SITES[s]
        if ts[0] == ts[-1]:
            ax.fill_between([ts[0] - 3, ts[0] + 3], [lo[0], lo[0]], [hi[0], hi[0]],
                            color=COLORS[s], alpha=0.25)
            ax.plot([ts[0]], [(lo[0] + hi[0]) / 2], "s", color=COLORS[s], ms=7,
                    label="corner box (measured T)")
        else:
            ax.fill_between(ts, lo, hi, color=COLORS[s], alpha=0.3, label="corner p-box band")
            ax.plot(ts, lo, color=COLORS[s], lw=1.2)
            ax.plot(ts, hi, color=COLORS[s], lw=1.2)
        blo, bhi = pbox[s]
        ax.axhline(blo, color=COLORS[s], ls="--", lw=0.9)
        ax.axhline(bhi, color=COLORS[s], ls="--", lw=0.9)
        ax.axvline(70.0, color="0.4", ls=":", lw=1.0)
        ax.text(70.4, ax.get_ylim()[1] * 0.02, "A2 70 °C", rotation=90, fontsize=7, color="0.4")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("reservoir temperature [°C]")
        ax.text(0.02, 0.97, sub, transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("30-yr H₂ loss p-box [%]")
    fig.suptitle("Per-site 30-yr H₂-loss p-boxes across the sourced temperature brackets\n"
                 "(dashed = site p-box extrema; band = 8 (k_m × A × salt) corners per T; "
                 "k_m slot zeroes above 70 °C, abstract-level A2)", fontsize=10)
    fig.text(0.01, -0.06, PROV, fontsize=6, va="top")
    fig.tight_layout()
    save(fig, Path(outdir) / "fig_a_loss_pbox_fan.png")


def fig_b(pgate, base, outdir):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = [1, 2, 3]
    conv = [pgate[s][0] for s in xs]
    best = [pgate[s][1] for s in xs]
    worst = [pgate[s][2] for s in xs]
    subonly = [pgate[s][3] for s in xs]
    for s, x in zip(xs, xs):
        ax.plot([x, x], [worst[x - 1], best[x - 1]], color=COLORS[s], lw=6, alpha=0.35,
                solid_capstyle="butt", zorder=2)
        ax.plot(x, conv[x - 1], "o", color=COLORS[s], ms=9, zorder=3,
                label="conventional composed (MC)" if s == 1 else None)
        ax.plot(x, subonly[x - 1], "^", color=COLORS[s], ms=8, mec="k", mew=0.4, zorder=3,
                label="subsurface-only coupling (R = 1)" if s == 1 else None)
    ax.axhline(base[0], color="k", ls="--", lw=1.1)
    ax.text(0.55, base[0] + 1.2, f"no-coupling baseline {base[0]:.3f} %", fontsize=8, ha="left")
    ax.axhline(base[1], color="k", ls="-.", lw=1.1)
    ax.text(0.55, base[1] + 1.2, f"valley composed (25 °C pinned) {base[1]:.3f} %", fontsize=8, ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([SITES[s][0] for s in xs], fontsize=9)
    ax.set_ylabel("P(dispensed < 6 EUR/kg) [%]")
    ax.set_ylim(-3, 105)
    ax.set_title("Gate probability per site vs the valley baselines\n"
                 "(thick band = distribution-free corner p-box on beating 6 EUR/kg)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.text(0.01, -0.06, PROV, fontsize=6, va="top")
    fig.tight_layout()
    save(fig, Path(outdir) / "fig_b_pgate_vs_baselines.png")


def fig_c(chain, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, s in zip(axes, (1, 2, 3)):
        dnom, dsub, dcmp = chain[s]
        name, _ = SITES[s]
        d_sub = dsub - dnom
        d_cmp = dcmp - dsub
        bottoms = [0.0, dnom, dnom + d_sub]
        vals = [dnom, d_sub, d_cmp]
        cols = ["0.6", COLORS[s], "#ff7f0e"]
        labels = ["nominal (interval mids)", "+ subsurface (site L30)", "+ compressor (R mid)"]
        for i, (b, v, c, lab) in enumerate(zip(bottoms, vals, cols, labels)):
            ax.bar(i, v, bottom=b, color=c, width=0.62,
                   label=lab if s == 1 else None)
            if i != 1:
                ax.text(i, b + v + 0.12, f"{b + v:.4f}" if i else f"{v:.4f}",
                        ha="center", fontsize=8)
        ax.text(1, 0.25, f"Δ = {d_sub:+.5f} EUR/kg", ha="center",
                fontsize=8, color=COLORS[s])
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["nominal", "+subsurface", "+compressor"], fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("dispensed cost at interval mids [EUR/kg]")
    fig.suptitle("Composed-chain build-up per site (analytic, interval mids, τ = 1 yr)\n"
                 "the subsurface step is sub-cent for every sourced site", fontsize=10)
    fig.legend(loc="center right", fontsize=8)
    fig.text(0.01, -0.06, PROV, fontsize=6, va="top")
    fig.tight_layout()
    save(fig, Path(outdir) / "fig_c_chain_waterfall.png")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_site_figures.py <demo_stdout.txt> <outdir>")
    text = Path(sys.argv[1]).read_text()
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    fans, pbox, pgate, chain, base = parse(text)
    fig_a(fans, pbox, outdir)
    fig_b(pgate, base, outdir)
    fig_c(chain, outdir)


if __name__ == "__main__":
    main()
