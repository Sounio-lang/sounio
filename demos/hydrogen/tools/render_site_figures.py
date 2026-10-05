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
#   fig_d_field_calibrated.png  per-site 30-yr loss: SOURCED LAW band vs
#                             FIELD-CALIBRATED LAW overlay (KMF p-box)
#
# Figs a-c parse only FIGDATA kinds that predate the 2026-08-01 field-
# calibration section ([A4]); that section is additive-only (verified by
# receipt diff), so figs a-c regenerate byte-identically from the new
# stdout. Fig d parses the [A4] FIGDATA kinds (FANF/PBOXF). Field
# observations (Lehen 3 %/285 d; Lobodice 17 %/7 mo) are SUB-YEAR
# extents and are deliberately NOT overlaid on the 30-yr fan — their
# receipt-level comparison lives in the demo's [A3]/[A4] sections.
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
# k_m(T) LAW path: Rosso 1993 CTMI (DOI 10.1006/jtbi.1993.1099) with a
# cardinal-temperature p-box (Zeikus & Wolfe 1972; Tyne et al. 2021;
# Head et al. 2014 / Wilhelms et al. 2001); economics from
# trieres_chain.sio / valley_chain_epistemic.sio.

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
        "abstract-level slots + public PHREEQC/PWP rates; k_m(T) LAW — Rosso 1993 (10.1006/jtbi.1993.1099) CTMI,\n"
        "cardinals Zeikus & Wolfe 1972 (10.1128/jb.109.2.707-713.1972), Tyne 2021 (10.1038/s41586-021-04153-3),\n"
        "Head 2014 (10.3389/fmicb.2014.00566) / Wilhelms 2001 (10.1038/35082535). Economics — trieres_chain/\n"
        "valley_chain_epistemic. Rendered deterministically from site_screening.sio stdout (lean_single, seeded MC n = 20000).")


def parse(text):
    fans = {1: [], 2: [], 3: []}
    fanl = {1: [], 2: [], 3: []}
    fanf = {1: [], 2: [], 3: []}
    pbox, pgate, chain, base = {}, {}, {}, None
    pboxl, pgatel, chainl = {}, {}, {}
    pboxf = {}
    for line in text.splitlines():
        tok = line.strip().split()
        if len(tok) < 3 or tok[0] != "FIGDATA":
            continue
        kind = tok[1]
        if kind == "FAN":
            s = int(tok[2][1])
            fans[s].append((float(tok[3]), float(tok[4]), float(tok[5])))
        elif kind == "FANL":
            s = int(tok[2][1])
            fanl[s].append((float(tok[3]), float(tok[4]), float(tok[5]),
                            float(tok[6]), float(tok[7])))
        elif kind == "FANF":
            s = int(tok[2][1])
            fanf[s].append((float(tok[3]), float(tok[4]), float(tok[5])))
        elif kind == "PBOX":
            pbox[int(tok[2][1])] = (float(tok[3]), float(tok[4]))
        elif kind == "PBOXL":
            pboxl[int(tok[2][1])] = (float(tok[3]), float(tok[4]))
        elif kind == "PBOXF":
            pboxf[int(tok[2][1])] = (float(tok[3]), float(tok[4]))
        elif kind == "PGATE":
            pgate[int(tok[2][1])] = tuple(float(x) for x in tok[3:7])
        elif kind == "PGATEL":
            pgatel[int(tok[2][1])] = tuple(float(x) for x in tok[3:7])
        elif kind == "CHAIN":
            chain[int(tok[2][1])] = tuple(float(x) for x in tok[3:6])
        elif kind == "CHAINL":
            chainl[int(tok[2][1])] = tuple(float(x) for x in tok[3:6])
        elif kind == "BASE":
            base = (float(tok[2]), float(tok[3]))
    for s in fans:
        fans[s].sort()
        fanl[s].sort()
        fanf[s].sort()
    if (base is None or len(pbox) != 3 or len(pgate) != 3 or len(chain) != 3
            or len(pboxl) != 3 or len(pgatel) != 3 or len(chainl) != 3
            or len(pboxf) != 3 or any(len(fanf[s]) == 0 for s in fanf)):
        raise SystemExit("FIGDATA parse incomplete — is this the full demo stdout?")
    return fans, pbox, pgate, chain, base, fanl, pboxl, pgatel, chainl, fanf, pboxf


def save(fig, path):
    fig.savefig(path, dpi=200, metadata={"Software": "render_site_figures.py", "CreationDate": None})
    plt.close(fig)
    print(f"wrote {path}")


def fig_a(fans, pbox, fanl, pboxl, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, s in zip(axes, (1, 2, 3)):
        pts = fans[s]
        ts = [p[0] for p in pts]
        lo = [p[1] for p in pts]
        hi = [p[2] for p in pts]
        lpts = fanl[s]
        lts = [p[0] for p in lpts]
        llo = [p[1] for p in lpts]
        lhi = [p[2] for p in lpts]
        name, sub = SITES[s]
        # thermal-death zone (Tmax p-box) behind everything
        ax.axvspan(75.0, 90.0, color="0.92", zorder=0,
                   label="Tₘₐₓ bracket [75,90] °C" if s == 1 else None)
        if ts[0] == ts[-1]:
            ax.fill_between([ts[0] - 3, ts[0] + 3], [lo[0], lo[0]], [hi[0], hi[0]],
                            color=COLORS[s], alpha=0.25)
            ax.plot([ts[0]], [(lo[0] + hi[0]) / 2], "s", color=COLORS[s], ms=7,
                    label="slot corner box (measured T)")
        else:
            ax.fill_between(ts, lo, hi, color=COLORS[s], alpha=0.25,
                            label="SLOT corner band (70 °C step)")
            ax.plot(ts, lo, color=COLORS[s], lw=1.0, alpha=0.6)
            ax.plot(ts, hi, color=COLORS[s], lw=1.0, alpha=0.6)
        # LAW overlay (dense T grid, CTMI thermal slide)
        if lts[0] == lts[-1]:
            ax.plot([lts[0]], [(llo[0] + lhi[0]) / 2], "D", color="k", ms=6,
                    label="LAW (degenerate, T > Tₘₐₓ)")
        else:
            ax.fill_between(lts, llo, lhi, color="none", edgecolor=COLORS[s],
                            hatch="////", lw=0.0, alpha=0.9,
                            label="LAW dense-grid band (CTMI)")
            ax.plot(lts, llo, color=COLORS[s], lw=1.8)
            ax.plot(lts, lhi, color=COLORS[s], lw=1.8)
        blo, bhi = pbox[s]
        ax.axhline(blo, color=COLORS[s], ls="--", lw=0.9)
        ax.axhline(bhi, color=COLORS[s], ls="--", lw=0.9)
        lbl_lo, lbl_hi = pboxl[s]
        ax.axhline(lbl_lo, color="k", ls="-.", lw=1.0)
        ax.axhline(lbl_hi, color="k", ls="-.", lw=1.0)
        ax.axvline(70.0, color="0.4", ls=":", lw=1.0)
        ax.text(70.4, ax.get_ylim()[1] * 0.02, "A2 slot 70 °C", rotation=90,
                fontsize=7, color="0.4")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("reservoir temperature [°C]")
        ax.text(0.02, 0.97, sub, transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("30-yr H₂ loss p-box [%]")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    proxies = [
        Patch(facecolor="0.92", label="Tₘₐₓ bracket [75,90] °C"),
        Patch(facecolor="0.45", alpha=0.35, label="SLOT corner band (70 °C step)"),
        Patch(facecolor="none", edgecolor="0.25", hatch="////",
              label="LAW dense-grid band (CTMI)"),
        Line2D([0], [0], color="0.25", lw=1.8, label="LAW band edges"),
        Line2D([0], [0], color="0.25", ls="--", lw=0.9, label="SLOT p-box extrema"),
        Line2D([0], [0], color="k", ls="-.", lw=1.0, label="LAW p-box extrema"),
    ]
    axes[0].legend(handles=proxies, loc="center", fontsize=7.5)
    fig.suptitle("Per-site 30-yr H₂-loss p-boxes: k_m SLOT (band, 70 °C step) vs SOURCED LAW (hatched, Rosso CTMI)\n"
                 "S3: the slot's hard cliff becomes a smooth thermal-death slide through the Tₘₐₓ bracket", fontsize=10)
    fig.text(0.01, -0.09, PROV, fontsize=6, va="top")
    fig.tight_layout()
    save(fig, Path(outdir) / "fig_a_loss_pbox_fan.png")


def fig_b(pgate, base, pgatel, outdir):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = [1, 2, 3]
    conv = [pgate[s][0] for s in xs]
    best = [pgate[s][1] for s in xs]
    worst = [pgate[s][2] for s in xs]
    subonly = [pgate[s][3] for s in xs]
    lconv = [pgatel[s][0] for s in xs]
    lbest = [pgatel[s][1] for s in xs]
    lworst = [pgatel[s][2] for s in xs]
    lsubonly = [pgatel[s][3] for s in xs]
    dx = 0.09
    for s, x in zip(xs, xs):
        ax.plot([x - dx, x - dx], [worst[x - 1], best[x - 1]], color=COLORS[s], lw=6,
                alpha=0.35, solid_capstyle="butt", zorder=2)
        ax.plot([x + dx, x + dx], [lworst[x - 1], lbest[x - 1]], color=COLORS[s], lw=3,
                alpha=0.55, solid_capstyle="butt", zorder=2)
        ax.plot(x - dx, conv[x - 1], "o", color=COLORS[s], ms=9, zorder=3,
                label="SLOT conventional composed (MC)" if s == 1 else None)
        ax.plot(x - dx, subonly[x - 1], "^", color=COLORS[s], ms=8, mec="k", mew=0.4,
                zorder=3, label="SLOT subsurface-only (R = 1)" if s == 1 else None)
        ax.plot(x + dx, lconv[x - 1], "D", color=COLORS[s], ms=7, mec="k", mew=0.5,
                zorder=3, label="LAW conventional composed (MC)" if s == 1 else None)
        ax.plot(x + dx, lsubonly[x - 1], "v", color=COLORS[s], ms=7, mec="k", mew=0.4,
                zorder=3, label="LAW subsurface-only (R = 1)" if s == 1 else None)
    ax.axhline(base[0], color="k", ls="--", lw=1.1)
    ax.text(0.55, base[0] + 1.2, f"no-coupling baseline {base[0]:.3f} %", fontsize=8, ha="left")
    ax.axhline(base[1], color="k", ls="-.", lw=1.1)
    ax.text(0.55, base[1] + 1.2, f"valley composed (25 °C pinned) {base[1]:.3f} %", fontsize=8, ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([SITES[s][0] for s in xs], fontsize=9)
    ax.set_ylabel("P(dispensed < 6 EUR/kg) [%]")
    ax.set_ylim(-3, 105)
    ax.set_title("Gate probability per site vs the valley baselines — SLOT (left) vs k_m LAW (right)\n"
                 "(bands = distribution-free corner p-boxes on beating 6 EUR/kg)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.text(0.01, -0.09, PROV, fontsize=6, va="top")
    fig.tight_layout()
    save(fig, Path(outdir) / "fig_b_pgate_vs_baselines.png")


def fig_c(chain, chainl, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, s in zip(axes, (1, 2, 3)):
        dnom, dsub, dcmp = chain[s]
        ldnom, ldsub, ldcmp = chainl[s]
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
        # LAW cumulative tops as black diamond markers
        ax.plot([1, 2], [ldsub, ldcmp], "D", color="k", ms=5, zorder=4,
                label="LAW cumulative tops" if s == 1 else None)
        ax.text(1, 0.25, f"Δ slot = {d_sub:+.5f}", ha="center",
                fontsize=8, color=COLORS[s])
        ax.text(1, 0.55, f"Δ law = {ldsub - ldnom:+.5f}", ha="center",
                fontsize=8, color="k")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["nominal", "+subsurface", "+compressor"], fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("dispensed cost at interval mids [EUR/kg]")
    fig.suptitle("Composed-chain build-up per site (analytic, interval mids, τ = 1 yr)\n"
                 "bars = SLOT path; black diamonds = k_m LAW cumulative tops — both sub-cent", fontsize=10)
    fig.legend(loc="center right", fontsize=8)
    fig.text(0.01, -0.09, PROV, fontsize=6, va="top")
    fig.tight_layout()
    save(fig, Path(outdir) / "fig_c_chain_waterfall.png")


PROVF = ("Provenance: FIELD-CALIBRATED k_m p-box [2.041617, 382.433772] at Topt (f = 1) — inverse calibration\n"
         "on Lehen (Hellerschmied 2024, 10.1038/s41560-024-01458-1) + Lobodice (Smigan 1990 / Buzek 1994 via\n"
         "Tremosa 2023, 10.3389/fenrg.2023.1145978), cross-anchored on the only in-situ rate measurement,\n"
         "Tyne 2021 (10.1038/s41586-021-04153-3; normalization-volume ambiguity documented); CTMI shape and\n"
         "SOURCED LAW band as in fig A. Field observations are sub-year extents — not overlaid on a 30-yr fan.\n"
         "Rendered deterministically from site_screening.sio stdout [A4] (lean_single, seeded MC n = 20000).")


def fig_d(fanl, pboxl, fanf, pboxf, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, s in zip(axes, (1, 2, 3)):
        lpts = fanl[s]
        lts = [p[0] for p in lpts]
        llo = [p[1] for p in lpts]
        lhi = [p[2] for p in lpts]
        fpts = fanf[s]
        fts = [p[0] for p in fpts]
        flo = [p[1] for p in fpts]
        fhi = [p[2] for p in fpts]
        name, sub = SITES[s]
        ax.axvspan(75.0, 90.0, color="0.92", zorder=0)
        if lts[0] == lts[-1]:
            ax.plot([lts[0]], [(llo[0] + lhi[0]) / 2], "D", color=COLORS[s], ms=6)
        else:
            ax.fill_between(lts, llo, lhi, color=COLORS[s], alpha=0.20)
            ax.plot(lts, llo, color=COLORS[s], lw=1.4)
            ax.plot(lts, lhi, color=COLORS[s], lw=1.4)
        if fts[0] == fts[-1]:
            ax.plot([fts[0]], [(flo[0] + fhi[0]) / 2], "s", color="k", ms=6)
        else:
            ax.fill_between(fts, flo, fhi, color="none", edgecolor="k",
                            hatch="\\\\\\\\", lw=0.0, alpha=0.9)
            ax.plot(fts, flo, color="k", lw=1.6)
            ax.plot(fts, fhi, color="k", lw=1.6)
        lbl_lo, lbl_hi = pboxl[s]
        ax.axhline(lbl_lo, color=COLORS[s], ls="--", lw=0.9)
        ax.axhline(lbl_hi, color=COLORS[s], ls="--", lw=0.9)
        fbl_lo, fbl_hi = pboxf[s]
        ax.axhline(fbl_lo, color="k", ls=":", lw=1.1)
        ax.axhline(fbl_hi, color="k", ls=":", lw=1.1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("reservoir temperature [°C]")
        ax.text(0.02, 0.97, sub, transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("30-yr H₂ loss p-box [%]")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    proxies = [
        Patch(facecolor="0.92", label="Tₘₐₓ bracket [75,90] °C"),
        Patch(facecolor="0.45", alpha=0.30, label="SOURCED LAW band (CTMI)"),
        Patch(facecolor="none", edgecolor="k", hatch="\\\\\\\\",
              label="FIELD-CALIBRATED LAW band (KMF)"),
        Line2D([0], [0], color="0.25", ls="--", lw=0.9, label="LAW p-box extrema"),
        Line2D([0], [0], color="k", ls=":", lw=1.1, label="FIELD p-box extrema"),
    ]
    axes[0].legend(handles=proxies, loc="center", fontsize=7.5)
    fig.suptitle("Per-site 30-yr H₂-loss p-boxes: SOURCED LAW (band) vs FIELD-CALIBRATED LAW (hatched, KMF)\n"
                 "S1 thermal death is anchor-independent [0, 0]; S2/S3 losses widen to the field-calibrated magnitude",
                 fontsize=10)
    fig.text(0.01, -0.11, PROVF, fontsize=6, va="top")
    fig.tight_layout()
    save(fig, Path(outdir) / "fig_d_field_calibrated.png")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_site_figures.py <demo_stdout.txt> <outdir>")
    text = Path(sys.argv[1]).read_text()
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    fans, pbox, pgate, chain, base, fanl, pboxl, pgatel, chainl, fanf, pboxf = parse(text)
    fig_a(fans, pbox, fanl, pboxl, outdir)
    fig_b(pgate, base, pgatel, outdir)
    fig_c(chain, chainl, outdir)
    fig_d(fanl, pboxl, fanf, pboxf, outdir)


if __name__ == "__main__":
    main()
