"""Figures from a core.analyze() result (JSON written by the CLI's --out or the Arena analysis).

fig_flip_vs_margin: binned flip rate inside the set vs human margin, logistic fit, permutation-null band.
fig_set_vs_all:     flip rate inside the set vs across all configs, with the null band and the noise floor.
fig_kappa:          each config's kappa against humans with its gap SE; Rashomon-set members filled.
Usage: python -m judge_rashomon.figures report.json outdir/
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, ACCENT, MUTED = "#222222", "#2f6fb0", "#bbbbbb"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_flip_vs_margin(m, path, bins=(0, 0.05, 0.1, 0.2, 0.5)):
    fl, mg = np.array(m["units"]["flip"]), np.array(m["units"]["margin"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    lo, hi = m["perm_null_band"]
    ax.axhspan(lo, hi, color=MUTED, alpha=0.35, lw=0, label="chance, all margins (permutation 95%)", zorder=0)
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (mg >= a) & (mg < b)
        if sel.sum():
            p, n = fl[sel].mean(), sel.sum()
            se = np.sqrt(p * (1 - p) / n)
            ax.errorbar((a + b) / 2, p, yerr=1.96 * se, fmt="o", color=ACCENT, capsize=3)
            ax.annotate(f"n={n}", ((a + b) / 2, p), textcoords="offset points", xytext=(6, 4), fontsize=7, color=INK)
    if m["slope"] == m["slope"]:
        b1 = m["slope"]
        b0 = np.log(m["p_at_005"] / (1 - m["p_at_005"])) - b1 * 0.05
        x = np.linspace(0, max(bins[-1], mg.max() if len(mg) else 0.5), 100)
        ax.plot(x, 1 / (1 + np.exp(-(b0 + b1 * x))), color=ACCENT, lw=1.5, label="logistic fit")
    _style(ax, "Flips among validated judges vs human closeness",
           "human margin |p_human - 0.5|", "P(flip) inside Rashomon set")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_set_vs_all(m, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    vals = [m["flip"], m["flip_all"]]
    cis = [m["ci"]["flip"], m["ci"]["flip_all"]]
    ax.bar([0, 1], vals, color=[ACCENT, MUTED], width=0.55, zorder=2)
    ax.errorbar([0, 1], vals, yerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]],
                fmt="none", ecolor=INK, capsize=4, zorder=3)
    lo, hi = m["perm_null_band"]
    ax.axhspan(lo, hi, color="#e8e8e8", lw=0, label="chance (permutation 95%)", zorder=0)
    if "noise_upper_95" in m:
        ax.axhline(m["noise_upper_95"], color=INK, ls="--", lw=1, label="re-run noise (upper 95%)")
    ax.set_xticks([0, 1], [f"Rashomon set\n({len(m['R'])} configs)", f"all configs\n({len(m['eligible'])})"])
    _style(ax, "Does passing validation buy stable rankings?", "", "flip rate")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_kappa(m, path):
    cs = sorted(m["kappa"], key=m["kappa"].get)
    fig, ax = plt.subplots(figsize=(5, 0.35 * len(cs) + 1.2))
    for i, c in enumerate(cs):
        inR = c in m["R"]
        se = m["gap_se"].get(c, 0)
        ax.errorbar(m["kappa"][c], i, xerr=se, fmt="o", color=ACCENT if inR else MUTED,
                    mfc=ACCENT if inR else "white", capsize=3)
    ax.set_yticks(range(len(cs)), cs)
    _style(ax, "Agreement with humans (filled = in Rashomon set)", "Cohen's kappa (± gap SE)", "")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def all_figures(m, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    fig_kappa(m, out / "kappa.png")
    if len(m["R"]) >= 2:
        fig_flip_vs_margin(m, out / "flip_vs_margin.png")
        fig_set_vs_all(m, out / "set_vs_all.png")
    return sorted(str(p) for p in out.glob("*.png"))


if __name__ == "__main__":
    print("\n".join(all_figures(json.load(open(sys.argv[1])), sys.argv[2])))
