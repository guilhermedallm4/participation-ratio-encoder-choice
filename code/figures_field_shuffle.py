#!/usr/bin/env python
"""Figures for the effective receptive field and for patch shuffling.

Identity NEVER by colour alone: each architecture has its own colour, marker
and dash pattern, because the figure goes into a paper that may be printed in
black and white.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical palette (slots 1-3, light mode, --pairs all):
# worst CVD pair dE 9.2 and normal vision dE 24.0. The contrast warning on the
# teal (2.74:1) is covered by the direct labels, as the relief rule allows.
COLOR = {"resnet50": "#2a78d6", "maxvit_t": "#eb6834", "vit_b16": "#1baf7a"}
MARKER = {"resnet50": "o", "maxvit_t": "s", "vit_b16": "^"}
DASH = {"resnet50": "-", "maxvit_t": "--", "vit_b16": ":"}
LABEL = {"resnet50": "ResNet-50", "maxvit_t": "MaxViT-T", "vit_b16": "ViT-B/16"}
DATASET_LABEL = {"mammography": "mammography", "dtd": "DTD (texture)",
                 "pet": "Pet (object)", "sun397": "SUN397 (scene)"}
INK, INK2 = "#0b0b0b", "#52514e"
ORDER = ["mammography", "dtd", "pet", "sun397"]


def _uniform(q, size=512):
    """Radius a UNIFORM gradient would accumulate, on the measure's own grid.

    It is the ceiling of the scale, and it is not 256: 256 px is the distance
    to the border, not the median of the mass. For q=0.5 the disc fits inside
    the square and equals sqrt(0.5 * size^2 / pi) = 204.3 px; for q=0.9 it is
    already clipped by the corners, so the value comes from the same radial
    cumulative used in _profile.
    """
    yy, xx = np.ogrid[:size, :size]
    r = np.sort(np.sqrt((yy - (size - 1) / 2) ** 2
                        + (xx - (size - 1) / 2) ** 2).ravel())
    return float(r[int(q * r.size)])


UNIFORM_50, UNIFORM_90 = _uniform(0.5), _uniform(0.9)


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5,
        "axes.edgecolor": "#b8b7b2", "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "axes.titlesize": 8,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "hatch.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 7,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _recessive(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_field(cr, out):
    """A: growth of the field with depth, per dataset.
       B: MaxViT opened up into conv / local attention / grid attention."""
    fig = plt.figure(figsize=(7.1, 4.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], hspace=0.95,
                          wspace=0.22, top=0.86)

    # --- A: block output of each family ------------------------------------
    # For MaxViT the "block" is the whole MaxxVitBlock, whose output is the
    # grid attention; comparing against the conv sub-block would compare half a
    # block with the whole block of the other two.
    block_out = cr[(cr.backbone != "maxvit_t") |
                   (cr.mechanism == "grid attention")]
    # The direct labels of the last panel overlap when two curves end close
    # together; push each one into its own lane.
    offset = {"vit_b16": 5, "maxvit_t": -6, "resnet50": 0}
    for k, dataset in enumerate(ORDER):
        ax = fig.add_subplot(gs[0, k])
        ax.axhline(UNIFORM_50, color="#b8b7b2", lw=0.8, ls=(0, (4, 3)),
                   zorder=1)
        if k == 0:
            ax.text(0.03, UNIFORM_50 * 1.1, "uniform gradient", fontsize=5.6,
                    color=INK2, va="bottom")
        for bb in LABEL:
            s_ = block_out[(block_out.dataset == dataset) &
                           (block_out.backbone == bb)]
            if s_.empty:
                continue
            x = np.linspace(0, 1, len(s_))
            ax.plot(x, s_.radius_50.to_numpy(), DASH[bb], color=COLOR[bb],
                    lw=1.4, marker=MARKER[bb], ms=2.8, mew=0, zorder=3,
                    label=LABEL[bb] if k == 0 else None)
            if k == len(ORDER) - 1:        # direct label on the last panel
                ax.annotate(LABEL[bb], (1.0, s_.radius_50.iloc[-1]),
                            xytext=(4, offset[bb]), textcoords="offset points",
                            fontsize=6.2, color=INK2, va="center")
        ax.set_yscale("log"); ax.set_ylim(2.5, 430)
        ax.set_yticks([4, 16, 64, 256])
        ax.set_yticklabels(["4", "16", "64", "256"])
        ax.set_xlim(-0.05, 1.05); ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "½", "1"])
        ax.set_xlabel("depth", labelpad=1.5, fontsize=6.8, color=INK2)
        ax.set_title(DATASET_LABEL[dataset], color=INK, pad=3)
        _recessive(ax)
        if k:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("effective field radius (px)")
        if k == len(ORDER) - 1:
            ax.set_xlim(-0.05, 1.34)
    handles = [plt.Line2D([], [], color=COLOR[b], ls=DASH[b], marker=MARKER[b],
                          ms=3.4, mew=0, lw=1.4, label=LABEL[b])
               for b in LABEL]
    fig.legend(handles=handles, loc="upper center", ncol=3, handlelength=2.0,
               bbox_to_anchor=(0.5, 1.005), columnspacing=2.2)

    # --- B: inside the MaxViT block ----------------------------------------
    # Two measures, because they disagree, and the disagreement IS the finding:
    # the core of the field (radius 50) stays narrow while the tail (radius 90)
    # already covers the whole image. Grid attention does not move the field,
    # it adds a long tail to a core that remains local.
    sub = cr[(cr.backbone == "maxvit_t") & (cr.dataset == "mammography")]
    stages = sub.block.str.split(".").str[:2].str.join(".").unique()
    # The hatch on the middle mechanism is the relief for B&W printing: in
    # grayscale the orange and the gray land at almost the same luminance.
    mech_order = [("conv (MBConv)", "#9a9994", "", "conv (MBConv)"),
                  ("7x7 window attention", "#eb6834", "///",
                   "7×7 window attention"),
                  ("grid attention", "#2a78d6", "", "grid attention")]
    width, base = 0.26, np.arange(len(stages))
    block_label = [e.replace("s", "").replace(".b", "·") for e in stages]
    for c, (col, title) in enumerate(
            [("radius_50", "core of the field (radius 50%)"),
             ("radius_90", "tail of the field (radius 90%)")]):
        ax = fig.add_subplot(gs[1, 2 * c:2 * c + 2])
        for i, (mech, color, hatch, lab) in enumerate(mech_order):
            v = [sub[(sub.block.str.startswith(e + ".")) &
                     (sub.mechanism == mech)][col].mean() for e in stages]
            # 0.93 of the width leaves the surface gap between neighbouring bars
            ax.bar(base + (i - 1) * width, v, width * 0.93, color=color,
                   label=lab if c == 0 else None, zorder=3, linewidth=0,
                   hatch=hatch, edgecolor="white")
        ax.axhline(UNIFORM_50 if col == "radius_50" else UNIFORM_90,
                   color="#b8b7b2", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_yscale("log"); ax.set_ylim(3, 600)
        ax.set_yticks([4, 16, 64, 256])
        ax.set_yticklabels(["4", "16", "64", "256"])
        ax.set_xticks(base); ax.set_xticklabels(block_label, fontsize=6.2)
        ax.set_xlabel("stage · block", labelpad=1.5, fontsize=6.8, color=INK2)
        ax.set_title(title, color=INK, pad=3)
        _recessive(ax)
        if c == 0:
            ax.set_ylabel("radius (px)")
            handles_b, labels_b = ax.get_legend_handles_labels()
    fig.text(0.5, 0.445, "Inside the MaxViT block (mammography)", ha="center",
             fontsize=8, color=INK)
    fig.legend(handles_b, labels_b, loc="upper center", ncol=3,
               handlelength=1.1, handleheight=0.8, fontsize=6.6,
               bbox_to_anchor=(0.5, 0.443), columnspacing=1.8)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"-> {out}")


def fig_shuffle(em, out):
    """ARRANGEMENT sensitivity: delta(global) - delta(local), by patch size.
    The subtraction is the control -- both modes create the same number of
    artificial seams, only the global one destroys the coarse arrangement."""
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.95), sharey=True)
    for k, (dataset, ax) in enumerate(zip(ORDER, axs)):
        for bb in LABEL:
            s = em[(em.dataset == dataset) & (em.backbone == bb)]
            if s.empty:
                continue
            p = s.pivot_table(index="patch_px", columns="mode", values="delta")
            p = p.sort_index(ascending=False)
            y = (p["global"] - p["local"]).to_numpy()
            ax.plot(p.index.to_numpy(), y, DASH[bb], color=COLOR[bb], lw=1.4,
                    marker=MARKER[bb], ms=3.4, mew=0, zorder=3,
                    label=LABEL[bb] if k == 0 else None)
            if k == len(ORDER) - 1:
                ax.annotate(LABEL[bb], (p.index[-1], y[-1]), xytext=(3, 0),
                            textcoords="offset points", fontsize=6.2,
                            color=INK2, va="center")
        ax.axhline(0, color="#b8b7b2", lw=0.7, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 64, 32, 16])
        ax.set_xticklabels(["128", "64", "32", "16"])
        ax.invert_xaxis()
        ax.set_title(DATASET_LABEL[dataset], color=INK, pad=3)
        ax.set_xlabel("patch (px)")
        _recessive(ax)
    axs[-1].set_xlim(axs[-1].get_xlim()[0], 11)
    axs[0].set_ylabel("arrangement sensitivity\n(Δ global − Δ local)")
    axs[0].legend(loc="upper left", handlelength=1.8)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"-> {out}")


def main():
    style()
    r = Path("results"); f = Path("figures"); f.mkdir(exist_ok=True)
    cr = pd.read_csv(r / "receptive_field.csv")
    em = pd.read_csv(r / "shuffling.csv")
    fig_field(cr, f / "receptive_field.pdf")
    fig_shuffle(em, f / "shuffling.pdf")


if __name__ == "__main__":
    main()
