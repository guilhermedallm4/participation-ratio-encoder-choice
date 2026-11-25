#!/usr/bin/env python
"""Reads pr_stability.json and answers the three questions of review item 1.

  ceiling     Do the measured PRs come near the limit n-1 = 199? If the largest
              is a small fraction of it, the rank constraint does not bite.

  order       Does PR estimated from 50, 100 and 150 images order the encoders
              the way PR from 200 does? The paper's conclusion is about RANK --
              Spearman, and "MAE sits at the bottom" -- so it is the order that
              has to be stable, not the magnitude (which grows with n by
              construction: more samples reveal more directions).

  conclusion  Does the paper's main result survive if PR is estimated from half
              the images? We redo the correlation with the probe and the
              sensitivity analysis using PR(m=100) in place of PR(200).
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from analysis_final import (ALL, ARCH, OBJECTIVE, DATASETS,         # noqa: E402
                            LABEL, z, perm_encoder)

MAE = "vit_b16_mae"
CLIPS = ("vit_b16_clip", "vit_b16_clip2b")
SIZES = [50, 100, 150, 200]


def main():
    r = ROOT / "results"
    st = json.loads((r / "pr_stability.json").read_text())
    pb = json.loads((r / "probe.json").read_text())

    def pr(e, b, m):
        return st[f"{e}|{b}"]["ladder"][str(m)]["mean"]

    missing = [f"{e}|{b}" for e in ALL for b in DATASETS if f"{e}|{b}" not in st]
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    p("# Review item 1 - is the participation ratio capped by n = 200?\n")
    if missing:
        p(f"**INCOMPLETE**: {len(missing)} pairs missing: {missing[:6]}\n")

    # ---- 1. ceiling ------------------------------------------------------
    p("## 1. Distance to the rank ceiling\n")
    p("The centred covariance of n images has rank at most n-1. With 200")
    p("images, no estimated PR can exceed **199**.\n")
    all_pr = [(pr(e, b, 200), e, b) for e in ALL for b in DATASETS
              if f"{e}|{b}" in st]
    mx, me, mb = max(all_pr)
    p(f"- largest PR observed across the 68 pairs: **{mx:.1f}** "
      f"({LABEL[me]}, {mb}) = **{100*mx/199:.1f} %** of the ceiling")
    p(f"- median of the 68: {np.median([v for v,_,_ in all_pr]):.1f} "
      f"({100*np.median([v for v,_,_ in all_pr])/199:.1f} % of the ceiling)")
    p(f"- smallest: {min(all_pr)[0]:.1f} ({LABEL[min(all_pr)[1]]}, "
      f"{min(all_pr)[2]})\n")

    # ---- 2. order --------------------------------------------------------
    p("## 2. Does the order among encoders survive a smaller n?\n")
    p("Spearman between PR(m) and PR(200), across the 17 encoders, within")
    p("each domain.\n")
    p("| domain | m=50 | m=100 | m=150 |")
    p("|---|---|---|---|")
    for b in DATASETS:
        if any(f"{e}|{b}" not in st for e in ALL):
            continue
        ref = [pr(e, b, 200) for e in ALL]
        row = [f"{spearmanr([pr(e, b, m) for e in ALL], ref).statistic:.3f}"
               for m in (50, 100, 150)]
        p(f"| {b} | " + " | ".join(row) + " |")
    p("")

    p("### Magnitude: how much PR grows with n (median over the 68 pairs)\n")
    p("| m | PR(m)/PR(200), median | min | max |")
    p("|---|---|---|---|")
    for m in SIZES:
        ratio = [pr(e, b, m) / pr(e, b, 200) for e in ALL for b in DATASETS
                 if f"{e}|{b}" in st]
        p(f"| {m} | {np.median(ratio):.3f} | {min(ratio):.3f} | "
          f"{max(ratio):.3f} |")
    p("")
    # the MAE case on its own: this is the paper's reading
    p("### The collapsed checkpoint, on its own\n")
    p("| encoder | domain | PR(50) | PR(100) | PR(150) | PR(200) |")
    p("|---|---|---|---|---|---|")
    for e in (MAE, "vit_b16_sup1k"):
        for b in DATASETS:
            if f"{e}|{b}" not in st:
                continue
            p(f"| {LABEL[e]} | {b} | " +
              " | ".join(f"{pr(e, b, m):.2f}" for m in SIZES) + " |")
    p("")

    # ---- 3. the conclusion -----------------------------------------------
    p("## 3. The main result with PR estimated from half the images\n")
    p("| set | k | rho with PR(200) | p | rho with PR(100) | p "
      "| rho with PR(50) | p |")
    p("|---|---|---|---|---|---|---|---|")
    SUB = [("all 17", ALL), ("without MAE", [e for e in ALL if e != MAE]),
           ("without MAE and the two CLIP",
            [e for e in ALL if e != MAE and e not in CLIPS]),
           ("9 pre-trainings", OBJECTIVE), ("9 architectures", ARCH)]
    for name, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in pb for b in DATASETS)
              and all(f"{e}|{b}" in st for b in DATASETS)]
        if len(es) < 4:
            continue
        P = [z([pb[f"{b}|{e}"]["value"] for e in es]) for b in DATASETS]
        cells = []
        for m in (200, 100, 50):
            G = [z([pr(e, b, m) for e in es]) for b in DATASETS]
            rho, pv, _ = perm_encoder(G, P, n=5000)
            cells.append(f"{rho:+.3f} | {pv:.3f}")
        p(f"| {name} | {len(es)} | " + " | ".join(cells) + " |")
    p("")

    out = r / "STABILITY.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
