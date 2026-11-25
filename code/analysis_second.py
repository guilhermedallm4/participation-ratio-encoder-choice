#!/usr/bin/env python
"""Reads second_collapse.json and judges the PREDICTION, without touching the
original set.

The paper's claim, in testable form: the participation ratio separates the
collapsed representation from the rest by a wide margin (for MAE, 2.3 to 3.4
times below the next encoder in any domain), and that representation probes
among the worst.

Three readings, in this order:

  1. where they land  PR and probe of the two new encoders against the
                      distribution of the seventeen.
  2. the prediction   was BEiT-3 flagged? was the untrained one? Both cases are
                      informative, and the second one bounds the claim.
  3. if they joined   what would happen to the main correlation if the new
                      encoders entered the set. This is not the test -- it is
                      transparency, so that nobody has to ask.

A NOTE ON THE PROBE POSITION. A held-out encoder is reported by how many of the
reference encoders probe WORSE than it, not by a rank inside a pool it was
never part of. Folding a held-out checkpoint back into the set to compute its
own rank would quietly undo the separation the test depends on, and it would
also invent a denominator the paper never defines.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from analysis_final import (ALL, DATASETS, LABEL, z,                # noqa: E402
                            perm_encoder)

MAE = "vit_b16_mae"
LABEL2 = {"vit_b16_beit3": "ViT-B/16 BEiT-3 (.pt)",
          "vit_b16_random": "ViT-B/16 untrained"}


def main():
    r = ROOT / "results"
    t = json.loads((r / "measures.json").read_text())
    pb = json.loads((r / "probe.json").read_text())
    nv = json.loads((r / "second_collapse.json").read_text())
    geo = {(x["backbone"], x["dataset"]): x
           for bl in ("ci", "architectures", "batch2", "batch3")
           for x in t.get(bl, []) if "dataset" in x}

    new = sorted({v["alias"] for v in nv.values()})
    missing = [f"{a}|{b}" for a in new for b in DATASETS if f"{a}|{b}" not in nv]
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    p("# Prediction test: a second collapse\n")
    if missing:
        p(f"**INCOMPLETE**: missing {missing}\n")

    def pr(e, b):
        return (nv[f"{e}|{b}"]["rank"][0] if f"{e}|{b}" in nv
                else geo[(e, b)]["rank"][0])

    def probe(e, b):
        return (nv[f"{e}|{b}"]["value"] if f"{e}|{b}" in nv
                else pb[f"{b}|{e}"]["value"])

    # ---- 1. where they land ----------------------------------------------
    p("## 1. Where the two of them land\n")
    p("| encoder | " + " | ".join(DATASETS) + " |")
    p("|---" * (len(DATASETS) + 1) + "|")
    for e in new + [MAE]:
        lab = LABEL2.get(e, LABEL.get(e, e))
        p(f"| **{lab}** - PR | "
          + " | ".join(f"{pr(e,b):.2f}" for b in DATASETS) + " |")
        p(f"| {lab} - probe | "
          + " | ".join(f"{probe(e,b):.3f}" for b in DATASETS) + " |")
    p("")
    p("Reference, the seventeen of the paper (PR):\n")
    p("| domain | minimum without MAE | median | maximum |")
    p("|---|---|---|---|")
    for b in DATASETS:
        v = np.array([geo[(e, b)]["rank"][0] for e in ALL if e != MAE])
        p(f"| {b} | {v.min():.2f} | {np.median(v):.2f} | {v.max():.2f} |")
    p("")

    # ---- 2. the prediction -----------------------------------------------
    p("## 2. The prediction\n")
    p("Criterion, the same as the paper's: *flagged* = PR below the minimum")
    p("over the remaining encoders in that domain, by a margin that is not")
    p("measurement noise.\n")
    p("| encoder | domain | PR | minimum of the others | ratio | flagged? "
      "| encoders probing worse |")
    p("|---|---|---|---|---|---|---|")
    for e in new + [MAE]:
        for b in DATASETS:
            others = [geo[(o, b)]["rank"][0] for o in ALL if o != e and o != MAE]
            mn = min(others)
            v = pr(e, b)
            # Position reported AGAINST the reference set, never inside a pool
            # the held-out encoder was not part of.
            ref = [o for o in ALL if o != e]
            worse = sum(1 for o in ref if probe(o, b) < probe(e, b))
            p(f"| {LABEL2.get(e, LABEL.get(e,e))} | {b} | {v:.2f} | {mn:.2f} | "
              f"{mn/v:.1f}x | {'**yes**' if v < mn else 'no'} | "
              f"{worse} of {len(ref)} |")
    p("")

    # ---- 3. if they joined ------------------------------------------------
    p("## 3. If they joined the set (transparency, this is not the test)\n")
    p("| set | k | rho | p |")
    p("|---|---|---|---|")
    for name, es in [("the 17 of the paper", ALL),
                     ("17 + BEiT-3", ALL + ["vit_b16_beit3"]),
                     ("17 + both new ones", ALL + new),
                     ("17 + BEiT-3, without MAE",
                      [e for e in ALL if e != MAE] + ["vit_b16_beit3"])]:
        es = [e for e in es if all((f"{e_}|{b}" in nv) or ((e_, b) in geo)
                                   for e_ in [e] for b in DATASETS)]
        G = [z([pr(e, b) for e in es]) for b in DATASETS]
        P = [z([probe(e, b) for e in es]) for b in DATASETS]
        rho, pv, _ = perm_encoder(G, P, n=20000)
        p(f"| {name} | {len(es)} | {rho:+.3f} | {pv:.3f} |")
    p("")

    out = r / "SECOND_COLLAPSE.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
