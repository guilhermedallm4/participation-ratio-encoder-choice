#!/usr/bin/env python
"""Third-review item 2.2: is width actually controlled for?

IT WAS NOT. The paper reported rho(PR, probe) = +0.41 and, beside it,
rho(width, probe) = +0.17, calling the second one "controlling for the
confounder". Reporting the confounder's correlation with the outcome controls
for nothing: it is just a second marginal correlation. The reviewer is right.

Here is the real control, in two forms:

  partial     Partial Spearman: ranks of PR, of the probe and of width; PR and
              probe are regressed on width and the RESIDUALS are correlated.
              This answers "is there association left once width is removed?".

  regression  Probe = b0 + b1 PR + b2 width, on values already standardised
              within each domain. The coefficient b1 is the partial slope.

NOTE ON THE p-VALUES BELOW. The null used here permutes PR and width TOGETHER,
which also destroys the width-probe association. A later review showed that
this is not the correct conditional null; `partial_fl.py` implements
Freedman-Lane, which preserves it, and its p-values supersede the ones in this
file. The point estimates are unaffected and agree between the two scripts.

Where width is constant (the nine ViT-B/16, all with 768 channels) there is
nothing to control for, and the script says so instead of returning nan.

It also recomputes the rank ceiling with the RIGHT m. The published point of
the participation ratio comes out of ic_sub, which subsamples without
replacement at m = n/2 = 100, so the ceiling is 99 and not 199. The earlier
version of this check used 199.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from analysis_final import (ALL, ARCH, OBJECTIVE, DATASETS,        # noqa: E402
                            LABEL, z)

MAE = "vit_b16_mae"
CLIPS = ("vit_b16_clip", "vit_b16_clip2b")


def load():
    r = ROOT / "results"
    t = json.loads((r / "measures.json").read_text())
    pb = json.loads((r / "probe.json").read_text())
    geo = {(x["backbone"], x["dataset"]): x
           for bl in ("ci", "architectures", "batch2", "batch3")
           for x in t.get(bl, []) if "dataset" in x}
    return geo, pb


def residual(y, x):
    """Residual of y regressed on x (least squares, with an intercept)."""
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def partial(a, b, c):
    """Partial Spearman of a and b given c: correlation of rank residuals."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    return float(np.corrcoef(residual(ra, rc), residual(rb, rc))[0, 1])


def partial_beta(probe, pr, width):
    X = np.column_stack([np.ones_like(pr), pr, width])
    b, *_ = np.linalg.lstsq(X, probe, rcond=None)
    return float(b[1]), float(b[2])


def permute(G, W, P, stat, n=20000, seed=0):
    """ONE permutation of encoders, applied to all four domains.

    PR and width are properties of the SAME encoder and are permuted together.
    """
    obs = stat(np.concatenate(G), np.concatenate(W), np.concatenate(P))
    rng = np.random.default_rng(seed)
    k = len(G[0])
    p = np.concatenate(P)
    null = []
    for q in (rng.permutation(k) for _ in range(n)):
        gq = np.concatenate([g[q] for g in G])
        wq = np.concatenate([w[q] for w in W])
        v = stat(gq, wq, p)
        if np.isfinite(v):
            null.append(abs(v))
    null = np.array(null)
    return obs, float((null >= abs(obs)).mean())


def main():
    geo, pb = load()
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    # ----- rank ceiling, with the right m --------------------------------
    p("# Review item 2.2 - width control, and the corrected rank ceiling\n")
    p("## 0. The ceiling, now with the correct m\n")
    p("The published point of the participation ratio comes from `ic_sub`,")
    p("which subsamples **without replacement at m = n/2 = 100**. The rank")
    p("ceiling is therefore **99**, not 199. The earlier check used 199 and")
    p("is corrected here.\n")
    vals = [(geo[(e, b)]["rank"][0], e, b) for e in ALL for b in DATASETS]
    mx, me, mb = max(vals)
    arr = np.array([v for v, _, _ in vals])
    p(f"- largest published PR (m = 100): **{mx:.1f}** ({LABEL[me]}, {mb}) = "
      f"**{100*mx/99:.1f} % of the ceiling of 99**")
    p(f"- median of the 68: {np.median(arr):.1f} "
      f"({100*np.median(arr)/99:.1f} %)")
    p(f"- above 50 % of the ceiling: {(arr > 49.5).sum()} of the 68 pairs")
    full = [geo[(e, b)].get("rank_n_full") for e in ALL for b in DATASETS]
    full = [c for c in full if c]
    if full:
        p(f"- for reference, with all 200 (ceiling 199): max {max(full):.1f} "
          f"({100*max(full)/199:.1f} %), median {np.median(full):.1f}\n")

    # ----- width control --------------------------------------------------
    p("## 1. Width, actually controlled for\n")
    p("| set | k | rho(PR, probe) | rho(width, probe) | "
      "partial rho (PR, probe; given width) | p perm |")
    p("|---|---|---|---|---|---|")
    SUB = [("all 17", ALL), ("without MAE", [e for e in ALL if e != MAE]),
           ("without MAE and the two CLIP",
            [e for e in ALL if e != MAE and e not in CLIPS]),
           ("9 architectures", ARCH), ("9 pre-trainings", OBJECTIVE)]
    for name, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in pb for b in DATASETS)
              and all((e, b) in geo for b in DATASETS)]
        k = len(es)
        width_raw = [geo[(e, DATASETS[0])]["channels"] for e in es]
        G = [z([geo[(e, b)]["rank"][0] for e in es]) for b in DATASETS]
        P = [z([pb[f"{b}|{e}"]["value"] for e in es]) for b in DATASETS]
        r_pr = spearmanr(np.concatenate(G), np.concatenate(P)).statistic
        if np.std(width_raw) == 0:
            p(f"| {name} | {k} | {r_pr:+.3f} | - | - | width constant "
              f"({width_raw[0]} channels): nothing to control for |")
            continue
        W = [z(width_raw) for _ in DATASETS]
        r_w = spearmanr(np.concatenate(W), np.concatenate(P)).statistic
        rp, pv = permute(G, W, P, lambda g, w, q: partial(g, q, w))
        p(f"| {name} | {k} | {r_pr:+.3f} | {r_w:+.3f} | **{rp:+.3f}** "
          f"| {pv:.3f} |")
    p("")

    # ----- regression -----------------------------------------------------
    p("## 2. The same thing as a regression\n")
    p("Probe = b0 + b1*PR + b2*width, everything standardised within the "
      "domain.\n")
    p("| set | k | b1 (PR) | p perm | b2 (width) |")
    p("|---|---|---|---|---|")
    for name, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in pb for b in DATASETS)
              and all((e, b) in geo for b in DATASETS)]
        width_raw = [geo[(e, DATASETS[0])]["channels"] for e in es]
        if np.std(width_raw) == 0:
            continue
        G = [z([geo[(e, b)]["rank"][0] for e in es]) for b in DATASETS]
        P = [z([pb[f"{b}|{e}"]["value"] for e in es]) for b in DATASETS]
        W = [z(width_raw) for _ in DATASETS]
        b1, b2 = partial_beta(np.concatenate(P), np.concatenate(G),
                              np.concatenate(W))
        _, pv = permute(G, W, P, lambda g, w, q: partial_beta(q, g, w)[0])
        p(f"| {name} | {len(es)} | {b1:+.3f} | {pv:.3f} | {b2:+.3f} |")
    p("")

    out = ROOT / "results" / "PARTIAL.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
