#!/usr/bin/env python
"""Correct CONDITIONAL inference for the partial correlation (Freedman-Lane).

The final review pointed out a real error in the earlier version of this
analysis.

WHAT WAS WRONG. The null permuted participation ratio and width TOGETHER
against the probe. That destroys the width-probe association as well, which the
conditional null hypothesis must PRESERVE: the null of interest is "the ratio
adds nothing once width is controlled for", not "neither the ratio nor the
width says anything about the probe". The old null is broader than it needs to
be, and the p it produces does not answer the question the text asks.

WHAT IS RIGHT. Freedman-Lane, the standard procedure for testing one term in
the presence of a nuisance covariate:

  1. regress the probe on width          ->  fit f and residual e
  2. permute ONLY the residuals e         ->  e*   (at the encoder level, the
                                               same permutation across all four
                                               domains, as everywhere else in
                                               the paper)
  3. rebuild the null probe               ->  y* = f + e*
  4. recompute the partial statistic of the ratio against y*, given width

The fit f enters every draw untouched, so the width-probe relation survives the
null. Only the part of the probe that width does NOT explain gets shuffled,
which is exactly the part the ratio is accused of informing about.

Everything is on the rank scale, like the partial Spearman in the paper. Note
that resid(y*|width) = resid(e*|width), because f lies in the span of width:
the null statistic is the correlation of the OBSERVED residual of the ratio
with the residual of the permuted residuals.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from analysis_final import ALL, ARCH, OBJECTIVE, DATASETS, z        # noqa: E402

MAE = "vit_b16_mae"
CLIPS = ("vit_b16_clip", "vit_b16_clip2b")
N = 20000


def load():
    r = ROOT / "results"
    t = json.loads((r / "measures.json").read_text())
    pb = json.loads((r / "probe.json").read_text())
    geo = {(x["backbone"], x["dataset"]): x
           for bl in ("ci", "architectures", "batch2", "batch3")
           for x in t.get(bl, []) if "dataset" in x}
    return geo, pb


def fit_ls(y, x):
    """Return (fit, residual) of y regressed on x, with an intercept."""
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    f = X @ b
    return f, y - f


def freedman_lane(rg, rp, rw, k, nd, n=N, seed=0):
    """Conditional p for the partial of rg and rp given rw.

    rg, rp, rw are stacked vectors: nd blocks of k encoders, with the encoders
    in the same order inside every block.
    """
    f, e = fit_ls(rp, rw)
    _, resid_g = fit_ls(rg, rw)                      # fixed across every draw
    obs = float(np.corrcoef(resid_g, e)[0, 1])

    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        q = rng.permutation(k)
        e_perm = np.concatenate([e[d * k:(d + 1) * k][q] for d in range(nd)])
        _, resid_p = fit_ls(f + e_perm, rw)
        null[i] = np.corrcoef(resid_g, resid_p)[0, 1]
    return obs, float((np.abs(null) >= abs(obs)).mean())


def freedman_lane_beta(rg, rp, rw, k, nd, n=N, seed=0):
    """The same null, for coefficient b1 of probe ~ ratio + width."""
    def b1(y):
        X = np.column_stack([np.ones_like(rg), rg, rw])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(b[1])

    f, e = fit_ls(rp, rw)
    obs = b1(rp)
    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        q = rng.permutation(k)
        e_perm = np.concatenate([e[d * k:(d + 1) * k][q] for d in range(nd)])
        null[i] = b1(f + e_perm)
    return obs, float((np.abs(null) >= abs(obs)).mean())


def joint_permutation(rg, rp, rw, k, nd, n=N, seed=0):
    """The OLD null, reproduced only to show the difference."""
    _, e = fit_ls(rp, rw)
    def stat(g, w):
        _, a = fit_ls(g, w); _, b = fit_ls(rp, w)
        return float(np.corrcoef(a, b)[0, 1])
    obs = stat(rg, rw)
    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        q = rng.permutation(k)
        gq = np.concatenate([rg[d * k:(d + 1) * k][q] for d in range(nd)])
        wq = np.concatenate([rw[d * k:(d + 1) * k][q] for d in range(nd)])
        null[i] = stat(gq, wq)
    return obs, float((np.abs(null) >= abs(obs)).mean())


def main():
    geo, pb = load()
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    p("# Partial correlation: conditional inference by Freedman-Lane\n")
    p("The final review is right. The earlier null permuted ratio and width")
    p("**together** against the probe, which also destroys the width-probe")
    p("association. The correct conditional null preserves that association")
    p("and shuffles only the part of the probe that width does not explain.")
    p("That is what Freedman-Lane does, and it is what follows.\n")
    p(f"Permutation at the encoder level, the same one across all four "
      f"domains, {N} draws.\n")

    SUB = [("all 17", ALL), ("without MAE", [e for e in ALL if e != MAE]),
           ("without MAE and the two CLIP",
            [e for e in ALL if e != MAE and e not in CLIPS]),
           ("9 architectures", ARCH), ("9 pre-trainings", OBJECTIVE)]

    p("| set | k | partial rho | **p Freedman-Lane** | p of the old null "
      "| b1 | p F-L of b1 |")
    p("|---|---|---|---|---|---|---|")
    for name, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in pb for b in DATASETS)
              and all((e, b) in geo for b in DATASETS)]
        k, nd = len(es), len(DATASETS)
        width_raw = [geo[(e, DATASETS[0])]["channels"] for e in es]
        if np.std(width_raw) == 0:
            p(f"| {name} | {k} | - | - | - | - | width constant "
              f"({width_raw[0]} channels): nothing to control for |")
            continue
        G = np.concatenate([z([geo[(e, b)]["rank"][0] for e in es])
                            for b in DATASETS])
        P = np.concatenate([z([pb[f"{b}|{e}"]["value"] for e in es])
                            for b in DATASETS])
        W = np.concatenate([z(width_raw) for _ in DATASETS])
        rg, rp, rw = rankdata(G), rankdata(P), rankdata(W)

        rho, pv = freedman_lane(rg, rp, rw, k, nd)
        _, pv_old = joint_permutation(rg, rp, rw, k, nd)
        b1, pv_b1 = freedman_lane_beta(rg, rp, rw, k, nd)
        p(f"| {name} | {k} | **{rho:+.3f}** | **{pv:.3f}** | {pv_old:.3f} | "
          f"{b1:+.3f} | {pv_b1:.3f} |")
    p("")
    p("The point estimate of the partial does not move: it never depended on")
    p("the null. What moves is the $p$, and the $p$ is what the review")
    p("questioned.\n")

    out = ROOT / "results" / "PARTIAL_FL.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
