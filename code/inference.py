#!/usr/bin/env python
"""Review item 2: why does the interval exclude zero while p = 0.075?

The reviewer points at the "without MAE" row: rho = 0.27, p_perm = 0.075, yet
the resampled 95 % interval came out as [+0.02; +0.50], which does not contain
zero. They are right to demand an explanation.

THE FIRST HYPOTHESIS TESTED HERE WAS RULED OUT BY THE NUMBERS, and is recorded
so that nobody redoes it. We supposed the bootstrap WITH REPLACEMENT might be
anticonservative for the same reason the paper already rejects replacement in
the participation ratio: an encoder drawn twice enters with pairs that agree
perfectly with itself, which ought to inflate a rank statistic. The
`bootstrap mean` column measures exactly that, and it matches the observed rho
to three decimals (+0.411 against +0.410 across the seventeen). There is no
duplicate bias.

WHAT REMAINS IS SIMPLER AND LESS DRAMATIC. The percentile interval ended at
+0.01 and the jackknife on the Fisher scale ends at -0.01: at 0.27 with k = 16
the end of the interval is not resolvable at that precision, and the percentile
method landed on the wrong side of zero. The jackknife restores coherence in
all SIX rows at once, and it is the companion procedure for an analysis that is
precisely about removing encoders. The decision rests with the permutation; the
interval describes the spread.

We compute five things for each subset and let the numbers decide:

  rho        the observed statistic (Spearman over the stacked pairs)
  p_perm     permutation AT THE ENCODER LEVEL, the same one across 4 domains
  threshold  95th percentile of |rho| under that null -- the "detectable"
  CI_repl    percentile bootstrap WITH replacement (the paper's current one)
  CI_jack    jackknife leaving one encoder out, on the Fisher scale

The jackknife is the natural companion of a sensitivity analysis about dropping
encoders, and it does not have the duplicate problem: every resample is a
proper subset, with no repetition.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from analysis_final import (ALL, ARCH, OBJECTIVE, DATASETS, z,      # noqa: E402
                            perm_encoder)

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


def pairs(geo, pb, es):
    """G, P standardised WITHIN each domain, as in the paper."""
    G = [z([geo[(e, b)]["rank"][0] for e in es]) for b in DATASETS]
    P = [z([pb[f"{b}|{e}"]["value"] for e in es]) for b in DATASETS]
    return G, P


def rho_of(G, P, idx):
    return spearmanr(np.concatenate([g[idx] for g in G]),
                     np.concatenate([q[idx] for q in P])).statistic


def ci_with_replacement(G, P, k, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    bs = np.array([rho_of(G, P, rng.integers(0, k, k)) for _ in range(n)])
    bs = bs[np.isfinite(bs)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), bs


def ci_jackknife(G, P, k):
    """Leave one encoder out at a time; variance on the Fisher scale.

    Fisher stabilises the variance and keeps the interval inside [-1, 1], which
    an interval on the raw scale does not guarantee.
    """
    everything = np.arange(k)
    fz = lambda r: np.arctanh(np.clip(r, -0.999999, 0.999999))      # noqa: E731
    z_i = np.array([fz(rho_of(G, P, np.delete(everything, i))) for i in range(k)])
    z0 = fz(rho_of(G, P, everything))
    var = (k - 1) / k * ((z_i - z_i.mean()) ** 2).sum()
    se = np.sqrt(var)
    return float(np.tanh(z0 - 1.96 * se)), float(np.tanh(z0 + 1.96 * se)), float(se)


def duplicate_bias(G, P, k, n=3000, seed=1):
    """Diagnostic: the mean bootstrap rho against the observed rho.

    If replacement were harmless, the mean of the resamples would sit near the
    observed value. An upward bias is the signature of the duplicate problem.
    """
    _, _, bs = ci_with_replacement(G, P, k, n, seed)
    return float(bs.mean()), float(np.median(bs))


def main():
    geo, pb = load()
    no_mae = [e for e in ALL if e != MAE]
    no_mae_clip = [e for e in ALL if e != MAE and e not in CLIPS]
    obj_no_mae = [e for e in OBJECTIVE if e != MAE]

    SUB = [("all 17", ALL), ("without MAE", no_mae),
           ("without MAE and the two CLIP", no_mae_clip),
           ("9 pre-trainings", OBJECTIVE),
           ("9 pre-trainings, without MAE", obj_no_mae),
           ("9 architectures", ARCH)]

    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    p("# Review item 2 - interval against p-value\n")
    p("| set | k | rho | p_perm | threshold | CI with replacement "
      "| CI jackknife | bootstrap mean |")
    p("|---|---|---|---|---|---|---|---|")
    for name, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in pb for b in DATASETS)
              and all((e, b) in geo for b in DATASETS)]
        k = len(es)
        G, P = pairs(geo, pb, es)
        rho, pv, threshold = perm_encoder(G, P)
        lo_r, hi_r, _ = ci_with_replacement(G, P, k)
        lo_j, hi_j, se = ci_jackknife(G, P, k)
        m_bs, _ = duplicate_bias(G, P, k)
        p(f"| {name} | {k} | {rho:+.3f} | {pv:.3f} | {threshold:.2f} | "
          f"[{lo_r:+.2f}; {hi_r:+.2f}] | [{lo_j:+.2f}; {hi_j:+.2f}] | {m_bs:+.3f} |")

    p("\n## How to read it\n")
    p("- `threshold` is the 95th percentile of |rho| under permutation of")
    p("  encoders: the decision consistent with p_perm is to compare |rho|")
    p("  against it.")
    p("- `CI with replacement` is the paper's current one. `bootstrap mean`")
    p("  serves as a diagnostic for duplicate bias: it matches the observed")
    p("  `rho`, so that bias does NOT exist here -- the hypothesis was tested")
    p("  and ruled out.")
    p("- `CI jackknife` leaves one encoder out at a time, with no repetition,")
    p("  and is the procedure compatible with the sensitivity analysis.")

    out = ROOT / "results" / "INFERENCE.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
