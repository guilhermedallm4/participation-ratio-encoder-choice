#!/usr/bin/env python
"""Comparative representation analysis across datasets.

Runs the measures of representation.py and residual_flow.py over several image
sets, to answer: how does the representation of each mechanism (local conv,
global attention, hierarchical local+global) respond to each TYPE of image?

Every measure is taken on FROZEN encoders, with no training -- therefore with
no seed variance and no split variance, the two sources that dominated the
mammography study.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from representation import (maps, participation_ratio, spatial_dispersion,
                            cka, band_sensitivity)
from residual_flow import measure as measure_flow

FAMILIES = [("resnet50", "pure convolution", "local, hierarchical"),
            ("maxvit_t", "conv + neighbourhood and grid attention",
             "local and global, hierarchical"),
            ("vit_b16", "pure global attention", "global, flat")]

BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]
BAND_LABEL = {"0.00-0.10": "very_low", "0.10-0.25": "low",
              "0.25-0.50": "medium", "0.50-1.00": "high"}


def images(d, n=None):
    qs = sorted(p for p in Path(d).iterdir() if p.suffix.lower() == ".png")
    return qs[:n] if n else qs


def analyse(name, paths, n_flow=64):
    print(f"\n{'=' * 70}\nDATASET: {name}  ({len(paths)} images)\n{'=' * 70}",
          flush=True)
    res, M, pooled = {}, {}, {}

    for alias, mech, reach in FAMILIES:
        m = maps(alias, paths)
        M[alias] = m
        pooled[alias] = m.mean(dim=(2, 3))
        res[alias] = dict(mechanism=mech, reach=reach,
                          channels=int(m.shape[1]), map=int(m.shape[2]))

    print("\n1. PARTICIPATION RATIO (directions the representation uses)",
          flush=True)
    for alias, _, _ in FAMILIES:
        X = pooled[alias]
        pr = participation_ratio(X)
        res[alias].update(participation_ratio=pr,
                          rank_fraction=pr / X.shape[1])
        print(f"   {alias:10} {pr:6.1f} of {X.shape[1]:5d} "
              f"({pr / X.shape[1]:5.1%})", flush=True)

    print("\n2. SPATIAL DISPERSION (variance between positions)", flush=True)
    for alias, _, _ in FAMILIES:
        v = spatial_dispersion(M[alias])
        res[alias]["spatial_dispersion"] = v
        print(f"   {alias:10} {v:.3f}", flush=True)

    print("\n3. CKA BETWEEN FAMILIES", flush=True)
    al = [x[0] for x in FAMILIES]
    res["cka"] = {x: {y: cka(pooled[x], pooled[y]) for y in al} for x in al}
    print(f"   {'':12}" + "".join(f"{x:>12}" for x in al), flush=True)
    for x in al:
        print(f"   {x:12}" + "".join(f"{res['cka'][x][y]:12.3f}" for y in al),
              flush=True)

    print("\n4. SENSITIVITY BY FREQUENCY BAND", flush=True)
    sub = paths[:48]
    print(f"   {'':10}" + "".join(f"{BAND_LABEL[f'{lo:.2f}-{hi:.2f}']:>14}"
                                  for lo, hi in BANDS), flush=True)
    for alias, _, _ in FAMILIES:
        s = band_sensitivity(alias, sub, BANDS)
        res[alias]["band"] = {BAND_LABEL[k]: v for k, v in s.items()}
        print(f"   {alias:10}" + "".join(f"{s[k]:14.3f}" for k in s),
              flush=True)

    del M, pooled
    torch.cuda.empty_cache()

    print("\n5. RESIDUAL FLOW (how much each mechanism rewrites)", flush=True)
    rows = []
    for alias, _, _ in FAMILIES:
        rows += measure_flow(alias, paths[:n_flow])
    import pandas as pd
    g = pd.DataFrame(rows)
    res["flow"] = g.groupby(["backbone", "mechanism"], sort=False)[
        ["branch", "flow", "cka_preserves"]].mean().round(4).to_dict("index")
    for (bb, mech), v in res["flow"].items():
        print(f"   {bb:10} {mech:24} branch {v['branch']:.3f}  "
              f"flow {v['flow']:.3f}  CKA {v['cka_preserves']:.3f}", flush=True)
    res["flow"] = {f"{k[0]}|{k[1]}": v for k, v in res["flow"].items()}
    return res, g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--datasets", default="dtd,sun397,pet")
    p.add_argument("--mammography", default=None,
                   help="directory of mammography PNGs, for reference")
    p.add_argument("--out", default="results")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    targets = {b: Path(a.samples) / b for b in a.datasets.split(",") if b}
    if a.mammography:
        targets["mammography"] = Path(a.mammography)

    everything, flows = {}, []
    for name, d in targets.items():
        if not d.exists():
            print(f"!! {name}: {d} does not exist, skipping", flush=True)
            continue
        qs = images(d, a.n)
        if len(qs) < 48:
            print(f"!! {name}: only {len(qs)} images, skipping", flush=True)
            continue
        everything[name], g = analyse(name, qs)
        g["dataset"] = name
        flows.append(g)
        (out / "representation_datasets.json").write_text(
            json.dumps(everything, indent=1, default=float))
        if flows:
            import pandas as pd
            pd.concat(flows).to_csv(out / "residual_flow_datasets.csv",
                                    index=False)
        print(f"\n-> partial results written ({len(everything)} datasets)",
              flush=True)

    print(f"\ndone: {list(everything)}", flush=True)


if __name__ == "__main__":
    main()
