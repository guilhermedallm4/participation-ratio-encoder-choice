#!/usr/bin/env python
"""Does geometry predict the probe?

Unit of analysis: the (encoder, domain) pair. That makes 6 x 4 = 24 points.

The correlation is computed WITHIN each domain, never across pooled domains:
the metric changes (AUC on mammography, accuracy elsewhere) and so does the
number of classes, so a point of accuracy on SUN397 is not comparable to a
point on DTD. Pooling would produce a high and empty correlation, driven by the
difference between tasks rather than between encoders.

THE CONTROL THAT MATTERS is width. An encoder with 2048 channels offers a
logistic regression more features than one with 384; if the probe tracks width,
there is nothing left to attribute to geometry. Width enters as if it were one
more geometric measure, and is judged by the same criterion.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = {"resnet50": "ResNet-50", "convnext_t": "ConvNeXt-T", "swin_t": "Swin-T",
     "maxvit_t": "MaxViT-T", "vit_s16": "ViT-S/16", "vit_b16": "ViT-B/16"}
B = ["mammography", "dtd", "pet", "sun397"]
root = Path("results")
s = json.loads((root / "probe.json").read_text())
t = json.loads((root / "measures.json").read_text())

geo = {}
for block in ("ci", "architectures"):
    for r in t[block]:
        if "dataset" in r:
            geo[(r["backbone"], r["dataset"])] = r

MEASURES = [("rank", lambda r: r["rank"][0], "participation ratio"),
            ("radius_px", lambda r: r["radius_px"][0], "field radius"),
            ("channels", lambda r: r["channels"], "width (control)")]

print("LINEAR PROBE, by domain")
print("=" * 78)
for b in B:
    rows = [(R[bb], s[f"{b}|{bb}"]) for bb in R if f"{b}|{bb}" in s]
    if not rows:
        continue
    m = rows[0][1]["metric"]
    print(f"\n{b}  ({m}, n={rows[0][1]['n']}, "
          f"{rows[0][1]['classes']} classes)")
    for name, d in sorted(rows, key=lambda x: -x[1]["value"]):
        print(f"   {name:12} {d['value']:.3f} [{d['lo']:.3f};{d['hi']:.3f}]")
    v = [d["value"] for _, d in rows]
    print(f"   spread across encoders: {max(v)-min(v):.3f}")

print("\n\nCORRELATION (Spearman) GEOMETRY x PROBE, within each domain")
print("=" * 78)
print(f"  {'':22}" + "".join(f"{b:>15}" for b in B) + f"{'median':>10}")
for field, f, lab in MEASURES:
    rs = []
    for b in B:
        bbs = [bb for bb in R if f"{b}|{bb}" in s and (bb, b) in geo]
        x = [f(geo[(bb, b)]) for bb in bbs]
        y = [s[f"{b}|{bb}"]["value"] for bb in bbs]
        rs.append(spearmanr(x, y).statistic if len(x) >= 4 else np.nan)
    print(f"  {lab:22}" + "".join(f"{r:>15.2f}" for r in rs)
          + f"{np.nanmedian(rs):>10.2f}")
print("\n  n = 6 encoders per domain; |rho| > 0.83 is p < 0.05 two-tailed.")
