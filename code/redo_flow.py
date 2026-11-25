#!/usr/bin/env python
"""Redo only the residual-flow measure, after the channels-last fix.

summarise() was averaging over (2,3) on MaxViT attention tensors shaped
(N,H,W,C), which returned a vector indexed by HEIGHT instead of by channel. The
norms (branch, flow) were unaffected -- they are taken over flatten(1), which
is insensitive to the axis -- but cka_preserves for MaxViT's local and global
sub-blocks was computed on the wrong object. That is the only column that
changes.
"""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from residual_flow import measure

DATASETS = ["mammography", "dtd", "pet", "sun397"]
FAMILIES = ["resnet50", "maxvit_t", "vit_b16"]

base = Path("data/samples"); out = Path("results")
everything = []
for dataset in DATASETS:
    qs = sorted(p for p in (base / dataset).iterdir()
                if p.suffix.lower() == ".png")[:64]
    print(f"\n== {dataset} ({len(qs)} images) ==", flush=True)
    rows = []
    for bb in FAMILIES:
        rows += measure(bb, qs)
    g = pd.DataFrame(rows); g["dataset"] = dataset
    everything.append(g)
    for (bb, mech), s in g.groupby(["backbone", "mechanism"], sort=False):
        print(f"   {bb:10} {mech:24} branch {s.branch.mean():.3f}  "
              f"CKA {s.cka_preserves.mean():.3f}", flush=True)
    pd.concat(everything).to_csv(out / "residual_flow_datasets.csv", index=False)

# sync the "flow" block of the JSON with the corrected CSV
j = json.loads((out / "representation_datasets.json").read_text())
for g in everything:
    dataset = g.dataset.iloc[0]
    if dataset not in j:
        continue
    m = g.groupby(["backbone", "mechanism"], sort=False)[
        ["branch", "flow", "cka_preserves"]].mean().round(4)
    j[dataset]["flow"] = {f"{k[0]}|{k[1]}": v
                          for k, v in m.to_dict("index").items()}
(out / "representation_datasets.json").write_text(
    json.dumps(j, indent=1, default=float))
print("\nresidual_flow_datasets.csv and representation_datasets.json updated",
      flush=True)
