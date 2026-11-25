#!/usr/bin/env python
"""PREDICTION test: does the participation ratio flag a collapse it never saw?

The structural weakness of the paper is that the title's claim rests on a
single collapsed checkpoint, MAE. Adding a second checkpoint TO THE SET and
redoing the correlation would be circular: the point enters the same broth the
claim came out of. What counts is to treat the new case as HELD OUT and to ask,
before looking, what the paper's claim predicts for it.

  vit_b16_beit3   BEiT-3 base, ".pt" checkpoint: PURE masked modelling, without
                  the fine-tuning the other timm beit checkpoints carry. Same
                  architecture, same 1025 tokens at 512 px.
                  PREDICTION: participation ratio in MAE's range (2-5) and
                  probe among the worst in the set.

  vit_b16_random  ViT-B/16 with no training at all. NOT a collapse candidate:
                  it is the reference point for "a representation one should
                  reject", with a known probe outcome. If the ratio does NOT
                  flag it, the claim is bounded -- it flags masked-modelling
                  collapse, not poor representations in general. Informative
                  either way.

Measured exactly as in the paper: participation ratio via ci_sub at
m = n/2 = 100 over 200 images, and a linear probe over three splits of the 387.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from measures import map_features, ci_sub                  # noqa: E402
from field_and_shuffle import images                       # noqa: E402
from representation import participation_ratio             # noqa: E402
from analysis_final import DATASETS                        # noqa: E402
import probe as PRB                                        # noqa: E402

NEW = ["vit_b16_beit3", "vit_b16_random"]


def main():
    out = ROOT / "results"
    fp = out / "second_collapse.json"
    data = json.loads(fp.read_text()) if fp.exists() else {}

    qs200 = {b: images(ROOT / "data" / "samples" / b, 200) for b in DATASETS}
    qs387 = {b: images(ROOT / "data" / "samples" / b, 387) for b in DATASETS}

    for alias in NEW:
        for b in DATASETS:
            key = f"{alias}|{b}"
            if key in data:
                print(f"  (already done) {key}", flush=True)
                continue
            t0 = time.time()

            # --- geometry, protocol identical to the paper's ---------------
            mp, _ = map_features(alias, qs200[b], 512)
            *rank, m_rank = ci_sub(mp, lambda i: participation_ratio(mp[i]))
            pr_full = float(participation_ratio(mp))

            # --- probe, identical protocol ---------------------------------
            X, _ = map_features(alias, qs387[b], 512)
            X = X.numpy().astype(np.float64)
            y, groups = PRB.labels(b, qs387[b])
            X, y, groups = PRB.prune(X, y, groups)
            metric, value, s_lo, s_hi, sd = PRB.probe(X, y, groups)

            data[key] = {"alias": alias, "dataset": b,
                         "channels": int(mp.shape[1]),
                         "rank": rank, "m_rank": m_rank,
                         "rank_n_full": pr_full,
                         "value": value, "ci": [s_lo, s_hi], "sd_split": sd,
                         "metric": metric,
                         "classes": int(len(set(y.tolist())))}
            fp.write_text(json.dumps(data, indent=1, default=float))
            print(f"  {alias:18} {b:11} PR={rank[0]:6.2f} "
                  f"[{rank[1]:.2f};{rank[2]:.2f}]  probe={value:.3f} "
                  f"[{s_lo:.3f};{s_hi:.3f}] ({metric})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            torch.cuda.empty_cache()

    print(f"\n-> {fp}", flush=True)


if __name__ == "__main__":
    main()
