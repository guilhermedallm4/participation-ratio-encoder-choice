#!/usr/bin/env python
"""Review item 1: is the participation ratio capped by n = 200?

The reviewer is right about the fact: the centred covariance of n images has
rank at most n-1, so with 200 images no estimated PR can exceed 199, however
many channels the encoder has. The question is whether that constraint BITES on
these data -- whether the measured PRs sit near the ceiling -- or whether it is
slack.

Two things are measured here, and both use ONE single extraction pass:

  ceiling   PR measured on all 200 against the limit of 199. If the largest PR
            in the set is a small fraction of the ceiling, the constraint does
            not bite.

  ladder    PR estimated on NESTED subsets of 50, 100, 150 and 200 images,
            without replacement, R repetitions. What matters is not the
            magnitude (which rises with n by construction, because more samples
            reveal more directions) but whether the ORDER among encoders holds.
            The paper's conclusion depends only on the order: the correlation
            is Spearman's and the reading is "MAE sits at the bottom".

Saves the per-image vectors to .npz so that any other statistic can be
recomputed later without touching the GPU again.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from measures import map_features                      # noqa: E402
from field_and_shuffle import images                   # noqa: E402
from representation import participation_ratio         # noqa: E402
from analysis_final import ALL, DATASETS, LABEL        # noqa: E402

SIZES = [50, 100, 150, 200]
R = 40                  # repetitions per size (the 200 one is exact, R=1)


def pr_of(X):
    return participation_ratio(torch.as_tensor(np.asarray(X, dtype=np.float64)))


def ladder(X, seed=0):
    """PR on subsets without replacement of 50, 100, 150 and 200 images."""
    rng = np.random.default_rng(seed)
    n = len(X)
    out = {}
    for m in SIZES:
        if m > n:
            continue
        if m == n:
            out[m] = {"mean": pr_of(X), "sd": 0.0, "r": 1, "ceiling": n - 1}
            continue
        v = [pr_of(X[rng.permutation(n)[:m]]) for _ in range(R)]
        out[m] = {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                  "r": R, "ceiling": m - 1}
    return out


def main():
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "pr_vectors"
    raw.mkdir(exist_ok=True)

    qs = {b: images(ROOT / "data" / "samples" / b, 200) for b in DATASETS}
    for b in DATASETS:
        print(f"{b}: {len(qs[b])} images", flush=True)

    fp = out / "pr_stability.json"
    data = json.loads(fp.read_text()) if fp.exists() else {}

    for alias in ALL:
        for b in DATASETS:
            key = f"{alias}|{b}"
            if key in data:
                print(f"  (already done) {key}", flush=True)
                continue
            t0 = time.time()
            try:
                mp, _ = map_features(alias, qs[b], 512)
            except Exception as e:                      # noqa: BLE001
                print(f"  !! {key}: {type(e).__name__}: {e}", flush=True)
                continue
            X = mp.numpy().astype(np.float64)
            np.savez_compressed(raw / f"{alias}__{b}.npz",
                                X=X.astype(np.float32))
            e = ladder(X)
            data[key] = {"alias": alias, "dataset": b, "n": int(X.shape[0]),
                         "channels": int(X.shape[1]),
                         "ladder": {str(k): v for k, v in e.items()}}
            fp.write_text(json.dumps(data, indent=1))
            pr200 = e[200]["mean"]
            print(f"  {LABEL.get(alias, alias):26} {b:11} C={X.shape[1]:5d} "
                  f"PR200={pr200:6.2f} ({100 * pr200 / 199:4.1f}% of ceiling)  "
                  f"PR50={e[50]['mean']:5.2f} PR100={e[100]['mean']:5.2f} "
                  f"PR150={e[150]['mean']:5.2f}  [{time.time() - t0:.0f}s]",
                  flush=True)
            torch.cuda.empty_cache()

    print(f"\n-> {fp}", flush=True)


if __name__ == "__main__":
    main()
