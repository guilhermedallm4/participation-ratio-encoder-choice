#!/usr/bin/env python
"""Does the geometry of the representation predict anything?

The paper motivates the geometric measures by saying that end-task accuracy is
too noisy in a small sample. But it never closes the loop: it does not show
that geometry PREDICTS performance. This is the probe that closes it.

A linear probe on the frozen features -- the SAME ones that feed the
participation ratio -- over 6 encoders x 4 domains, and its correlation with
the geometry. A null result is also useful: "geometry does not predict the
probe" is a conclusion, not a failure.

TWO DIFFERENT SPLITTING RULES, on purpose:
  natural datasets  independent images -> class-stratified CV
  mammography       103 patients across 387 images, several views per patient
                    -> StratifiedGroupKFold BY PATIENT, and resampling BY
                    PATIENT. Without this the same breast appears in train and
                    test, and the AUC inflates. That was the central point of
                    the earlier study.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from field_and_shuffle import images
from measures import map_features, ci

FAMILIES = ["resnet18", "resnet50", "convnext_t", "efficientnet_b3", "swin_t",
            "maxvit_t", "maxvit_s", "vit_s16",
            # the nine objectives on the SAME ViT-B/16
            "vit_b16", "vit_b16_sup21k", "vit_b16_sup1k", "vit_b16_deit3",
            "vit_b16_dino", "vit_b16_mae", "vit_b16_clip", "vit_b16_clip2b",
            "vit_b16_siglip"]
SEEDS = (0, 1, 2)
DATASETS = ["mammography", "dtd", "pet", "sun397"]
# Local mount holding the INbreast derivatives. The dataset is not
# redistributed here: it has to be requested from its authors.
COLD = Path("/frio/guilhermelima/quantum-transfer-learning-bcdr/data")
C_GRID = {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]}


def labels(dataset, paths):
    """(y, groups). groups is None where the images are independent."""
    if dataset != "mammography":
        d = json.loads((paths[0].parent / "labels.json").read_text())
        y = np.array([d[q.name] for q in paths])
        _, y = np.unique(y, return_inverse=True)
        return y, None
    # the sample was generated in alphabetical order from the cold mount;
    # verified byte by byte, so the k-th PNG matches the k-th sorted name
    sources = sorted(p.name for p in (COLD / "inbreast_png_896").glob("*.png"))
    meta = pd.read_csv(COLD / "inbreast_images.csv").set_index(
        "processed_image_filename")
    idx = [int(q.stem) for q in paths]
    sub = meta.loc[[sources[i] for i in idx]]
    return sub.classification.to_numpy(), sub.patient_id.to_numpy()


def prune(X, y, groups, minimum=5):
    """Drop classes too rare to stratify over 5 folds."""
    v, c = np.unique(y, return_counts=True)
    keep = np.isin(y, v[c >= minimum])
    return X[keep], y[keep], (None if groups is None else groups[keep])


def _one_split(X, y, groups, seed):
    """Out-of-fold predictions for ONE split."""
    binary = len(np.unique(y)) == 2
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=seed)
          if groups is not None
          else StratifiedKFold(5, shuffle=True, random_state=seed))
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    oof = np.zeros(len(y), float)
    for tr, te in cv.split(X, y, groups):
        b = GridSearchCV(pipe, C_GRID, cv=3, n_jobs=4,
                         scoring="roc_auc" if binary else "accuracy")
        b.fit(X[tr], y[tr])
        oof[te] = (b.predict_proba(X[te])[:, 1] if binary
                   else (b.predict(X[te]) == y[te]).astype(float))
    return oof, binary


def probe(X, y, groups, seeds=SEEDS, n_boot=200):
    """Out-of-fold metric over SEVERAL splits.

    The encoder is frozen, but the PROBE is not: it is trained, and therefore
    carries split variance. Resampling only the images, with one fixed split,
    would measure half the uncertainty. Here each seed gives one split; the
    point estimate is the mean over them and the interval pools the resamples
    of ALL of them, so that it carries both sources.

    The regularisation C comes out of an inner CV: fixing it would penalise
    encoders with 384 and with 2048 channels differently, measuring width.
    """
    points, draws = [], []
    for sd in seeds:
        oof, binary = _one_split(X, y, groups, sd)
        if binary:
            gs = np.unique(groups)
            def stat(i, f=oof):
                m = np.isin(groups, gs[i])
                return (roc_auc_score(y[m], f[m]) if len(np.unique(y[m])) == 2
                        else np.nan)
            points.append(roc_auc_score(y, oof))
            rng = np.random.default_rng(sd)
            draws += [stat(rng.integers(0, len(gs), len(gs)))
                      for _ in range(n_boot)]
        else:
            points.append(float(oof.mean()))
            rng = np.random.default_rng(sd)
            draws += [float(oof[rng.integers(0, len(oof), len(oof))].mean())
                      for _ in range(n_boot)]
    d = np.array([x for x in draws if np.isfinite(x)])
    name = "AUC" if len(np.unique(y)) == 2 else "accuracy"
    return (name, float(np.mean(points)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float(np.std(points)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--out", default="results")
    p.add_argument("--n", type=int, default=387)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fp = out / "probe.json"
    data = json.loads(fp.read_text()) if fp.exists() else {}

    for dataset in DATASETS:
        qs = images(Path(a.samples) / dataset, a.n)
        y, g = labels(dataset, qs)
        print(f"\n== {dataset}: {len(qs)} images, {len(np.unique(y))} classes"
              f"{'' if g is None else f', {len(np.unique(g))} patients'} ==",
              flush=True)
        for bb in FAMILIES:
            key = f"{dataset}|{bb}"
            if key in data:
                continue
            X, _ = map_features(bb, qs, 512)
            Xp, yp, gp = prune(X.numpy(), y, g)
            name, pt, lo, hi, sd = probe(Xp, yp, gp)
            data[key] = dict(dataset=dataset, backbone=bb, metric=name,
                             value=pt, lo=lo, hi=hi, sd_split=sd,
                             n=len(yp), classes=int(len(np.unique(yp))),
                             channels=X.shape[1])
            fp.write_text(json.dumps(data, indent=1, default=float))
            print(f"   {bb:16} {name} {pt:.3f} [{lo:.3f};{hi:.3f}]"
                  f"  sd across splits {sd:.3f}", flush=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
