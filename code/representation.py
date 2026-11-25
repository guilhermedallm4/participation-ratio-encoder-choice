#!/usr/bin/env python3
"""How each architecture family represents mammography.

The axis is the degree and the reach of spatial mixing:

    ResNet-50   pure convolution          local, hierarchical
    MaxViT-T    conv + neighbourhood+grid local AND global, hierarchical
    ViT-B/16    pure global attention     global, flat

NOTHING HERE TRAINS. These are forward passes through frozen encoders. That
matters for a reason specific to this work: we measured that changing only the
initialisation seed moves the AUC by 0.059 on ResNet-50 and 0.224 on ResNet-18.
Any claim about PERFORMANCE across families sinks in that noise. The measures
below do not depend on training, so they carry no seed variance, and they are
the part of the comparison that stays measurable.

MEASURES
  participation ratio  how many directions the representation actually uses
  spatial dispersion   how much of the variance sits BETWEEN image positions
  CKA                  do two networks encode the same similarity geometry?
  frequency band       which spatial scale each family responds to
  position probe       is the lesion position linearly recoverable?

The control that gives the participation ratio meaning is comparing mammography
against natural images: if ImageNet activates many directions and mammography
few, the number quantifies how narrow the domain is.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
import multitask as MT
from PIL import Image
from torchvision import transforms

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FAMILIES = [
    ("resnet50", "pure convolution", "local, hierarchical"),
    ("maxvit_t", "conv + neighbourhood and grid attention",
     "local and global, hierarchical"),
    ("vit_b16", "pure global attention", "global, flat"),
]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def transformation(norm):
    return transforms.Compose(
        [transforms.ToTensor(),
         MT.NORM_IMAGENET if norm == "imagenet" else MT.NORM_HALF])


def maps(alias, paths, size=512, bs=8, transform=None):
    """Feature map (N, C, H, W) of the frozen encoder."""
    bb, channels, norm = MT.build_backbone(alias, pre=True, trainable=False)
    bb = bb.to(DEV).eval()
    tf = transformation(norm)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = []
            for p in paths[i:i + bs]:
                im = Image.open(p).convert("L").resize(
                    (size, size), Image.Resampling.BILINEAR)
                a = np.asarray(im, dtype=np.float32) / 255.0
                if transform is not None:
                    a = transform(a)
                batch.append(tf(Image.fromarray(
                    (np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB")))
            out.append(bb(torch.stack(batch).to(DEV)).float().cpu())
    del bb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def participation_ratio(X):
    """Participation ratio of the covariance eigenvalues.

    (sum L)^2 / sum L^2. It equals 1 if all the variance lies in one direction
    and C if it is spread uniformly. It is a continuous count of "how many
    directions the representation uses", robust to tiny eigenvalues in a way
    the algebraic rank is not.
    """
    X = (X - X.mean(0, keepdim=True)).double()
    v = torch.linalg.svdvals(X) ** 2
    return float((v.sum() ** 2) / (v ** 2).sum())


def spatial_dispersion(M):
    """Fraction of the variance that lies BETWEEN positions, not between images.

    High = the representation distinguishes places inside the image, which is
    what localising requires. Low = the map is nearly uniform in space and only
    the global average carries information.
    """
    n, c, h, w = M.shape
    v = M.reshape(n, c, h * w).double()
    between_pos = v.var(dim=2).mean()
    between_img = v.mean(dim=2).var(dim=0).mean()
    return float(between_pos / (between_pos + between_img))


def cka(X, Y):
    """Linear CKA. Invariant to rotation and isotropic scaling, so it compares
    representations of different widths. 1 = the same similarity geometry
    across the images; 0 = no relation."""
    X = (X - X.mean(0)).double()
    Y = (Y - Y.mean(0)).double()
    return float((X.T @ Y).pow(2).sum() /
                 ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))


def band_filter(a, low, high):
    """Keep only the spatial frequencies between two fractions of Nyquist.

    A mammographic lesion is high-frequency texture over a low-frequency
    background. Perturbing band by band and measuring how much the
    representation moves says which scale each family responds to.
    """
    F = np.fft.fftshift(np.fft.fft2(a))
    h, w = a.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)
    F[(r < low) | (r >= high)] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def band_sensitivity(alias, paths, bands, size=512):
    """How much the representation moves when only one frequency band survives.

    Normalised by the norm of the original representation, so that families
    with different activation scales stay comparable.
    """
    base = maps(alias, paths, size).flatten(1)
    nb = base.norm(dim=1, keepdim=True).clamp_min(1e-6)
    out = {}
    for lo, hi in bands:
        m = maps(alias, paths, size,
                 transform=lambda a, lo=lo, hi=hi: band_filter(a, lo, hi)
                 ).flatten(1)
        out[f"{lo:.2f}-{hi:.2f}"] = float(
            ((m - base).norm(dim=1) / nb.squeeze()).mean())
    return out


def position_probe(M, target, n_comp=64, cv=5):
    """Is the lesion position recoverable LINEARLY from the map?

    Flattening the map gives tens of thousands of features for a few dozen
    images, which leaves the probe underdetermined. A PCA first solves it: it
    keeps the information and brings the dimension below the sample count.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    X = M.flatten(1).numpy()

    # 1. Drop nearly constant features. In a typical ReLU map, ~98% of the
    #    positions are dead across all images; standardising them divides by
    #    ~0 and produces huge values that wreck the regression.
    v = X.var(0)
    X = X[:, v > 1e-8 * max(v.max(), 1e-12)]
    if X.shape[1] == 0:
        return float("nan")

    # 2. The PCA runs INSIDE each training fold, which holds (cv-1)/cv of the
    #    samples. With n_comp ~ n_train the fit becomes determined and
    #    extrapolates without control on the test fold, so we cap it at half
    #    the training samples.
    n_train = int(len(X) * (cv - 1) / cv)
    n_comp = min(n_comp, max(2, n_train // 2), X.shape[1])

    # NO StandardScaler: in a ReLU map with ~10^5 surviving features, many have
    # tiny but nonzero variance, and standardising amplifies them until they
    # dominate the PCA (ResNet-50's R2 collapsed to -37). The PCA already
    # centres the data.
    pipe = make_pipeline(PCA(n_components=n_comp, random_state=0),
                         RidgeCV(alphas=np.logspace(-2, 5, 16)))
    return float(cross_val_score(pipe, X, target, cv=cv, scoring="r2").mean())


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="representation analysis")
    p.add_argument("--images", default="../data/inbreast_png_896")
    p.add_argument("--csv", default="../data/inbreast_images.csv")
    p.add_argument("--boxes", default="../data/inbreast_boxes.csv")
    p.add_argument("--out", default="../data/representation")
    p.add_argument("--n", type=int, default=200, help="mammography images")
    p.add_argument("--control", default=None,
                   help="directory of natural images for the domain contrast")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    imgs = Path(a.images)
    # Column names below (processed_image_filename, cx, cy, has_lesion) belong
    # to the INbreast derivative tables, which are not redistributed here.
    d = pd.read_csv(a.csv).sample(min(a.n, len(pd.read_csv(a.csv))),
                                  random_state=0)
    paths = [imgs / f for f in d.processed_image_filename]
    print(f"{len(paths)} mammograms | device {DEV}\n")

    M, pooled, res = {}, {}, {}
    for alias, mechanism, reach in FAMILIES:
        m = maps(alias, paths)
        M[alias] = m
        pooled[alias] = m.mean(dim=(2, 3))
        res[alias] = dict(mechanism=mechanism, reach=reach,
                          channels=int(m.shape[1]), map=int(m.shape[2]),
                          cell_px=512 // int(m.shape[2]))
        print(f"{alias:10} {mechanism:38} map {m.shape[2]}x{m.shape[3]}, "
              f"{m.shape[1]} channels, cell {512//m.shape[2]} px")

    print("\n=== 1. PARTICIPATION RATIO (directions the representation uses) ===")
    for alias, _, _ in FAMILIES:
        X = pooled[alias]
        pr = participation_ratio(X)
        res[alias]["participation_ratio"] = pr
        res[alias]["rank_fraction"] = pr / X.shape[1]
        print(f"  {alias:10} {pr:6.1f} of {X.shape[1]:5d} channels "
              f"({pr/X.shape[1]:5.1%})")

    if a.control:
        print("\n  -- control: natural images --")
        nat = sorted(Path(a.control).glob("*"))[:len(paths)]
        nat = [q for q in nat if q.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(nat) >= 32:
            for alias, _, _ in FAMILIES:
                X = maps(alias, nat).mean(dim=(2, 3))
                pr = participation_ratio(X)
                res[alias]["participation_ratio_natural"] = pr
                ratio = pr / res[alias]["participation_ratio"]
                print(f"  {alias:10} {pr:6.1f} ({pr/X.shape[1]:5.1%})  "
                      f"-> {ratio:.1f}x the mammography one")
        else:
            print(f"  too few control images ({len(nat)}); skipping")

    print("\n=== 2. SPATIAL DISPERSION (variance between positions) ===")
    for alias, _, _ in FAMILIES:
        v = spatial_dispersion(M[alias])
        res[alias]["spatial_dispersion"] = v
        print(f"  {alias:10} {v:.3f}")

    print("\n=== 3. CKA BETWEEN FAMILIES ===")
    al = [x[0] for x in FAMILIES]
    print(f"  {'':12}" + "".join(f"{x:>12}" for x in al))
    mat = {}
    for x in al:
        row = [cka(pooled[x], pooled[y]) for y in al]
        mat[x] = dict(zip(al, row))
        print(f"  {x:12}" + "".join(f"{v:12.3f}" for v in row))
    res["cka"] = mat

    print("\n=== 4. SENSITIVITY BY FREQUENCY BAND ===")
    print("    (relative change of the representation when only the band "
          "survives)")
    bands = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]
    lab = {"0.00-0.10": "very low", "0.10-0.25": "low",
           "0.25-0.50": "medium", "0.50-1.00": "high"}
    sub = paths[:48]
    print(f"  {'':10}" + "".join(f"{lab[f'{lo:.2f}-{hi:.2f}']:>13}"
                                 for lo, hi in bands))
    for alias, _, _ in FAMILIES:
        s = band_sensitivity(alias, sub, bands)
        res[alias]["band"] = s
        print(f"  {alias:10}" + "".join(f"{s[k]:13.3f}" for k in s))

    print("\n=== 5. LINEAR PROBE FOR LESION POSITION ===")
    bx = pd.read_csv(a.boxes)
    j = d.merge(bx[bx.has_lesion], on="processed_image_filename")
    idx = [i for i, f in enumerate(d.processed_image_filename)
           if f in set(j.processed_image_filename)]
    if len(idx) >= 30:
        order = {f: i for i, f in enumerate(d.processed_image_filename)}
        j = j.sort_values("processed_image_filename", key=lambda s: s.map(order))
        target = j[["cx", "cy"]].to_numpy()
        print(f"    n = {len(idx)} images with a lesion | cross-validated R2")
        for alias, _, _ in FAMILIES:
            r2 = position_probe(M[alias][idx], target)
            res[alias]["position_probe_r2"] = r2
            print(f"  {alias:10} R2 = {r2:+.3f}")
    else:
        print(f"  too few lesions ({len(idx)}); raise --n")

    (out / "representation.json").write_text(
        json.dumps(res, indent=1, default=float))
    print(f"\nwritten to {out}/representation.json")


if __name__ == "__main__":
    main()
