#!/usr/bin/env python
"""Two measures of the spatial REACH of each mechanism.

The measures in representation.py look at the final output; those in
residual_flow.py look at how much each block rewrites. Neither says AT WHAT
DISTANCE each mechanism looks. That is exactly the axis separating the three
families (local / global / hierarchical local+global), so it is what is left to
measure.

  1. EFFECTIVE RECEPTIVE FIELD (from the inside out)
     Gradient of the CENTRAL unit of a block with respect to the input image.
     The THEORETICAL receptive field of global attention is the whole image
     already at the first block; the EFFECTIVE one is not, and the effective
     one is what describes the model. Luo et al. (2016) showed that in a ResNet
     it grows with the square root of depth and stays far below the theoretical
     value. The question here is how the growth curve differs between
     convolution, global attention and the hybrid.

  2. PATCH SHUFFLING (from the outside in)
     Cut the image into a g x g grid, permute the patches and measure how much
     the POOLED representation moves. The pooling is what makes the measure
     informative: for a purely local encoder whose field is smaller than a
     patch, moving a patch moves its features along with it, and the spatial
     average does not change -- delta ~ 0. The larger the delta, the more the
     representation depends on the ARRANGEMENT and not only on the content.

     A confound to control: shuffling creates artificial seams at the patch
     borders, and the number of seams grows with g. So each grid runs in two
     modes with the SAME number of seams:
         global  free permutation among the g^2 patches (destroys arrangement)
         local   permutation only within 2x2 neighbourhoods (keeps the coarse
                 arrangement)
     The difference between the two isolates arrangement sensitivity from the
     border artefact, which affects both equally.

Everything on a FROZEN encoder, with no training: no seed variance and no split
variance.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import multitask as MT
from representation import transformation
from residual_flow import sub_blocks, channels_last

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILIES = ["resnet50", "maxvit_t", "vit_b16"]


def load_batch(paths, tf, size=512, transform=None, gray=True):
    """Batch (N,3,size,size), already normalised, optionally perturbed.

    gray=False keeps the colour. It is the control for the objection that
    converting everything to grayscale -- done so that colour does not confound
    the comparison with mammography -- might penalise models trained with
    language over colour images, such as CLIP, more than the others. The
    perturbations (mask, shuffling) operate in 2D and only apply in grayscale.
    """
    batch = []
    for q in paths:
        if not gray:
            batch.append(tf(Image.open(q).convert("RGB")
                            .resize((size, size), Image.Resampling.BILINEAR)))
            continue
        im = Image.open(q).convert("L").resize((size, size),
                                               Image.Resampling.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        if transform is not None:
            a = transform(a)
        batch.append(tf(Image.fromarray(
            (np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB")))
    return torch.stack(batch)


# ---------------------------------------------------------------------------
# 1. Effective receptive field
# ---------------------------------------------------------------------------

def _central_unit(out, mod, n_prefix):
    """Sum of the channels at the central position -- one scalar per image.

    Three layouts coexist in these models, and confusing them raises no error:
    it returns a wrong number with the right shape.
      (N,C,H,W)  MBConv, ResNet
      (N,H,W,C)  MaxViT attention (PartitionAttentionCl, channels-last)
      (N,L,C)    ViT tokens, with L = prefix (CLS) + grid
    """
    if out.dim() == 4:
        if channels_last(mod):                           # (N,H,W,C)
            h, w = out.shape[1], out.shape[2]
            return out[:, h // 2, w // 2, :].sum()
        h, w = out.shape[2], out.shape[3]                # (N,C,H,W)
        return out[:, :, h // 2, w // 2].sum()
    g = int(round((out.shape[1] - n_prefix) ** 0.5))     # (N,L,C)
    t = out[:, n_prefix:, :].reshape(out.shape[0], g, g, -1)
    return t[:, g // 2, g // 2, :].sum()


def _profile(G, size):
    """From a normalised (H,W) gradient map to the reach statistics.

    radius_50 / radius_90  distance from the centre accumulating 50% / 90% of
                           the mass. For reference, a UNIFORM gradient over the
                           whole image gives radius_50 ~ 205 px and
                           radius_90 ~ 280 px at 512 px, so that is the
                           ceiling, not 256.
    area_90                fraction of the pixels needed to accumulate 90% of
                           the mass, ordered by magnitude. It measures
                           CONCENTRATION without assuming the field is centred.
    """
    h, w = G.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2).ravel()
    g = G.ravel()
    o = np.argsort(r)
    cum = np.cumsum(g[o])
    rs = r[o]
    radius = {f"radius_{int(q*100)}": float(rs[np.searchsorted(cum, q)])
              for q in (0.5, 0.9)}
    gs = np.sort(g)[::-1]
    a90 = int(np.searchsorted(np.cumsum(gs), 0.9)) + 1
    radius["area_90"] = a90 / g.size
    return radius


def receptive_field(backbone, paths, size=512, bs=2):
    """Effective receptive field of each sub-block, in input pixels."""
    model, _, norm = MT.build_backbone(backbone, pre=True, trainable=False)
    model = model.to(DEV).eval()
    inner = model.m if hasattr(model, "m") else model
    blocks = sub_blocks(backbone,
                        model if backbone.startswith("resnet") else inner)

    store = {}
    hooks = [mod.register_forward_hook(
        lambda m, e, s, n=n: store.__setitem__(
            n, s if torch.is_tensor(s) else s[0]))
        for n, _, mod in blocks]

    n_pref = int(getattr(inner, "num_prefix_tokens", 0) or 0)
    tf = transformation(norm)
    total = {n: np.zeros((size, size), dtype=np.float64) for n, _, _ in blocks}
    seen = 0
    for i in range(0, len(paths), bs):
        x = load_batch(paths[i:i + bs], tf, size).to(DEV).requires_grad_(True)
        store.clear()
        model(x)
        for j, (n, _, mod) in enumerate(blocks):
            if n not in store:
                continue
            target = _central_unit(store[n], mod, n_pref)
            # retain_graph until the last block: one backward per block over
            # the same graph is far cheaper than one forward per block.
            g, = torch.autograd.grad(target, x,
                                     retain_graph=(j < len(blocks) - 1))
            a = g.abs().sum(1).double().cpu().numpy()     # sum the 3 channels
            a = a / np.maximum(a.sum(axis=(1, 2), keepdims=True), 1e-30)
            total[n] += a.sum(0)
        seen += x.shape[0]
        del x, g
        torch.cuda.empty_cache()
    for h in hooks:
        h.remove()

    rows = []
    for n, mech, _ in blocks:
        if total[n].sum() == 0:
            continue
        G = total[n] / total[n].sum()
        rows.append(dict(backbone=backbone, block=n, mechanism=mech,
                         n_images=seen, **_profile(G, size)))
    del model
    torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# 2. Patch shuffling
# ---------------------------------------------------------------------------

class Shuffler:
    """Permutes the patches of a g x g grid, in global or local mode.

    It holds its own generator and is reinitialised on every call, so the three
    families see EXACTLY the same permutations on the same images -- the
    comparison between them carries no sampling noise.
    """

    def __init__(self, g, mode, seed=0):
        self.g, self.mode, self.rng = g, mode, np.random.default_rng(seed)

    def _permutation(self):
        g = self.g
        idx = np.arange(g * g).reshape(g, g)
        if self.mode == "global":
            return self.rng.permutation(idx.ravel()).reshape(g, g)
        # local: permute only within each 2x2 neighbourhood of patches
        out = idx.copy()
        for i in range(0, g, 2):
            for j in range(0, g, 2):
                b = idx[i:i + 2, j:j + 2].ravel()
                out[i:i + 2, j:j + 2] = self.rng.permutation(b).reshape(2, 2)
        return out

    def __call__(self, a):
        g, (h, w) = self.g, a.shape
        ph, pw = h // g, w // g
        r = a[:ph * g, :pw * g].reshape(g, ph, g, pw).transpose(0, 2, 1, 3)
        p = self._permutation().ravel()
        r = r.reshape(g * g, ph, pw)[p].reshape(g, g, ph, pw)
        return r.transpose(0, 2, 1, 3).reshape(ph * g, pw * g)


def _pooled(backbone, paths, tf, size, bs, transform=None, model=None):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            x = load_batch(paths[i:i + bs], tf, size, transform).to(DEV)
            o = model(x).float()
            out.append((o.mean(dim=(2, 3)) if o.dim() == 4
                        else o.mean(1)).cpu())
    return torch.cat(out)


def cka_gram(X, Y):
    X = (X - X.mean(0)).double(); Y = (Y - Y.mean(0)).double()
    K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / (K.norm() * L.norm()).clamp_min(1e-12))


def shuffling(backbone, paths, grids=(4, 8, 16, 32), size=512, bs=8):
    model, _, norm = MT.build_backbone(backbone, pre=True, trainable=False)
    model = model.to(DEV).eval()
    tf = transformation(norm)

    base = _pooled(backbone, paths, tf, size, bs, None, model)
    nb = base.norm(dim=1).clamp_min(1e-8)
    rows = []
    for g in grids:
        for mode in ("global", "local"):
            p = _pooled(backbone, paths, tf, size, bs,
                        Shuffler(g, mode, seed=g * 10 + (mode == "local")),
                        model)
            rows.append(dict(backbone=backbone, grid=g, patch_px=size // g,
                             mode=mode,
                             delta=float(((p - base).norm(dim=1) / nb).mean()),
                             cka=cka_gram(base, p)))
            print(f"   {backbone:10} g={g:2d} ({size//g:3d} px) {mode:7} "
                  f"delta {rows[-1]['delta']:.3f}  CKA {rows[-1]['cka']:.3f}",
                  flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------

def images(d, n):
    return sorted(p for p in Path(d).iterdir()
                  if p.suffix.lower() == ".png")[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--datasets", default="mammography,dtd,pet,sun397")
    p.add_argument("--out", default="results")
    p.add_argument("--n-field", type=int, default=16)
    p.add_argument("--n-shuffle", type=int, default=96)
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fields, shuffles = [], []
    for dataset in [b for b in a.datasets.split(",") if b]:
        d = Path(a.samples) / dataset
        if not d.exists():
            print(f"!! {dataset}: {d} does not exist, skipping", flush=True)
            continue
        print(f"\n{'=' * 70}\nDATASET: {dataset}\n{'=' * 70}", flush=True)

        print("\n1. EFFECTIVE RECEPTIVE FIELD", flush=True)
        qs = images(d, a.n_field)
        for bb in FAMILIES:
            l = receptive_field(bb, qs)
            for r in l:
                r["dataset"] = dataset
            fields += l
            g = pd.DataFrame(l)
            for mech, s in g.groupby("mechanism", sort=False):
                print(f"   {bb:10} {mech:24} n={len(s):2d}  radius50 "
                      f"{s.radius_50.min():5.1f}->{s.radius_50.max():5.1f} px",
                      flush=True)
            pd.DataFrame(fields).to_csv(out / "receptive_field.csv",
                                        index=False)

        print("\n2. PATCH SHUFFLING", flush=True)
        qs = images(d, a.n_shuffle)
        for bb in FAMILIES:
            l = shuffling(bb, qs)
            for r in l:
                r["dataset"] = dataset
            shuffles += l
            pd.DataFrame(shuffles).to_csv(out / "shuffling.csv", index=False)
        print(f"\n-> partial results written", flush=True)

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
