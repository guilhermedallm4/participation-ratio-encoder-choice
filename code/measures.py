#!/usr/bin/env python
"""The five controls that separate a claim from an observation.

The earlier measures (participation ratio, receptive field, residual flow,
shuffling) were taken ONCE, on three architectures that differ in four things
at the same time: mechanism, pre-training corpus, native resolution and
parameter count. Each test here attacks one of those problems.

  ci             Interval by resampling images. Without a noise floor,
                 "1.1x against 1.5x" is an observation, not a claim -- which is
                 exactly the lesson of the mammography study.
  resolution     ViT-B/16 was pre-trained at 224 px and we run it at 512, with
                 the positional encodings interpolated. Running all three sizes
                 separates "global attention" from "position extrapolation".
  background     Mammography has a lot of empty background. Applying the REAL
                 mammography MASKS over natural images separates "the domain is
                 narrow" from "there is nowhere to look".
  architectures  ConvNeXt-T (convolution, modern recipe), Swin-T (hierarchical
                 with local attention only) and ViT-S/16 (global attention,
                 IN-1k). They take n from 3 to 6 and break the
                 mechanism-recipe tie.
  ablation       Replace MaxViT's grid attention with the identity. This is the
                 only causal test: nothing else changes.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import multitask as MT
from representation import transformation, participation_ratio
from field_and_shuffle import load_batch, Shuffler, cka_gram, images
from residual_flow import channels_last

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 400
GRIDS = (4, 8, 16, 32)


class Shifted:
    """Shuffles on a grid shifted by `d` pixels.

    ALIGNMENT control. With 16 px patches in a 512 px image, the shuffling grid
    coincides with the token grid of a ViT-B/16 -- each patch is exactly one
    token. Shifting by half a token makes every patch straddle two tokens. If
    arrangement sensitivity collapses under the shift, the measure was reading
    the coincidence of the grids, not the dependence on arrangement.

    The roll is undone afterwards, so the content returns to its place; only
    the partition changes. The seam the roll creates at the borders exists in
    both modes (global and local) and cancels in the difference.
    """

    def __init__(self, shuffler, d):
        self.shuffler, self.d = shuffler, d

    def reset(self):
        if hasattr(self.shuffler, "reset"):
            self.shuffler.reset()

    def __call__(self, a):
        d = self.d
        return np.roll(self.shuffler(np.roll(a, (-d, -d), (0, 1))),
                       (d, d), (0, 1))


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def ci_sub(samples, stat, m=None, n_boot=N_BOOT, seed=0):
    """Interval by SUBSAMPLING without replacement, for spectral statistics.

    The participation ratio cannot be resampled with replacement: a duplicated
    image adds no direction to the covariance, so every resample has lower rank
    than the original sample and the interval does not even contain the point.
    Subsampling without replacement avoids the duplication, at the price of
    measuring at m < n -- and the participation ratio GROWS with n. So the
    point estimate is reported at m as well: point and interval stay on the
    same scale, and every comparison across architectures and datasets uses the
    same m.
    """
    rng = np.random.default_rng(seed)
    n = len(samples)
    m = m or max(2, n // 2)
    v = np.array([stat(rng.permutation(n)[:m]) for _ in range(n_boot)])
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(np.percentile(v, 2.5)),
            float(np.percentile(v, 97.5)), m)


def ci(samples, stat, n_boot=N_BOOT, seed=0):
    """(point, lower, upper) by resampling IMAGES with replacement.

    The resampled unit is the image, because the image is what varies between
    runs of this measure -- there is no training seed and no split, the
    encoders are frozen. The percentile is the 95% one.
    """
    rng = np.random.default_rng(seed)
    n = len(samples)
    v = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    v = np.array([x for x in v if np.isfinite(x)])
    return (float(stat(np.arange(n))), float(np.percentile(v, 2.5)),
            float(np.percentile(v, 97.5)))


# ---------------------------------------------------------------------------
# Per-image primitives (they keep enough to resample later)
# ---------------------------------------------------------------------------

#: When set, _model returns THIS (model, transformation) pair instead of
#: building one from the registry. That is how the ablation gets in: the rest
#: of the code does not need to know the model was modified.
SUBSTITUTE = None


def reset(t):
    """Zero the counter of a stateful transform, before each pass.

    ApplyMask cycles through the masks in the order the images arrive. If the
    counter is not zeroed between passes, the SAME image gets different masks
    in the base pass and in the shuffled one, and the delta would measure the
    change of mask instead of the shuffling.
    """
    if hasattr(t, "reset"):
        t.reset()
    return t


def _model(alias):
    if SUBSTITUTE is not None:
        return SUBSTITUTE
    m, _, norm = MT.build_backbone(alias, pre=True, trainable=False)
    return m.to(DEV).eval(), transformation(norm)


def map_features(alias, paths, size, bs=8, transform=None, model=None,
                 tf=None, gray=True):
    """Per image: spatial mean (C,), variance between positions (C,).

    Keeping those two vectors instead of the whole map lets the participation
    ratio and the spatial dispersion be recomputed a thousand times for free,
    and reproduces both measures exactly -- no approximation.
    """
    own = model is None and SUBSTITUTE is None
    if model is None:
        model, tf = _model(alias)
    reset(transform)
    mp, vp = [], []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            x = load_batch(paths[i:i + bs], tf, size, transform, gray).to(DEV)
            o = model(x).float()
            v = o.flatten(2)                      # (N, C, HW)
            mp.append(v.mean(2).cpu()); vp.append(v.var(2).cpu())
    if own:
        del model; torch.cuda.empty_cache()
    return torch.cat(mp), torch.cat(vp)


def final_field(alias, paths, size, bs=2, transform=None):
    """Gradient map of the central unit of the OUTPUT, per image, normalised.

    Only the final output, not block by block: this is the number that goes
    into the tables, and keeping one map per image allows resampling the exact
    estimator (profile of the mean of the maps) instead of the mean of the
    profiles, which is not the same thing.
    """
    model, tf = _model(alias)
    reset(transform)
    out = []
    for i in range(0, len(paths), bs):
        x = load_batch(paths[i:i + bs], tf, size,
                       transform).to(DEV).requires_grad_(True)
        o = model(x)
        h, w = o.shape[2], o.shape[3]
        g, = torch.autograd.grad(o[:, :, h // 2, w // 2].sum(), x)
        a = g.abs().sum(1).double()
        a = a / a.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        out.append(a.float().cpu())
        del x, g, o
    torch.cuda.empty_cache()
    if SUBSTITUTE is None:
        del model
    return torch.cat(out).numpy()


def radius_profile(G, q=0.5):
    """Radius accumulating fraction q of the mass of an (H,W) normalised map."""
    h, w = G.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2).ravel()
    o = np.argsort(r)
    return float(r[o][np.searchsorted(np.cumsum(G.ravel()[o]), q)])


class Composed:
    """Applies the image preparation and only then the shuffling."""

    def __init__(self, before, after):
        self.before, self.after = before, after

    def reset(self):
        reset(self.before); reset(self.after)

    def __call__(self, a):
        return self.after(self.before(a) if self.before is not None else a)


def shuffle_deltas(alias, paths, size, bs=8, grids=GRIDS, transform=None):
    """PER-IMAGE delta for each (grid, mode). Allows resampling the delta."""
    model, tf = _model(alias)
    base, _ = map_features(alias, paths, size, bs, transform, model, tf)
    nb = base.norm(dim=1).clamp_min(1e-8)
    out = {}
    for g in grids:
        for mode in ("global", "local"):
            sh = Shuffler(g, mode, seed=g * 10 + (mode == "local"))
            p, _ = map_features(alias, paths, size, bs,
                                Composed(transform, sh), model, tf)
            out[(g, mode)] = ((p - base).norm(dim=1) / nb).numpy()
    if SUBSTITUTE is None:
        del model
    torch.cuda.empty_cache()
    return out, base


# ---------------------------------------------------------------------------
# The full profile of one (architecture, image set, resolution)
# ---------------------------------------------------------------------------

def _disp(mp, vp, idx):
    between_pos = vp[idx].mean(0).mean()
    between_img = mp[idx].var(0).mean()
    return float(between_pos / (between_pos + between_img))


def profile(alias, paths, size=512, label="", transform=None,
            n_field=32, n_shuffle=96, ablated_model=None):
    mp, vp = map_features(alias, paths, size, transform=transform)
    *rank, m_rank = ci_sub(mp, lambda i: participation_ratio(mp[i]))
    rank_n = float(participation_ratio(mp))   # at full n, for continuity
    disp = ci(mp, lambda i: _disp(mp, vp, i))

    G = final_field(alias, paths[:n_field], size, transform=transform)
    # the estimator is the profile of the MEAN of the maps; resampling the
    # maps reproduces it
    radius = ci(G, lambda i: radius_profile(G[i].mean(0)))

    d, _ = shuffle_deltas(alias, paths[:n_shuffle], size, transform=transform)
    arrangement = {}
    for g in GRIDS:
        dg, dl = d[(g, "global")], d[(g, "local")]
        arrangement[g] = ci(dg, lambda i: float(dg[i].mean() - dl[i].mean()))

    r = dict(backbone=alias, label=label, size=size, channels=int(mp.shape[1]),
             n_images=len(paths), m_rank=m_rank, rank_n_full=rank_n,
             rank=rank, rank_frac=tuple(x / mp.shape[1] for x in rank),
             dispersion=disp, radius_px=radius,
             radius_frac=tuple(x / size for x in radius),
             arrangement={str(g): arrangement[g] for g in GRIDS})
    print(f"   {alias:11} {label:22} rank {rank[0]:5.1f} "
          f"[{rank[1]:.1f};{rank[2]:.1f}] (n {rank_n:5.1f})"
          f"  radius {radius[0]:5.1f} [{radius[1]:.1f};{radius[2]:.1f}] px"
          f"  arrangement32 {arrangement[32][0]:+.3f} "
          f"[{arrangement[32][1]:+.3f};{arrangement[32][2]:+.3f}]",
          flush=True)
    return r


def flow_ci(alias, paths, size=512):
    """Fraction of the stream each MECHANISM rewrites, with an interval.

    The blocks of one mechanism are averaged within each image before
    resampling, because the resampling unit is the image, not the block: blocks
    of the same image are not independent.
    """
    from residual_flow import measure
    _, det = measure(alias, paths, size, detail=True)
    per_mech = {}
    for name, (mech, v) in det.items():
        per_mech.setdefault(mech, []).append(v)
    out = {}
    for mech, vs in per_mech.items():
        M = np.stack(vs)                   # (blocks, images)
        m = M.mean(0)                      # mean over blocks, per image
        out[mech] = ci(m, lambda i: float(m[i].mean()))
        print(f"   {alias:11} {mech:24} branch {out[mech][0]:.3f} "
              f"[{out[mech][1]:.3f};{out[mech][2]:.3f}]", flush=True)
    return out


# ---------------------------------------------------------------------------
# Background control
# ---------------------------------------------------------------------------

def masks(paths, size=512, threshold=0.06):
    """Binary foreground masks of the mammograms.

    A plain threshold on intensity: in an FFDM the background is air,
    practically zero, and the breast is everything above. It does not need to
    be good segmentation -- it needs to be the SAME geometry of emptiness the
    model sees in mammography.
    """
    out = []
    for q in paths:
        a = np.asarray(Image.open(q).convert("L").resize(
            (size, size), Image.Resampling.BILINEAR),
            dtype=np.float32) / 255.0
        out.append(a > threshold)
    return out


class ApplyMask:
    """Zeroes the image outside the mask, cycling through the mask set."""

    def __init__(self, ms, seed=0):
        self.ms = ms
        self.order = np.random.default_rng(seed).permutation(len(ms))
        self.k = 0

    def reset(self):
        self.k = 0

    def __call__(self, a):
        m = self.ms[self.order[self.k % len(self.ms)]]
        self.k += 1
        return a * m


# ---------------------------------------------------------------------------

def grid_ablated(alias="maxvit_t"):
    """MaxViT with attn_grid replaced by the identity, everything else equal."""
    m, _, norm = MT.build_backbone(alias, pre=True, trainable=False)
    n = 0
    for stage in m.m.stages:
        for b in stage.blocks:
            b.attn_grid = torch.nn.Identity(); n += 1
    print(f"   (grid attention removed from {n} blocks)", flush=True)
    return m.to(DEV).eval(), transformation(norm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests",
                   default="ci,resolution,background,architectures,ablation")
    p.add_argument("--samples", default="data/samples")
    p.add_argument("--out", default="results")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    which = [t for t in a.tests.split(",") if t]
    fp = out / "measures.json"
    data = json.loads(fp.read_text()) if fp.exists() else {}

    def save(key, rows):
        data[key] = rows
        fp.write_text(json.dumps(data, indent=1, default=float))
        print(f"-> {key} saved", flush=True)

    datasets = ["mammography", "dtd", "pet", "sun397"]
    qs = {b: images(Path(a.samples) / b, a.n) for b in datasets}

    if "ci" in which:
        print("\n== 1. INTERVAL BY RESAMPLING (3 architectures x 4 datasets) ==",
              flush=True)
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb in ("resnet50", "maxvit_t", "vit_b16"):
                r = profile(bb, qs[b], 512, b); r["dataset"] = b; L.append(r)
            save("ci", L)

    if "resolution" in which:
        print("\n== 2. RESOLUTION: the ViT at 224, 384 and 512 ==", flush=True)
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb, sizes in (("vit_b16", (224, 384, 512)),
                              ("vit_s16", (224, 512))):
                for t in sizes:
                    r = profile(bb, qs[b], t, f"{b} @{t}")
                    r["dataset"] = b; L.append(r)
            save("resolution", L)

    if "background" in which:
        print("\n== 3. BACKGROUND CONTROL ==", flush=True)
        ms = masks(qs["mammography"][:a.n])
        frac = float(np.mean([m.mean() for m in ms]))
        print(f"   mean foreground fraction in mammography: {frac:.3f}",
              flush=True)
        L = [{"mammography_foreground_fraction": frac}]
        for b in ("dtd", "pet", "sun397"):
            print(f"  -- {b} with a mammography mask --", flush=True)
            for bb in ("resnet50", "maxvit_t", "vit_b16"):
                r = profile(bb, qs[b], 512, f"{b} masked",
                            transform=ApplyMask(ms))
                r["dataset"] = b; L.append(r)
            save("background", L)

    if "architectures" in which:
        print("\n== 4. THREE CONTROL ARCHITECTURES ==", flush=True)
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb in ("convnext_t", "swin_t", "vit_s16"):
                r = profile(bb, qs[b], 512, b); r["dataset"] = b; L.append(r)
            save("architectures", L)

    if "flow" in which:
        print("\n== 6. RESIDUAL FLOW WITH AN INTERVAL ==", flush=True)
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb in ("resnet50", "maxvit_t", "vit_b16"):
                L.append(dict(dataset=b, backbone=bb,
                              mech=flow_ci(bb, qs[b][:64])))
            save("flow", L)

    if "alignment" in which:
        print("\n== 7. GRID ALIGNMENT CONTROL ==", flush=True)
        L = []
        for b in ("mammography", "pet"):
            print(f"  -- {b} --", flush=True)
            for bb in ("resnet50", "maxvit_t", "vit_b16", "convnext_t",
                       "swin_t", "vit_s16"):
                mod, tf = _model(bb)
                base, _ = map_features(bb, qs[b][:96], 512, 8, None, mod, tf)
                nb = base.norm(dim=1).clamp_min(1e-8)
                row = dict(dataset=b, backbone=bb)
                for d in (0, 2, 8):
                    dd = {}
                    for mode in ("global", "local"):
                        e = Shuffler(32, mode, seed=320 + (mode == "local"))
                        t = e if d == 0 else Shifted(e, d)
                        pm, _ = map_features(bb, qs[b][:96], 512, 8, t, mod, tf)
                        dd[mode] = ((pm - base).norm(dim=1) / nb).numpy()
                    g, l = dd["global"], dd["local"]
                    row[f"shift{d}"] = ci(
                        g, lambda i: float(g[i].mean() - l[i].mean()))
                    print(f"   {bb:11} {b:11} shift {d:2d} px  "
                          f"arrangement {row[f'shift{d}'][0]:+.3f} "
                          f"[{row[f'shift{d}'][1]:+.3f};"
                          f"{row[f'shift{d}'][2]:+.3f}]", flush=True)
                del mod; torch.cuda.empty_cache()
                L.append(row)
            save("alignment", L)

    if "batch2" in which:
        print("\n== 8. SECOND BATCH OF ENCODERS ==", flush=True)
        new = ["vit_b16_dino", "vit_b16_mae", "vit_b16_clip", "maxvit_s",
               "efficientnet_b3", "resnet18"]
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb in new:
                r = profile(bb, qs[b], 512, b); r["dataset"] = b; L.append(r)
            save("batch2", L)

    if "flow2" in which:
        # the residual flow would be "of the architecture" -- but the four
        # ViT-B/16 share the SAME architecture and differ in pre-training
        # objective. If the rewritten fraction changes between them, the factor
        # is not the architecture but the architecture PLUS the objective, and
        # the paper's claim has to change.
        print("\n== 9. RESIDUAL FLOW BY PRE-TRAINING OBJECTIVE ==", flush=True)
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb in ("vit_b16", "vit_b16_dino", "vit_b16_mae",
                       "vit_b16_clip"):
                L.append(dict(dataset=b, backbone=bb,
                              mech=flow_ci(bb, qs[b][:64])))
            save("flow2", L)

    if "batch3" in which:
        print("\n== 10. NINE OBJECTIVES ON THE SAME ViT-B/16 ==", flush=True)
        new = ["vit_b16_sup21k", "vit_b16_sup1k", "vit_b16_deit3",
               "vit_b16_clip2b", "vit_b16_siglip"]
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb in new:
                r = profile(bb, qs[b], 512, b); r["dataset"] = b; L.append(r)
            save("batch3", L)

    if "flow3" in which:
        print("\n== 11. RESIDUAL FLOW ACROSS THE NINE OBJECTIVES ==",
              flush=True)
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb in ("vit_b16", "vit_b16_sup21k", "vit_b16_sup1k",
                       "vit_b16_deit3", "vit_b16_dino", "vit_b16_mae",
                       "vit_b16_clip", "vit_b16_clip2b", "vit_b16_siglip"):
                L.append(dict(dataset=b, backbone=bb,
                              mech=flow_ci(bb, qs[b][:64])))
            save("flow3", L)

    if "rgb" in which:
        # colour control: participation ratio only, on the nine ViT-B/16,
        # with no conversion
        print("\n== 12. RGB CONTROL (no conversion to grayscale) ==",
              flush=True)
        L = []
        for b in datasets:
            print(f"  -- {b} --", flush=True)
            for bb in ("vit_b16", "vit_b16_sup21k", "vit_b16_sup1k",
                       "vit_b16_deit3", "vit_b16_dino", "vit_b16_mae",
                       "vit_b16_clip", "vit_b16_clip2b", "vit_b16_siglip"):
                mp, _ = map_features(bb, qs[b], 512, gray=False)
                *rk, m = ci_sub(mp, lambda i: participation_ratio(mp[i]))
                L.append(dict(dataset=b, backbone=bb, rank=rk, m_rank=m))
                print(f"   {bb:16} {b:11} RGB rank {rk[0]:5.1f} "
                      f"[{rk[1]:.1f};{rk[2]:.1f}]", flush=True)
            save("rgb", L)

    if "ablation" in which:
        print("\n== 5. ABLATION OF MAXVIT'S GRID ATTENTION ==", flush=True)
        # Do NOT use "import measures as ME; ME.SUBSTITUTE = ...": running as
        # __main__, that import creates a SECOND module object, and the
        # assignment is not seen by _model, which reads this one's global. The
        # symptom is silent -- the ablation simply does not happen and the
        # numbers come out identical to the intact model.
        global SUBSTITUTE
        L = []
        SUBSTITUTE = grid_ablated()
        try:
            for b in datasets:
                print(f"  -- {b} --", flush=True)
                r = profile("maxvit_t", qs[b], 512, f"{b} without grid")
                r["dataset"] = b; r["variant"] = "no_grid"; L.append(r)
                save("ablation", L)
        finally:
            SUBSTITUTE = None

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
