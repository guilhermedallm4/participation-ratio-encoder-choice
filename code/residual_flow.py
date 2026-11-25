#!/usr/bin/env python
"""How much each mechanism MOVES the representation, along the depth.

The five measures in representation.py look at the encoder's final output.
This one looks at the path to it. Every modern architecture is a residual
stream:

    x_{n+1} = x_n + F(x_n)

where F is a convolution (ResNet), a windowed attention (MaxViT attn_block), a
grid attention (MaxViT attn_grid) or a global attention (ViT). Measuring
||F(x)|| / ||x|| says what fraction of the representation each mechanism
REWRITES, and CKA(x, x+F(x)) says how much of the geometry it PRESERVES.

It answers two questions directly:
  - does the ResNet residual preserve or rewrite the spatial information?
  - do MaxViT's local and global attentions compete or split the work?
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import multitask as MT
from representation import transformation

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sub_blocks(backbone, model):
    """List of (name, mechanism, module) for the residual sub-blocks, in order."""
    out = []
    if backbone.startswith("resnet"):
        r = model[0] if isinstance(model, torch.nn.Sequential) else model
        # nn.Sequential(*children[:-2]) -> indices 4..7 are layer1..layer4
        for e, i in enumerate((4, 5, 6, 7), start=1):
            for j, block in enumerate(model[i]):
                out.append((f"s{e}.b{j}", "residual conv", block))
    elif backbone.startswith("maxvit"):
        m = model.m if hasattr(model, "m") else model
        for e, stage in enumerate(m.stages, start=1):
            for j, block in enumerate(stage.blocks):
                out.append((f"s{e}.b{j}.conv", "conv (MBConv)", block.conv))
                out.append((f"s{e}.b{j}.local", "7x7 window attention",
                            block.attn_block))
                out.append((f"s{e}.b{j}.global", "grid attention",
                            block.attn_grid))
    elif backbone.startswith("vit_"):
        m = model.m if hasattr(model, "m") else model
        # Hooking block.attn would be WRONG: its input is norm1(x), not the
        # residual stream x, and its output does not include the sum. The
        # measured difference would not be the residual update. The whole Block
        # does x = x + attn(norm1(x)); x = x + mlp(norm2(x)) -- stream in,
        # stream out, exactly like MaxViT's PartitionAttentionCl (which also
        # embeds its own MLP) and like the ResNet Bottleneck.
        for j, block in enumerate(m.blocks):
            out.append((f"b{j}", "global attention + MLP", block))
    return out


def cka_gram(X, Y):
    """Linear CKA in Gram form: O(n^2) instead of O(d^2).

    ||X^T Y||_F^2 = tr(K L) with K = XX^T, L = YY^T, so
    CKA = tr(KL) / (||K||_F ||L||_F). With n=64 samples and d~10^6 features,
    the feature-space form would allocate a d x d matrix; this one allocates
    n x n.
    """
    X = (X - X.mean(0)).double(); Y = (Y - Y.mean(0)).double()
    K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / (K.norm() * L.norm()).clamp_min(1e-12))


def channels_last(mod):
    """Does the module return (N,H,W,C) instead of (N,C,H,W)?

    timm runs MaxViT attention channels-last (PartitionAttentionCl, the Cl
    suffix), while the MBConv of the same block runs NCHW. Both formats coexist
    inside the same model, so anything that averages spatially has to know
    which is which: averaging (2,3) on an NHWC tensor would average width and
    CHANNEL, returning a vector indexed by HEIGHT -- a meaningless object that
    still has the right shape and slips through unnoticed.
    """
    return type(mod).__name__.endswith("Cl")


def summarise(t, nhwc=False):
    """(N,C,H,W), (N,H,W,C) or (N,L,C) -> (N,C) by spatial averaging.

    Keeping the full map for 16 sub-blocks blows up memory; the similarity
    geometry across images, which is what CKA measures, survives spatial
    averaging -- and that is how representation.py already computes the final
    CKA.
    """
    if t.dim() == 4:
        return t.mean(dim=(1, 2)) if nhwc else t.mean(dim=(2, 3))
    if t.dim() == 3:
        return t.mean(dim=1)
    return t.reshape(t.shape[0], -1)


def measure(backbone, paths, size=512, bs=8, detail=False):
    model, channels, norm = MT.build_backbone(backbone, pre=True,
                                              trainable=False)
    model = model.to(DEV).eval()
    inner = model.m if hasattr(model, "m") else model
    blocks = sub_blocks(backbone,
                        model if backbone.startswith("resnet") else inner)

    reg = {n: {"mech": mech, "in": [], "out": [], "frac": [], "shape_ok": True}
           for n, mech, _ in blocks}
    branch = {}        # name -> norm of the residual branch for this batch
    hooks = []

    # The ResNet Bottleneck does out = relu(identity + F(x)): the ReLU comes
    # AFTER the sum, so (output - input) is NOT the residual branch, it
    # includes the ReLU clipping. In ViT and MaxViT, which are PreNorm with a
    # plain sum, (output - input) IS exactly the branch. To measure the same
    # quantity in all three, we capture F directly at the last BN of the
    # ResNet branch.
    def make_branch(name):
        def g(mod, inp, out_):
            branch[name] = out_.detach().flatten(1).norm(dim=1).cpu()
        return g

    if backbone.startswith("resnet"):
        for n, _, mod in blocks:
            hooks.append(mod.bn3.register_forward_hook(make_branch(n)))

    def make(name):
        def g(mod, inp, out_):
            cl = channels_last(mod)
            e = inp[0].detach()
            s = out_.detach() if torch.is_tensor(out_) else out_[0].detach()
            if e.shape != s.shape:   # shape changes: not a pure residual step
                reg[name]["shape_ok"] = False
                return
            # rewritten fraction = ||residual branch|| / ||incoming stream||
            pe = e.flatten(1); ps = s.flatten(1)
            ne = pe.norm(dim=1).clamp_min(1e-8).cpu()
            flow = (ps - pe).norm(dim=1).cpu() / ne   # how much the STREAM moves
            nr = branch.get(name)                     # ResNet: F at bn3
            br = (nr / ne) if nr is not None else flow
            reg[name]["frac"].append(br)
            reg[name].setdefault("flow", []).append(flow)
            reg[name]["in"].append(summarise(e, cl).float().cpu())
            reg[name]["out"].append(summarise(s, cl).float().cpu())
        return g

    for n, _, mod in blocks:
        hooks.append(mod.register_forward_hook(make(n)))

    tf = transformation(norm)
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = []
            for q in paths[i:i + bs]:
                im = Image.open(q).convert("L").resize(
                    (size, size), Image.Resampling.BILINEAR)
                batch.append(tf(im.convert("RGB")))
            model(torch.stack(batch).to(DEV))
    for h in hooks:
        h.remove()

    rows, per_image = [], {}
    for n, mech, _ in blocks:
        if not reg[n]["shape_ok"] or not reg[n]["frac"]:
            continue
        # keep the branch IMAGE BY IMAGE, so the interval can be computed by
        # resampling images -- an already aggregated mean does not allow it
        per_image[n] = (mech, torch.cat(reg[n]["frac"]).numpy())
        E = torch.cat(reg[n]["in"]); S = torch.cat(reg[n]["out"])
        rows.append(dict(backbone=backbone, block=n, mechanism=mech,
                         branch=float(torch.cat(reg[n]["frac"]).mean()),
                         flow=float(torch.cat(reg[n]["flow"]).mean()),
                         cka_preserves=cka_gram(E, S)))
    del model
    torch.cuda.empty_cache()
    return (rows, per_image) if detail else rows


def main():
    p = argparse.ArgumentParser(description="residual flow and attention")
    p.add_argument("--images", default="../data/inbreast_png_896")
    p.add_argument("--csv", default="../data/inbreast_images.csv")
    p.add_argument("--out", default="../data/representation")
    p.add_argument("--n", type=int, default=64)
    a = p.parse_args()

    d = pd.read_csv(a.csv).sample(a.n, random_state=0)
    paths = [Path(a.images) / f for f in d.processed_image_filename]
    print(f"{len(paths)} mammograms | {DEV}\n")

    everything = []
    for b in ("resnet50", "maxvit_t", "vit_b16"):
        print(f"-- {b} --")
        rows = measure(b, paths)
        everything += rows
        g = pd.DataFrame(rows)
        for mech, sub in g.groupby("mechanism", sort=False):
            print(f"   {mech:22} n={len(sub):3d}  branch {sub.branch.mean():.3f}"
                  f"  flow {sub.flow.mean():.3f}"
                  f"  CKA {sub.cka_preserves.mean():.3f}")
    df = pd.DataFrame(everything)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "residual_flow.csv", index=False)
    print(f"\nwritten to {out}/residual_flow.csv  ({len(df)} sub-blocks)")


if __name__ == "__main__":
    main()
