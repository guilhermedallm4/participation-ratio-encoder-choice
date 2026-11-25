#!/usr/bin/env python
"""Download dataset samples from Hugging Face mirrors.

The Oxford servers (thor.robots.ox.ac.uk), the source of DTD and Oxford-IIIT
Pet in torchvision, are unreachable from this machine. We use HF mirrors.

We sample the SAME N as mammography (387) from every dataset. Without that we
would confound "what kind of data" with "how much data" -- and the amount was
precisely the explanation for ViT-B/16's failure in that earlier study.
"""
import argparse, io, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

N_MAMMOGRAPHY = 387
SIZE = 512


def square(img, size=SIZE):
    """Central square crop and resize -- without distorting the aspect ratio."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    return img.resize((size, size), Image.Resampling.BILINEAR)


def from_files(repo, n, dest, seed=0):
    """Mirror with loose image files: download only the N we need, picked
    round-robin across the class folders."""
    fs = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
          if f.lower().endswith((".jpg", ".jpeg", ".png")) and "/" in f]
    per_class = defaultdict(list)
    for f in fs:
        per_class[Path(f).parent.name].append(f)
    classes = sorted(per_class)
    rng = np.random.RandomState(seed)
    chosen, i = [], 0
    while len(chosen) < n:          # round-robin across classes
        c = classes[i % len(classes)]
        pool = per_class[c]
        if pool:
            chosen.append(pool.pop(rng.randint(len(pool))))
        i += 1
        if i > n * 20:
            break
    labels = {}
    dest.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(sorted(chosen)):
        p = hf_hub_download(repo, f, repo_type="dataset")
        square(Image.open(p)).save(dest / f"{k:04d}.png")
        labels[f"{k:04d}.png"] = Path(f).parent.name
        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(chosen)}", flush=True)
    return labels


def from_parquet(repo, pattern, n, dest, seed=0):
    """Parquet mirror: download ONE shard and sample inside it."""
    import pandas as pd
    fs = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
          if f.endswith(".parquet") and pattern in f]
    p = hf_hub_download(repo, sorted(fs)[0], repo_type="dataset")
    d = pd.read_parquet(p)
    col_img = next(c for c in d.columns if c in ("image", "img", "jpg", "png"))
    col_lab = next((c for c in d.columns
                    if c in ("label", "labels", "cls", "category")), None)
    idx = np.random.RandomState(seed).choice(len(d), min(n, len(d)),
                                             replace=False)
    labels = {}
    dest.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(sorted(int(x) for x in idx)):
        cell = d.iloc[i][col_img]
        b = cell["bytes"] if isinstance(cell, dict) else cell
        square(Image.open(io.BytesIO(b))).save(dest / f"{k:04d}.png")
        labels[f"{k:04d}.png"] = int(d.iloc[i][col_lab]) if col_lab else -1
        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(idx)}", flush=True)
    return labels


DATASETS = {
    "dtd":    ("files",    "cansa/Describable-Textures-Dataset-DTD", None),
    "pet":    ("parquet",  "timm/oxford-iiit-pet", "train"),
    "sun397": ("parquet",  "tanganke/sun397", "train-00000"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/samples")
    p.add_argument("--n", type=int, default=N_MAMMOGRAPHY)
    p.add_argument("--datasets", default="dtd,pet,sun397")
    a = p.parse_args()

    root = Path(a.root)
    summary = {}
    for name in a.datasets.split(","):
        kind, repo, pattern = DATASETS[name]
        dest = root / name
        print(f"\n=== {name} ({repo}) ===", flush=True)
        lab = (from_files(repo, a.n, dest) if kind == "files"
               else from_parquet(repo, pattern, a.n, dest))
        n_cls = len(set(lab.values()))
        summary[name] = dict(repo=repo, images=len(lab), classes=n_cls,
                             dest=str(dest))
        (dest / "labels.json").write_text(json.dumps(lab, indent=1))
        print(f"   -> {len(lab)} images, {n_cls} classes", flush=True)

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
