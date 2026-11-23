#!/usr/bin/env python
"""Baixa amostras dos bancos a partir de espelhos no Hugging Face.

Os servidores de Oxford (thor.robots.ox.ac.uk), origem do DTD e do Oxford-IIIT
Pet no torchvision, estao inacessiveis desta maquina. Usamos espelhos no HF.

Amostramos o MESMO N da mamografia (387) em todos os bancos. Sem isso,
confundiriamos "que tipo de dado" com "quanto dado" -- e a quantidade foi
justamente a explicacao do fracasso do ViT-B/16 naquele estudo.
"""
import argparse, io, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

N_MAMOGRAFIA = 387
TAM = 512


def quadra(img, tam=TAM):
    """Recorte central quadrado e redimensionamento -- sem distorcer proporcao."""
    img = img.convert("RGB")
    w, h = img.size
    lado = min(w, h)
    img = img.crop(((w - lado) // 2, (h - lado) // 2,
                    (w - lado) // 2 + lado, (h - lado) // 2 + lado))
    return img.resize((tam, tam), Image.Resampling.BILINEAR)


def de_arquivos(repo, n, destino, seed=0):
    """Espelho com imagens soltas: baixa so as N que precisamos, estratificado
    pela pasta de classe."""
    fs = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
          if f.lower().endswith((".jpg", ".jpeg", ".png")) and "/" in f]
    por_classe = defaultdict(list)
    for f in fs:
        por_classe[Path(f).parent.name].append(f)
    classes = sorted(por_classe)
    rng = np.random.RandomState(seed)
    escolhidas, i = [], 0
    while len(escolhidas) < n:          # round-robin entre classes
        c = classes[i % len(classes)]
        pool = por_classe[c]
        if pool:
            escolhidas.append(pool.pop(rng.randint(len(pool))))
        i += 1
        if i > n * 20:
            break
    rotulos = {}
    destino.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(sorted(escolhidas)):
        p = hf_hub_download(repo, f, repo_type="dataset")
        quadra(Image.open(p)).save(destino / f"{k:04d}.png")
        rotulos[f"{k:04d}.png"] = Path(f).parent.name
        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(escolhidas)}", flush=True)
    return rotulos


def de_parquet(repo, padrao, n, destino, seed=0):
    """Espelho em parquet: baixa UM fragmento e amostra dentro dele."""
    import pandas as pd
    fs = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
          if f.endswith(".parquet") and padrao in f]
    p = hf_hub_download(repo, sorted(fs)[0], repo_type="dataset")
    d = pd.read_parquet(p)
    col_img = next(c for c in d.columns if c in ("image", "img", "jpg", "png"))
    col_rot = next((c for c in d.columns
                    if c in ("label", "labels", "cls", "category")), None)
    idx = np.random.RandomState(seed).choice(len(d), min(n, len(d)), replace=False)
    rotulos = {}
    destino.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(sorted(int(x) for x in idx)):
        cel = d.iloc[i][col_img]
        b = cel["bytes"] if isinstance(cel, dict) else cel
        quadra(Image.open(io.BytesIO(b))).save(destino / f"{k:04d}.png")
        rotulos[f"{k:04d}.png"] = int(d.iloc[i][col_rot]) if col_rot else -1
        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(idx)}", flush=True)
    return rotulos


BANCOS = {
    "dtd":    ("arquivos", "cansa/Describable-Textures-Dataset-DTD", None),
    "pet":    ("parquet",  "timm/oxford-iiit-pet", "train"),
    "sun397": ("parquet",  "tanganke/sun397", "train-00000"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default="data/amostras")
    p.add_argument("--n", type=int, default=N_MAMOGRAFIA)
    p.add_argument("--bancos", default="dtd,pet,sun397")
    a = p.parse_args()

    raiz = Path(a.raiz)
    resumo = {}
    for nome in a.bancos.split(","):
        tipo, repo, padrao = BANCOS[nome]
        destino = raiz / nome
        print(f"\n=== {nome} ({repo}) ===", flush=True)
        rot = (de_arquivos(repo, a.n, destino) if tipo == "arquivos"
               else de_parquet(repo, padrao, a.n, destino))
        n_cls = len(set(rot.values()))
        resumo[nome] = dict(repo=repo, imagens=len(rot), classes=n_cls,
                            destino=str(destino))
        (destino / "rotulos.json").write_text(json.dumps(rot, indent=1))
        print(f"   -> {len(rot)} imagens, {n_cls} classes", flush=True)

    raiz.mkdir(parents=True, exist_ok=True)
    (raiz / "resumo.json").write_text(json.dumps(resumo, indent=1))
    print("\n" + json.dumps(resumo, indent=1), flush=True)


if __name__ == "__main__":
    main()
