#!/usr/bin/env python
"""A geometria prediz a sonda?

Unidade de analise: o par (extrator, dominio). Sao 6 x 4 = 24 pontos.

A correlacao e calculada DENTRO de cada dominio, nunca entre dominios
agrupados: a metrica muda (AUC na mamografia, acuracia nos outros) e o numero
de classes muda, entao um ponto de acuracia no SUN397 nao e comparavel a um
ponto no DTD. Agrupar produziria uma correlacao alta e vazia, dirigida pela
diferenca entre tarefas e nao entre extratores.

O CONTROLE QUE IMPORTA e a largura. Um extrator com 2048 canais oferece mais
features a uma regressao logistica do que um de 384; se a sonda seguir a
largura, nao ha o que atribuir a geometria. A largura entra como se fosse mais
uma medida de geometria, e e julgada pelo mesmo criterio.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = {"resnet50": "ResNet-50", "convnext_t": "ConvNeXt-T", "swin_t": "Swin-T",
     "maxvit_t": "MaxViT-T", "vit_s16": "ViT-S/16", "vit_b16": "ViT-B/16"}
B = ["mamografia", "dtd", "pet", "sun397"]
raiz = Path("resultados")
s = json.loads((raiz / "sonda.json").read_text())
t = json.loads((raiz / "testes.json").read_text())

geo = {}
for bloco in ("ic", "arquiteturas"):
    for r in t[bloco]:
        if "banco" in r:
            geo[(r["backbone"], r["banco"])] = r

MEDIDAS = [("posto", lambda r: r["posto"][0], "posto efetivo"),
           ("raio_px", lambda r: r["raio_px"][0], "raio do campo"),
           ("canais", lambda r: r["canais"], "largura (controle)")]

print("SONDA LINEAR, por dominio")
print("=" * 78)
for b in B:
    linhas = [(R[bb], s[f"{b}|{bb}"]) for bb in R if f"{b}|{bb}" in s]
    if not linhas:
        continue
    m = linhas[0][1]["metrica"]
    print(f"\n{b}  ({m}, n={linhas[0][1]['n']}, "
          f"{linhas[0][1]['classes']} classes)")
    for nome, d in sorted(linhas, key=lambda x: -x[1]["valor"]):
        print(f"   {nome:12} {d['valor']:.3f} [{d['lo']:.3f};{d['hi']:.3f}]")
    v = [d["valor"] for _, d in linhas]
    print(f"   amplitude entre extratores: {max(v)-min(v):.3f}")

print("\n\nCORRELACAO (Spearman) GEOMETRIA x SONDA, dentro de cada dominio")
print("=" * 78)
print(f"  {'':22}" + "".join(f"{b:>15}" for b in B) + f"{'mediana':>10}")
for campo, f, rot in MEDIDAS:
    rs = []
    for b in B:
        bbs = [bb for bb in R if f"{b}|{bb}" in s and (bb, b) in geo]
        x = [f(geo[(bb, b)]) for bb in bbs]
        y = [s[f"{b}|{bb}"]["valor"] for bb in bbs]
        rs.append(spearmanr(x, y).statistic if len(x) >= 4 else np.nan)
    print(f"  {rot:22}" + "".join(f"{r:>15.2f}" for r in rs)
          + f"{np.nanmedian(rs):>10.2f}")
print("\n  n = 6 extratores por dominio; |rho| > 0,83 e p < 0,05 bicaudal.")
