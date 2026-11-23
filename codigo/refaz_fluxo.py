#!/usr/bin/env python
"""Refaz so a medida de fluxo residual, apos a correcao de channels-last.

O resume() mediava (2,3) em tensores (N,H,W,C) da atencao do MaxViT, o que
devolvia um vetor indexado por ALTURA em vez de por canal. As normas (ramo,
fluxo) nao eram afetadas -- sao sobre flatten(1), insensivel ao eixo -- mas o
cka_preserva dos sub-blocos local e global do MaxViT era calculado sobre o
objeto errado. Esta e a unica coluna que muda.
"""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fluxo_residual import mede

BANCOS = ["mamografia", "dtd", "pet", "sun397"]
FAMILIAS = ["resnet50", "maxvit_t", "vit_b16"]

base = Path("data/amostras"); saida = Path("resultados")
todas = []
for banco in BANCOS:
    qs = sorted(p for p in (base / banco).iterdir() if p.suffix.lower() == ".png")[:64]
    print(f"\n== {banco} ({len(qs)} imagens) ==", flush=True)
    linhas = []
    for bb in FAMILIAS:
        linhas += mede(bb, qs)
    g = pd.DataFrame(linhas); g["banco"] = banco
    todas.append(g)
    for (bb, mec), s in g.groupby(["backbone", "mecanismo"], sort=False):
        print(f"   {bb:10} {mec:24} ramo {s.ramo.mean():.3f}  "
              f"CKA {s.cka_preserva.mean():.3f}", flush=True)
    pd.concat(todas).to_csv(saida / "fluxo_bancos.csv", index=False)

# sincroniza o bloco "fluxo" do JSON com o CSV corrigido
j = json.loads((saida / "representacao_bancos.json").read_text())
for g in todas:
    banco = g.banco.iloc[0]
    if banco not in j:
        continue
    m = g.groupby(["backbone", "mecanismo"], sort=False)[
        ["ramo", "fluxo", "cka_preserva"]].mean().round(4)
    j[banco]["fluxo"] = {f"{k[0]}|{k[1]}": v for k, v in m.to_dict("index").items()}
(saida / "representacao_bancos.json").write_text(json.dumps(j, indent=1, default=float))
print("\nfluxo_bancos.csv e representacao_bancos.json atualizados", flush=True)
