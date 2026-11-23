#!/usr/bin/env python
"""Analise de representacao comparada entre bancos.

Roda as medidas de representacao.py e fluxo_residual.py sobre varios conjuntos
de imagens, para responder: como a representacao de cada mecanismo (conv local,
atencao global, local+global hierarquico) responde a cada TIPO de imagem?

Todas as medidas sao feitas em extratores CONGELADOS, sem treino -- portanto
sem variancia de semente nem de particao, as duas fontes que dominaram o
estudo de mamografia.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from representacao import (mapas, posto_efetivo, dispersao_espacial, cka,
                           sensibilidade_banda)
from fluxo_residual import mede as mede_fluxo

FAMILIAS = [("resnet50", "convolucao pura", "local, hierarquica"),
            ("maxvit_t", "conv + atencao bairro + grade", "local e global, hierarquica"),
            ("vit_b16", "atencao global pura", "global, plana")]

BANDAS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]
ROT_BANDA = {"0.00-0.10": "muito_baixa", "0.10-0.25": "baixa",
             "0.25-0.50": "media", "0.50-1.00": "alta"}


def imagens(d, n=None):
    qs = sorted(p for p in Path(d).iterdir() if p.suffix.lower() == ".png")
    return qs[:n] if n else qs


def analisa(nome, caminhos, n_fluxo=64):
    print(f"\n{'=' * 70}\nBANCO: {nome}  ({len(caminhos)} imagens)\n{'=' * 70}", flush=True)
    res, M, agr = {}, {}, {}

    for alias, mec, alc in FAMILIAS:
        m = mapas(alias, caminhos)
        M[alias] = m
        agr[alias] = m.mean(dim=(2, 3))
        res[alias] = dict(mecanismo=mec, alcance=alc, canais=int(m.shape[1]),
                          mapa=int(m.shape[2]))

    print("\n1. POSTO EFETIVO (quantas direcoes a representacao usa)", flush=True)
    for alias, _, _ in FAMILIAS:
        X = agr[alias]
        pe = posto_efetivo(X)
        res[alias].update(posto_efetivo=pe, posto_fracao=pe / X.shape[1])
        print(f"   {alias:10} {pe:6.1f} de {X.shape[1]:5d} ({pe / X.shape[1]:5.1%})", flush=True)

    print("\n2. DISPERSAO ESPACIAL (variancia entre posicoes)", flush=True)
    for alias, _, _ in FAMILIAS:
        v = dispersao_espacial(M[alias])
        res[alias]["dispersao_espacial"] = v
        print(f"   {alias:10} {v:.3f}", flush=True)

    print("\n3. CKA ENTRE FAMILIAS", flush=True)
    al = [x[0] for x in FAMILIAS]
    res["cka"] = {x: {y: cka(agr[x], agr[y]) for y in al} for x in al}
    print(f"   {'':12}" + "".join(f"{x:>12}" for x in al), flush=True)
    for x in al:
        print(f"   {x:12}" + "".join(f"{res['cka'][x][y]:12.3f}" for y in al), flush=True)

    print("\n4. SENSIBILIDADE POR BANDA DE FREQUENCIA", flush=True)
    sub = caminhos[:48]
    print(f"   {'':10}" + "".join(f"{ROT_BANDA[f'{lo:.2f}-{hi:.2f}']:>14}" for lo, hi in BANDAS), flush=True)
    for alias, _, _ in FAMILIAS:
        s = sensibilidade_banda(alias, sub, BANDAS)
        res[alias]["banda"] = {ROT_BANDA[k]: v for k, v in s.items()}
        print(f"   {alias:10}" + "".join(f"{s[k]:14.3f}" for k in s), flush=True)

    del M, agr
    torch.cuda.empty_cache()

    print("\n5. FLUXO RESIDUAL (quanto cada mecanismo reescreve)", flush=True)
    linhas = []
    for alias, _, _ in FAMILIAS:
        linhas += mede_fluxo(alias, caminhos[:n_fluxo])
    import pandas as pd
    g = pd.DataFrame(linhas)
    res["fluxo"] = g.groupby(["backbone", "mecanismo"], sort=False)[
        ["ramo", "fluxo", "cka_preserva"]].mean().round(4).to_dict("index")
    for (bb, mec), v in res["fluxo"].items():
        print(f"   {bb:10} {mec:24} ramo {v['ramo']:.3f}  fluxo {v['fluxo']:.3f}"
              f"  CKA {v['cka_preserva']:.3f}", flush=True)
    res["fluxo"] = {f"{k[0]}|{k[1]}": v for k, v in res["fluxo"].items()}
    return res, g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amostras", default="data/amostras")
    p.add_argument("--bancos", default="dtd,sun397,pet")
    p.add_argument("--mamografia", default=None,
                   help="diretorio de PNGs da mamografia, para referencia")
    p.add_argument("--out", default="resultados")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()

    saida = Path(a.out); saida.mkdir(parents=True, exist_ok=True)
    alvos = {b: Path(a.amostras) / b for b in a.bancos.split(",") if b}
    if a.mamografia:
        alvos["mamografia"] = Path(a.mamografia)

    tudo, fluxos = {}, []
    for nome, d in alvos.items():
        if not d.exists():
            print(f"!! {nome}: {d} nao existe, pulando", flush=True)
            continue
        qs = imagens(d, a.n)
        if len(qs) < 48:
            print(f"!! {nome}: so {len(qs)} imagens, pulando", flush=True)
            continue
        tudo[nome], g = analisa(nome, qs)
        g["banco"] = nome
        fluxos.append(g)
        (saida / "representacao_bancos.json").write_text(json.dumps(tudo, indent=1, default=float))
        if fluxos:
            import pandas as pd
            pd.concat(fluxos).to_csv(saida / "fluxo_bancos.csv", index=False)
        print(f"\n-> parcial gravado ({len(tudo)} bancos)", flush=True)

    print(f"\nconcluido: {list(tudo)}", flush=True)


if __name__ == "__main__":
    main()
