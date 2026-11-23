#!/usr/bin/env python
"""Le segundo_colapso.json e julga a PREDICAO, sem mexer no conjunto original.

A afirmacao do artigo, em forma testavel: a razao de participacao separa a
representacao colapsada do resto por uma margem grande (no MAE, de tres a dez
vezes abaixo do extrator seguinte em qualquer dominio), e essa representacao
sonda entre as piores.

Tres leituras, nesta ordem:

  1. onde caem     PR e sonda dos dois novos contra a distribuicao dos 17.
  2. a predicao    o BEiT-3 foi flagrado? o aleatorio foi? Os dois casos sao
                   informativos, e o segundo delimita a afirmacao.
  3. se entrassem  o que aconteceria com a correlacao principal se os novos
                   entrassem no conjunto. Nao e o teste -- e transparencia,
                   para que ninguem precise perguntar.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent

from analise_final import TODOS, BANCOS, ROT, z, perm_extrator       # noqa: E402

MAE = "vit_b16_mae"
ROT2 = {"vit_b16_beit3": "ViT-B/16 BEiT-3 (.pt)",
        "vit_b16_aleatorio": "ViT-B/16 sem treino"}


def main():
    r = RAIZ / "resultados"
    t = json.loads((r / "testes.json").read_text())
    sd = json.loads((r / "sonda.json").read_text())
    nv = json.loads((r / "segundo_colapso.json").read_text())
    geo = {(x["backbone"], x["banco"]): x
           for bl in ("ic", "arquiteturas", "leva2", "leva3") for x in t.get(bl, [])
           if "banco" in x}

    novos = sorted({v["alias"] for v in nv.values()})
    falta = [f"{a}|{b}" for a in novos for b in BANCOS if f"{a}|{b}" not in nv]
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    p("# Teste de predição: um segundo colapso\n")
    if falta:
        p(f"**INCOMPLETO**: faltam {falta}\n")

    def pr(e, b):
        return (nv[f"{e}|{b}"]["posto"][0] if f"{e}|{b}" in nv
                else geo[(e, b)]["posto"][0])

    def so(e, b):
        return (nv[f"{e}|{b}"]["valor"] if f"{e}|{b}" in nv
                else sd[f"{b}|{e}"]["valor"])

    # ---- 1. onde caem ----------------------------------------------------
    p("## 1. Onde os dois caem\n")
    p("| extrator | " + " | ".join(BANCOS) + " |")
    p("|---" * (len(BANCOS) + 1) + "|")
    for e in novos + [MAE]:
        rot = ROT2.get(e, ROT.get(e, e))
        p(f"| **{rot}** — PR | " + " | ".join(f"{pr(e,b):.2f}" for b in BANCOS) + " |")
        p(f"| {rot} — sonda | " + " | ".join(f"{so(e,b):.3f}" for b in BANCOS) + " |")
    p("")
    p("Referência, os 17 do artigo (PR):\n")
    p("| domínio | mínimo sem MAE | mediana | máximo |")
    p("|---|---|---|---|")
    for b in BANCOS:
        v = np.array([geo[(e, b)]["posto"][0] for e in TODOS if e != MAE])
        p(f"| {b} | {v.min():.2f} | {np.median(v):.2f} | {v.max():.2f} |")
    p("")

    # ---- 2. a predicao ---------------------------------------------------
    p("## 2. A predição\n")
    p("Critério, o mesmo do artigo: *flagrado* = PR abaixo do mínimo dos demais")
    p("extratores naquele domínio, por uma margem que não seja de medida.\n")
    p("| extrator | domínio | PR | mínimo dos outros | razão | flagrado? | posto na sonda |")
    p("|---|---|---|---|---|---|---|")
    for e in novos + [MAE]:
        for b in BANCOS:
            outros = [geo[(o, b)]["posto"][0] for o in TODOS if o != e and o != MAE]
            mn = min(outros)
            v = pr(e, b)
            # posto da sonda entre os 17 + este
            todos_s = sorted([(so(o, b), o) for o in TODOS if o != e] + [(so(e, b), e)])
            pos = [o for _, o in todos_s].index(e) + 1
            p(f"| {ROT2.get(e, ROT.get(e,e))} | {b} | {v:.2f} | {mn:.2f} | "
              f"{mn/v:.1f}× | {'**sim**' if v < mn else 'não'} | "
              f"{pos}º de {len(todos_s)} |")
    p("")

    # ---- 3. se entrassem --------------------------------------------------
    p("## 3. Se entrassem no conjunto (transparência, não é o teste)\n")
    p("| conjunto | k | ρ | p |")
    p("|---|---|---|---|")
    for nome, es in [("os 17 do artigo", TODOS),
                     ("17 + BEiT-3", TODOS + ["vit_b16_beit3"]),
                     ("17 + os dois novos", TODOS + novos),
                     ("17 + BEiT-3, sem o MAE",
                      [e for e in TODOS if e != MAE] + ["vit_b16_beit3"])]:
        es = [e for e in es if all((f"{e_}|{b}" in nv) or ((e_, b) in geo)
                                   for e_ in [e] for b in BANCOS)]
        G = [z([pr(e, b) for e in es]) for b in BANCOS]
        P = [z([so(e, b) for e in es]) for b in BANCOS]
        rho, pv, _ = perm_extrator(G, P, n=20000)
        p(f"| {nome} | {len(es)} | {rho:+.3f} | {pv:.3f} |")
    p("")

    saida = r / "SEGUNDO_COLAPSO.md"
    saida.write_text("\n".join(L) + "\n")
    print(f"\n-> {saida}", flush=True)


if __name__ == "__main__":
    main()
