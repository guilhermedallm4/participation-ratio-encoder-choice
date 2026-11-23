#!/usr/bin/env python
"""Le estabilidade_pr.json e responde as tres perguntas do item 1.

  teto     As PR medidas chegam perto do limite n-1 = 199? Se a maior for uma
           fracao pequena, a restricao de posto nao morde.

  ordem    A PR estimada com 50, 100 e 150 imagens ordena os extratores como a
           de 200? A conclusao do artigo e de POSTO -- Spearman e "o MAE esta
           embaixo" --, entao e a ordem que precisa ser estavel, nao a
           magnitude (que sobe com n por construcao: mais amostras revelam mais
           direcoes).

  conclusao  O resultado principal do artigo sobrevive se a PR for estimada com
           metade das imagens? Refazemos a correlacao com a sonda e a analise
           de sensibilidade usando PR(m=100) no lugar de PR(200).
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent

from analise_final import TODOS, ARQ, OBJ, BANCOS, ROT, z, perm_extrator  # noqa: E402

MAE = "vit_b16_mae"
CLIPS = ("vit_b16_clip", "vit_b16_clip2b")
TAMANHOS = [50, 100, 150, 200]


def main():
    r = RAIZ / "resultados"
    est = json.loads((r / "estabilidade_pr.json").read_text())
    sd = json.loads((r / "sonda.json").read_text())

    def pr(e, b, m):
        return est[f"{e}|{b}"]["escada"][str(m)]["media"]

    falta = [f"{e}|{b}" for e in TODOS for b in BANCOS if f"{e}|{b}" not in est]
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    p("# Item 1 — a razão de participação está limitada por n = 200?\n")
    if falta:
        p(f"**INCOMPLETO**: faltam {len(falta)} pares: {falta[:6]}\n")

    # ---- 1. teto ---------------------------------------------------------
    p("## 1. Distância até o teto de posto\n")
    p("A covariância centrada de n imagens tem posto no máximo n−1. Com 200")
    p("imagens, nenhuma PR estimada pode passar de **199**.\n")
    todos_pr = [(pr(e, b, 200), e, b) for e in TODOS for b in BANCOS
                if f"{e}|{b}" in est]
    mx, me, mb = max(todos_pr)
    p(f"- maior PR observada nos 68 pares: **{mx:.1f}** "
      f"({ROT[me]}, {mb}) = **{100*mx/199:.1f} %** do teto")
    p(f"- mediana das 68: {np.median([v for v,_,_ in todos_pr]):.1f} "
      f"({100*np.median([v for v,_,_ in todos_pr])/199:.1f} % do teto)")
    p(f"- menor: {min(todos_pr)[0]:.1f} ({ROT[min(todos_pr)[1]]}, {min(todos_pr)[2]})\n")

    # ---- 2. ordem --------------------------------------------------------
    p("## 2. A ordem entre extratores sobrevive à redução de n?\n")
    p("Spearman entre PR(m) e PR(200), nos 17 extratores, dentro de cada domínio.\n")
    p("| domínio | m=50 | m=100 | m=150 |")
    p("|---|---|---|---|")
    for b in BANCOS:
        if any(f"{e}|{b}" not in est for e in TODOS):
            continue
        ref = [pr(e, b, 200) for e in TODOS]
        linha = [f"{spearmanr([pr(e, b, m) for e in TODOS], ref).statistic:.3f}"
                 for m in (50, 100, 150)]
        p(f"| {b} | " + " | ".join(linha) + " |")
    p("")

    p("### Magnitude: quanto a PR cresce com n (mediana entre os 68 pares)\n")
    p("| m | PR(m)/PR(200), mediana | mín | máx |")
    p("|---|---|---|---|")
    for m in TAMANHOS:
        raz = [pr(e, b, m) / pr(e, b, 200) for e in TODOS for b in BANCOS
               if f"{e}|{b}" in est]
        p(f"| {m} | {np.median(raz):.3f} | {min(raz):.3f} | {max(raz):.3f} |")
    p("")
    # o caso do MAE, separado: e a leitura do artigo
    p("### O checkpoint colapsado, separado\n")
    p("| extrator | domínio | PR(50) | PR(100) | PR(150) | PR(200) |")
    p("|---|---|---|---|---|---|")
    for e in (MAE, "vit_b16_sup1k"):
        for b in BANCOS:
            if f"{e}|{b}" not in est:
                continue
            p(f"| {ROT[e]} | {b} | " +
              " | ".join(f"{pr(e, b, m):.2f}" for m in TAMANHOS) + " |")
    p("")

    # ---- 3. a conclusao --------------------------------------------------
    p("## 3. O resultado principal com a PR estimada em metade das imagens\n")
    p("| conjunto | k | ρ com PR(200) | p | ρ com PR(100) | p | ρ com PR(50) | p |")
    p("|---|---|---|---|---|---|---|---|")
    SUB = [("todos os 17", TODOS), ("sem o MAE", [e for e in TODOS if e != MAE]),
           ("sem MAE e os dois CLIP",
            [e for e in TODOS if e != MAE and e not in CLIPS]),
           ("9 pré-treinos", OBJ), ("9 arquiteturas", ARQ)]
    for nome, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in sd for b in BANCOS)
              and all(f"{e}|{b}" in est for b in BANCOS)]
        if len(es) < 4:
            continue
        P = [z([sd[f"{b}|{e}"]["valor"] for e in es]) for b in BANCOS]
        cel = []
        for m in (200, 100, 50):
            G = [z([pr(e, b, m) for e in es]) for b in BANCOS]
            rho, pv, _ = perm_extrator(G, P, n=5000)
            cel.append(f"{rho:+.3f} | {pv:.3f}")
        p(f"| {nome} | {len(es)} | " + " | ".join(cel) + " |")
    p("")

    saida = r / "ESTABILIDADE.md"
    saida.write_text("\n".join(L) + "\n")
    print(f"\n-> {saida}", flush=True)


if __name__ == "__main__":
    main()
