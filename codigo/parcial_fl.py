#!/usr/bin/env python
"""Inferencia CONDICIONAL correta para a correlacao parcial (Freedman-Lane).

O parecer final apontou um erro real na versao anterior desta analise.

O QUE ESTAVA ERRADO. O nulo permutava razao de participacao e largura JUNTAS
contra a sonda. Isso destroi tambem a associacao largura-sonda, que a hipotese
nula condicional deve PRESERVAR: o nulo de interesse e "a razao nao acrescenta
nada depois de controlar a largura", nao "nem a razao nem a largura dizem algo
sobre a sonda". O nulo antigo e mais amplo do que o necessario, e o p que sai
dele nao responde a pergunta que o texto faz.

O QUE ESTA CERTO. Freedman-Lane, que e o procedimento padrao para testar um
termo na presenca de um covariavel de incomodo:

  1. regride-se a sonda na largura        ->  ajuste f e residuo e
  2. permutam-se SO os residuos e          ->  e*   (no nivel do extrator, a
                                                mesma permutacao nos quatro
                                                dominios, como no resto do
                                                artigo)
  3. remonta-se a sonda nula               ->  y* = f + e*
  4. recalcula-se a estatistica parcial de razao contra y*, dado largura

O ajuste f entra intacto em todo sorteio, entao a relacao largura-sonda
sobrevive ao nulo. So a parte da sonda que a largura NAO explica e embaralhada
-- que e exatamente a parte sobre a qual a razao e acusada de informar.

Tudo na escala de postos, como a Spearman parcial do artigo. Note que
resid(y*|largura) = resid(e*|largura), porque f esta no espaco da largura:
a estatistica nula e a correlacao do residuo OBSERVADO da razao com o residuo
dos residuos permutados.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent

from analise_final import TODOS, ARQ, OBJ, BANCOS, z              # noqa: E402

MAE = "vit_b16_mae"
CLIPS = ("vit_b16_clip", "vit_b16_clip2b")
N = 20000


def carrega():
    r = RAIZ / "resultados"
    t = json.loads((r / "testes.json").read_text())
    sd = json.loads((r / "sonda.json").read_text())
    geo = {(x["backbone"], x["banco"]): x
           for bl in ("ic", "arquiteturas", "leva2", "leva3") for x in t.get(bl, [])
           if "banco" in x}
    return geo, sd


def ajusta(y, x):
    """Devolve (ajuste, residuo) de y regredido em x, com intercepto."""
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    f = X @ b
    return f, y - f


def freedman_lane(rg, rp, rw, k, nd, n=N, semente=0):
    """p condicional para a parcial de rg e rp dado rw.

    rg, rp, rw sao vetores empilhados: nd blocos de k extratores, na mesma
    ordem de extratores em cada bloco.
    """
    f, e = ajusta(rp, rw)
    _, resid_g = ajusta(rg, rw)                      # fixo em todo sorteio
    obs = float(np.corrcoef(resid_g, e)[0, 1])

    rng = np.random.default_rng(semente)
    nulo = np.empty(n)
    for i in range(n):
        q = rng.permutation(k)
        e_perm = np.concatenate([e[d * k:(d + 1) * k][q] for d in range(nd)])
        _, resid_p = ajusta(f + e_perm, rw)
        nulo[i] = np.corrcoef(resid_g, resid_p)[0, 1]
    return obs, float((np.abs(nulo) >= abs(obs)).mean())


def freedman_lane_beta(rg, rp, rw, k, nd, n=N, semente=0):
    """O mesmo nulo, para o coeficiente b1 da regressao sonda ~ razao+largura."""
    def b1(y):
        X = np.column_stack([np.ones_like(rg), rg, rw])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(b[1])

    f, e = ajusta(rp, rw)
    obs = b1(rp)
    rng = np.random.default_rng(semente)
    nulo = np.empty(n)
    for i in range(n):
        q = rng.permutation(k)
        e_perm = np.concatenate([e[d * k:(d + 1) * k][q] for d in range(nd)])
        nulo[i] = b1(f + e_perm)
    return obs, float((np.abs(nulo) >= abs(obs)).mean())


def conjunto_permuta(rg, rp, rw, k, nd, n=N, semente=0):
    """O nulo ANTIGO, reproduzido só para mostrar a diferença."""
    _, e = ajusta(rp, rw)
    def est(g, w):
        _, a = ajusta(g, w); _, b = ajusta(rp, w)
        return float(np.corrcoef(a, b)[0, 1])
    obs = est(rg, rw)
    rng = np.random.default_rng(semente)
    nulo = np.empty(n)
    for i in range(n):
        q = rng.permutation(k)
        gq = np.concatenate([rg[d * k:(d + 1) * k][q] for d in range(nd)])
        wq = np.concatenate([rw[d * k:(d + 1) * k][q] for d in range(nd)])
        nulo[i] = est(gq, wq)
    return obs, float((np.abs(nulo) >= abs(obs)).mean())


def main():
    geo, sd = carrega()
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    p("# Correlação parcial: inferência condicional por Freedman–Lane\n")
    p("O parecer final está certo. O nulo anterior permutava razão e largura")
    p("**juntas** contra a sonda, o que destrói também a associação")
    p("largura–sonda. A hipótese nula condicional correta preserva essa")
    p("associação e embaralha apenas a parte da sonda que a largura não")
    p("explica. É o que Freedman–Lane faz, e é o que está abaixo.\n")
    p(f"Permutação no nível do extrator, a mesma nos quatro domínios, {N} sorteios.\n")

    SUB = [("todos os 17", TODOS), ("sem o MAE", [e for e in TODOS if e != MAE]),
           ("sem MAE e os dois CLIP",
            [e for e in TODOS if e != MAE and e not in CLIPS]),
           ("9 arquiteturas", ARQ), ("9 pré-treinos", OBJ)]

    p("| conjunto | k | ρ parcial | **p Freedman–Lane** | p do nulo antigo | b1 | p F–L de b1 |")
    p("|---|---|---|---|---|---|---|")
    for nome, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in sd for b in BANCOS)
              and all((e, b) in geo for b in BANCOS)]
        k, nd = len(es), len(BANCOS)
        larg_cru = [geo[(e, BANCOS[0])]["canais"] for e in es]
        if np.std(larg_cru) == 0:
            p(f"| {nome} | {k} | — | — | — | — | largura constante "
              f"({larg_cru[0]} canais): nada a controlar |")
            continue
        G = np.concatenate([z([geo[(e, b)]["posto"][0] for e in es]) for b in BANCOS])
        P = np.concatenate([z([sd[f"{b}|{e}"]["valor"] for e in es]) for b in BANCOS])
        W = np.concatenate([z(larg_cru) for _ in BANCOS])
        rg, rp, rw = rankdata(G), rankdata(P), rankdata(W)

        rho, pv = freedman_lane(rg, rp, rw, k, nd)
        _, pv_antigo = conjunto_permuta(rg, rp, rw, k, nd)
        b1, pv_b1 = freedman_lane_beta(rg, rp, rw, k, nd)
        p(f"| {nome} | {k} | **{rho:+.3f}** | **{pv:.3f}** | {pv_antigo:.3f} | "
          f"{b1:+.3f} | {pv_b1:.3f} |")
    p("")
    p("O ponto estimado da parcial não muda --- ele nunca dependeu do nulo.")
    p("O que muda é o $p$, e é ele que o parecer questionou.\n")

    saida = RAIZ / "resultados" / "PARCIAL_FL.md"
    saida.write_text("\n".join(L) + "\n")
    print(f"\n-> {saida}", flush=True)


if __name__ == "__main__":
    main()
