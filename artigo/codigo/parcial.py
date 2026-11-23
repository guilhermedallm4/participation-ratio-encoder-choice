#!/usr/bin/env python
"""Item 2.2 do terceiro parecer: a largura esta mesmo controlada?

NAO ESTAVA. O artigo reportava rho(PR, sonda) = +0,41 e, ao lado,
rho(largura, sonda) = +0,17, e chamava o segundo de "controle do confundidor".
Reportar a correlacao do confundidor com o desfecho nao controla coisa
nenhuma -- e so uma segunda correlacao marginal. O revisor esta certo.

Aqui esta o controle de verdade, em duas formas:

  parcial    Spearman parcial: posto de PR, da sonda e da largura; regride-se
             PR e sonda sobre a largura e correlacionam-se os RESIDUOS. E o
             que responde "sobra associacao depois de tirar a largura?".

  regressao  Sonda = b0 + b1 PR + b2 largura, nos valores ja padronizados
             dentro de cada dominio. O coeficiente b1 e a inclinacao parcial.

A inferencia das duas e a MESMA do artigo: permutacao no nivel do extrator, a
mesma permutacao aplicada aos quatro dominios. PR e largura sao as duas
propriedades do extrator e viajam JUNTAS na permutacao -- separa-las criaria um
nulo que quebra a associacao entre elas, que nao e o que se quer testar.

Onde a largura e constante (os nove ViT-B/16, todos com 768 canais) nao ha o
que controlar, e o script diz isso em vez de devolver nan.

Tambem recalcula o teto de posto com o m CERTO. O ponto publicado da razao de
participacao sai de ic_sub, que subamostra sem reposicao em m = n/2 = 100,
entao o teto e 99 e nao 199. A versao anterior desta checagem usou 199.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent

from analise_final import TODOS, ARQ, OBJ, BANCOS, ROT, z          # noqa: E402

MAE = "vit_b16_mae"
CLIPS = ("vit_b16_clip", "vit_b16_clip2b")


def carrega():
    r = RAIZ / "resultados"
    t = json.loads((r / "testes.json").read_text())
    sd = json.loads((r / "sonda.json").read_text())
    geo = {(x["backbone"], x["banco"]): x
           for bl in ("ic", "arquiteturas", "leva2", "leva3") for x in t.get(bl, [])
           if "banco" in x}
    return geo, sd


def residuo(y, x):
    """Residuo de y regredido em x (minimos quadrados, com intercepto)."""
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def parcial(a, b, c):
    """Spearman parcial de a e b dado c: correlacao dos residuos dos postos."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    return float(np.corrcoef(residuo(ra, rc), residuo(rb, rc))[0, 1])


def beta_parcial(sonda, pr, larg):
    X = np.column_stack([np.ones_like(pr), pr, larg])
    b, *_ = np.linalg.lstsq(X, sonda, rcond=None)
    return float(b[1]), float(b[2])


def permuta(G, W, P, estat, n=20000, semente=0):
    """UMA permutacao de extratores, aplicada aos quatro dominios.

    PR e largura sao propriedades do MESMO extrator e sao permutadas juntas.
    """
    obs = estat(np.concatenate(G), np.concatenate(W), np.concatenate(P))
    rng = np.random.default_rng(semente)
    k = len(G[0])
    p = np.concatenate(P)
    nulo = []
    for q in (rng.permutation(k) for _ in range(n)):
        gq = np.concatenate([g[q] for g in G])
        wq = np.concatenate([w[q] for w in W])
        v = estat(gq, wq, p)
        if np.isfinite(v):
            nulo.append(abs(v))
    nulo = np.array(nulo)
    return obs, float((nulo >= abs(obs)).mean())


def main():
    geo, sd = carrega()
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    # ----- teto de posto, com o m certo ---------------------------------
    p("# Item 2.2 — controle de largura, e o teto de posto corrigido\n")
    p("## 0. O teto, agora com o m correto\n")
    p("O ponto publicado da razão de participação vem de `ic_sub`, que")
    p("subamostra **sem reposição em m = n/2 = 100**. O teto de posto é")
    p("portanto **99**, não 199. A checagem anterior usou 199 e está corrigida.\n")
    vals = [(geo[(e, b)]["posto"][0], e, b) for e in TODOS for b in BANCOS]
    mx, me, mb = max(vals)
    arr = np.array([v for v, _, _ in vals])
    p(f"- maior PR publicada (m = 100): **{mx:.1f}** ({ROT[me]}, {mb}) = "
      f"**{100*mx/99:.1f} % do teto de 99**")
    p(f"- mediana das 68: {np.median(arr):.1f} ({100*np.median(arr)/99:.1f} %)")
    p(f"- acima de 50 % do teto: {(arr > 49.5).sum()} dos 68 pares")
    cheio = [geo[(e, b)].get("posto_n_cheio") for e in TODOS for b in BANCOS]
    cheio = [c for c in cheio if c]
    if cheio:
        p(f"- para referência, com os 200 (teto 199): máx {max(cheio):.1f} "
          f"({100*max(cheio)/199:.1f} %), mediana {np.median(cheio):.1f}\n")

    # ----- controle de largura -------------------------------------------
    p("## 1. A largura, controlada de verdade\n")
    p("| conjunto | k | ρ(PR, sonda) | ρ(largura, sonda) | ρ parcial (PR·sonda | largura) | p perm |")
    p("|---|---|---|---|---|---|")
    SUB = [("todos os 17", TODOS), ("sem o MAE", [e for e in TODOS if e != MAE]),
           ("sem MAE e os dois CLIP",
            [e for e in TODOS if e != MAE and e not in CLIPS]),
           ("9 arquiteturas", ARQ), ("9 pré-treinos", OBJ)]
    for nome, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in sd for b in BANCOS)
              and all((e, b) in geo for b in BANCOS)]
        k = len(es)
        larg_cru = [geo[(e, BANCOS[0])]["canais"] for e in es]
        G = [z([geo[(e, b)]["posto"][0] for e in es]) for b in BANCOS]
        P = [z([sd[f"{b}|{e}"]["valor"] for e in es]) for b in BANCOS]
        r_pr = spearmanr(np.concatenate(G), np.concatenate(P)).statistic
        if np.std(larg_cru) == 0:
            p(f"| {nome} | {k} | {r_pr:+.3f} | — | — | largura constante "
              f"({larg_cru[0]} canais): nada a controlar |")
            continue
        W = [z(larg_cru) for _ in BANCOS]
        r_w = spearmanr(np.concatenate(W), np.concatenate(P)).statistic
        rp, pv = permuta(G, W, P, lambda g, w, q: parcial(g, q, w))
        p(f"| {nome} | {k} | {r_pr:+.3f} | {r_w:+.3f} | **{rp:+.3f}** | {pv:.3f} |")
    p("")

    # ----- regressao ------------------------------------------------------
    p("## 2. A mesma coisa como regressão\n")
    p("Sonda = b0 + b1·PR + b2·largura, tudo padronizado dentro do domínio.\n")
    p("| conjunto | k | b1 (PR) | p perm | b2 (largura) |")
    p("|---|---|---|---|---|")
    for nome, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in sd for b in BANCOS)
              and all((e, b) in geo for b in BANCOS)]
        larg_cru = [geo[(e, BANCOS[0])]["canais"] for e in es]
        if np.std(larg_cru) == 0:
            continue
        G = [z([geo[(e, b)]["posto"][0] for e in es]) for b in BANCOS]
        P = [z([sd[f"{b}|{e}"]["valor"] for e in es]) for b in BANCOS]
        W = [z(larg_cru) for _ in BANCOS]
        b1, b2 = beta_parcial(np.concatenate(P), np.concatenate(G),
                              np.concatenate(W))
        _, pv = permuta(G, W, P,
                        lambda g, w, q: beta_parcial(q, g, w)[0])
        p(f"| {nome} | {len(es)} | {b1:+.3f} | {pv:.3f} | {b2:+.3f} |")
    p("")

    saida = RAIZ / "resultados" / "PARCIAL.md"
    saida.write_text("\n".join(L) + "\n")
    print(f"\n-> {saida}", flush=True)


if __name__ == "__main__":
    main()
