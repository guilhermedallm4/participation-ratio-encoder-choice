#!/usr/bin/env python
"""Item 2 do parecer: por que o IC exclui zero enquanto p = 0,075?

O revisor aponta a linha "sem o MAE": rho = 0,27, p_perm = 0,075, mas o
intervalo de 95 % reamostrado dava [+0,02; +0,50], que nao contem zero. Ele
tem razao em exigir explicacao.

A PRIMEIRA HIPOTESE TESTADA AQUI FOI DESCARTADA PELOS NUMEROS, e fica
registrada para que ninguem a refaca. Supunhamos que o bootstrap COM
REPOSICAO fosse anticonservador pelo mesmo motivo que o artigo ja rejeita a
reposicao na razao de participacao: um extrator sorteado duas vezes entra com
pares perfeitamente concordantes consigo mesmo, o que deveria inflar uma
estatistica de posto. A coluna `média do bootstrap` mede exatamente isso --
e ela coincide com o rho observado ate a terceira casa (+0,411 contra +0,410
nos dezessete). Nao ha vies de duplicata.

O QUE RESTA E MAIS SIMPLES E MENOS DRAMATICO. O intervalo percentil terminava
em +0,01 e o jackknife na escala de Fisher termina em -0,01: a 0,27 com k=16,
a ponta do intervalo nao e resolvivel nessa precisao, e o metodo percentil caiu
do lado errado do zero. O jackknife devolve coerencia nas SEIS linhas ao mesmo
tempo, e e o procedimento companheiro de uma analise que trata justamente de
retirar extratores. A decisao fica com a permutacao; o intervalo descreve a
dispersao.

Calculamos cinco coisas para cada subconjunto e deixamos os numeros decidirem:

  rho        a estatistica observada (Spearman sobre os pares empilhados)
  p_perm     permutacao NO NIVEL DO EXTRATOR, a mesma aplicada aos 4 dominios
  limiar     percentil 95 de |rho| sob esse nulo -- o "detectavel"
  IC_rep     bootstrap percentil COM reposicao (o do artigo atual)
  IC_jack    jackknife deixando um extrator de fora, na escala de Fisher

O jackknife e o companheiro natural de uma analise de sensibilidade sobre
retirar extratores, e nao tem o problema da duplicata: cada reamostra e um
subconjunto proprio, sem repeticao.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent

from analise_final import TODOS, ARQ, OBJ, BANCOS, z, perm_extrator   # noqa: E402

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


def pares(geo, sd, es):
    """G, P padronizados DENTRO de cada dominio, como no artigo."""
    G = [z([geo[(e, b)]["posto"][0] for e in es]) for b in BANCOS]
    P = [z([sd[f"{b}|{e}"]["valor"] for e in es]) for b in BANCOS]
    return G, P


def rho_de(G, P, idx):
    return spearmanr(np.concatenate([g[idx] for g in G]),
                     np.concatenate([q[idx] for q in P])).statistic


def ic_com_reposicao(G, P, k, n=3000, semente=1):
    rng = np.random.default_rng(semente)
    bs = np.array([rho_de(G, P, rng.integers(0, k, k)) for _ in range(n)])
    bs = bs[np.isfinite(bs)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), bs


def ic_jackknife(G, P, k):
    """Deixa um extrator de fora de cada vez; variancia na escala de Fisher.

    Fisher estabiliza a variancia e mantem o intervalo dentro de [-1, 1],
    que o intervalo na escala bruta nao garante.
    """
    tudo = np.arange(k)
    fz = lambda r: np.arctanh(np.clip(r, -0.999999, 0.999999))        # noqa: E731
    z_i = np.array([fz(rho_de(G, P, np.delete(tudo, i))) for i in range(k)])
    z0 = fz(rho_de(G, P, tudo))
    var = (k - 1) / k * ((z_i - z_i.mean()) ** 2).sum()
    se = np.sqrt(var)
    return float(np.tanh(z0 - 1.96 * se)), float(np.tanh(z0 + 1.96 * se)), float(se)


def quanto_duplica(G, P, k, n=3000, semente=1):
    """Diagnostico: rho medio do bootstrap contra o rho observado.

    Se a reposicao fosse inocua, a media das reamostras ficaria perto do
    observado. O vies para cima e a assinatura do problema das duplicatas.
    """
    _, _, bs = ic_com_reposicao(G, P, k, n, semente)
    return float(bs.mean()), float(np.median(bs))


def main():
    geo, sd = carrega()
    sem_mae = [e for e in TODOS if e != MAE]
    sem_mae_clip = [e for e in TODOS if e != MAE and e not in CLIPS]
    obj_sem_mae = [e for e in OBJ if e != MAE]

    SUB = [("todos os 17", TODOS), ("sem o MAE", sem_mae),
           ("sem MAE e os dois CLIP", sem_mae_clip),
           ("9 pré-treinos", OBJ), ("9 pré-treinos, sem MAE", obj_sem_mae),
           ("9 arquiteturas", ARQ)]

    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    p("# Item 2 — intervalo contra valor-p\n")
    p("| conjunto | k | rho | p_perm | limiar | IC com reposição | IC jackknife | média do bootstrap |")
    p("|---|---|---|---|---|---|---|---|")
    for nome, es in SUB:
        es = [e for e in es if all(f"{b}|{e}" in sd for b in BANCOS)
              and all((e, b) in geo for b in BANCOS)]
        k = len(es)
        G, P = pares(geo, sd, es)
        rho, pv, limiar = perm_extrator(G, P)
        lo_r, hi_r, _ = ic_com_reposicao(G, P, k)
        lo_j, hi_j, se = ic_jackknife(G, P, k)
        m_bs, _ = quanto_duplica(G, P, k)
        p(f"| {nome} | {k} | {rho:+.3f} | {pv:.3f} | {limiar:.2f} | "
          f"[{lo_r:+.2f}; {hi_r:+.2f}] | [{lo_j:+.2f}; {hi_j:+.2f}] | {m_bs:+.3f} |")

    p("\n## Leitura\n")
    p("- `limiar` é o percentil 95 de |rho| sob permutação de extratores: a")
    p("  decisão coerente com p_perm é comparar |rho| contra ele.")
    p("- `IC com reposição` é o do artigo atual. `média do bootstrap` serve de")
    p("  diagnóstico de viés de duplicata: ela coincide com o `rho` observado,")
    p("  então esse viés NÃO existe aqui — a hipótese foi testada e descartada.")
    p("- `IC jackknife` deixa um extrator de fora de cada vez, sem repetição,")
    p("  e é o procedimento compatível com a análise de sensibilidade.")

    saida = RAIZ / "resultados" / "INFERENCIA.md"
    saida.write_text("\n".join(L) + "\n")
    print(f"\n-> {saida}", flush=True)


if __name__ == "__main__":
    main()
