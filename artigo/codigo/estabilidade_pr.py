#!/usr/bin/env python
"""Item 1 do parecer: a razao de participacao esta limitada por n = 200?

O revisor esta certo no fato: a covariancia centrada de n imagens tem posto no
maximo n-1, entao com 200 imagens nenhuma PR estimada pode passar de 199, por
mais canais que o extrator tenha. A pergunta e se essa restricao MORDE nestes
dados -- se as PR medidas estao perto do teto -- ou se e folgada.

Duas coisas sao medidas aqui, e as duas usam UMA unica passagem de extracao:

  teto      PR medida com os 200 contra o limite 199. Se a maior PR do
            conjunto for uma fracao pequena do teto, a restricao nao morde.

  escada    PR estimada em subconjuntos ANINHADOS de 50, 100, 150 e 200
            imagens, sem reposicao, R repeticoes. O que interessa nao e a
            magnitude (que sobe com n por construcao, porque mais amostras
            revelam mais direcoes), e sim se a ORDEM entre extratores se
            mantem. A conclusao do artigo depende so da ordem: a correlacao
            e de Spearman e a leitura e "o MAE esta embaixo".

Guarda os vetores por imagem em .npz para que qualquer outra estatistica possa
ser recalculada depois sem tocar na GPU de novo.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent

from testes import pecas_mapa                      # noqa: E402
from campo_e_embaralho import imagens              # noqa: E402
from representacao import posto_efetivo            # noqa: E402
from analise_final import TODOS, BANCOS, ROT       # noqa: E402

TAMANHOS = [50, 100, 150, 200]
R = 40                      # repeticoes por tamanho (o de 200 e exato, R=1)


def pr_de(X):
    return posto_efetivo(torch.as_tensor(np.asarray(X, dtype=np.float64)))


def escada(X, semente=0):
    """PR em subconjuntos sem reposicao de 50, 100, 150 e 200 imagens."""
    rng = np.random.default_rng(semente)
    n = len(X)
    fora = {}
    for m in TAMANHOS:
        if m > n:
            continue
        if m == n:
            fora[m] = {"media": pr_de(X), "dp": 0.0, "r": 1, "teto": n - 1}
            continue
        v = [pr_de(X[rng.permutation(n)[:m]]) for _ in range(R)]
        fora[m] = {"media": float(np.mean(v)), "dp": float(np.std(v)),
                   "r": R, "teto": m - 1}
    return fora


def main():
    saida = RAIZ / "resultados"
    saida.mkdir(parents=True, exist_ok=True)
    bruto = saida / "pr_vetores"
    bruto.mkdir(exist_ok=True)

    qs = {b: imagens(RAIZ / "data" / "amostras" / b, 200) for b in BANCOS}
    for b in BANCOS:
        print(f"{b}: {len(qs[b])} imagens", flush=True)

    arq = saida / "estabilidade_pr.json"
    tudo = json.loads(arq.read_text()) if arq.exists() else {}

    for alias in TODOS:
        for b in BANCOS:
            chave = f"{alias}|{b}"
            if chave in tudo:
                print(f"  (ja feito) {chave}", flush=True)
                continue
            t0 = time.time()
            try:
                mp, _ = pecas_mapa(alias, qs[b], 512)
            except Exception as e:                      # noqa: BLE001
                print(f"  !! {chave}: {type(e).__name__}: {e}", flush=True)
                continue
            X = mp.numpy().astype(np.float64)
            np.savez_compressed(bruto / f"{alias}__{b}.npz", X=X.astype(np.float32))
            e = escada(X)
            tudo[chave] = {"alias": alias, "banco": b, "n": int(X.shape[0]),
                           "canais": int(X.shape[1]),
                           "escada": {str(k): v for k, v in e.items()}}
            arq.write_text(json.dumps(tudo, indent=1))
            pr200 = e[200]["media"]
            print(f"  {ROT.get(alias, alias):26} {b:11} C={X.shape[1]:5d} "
                  f"PR200={pr200:6.2f} ({100 * pr200 / 199:4.1f}% do teto)  "
                  f"PR50={e[50]['media']:5.2f} PR100={e[100]['media']:5.2f} "
                  f"PR150={e[150]['media']:5.2f}  [{time.time() - t0:.0f}s]",
                  flush=True)
            torch.cuda.empty_cache()

    print(f"\n-> {arq}", flush=True)


if __name__ == "__main__":
    main()
