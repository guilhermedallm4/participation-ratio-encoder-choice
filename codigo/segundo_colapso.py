#!/usr/bin/env python
"""Teste de PREDICAO: a razao de participacao flagra um colapso que ela nao viu?

A fraqueza estrutural do artigo e que a afirmacao do titulo repousa num unico
checkpoint colapsado, o MAE. Acrescentar um segundo checkpoint AO CONJUNTO e
refazer a correlacao seria circular -- o ponto entra no mesmo caldo de onde a
afirmacao saiu. O que vale e tratar o novo caso como RETIDO e perguntar, antes
de olhar, o que a afirmacao do artigo prediz para ele.

  vit_b16_beit3      BEiT-3 base, checkpoint ".pt": modelagem mascarada PURA,
                     sem o ajuste fino que os demais beit do timm carregam.
                     Mesma arquitetura, mesmas 1025 fichas a 512 px.
                     PREDICAO: razao de participacao na faixa do MAE (2-5) e
                     sonda entre as piores do conjunto.

  vit_b16_aleatorio  ViT-B/16 sem treino nenhum. NAO e candidato a colapso: e
                     o referencial de "representacao que se deve rejeitar",
                     com resposta conhecida na sonda. Se a razao NAO o
                     sinalizar, a afirmacao fica delimitada -- ela flagra
                     colapso de modelagem mascarada, nao representacao ruim em
                     geral. Interessa nos dois sentidos.

Mede exatamente como o artigo: razao de participacao por ic_sub em m = n/2 =
100 sobre 200 imagens, e sonda linear em tres particoes sobre as 387.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent

from testes import pecas_mapa, ic_sub                      # noqa: E402
from campo_e_embaralho import imagens                      # noqa: E402
from representacao import posto_efetivo                    # noqa: E402
from analise_final import BANCOS                           # noqa: E402
import sonda as SND                                        # noqa: E402

NOVOS = ["vit_b16_beit3", "vit_b16_aleatorio"]


def main():
    saida = RAIZ / "resultados"
    arq = saida / "segundo_colapso.json"
    tudo = json.loads(arq.read_text()) if arq.exists() else {}

    qs200 = {b: imagens(RAIZ / "data" / "amostras" / b, 200) for b in BANCOS}
    qs387 = {b: imagens(RAIZ / "data" / "amostras" / b, 387) for b in BANCOS}

    for alias in NOVOS:
        for b in BANCOS:
            chave = f"{alias}|{b}"
            if chave in tudo:
                print(f"  (já feito) {chave}", flush=True)
                continue
            t0 = time.time()

            # --- geometria, protocolo identico ao do artigo -----------------
            mp, _ = pecas_mapa(alias, qs200[b], 512)
            *posto, m_posto = ic_sub(mp, lambda i: posto_efetivo(mp[i]))
            pr_cheio = float(posto_efetivo(mp))

            # --- sonda, protocolo identico ---------------------------------
            X, _ = pecas_mapa(alias, qs387[b], 512)
            X = X.numpy().astype(np.float64)
            y, grupos = SND.rotulos(b, qs387[b])
            X, y, grupos = SND.poda(X, y, grupos)
            metrica, valor, s_lo, s_hi, dp = SND.sonda(X, y, grupos)

            tudo[chave] = {"alias": alias, "banco": b, "canais": int(mp.shape[1]),
                           "posto": posto, "m_posto": m_posto,
                           "posto_n_cheio": pr_cheio,
                           "valor": valor, "ic": [s_lo, s_hi], "dp_particao": dp,
                           "metrica": metrica, "classes": int(len(set(y.tolist())))}
            arq.write_text(json.dumps(tudo, indent=1, default=float))
            print(f"  {alias:18} {b:11} PR={posto[0]:6.2f} "
                  f"[{posto[1]:.2f};{posto[2]:.2f}]  sonda={valor:.3f} "
                  f"[{s_lo:.3f};{s_hi:.3f}] ({metrica})  [{time.time()-t0:.0f}s]",
                  flush=True)
            torch.cuda.empty_cache()

    print(f"\n-> {arq}", flush=True)


if __name__ == "__main__":
    main()
