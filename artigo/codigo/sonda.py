#!/usr/bin/env python
"""A geometria da representacao prediz alguma coisa?

O artigo motiva as medidas de geometria dizendo que a acuracia na tarefa final
e ruidosa demais em amostra pequena. Mas nunca fecha o laco: nao mostra que a
geometria PREDIZ desempenho. Esta e a sonda que fecha.

Sonda linear sobre as caracteristicas congeladas -- as MESMAS que alimentam o
posto efetivo -- em 6 extratores x 4 dominios, e correlacao com a geometria.
Um resultado nulo tambem serve: "a geometria nao prediz a sonda" e conclusao,
nao fracasso.

DUAS REGRAS DE DIVISAO DIFERENTES, de proposito:
  bancos naturais  imagens independentes -> CV estratificada por classe
  mamografia       103 pacientes em 387 imagens, varias vistas por paciente ->
                   StratifiedGroupKFold POR PACIENTE, e reamostragem por
                   PACIENTE. Sem isso a mesma mama aparece no treino e no
                   teste, e a AUC infla. Era o ponto central do estudo anterior.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from campo_e_embaralho import imagens
from testes import pecas_mapa, ic

FAMILIAS = ["resnet18", "resnet50", "convnext_t", "efficientnet_b3", "swin_t",
            "maxvit_t", "maxvit_s", "vit_s16",
            # os nove objetivos sobre o MESMO ViT-B/16
            "vit_b16", "vit_b16_sup21k", "vit_b16_sup1k", "vit_b16_deit3",
            "vit_b16_dino", "vit_b16_mae", "vit_b16_clip", "vit_b16_clip2b",
            "vit_b16_siglip"]
SEMENTES = (0, 1, 2)
BANCOS = ["mamografia", "dtd", "pet", "sun397"]
FRIO = Path("/frio/guilhermelima/quantum-transfer-learning-bcdr/data")
GRADE_C = {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]}


def rotulos(banco, caminhos):
    """(y, grupos). grupos e None onde as imagens sao independentes."""
    if banco != "mamografia":
        d = json.loads((caminhos[0].parent / "rotulos.json").read_text())
        y = np.array([d[q.name] for q in caminhos])
        _, y = np.unique(y, return_inverse=True)
        return y, None
    # a amostra foi gerada em ordem alfabetica a partir do /frio; verificado
    # byte a byte, entao o k-esimo PNG corresponde ao k-esimo nome ordenado
    fontes = sorted(p.name for p in (FRIO / "inbreast_png_896").glob("*.png"))
    meta = pd.read_csv(FRIO / "inbreast_images.csv").set_index("processed_image_filename")
    idx = [int(q.stem) for q in caminhos]
    sub = meta.loc[[fontes[i] for i in idx]]
    return sub.classification.to_numpy(), sub.patient_id.to_numpy()


def poda(X, y, grupos, minimo=5):
    """Descarta classes raras demais para estratificar em 5 dobras."""
    v, c = np.unique(y, return_counts=True)
    manter = np.isin(y, v[c >= minimo])
    return X[manter], y[manter], (None if grupos is None else grupos[manter])


def _uma_divisao(X, y, grupos, semente):
    """Previsoes fora da dobra para UMA particao."""
    binaria = len(np.unique(y)) == 2
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=semente) if grupos is not None
          else StratifiedKFold(5, shuffle=True, random_state=semente))
    tubo = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    fora = np.zeros(len(y), float)
    for tr, te in cv.split(X, y, grupos):
        b = GridSearchCV(tubo, GRADE_C, cv=3, n_jobs=4,
                         scoring="roc_auc" if binaria else "accuracy")
        b.fit(X[tr], y[tr])
        fora[te] = (b.predict_proba(X[te])[:, 1] if binaria
                    else (b.predict(X[te]) == y[te]).astype(float))
    return fora, binaria


def sonda(X, y, grupos, sementes=SEMENTES, n_boot=200):
    """Metrica fora da dobra sobre VARIAS particoes.

    O extrator esta congelado, mas a SONDA nao: ela e treinada, e portanto tem
    variancia de particao. Reamostrar so as imagens, com uma divisao fixa,
    mediria metade da incerteza. Aqui cada semente da uma particao; o ponto e
    a media entre elas e o intervalo junta as reamostras de TODAS, de modo que
    carrega as duas fontes.

    O C da regularizacao sai de CV interna: fixa-lo penalizaria de forma
    diferente extratores de 384 e de 2048 canais, medindo a largura.
    """
    pontos, draws = [], []
    for sem in sementes:
        fora, binaria = _uma_divisao(X, y, grupos, sem)
        if binaria:
            gs = np.unique(grupos)
            def est(i, f=fora):
                m = np.isin(grupos, gs[i])
                return (roc_auc_score(y[m], f[m]) if len(np.unique(y[m])) == 2
                        else np.nan)
            pontos.append(roc_auc_score(y, fora))
            rng = np.random.default_rng(sem)
            draws += [est(rng.integers(0, len(gs), len(gs))) for _ in range(n_boot)]
        else:
            pontos.append(float(fora.mean()))
            rng = np.random.default_rng(sem)
            draws += [float(fora[rng.integers(0, len(fora), len(fora))].mean())
                      for _ in range(n_boot)]
    d = np.array([x for x in draws if np.isfinite(x)])
    nome = "AUC" if len(np.unique(y)) == 2 else "acurácia"
    return (nome, float(np.mean(pontos)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float(np.std(pontos)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amostras", default="data/amostras")
    p.add_argument("--out", default="resultados")
    p.add_argument("--n", type=int, default=387)
    a = p.parse_args()
    saida = Path(a.out); saida.mkdir(parents=True, exist_ok=True)
    arq = saida / "sonda.json"
    tudo = json.loads(arq.read_text()) if arq.exists() else {}

    for banco in BANCOS:
        qs = imagens(Path(a.amostras) / banco, a.n)
        y, g = rotulos(banco, qs)
        print(f"\n== {banco}: {len(qs)} imagens, {len(np.unique(y))} classes"
              f"{'' if g is None else f', {len(np.unique(g))} pacientes'} ==", flush=True)
        for bb in FAMILIAS:
            chave = f"{banco}|{bb}"
            if chave in tudo:
                continue
            X, _ = pecas_mapa(bb, qs, 512)
            Xp, yp, gp = poda(X.numpy(), y, g)
            nome, pt, lo, hi, dp = sonda(Xp, yp, gp)
            tudo[chave] = dict(banco=banco, backbone=bb, metrica=nome,
                               valor=pt, lo=lo, hi=hi, dp_particao=dp,
                               n=len(yp), classes=int(len(np.unique(yp))),
                               canais=X.shape[1])
            arq.write_text(json.dumps(tudo, indent=1, default=float))
            print(f"   {bb:16} {nome} {pt:.3f} [{lo:.3f};{hi:.3f}]"
                  f"  dp entre partições {dp:.3f}", flush=True)
    print("\nconcluido", flush=True)


if __name__ == "__main__":
    main()
