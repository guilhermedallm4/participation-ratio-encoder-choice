#!/usr/bin/env python3
"""Como cada familia de arquitetura representa mamografia.

O eixo e o grau e o alcance da mistura espacial:

    ResNet-50   convolucao pura        local, hierarquica
    MaxViT-T    conv + bairro + grade  local E global, hierarquica
    ViT-B/16    atencao global pura    global, plana

NADA AQUI TREINA. Sao passagens para frente em extratores congelados. Isso
importa por um motivo especifico deste trabalho: medimos que trocar apenas a
semente de inicializacao move a AUC em 0,059 no ResNet-50 e 0,224 no
ResNet-18. Qualquer afirmacao sobre DESEMPENHO entre familias afunda nesse
ruido. As medidas abaixo nao dependem de treino, logo nao tem variancia de
semente, e sao a parte da comparacao que continua mensuravel.

MEDIDAS
  posto efetivo      quantas direcoes a representacao de fato usa
  dispersao espacial quanto da variancia esta ENTRE posicoes da imagem
  CKA                duas redes codificam a mesma geometria de similaridade?
  banda de frequencia a qual escala espacial cada familia responde
  sonda de posicao   a posicao da lesao e recuperavel linearmente?

O controle que da sentido ao posto efetivo e comparar mamografia com imagem
natural: se a ImageNet ativa muitas direcoes e a mamografia poucas, o numero
quantifica o quanto o dominio e estreito.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
import multitarefa as MT
from PIL import Image
from torchvision import transforms

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FAMILIAS = [
    ("resnet50", "convolucao pura", "local, hierarquica"),
    ("maxvit_t", "conv + atencao bairro + grade", "local e global, hierarquica"),
    ("vit_b16", "atencao global pura", "global, plana"),
]


# ---------------------------------------------------------------------------
# Extracao
# ---------------------------------------------------------------------------

def transformacao(norm):
    return transforms.Compose([transforms.ToTensor(),
                               MT.NORM_IMAGENET if norm == "imagenet" else MT.NORM_MEIO])


def mapas(alias, caminhos, tam=512, bs=8, transformar=None):
    """Mapa de features (N, C, H, W) do extrator congelado."""
    bb, canais, norm = MT.monta_backbone(alias, pre=True, treinavel=False)
    bb = bb.to(DEV).eval()
    tf = transformacao(norm)
    out = []
    with torch.no_grad():
        for i in range(0, len(caminhos), bs):
            lote = []
            for p in caminhos[i:i + bs]:
                im = Image.open(p).convert("L").resize((tam, tam), Image.Resampling.BILINEAR)
                a = np.asarray(im, dtype=np.float32) / 255.0
                if transformar is not None:
                    a = transformar(a)
                lote.append(tf(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)
                                               ).convert("RGB")))
            out.append(bb(torch.stack(lote).to(DEV)).float().cpu())
    del bb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Medidas
# ---------------------------------------------------------------------------

def posto_efetivo(X):
    """Razao de participacao dos autovalores da covariancia.

    (sum L)^2 / sum L^2. Vale 1 se toda a variancia esta numa direcao e C se
    esta uniformemente distribuida. E uma contagem continua de "quantas
    direcoes a representacao usa", robusta a autovalores minusculos, que o
    posto algebrico nao e.
    """
    X = (X - X.mean(0, keepdim=True)).double()
    v = torch.linalg.svdvals(X) ** 2
    return float((v.sum() ** 2) / (v ** 2).sum())


def dispersao_espacial(M):
    """Fracao da variancia que esta ENTRE posicoes, e nao entre imagens.

    Alto = a representacao distingue lugares dentro da imagem, que e o que
    localizar exige. Baixo = o mapa e quase uniforme no espaco e so a media
    global carrega informacao.
    """
    n, c, h, w = M.shape
    v = M.reshape(n, c, h * w).double()
    entre_pos = v.var(dim=2).mean()
    entre_img = v.mean(dim=2).var(dim=0).mean()
    return float(entre_pos / (entre_pos + entre_img))


def cka(X, Y):
    """CKA linear. Invariante a rotacao e escala isotropica, entao compara
    representacoes de larguras diferentes. 1 = mesma geometria de similaridade
    entre as imagens; 0 = nenhuma relacao."""
    X = (X - X.mean(0)).double()
    Y = (Y - Y.mean(0)).double()
    return float((X.T @ Y).pow(2).sum() /
                 ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))


def filtro_banda(a, baixa, alta):
    """Mantem apenas as frequencias espaciais entre duas fracoes de Nyquist.

    Lesao em mamografia e textura de alta frequencia sobre fundo de baixa.
    Perturbar por banda e medir a mudanca da representacao diz a qual escala
    cada familia responde.
    """
    F = np.fft.fftshift(np.fft.fft2(a))
    h, w = a.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)
    F[(r < baixa) | (r >= alta)] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def sensibilidade_banda(alias, caminhos, bandas, tam=512):
    """Quanto a representacao muda quando so uma banda de frequencia sobrevive.

    Normalizada pela norma da representacao original, para comparar familias
    com escalas de ativacao diferentes.
    """
    base = mapas(alias, caminhos, tam).flatten(1)
    nb = base.norm(dim=1, keepdim=True).clamp_min(1e-6)
    saida = {}
    for lo, hi in bandas:
        m = mapas(alias, caminhos, tam,
                  transformar=lambda a, lo=lo, hi=hi: filtro_banda(a, lo, hi)).flatten(1)
        saida[f"{lo:.2f}-{hi:.2f}"] = float(((m - base).norm(dim=1) / nb.squeeze()).mean())
    return saida


def sonda_posicao(M, alvo, n_comp=64, cv=5):
    """A posicao da lesao e recuperavel LINEARMENTE do mapa?

    Achatar o mapa da dezenas de milhares de features para poucas dezenas de
    imagens, o que deixa a sondagem subdeterminada. PCA antes resolve: mantem
    a informacao e traz a dimensao para baixo do numero de amostras.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    X = M.flatten(1).numpy()

    # 1. Descartar features quase constantes. Num mapa ReLU tipico, ~98% das
    #    posicoes estao mortas em todas as imagens; padroniza-las divide por
    #    ~0 e produz valores enormes que arruinam a regressao.
    v = X.var(0)
    X = X[:, v > 1e-8 * max(v.max(), 1e-12)]
    if X.shape[1] == 0:
        return float("nan")

    # 2. O PCA roda DENTRO de cada dobra de treino, que tem (cv-1)/cv das
    #    amostras. Com n_comp ~ n_treino o ajuste fica determinado e extrapola
    #    sem controle no teste, entao limitamos a metade das amostras de treino.
    n_treino = int(len(X) * (cv - 1) / cv)
    n_comp = min(n_comp, max(2, n_treino // 2), X.shape[1])

    # SEM StandardScaler: num mapa ReLU com ~10^5 features sobreviventes, muitas
    # tem variancia minuscula mas nao nula, e padronizar as amplifica ate
    # dominarem o PCA (o R2 do ResNet-50 despencava para -37). O PCA ja centra.
    tubo = make_pipeline(PCA(n_components=n_comp, random_state=0),
                         RidgeCV(alphas=np.logspace(-2, 5, 16)))
    return float(cross_val_score(tubo, X, alvo, cv=cv, scoring="r2").mean())


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="analise de representacao")
    p.add_argument("--imagens", default="../data/inbreast_png_896")
    p.add_argument("--csv", default="../data/inbreast_images.csv")
    p.add_argument("--caixas", default="../data/inbreast_caixas.csv")
    p.add_argument("--out", default="../data/representacao")
    p.add_argument("--n", type=int, default=200, help="imagens de mamografia")
    p.add_argument("--controle", default=None,
                   help="diretorio com imagens naturais para o contraste de dominio")
    a = p.parse_args()

    saida = Path(a.out); saida.mkdir(parents=True, exist_ok=True)
    imgs = Path(a.imagens)
    d = pd.read_csv(a.csv).sample(min(a.n, len(pd.read_csv(a.csv))), random_state=0)
    caminhos = [imgs / f for f in d.processed_image_filename]
    print(f"{len(caminhos)} mamografias | dispositivo {DEV}\n")

    M, agr, res = {}, {}, {}
    for alias, mecanismo, alcance in FAMILIAS:
        m = mapas(alias, caminhos)
        M[alias] = m
        agr[alias] = m.mean(dim=(2, 3))
        res[alias] = dict(mecanismo=mecanismo, alcance=alcance,
                          canais=int(m.shape[1]), mapa=int(m.shape[2]),
                          celula_px=512 // int(m.shape[2]))
        print(f"{alias:10} {mecanismo:30} mapa {m.shape[2]}x{m.shape[3]}, "
              f"{m.shape[1]} canais, celula {512//m.shape[2]} px")

    print("\n=== 1. POSTO EFETIVO (quantas direcoes a representacao usa) ===")
    for alias, _, _ in FAMILIAS:
        X = agr[alias]
        pe = posto_efetivo(X)
        res[alias]["posto_efetivo"] = pe
        res[alias]["posto_fracao"] = pe / X.shape[1]
        print(f"  {alias:10} {pe:6.1f} de {X.shape[1]:5d} canais ({pe/X.shape[1]:5.1%})")

    if a.controle:
        print("\n  -- controle: imagens naturais --")
        nat = sorted(Path(a.controle).glob("*"))[:len(caminhos)]
        nat = [q for q in nat if q.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(nat) >= 32:
            for alias, _, _ in FAMILIAS:
                X = mapas(alias, nat).mean(dim=(2, 3))
                pe = posto_efetivo(X)
                res[alias]["posto_efetivo_natural"] = pe
                razao = pe / res[alias]["posto_efetivo"]
                print(f"  {alias:10} {pe:6.1f} ({pe/X.shape[1]:5.1%})  "
                      f"-> {razao:.1f}x o da mamografia")
        else:
            print(f"  poucas imagens de controle ({len(nat)}); pulando")

    print("\n=== 2. DISPERSAO ESPACIAL (variancia entre posicoes) ===")
    for alias, _, _ in FAMILIAS:
        v = dispersao_espacial(M[alias])
        res[alias]["dispersao_espacial"] = v
        print(f"  {alias:10} {v:.3f}")

    print("\n=== 3. CKA ENTRE FAMILIAS ===")
    al = [x[0] for x in FAMILIAS]
    print(f"  {'':12}" + "".join(f"{x:>12}" for x in al))
    mat = {}
    for x in al:
        linha = [cka(agr[x], agr[y]) for y in al]
        mat[x] = dict(zip(al, linha))
        print(f"  {x:12}" + "".join(f"{v:12.3f}" for v in linha))
    res["cka"] = mat

    print("\n=== 4. SENSIBILIDADE POR BANDA DE FREQUENCIA ===")
    print("    (mudanca relativa da representacao quando so a banda sobrevive)")
    bandas = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]
    rot = {"0.00-0.10": "muito baixa", "0.10-0.25": "baixa",
           "0.25-0.50": "media", "0.50-1.00": "alta"}
    sub = caminhos[:48]
    print(f"  {'':10}" + "".join(f"{rot[f'{lo:.2f}-{hi:.2f}']:>13}" for lo, hi in bandas))
    for alias, _, _ in FAMILIAS:
        s = sensibilidade_banda(alias, sub, bandas)
        res[alias]["banda"] = s
        print(f"  {alias:10}" + "".join(f"{s[k]:13.3f}" for k in s))

    print("\n=== 5. SONDA LINEAR DE POSICAO DA LESAO ===")
    cx = pd.read_csv(a.caixas)
    j = d.merge(cx[cx.tem_lesao], on="processed_image_filename")
    idx = [i for i, f in enumerate(d.processed_image_filename)
           if f in set(j.processed_image_filename)]
    if len(idx) >= 30:
        ordem = {f: i for i, f in enumerate(d.processed_image_filename)}
        j = j.sort_values("processed_image_filename", key=lambda s: s.map(ordem))
        alvo = j[["cx", "cy"]].to_numpy()
        print(f"    n = {len(idx)} imagens com lesao | R2 por validacao cruzada")
        for alias, _, _ in FAMILIAS:
            r2 = sonda_posicao(M[alias][idx], alvo)
            res[alias]["sonda_posicao_r2"] = r2
            print(f"  {alias:10} R2 = {r2:+.3f}")
    else:
        print(f"  poucas lesoes ({len(idx)}); aumente --n")

    (saida / "representacao.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\ngravado em {saida}/representacao.json")


if __name__ == "__main__":
    main()
