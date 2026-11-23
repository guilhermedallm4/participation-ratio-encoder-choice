#!/usr/bin/env python
"""Duas medidas do ALCANCE espacial de cada mecanismo.

As medidas de representacao.py olham a saida final; as de fluxo_residual.py
olham quanto cada bloco reescreve. Nenhuma das duas diz a QUE DISTANCIA cada
mecanismo olha. Isso e exatamente o eixo que separa as tres familias
(local / global / local+global hierarquico), entao e o que falta medir.

  1. CAMPO RECEPTIVO EFETIVO (de dentro para fora)
     Gradiente da unidade CENTRAL de um bloco em relacao a imagem de entrada.
     O campo receptivo TEORICO da atencao global e a imagem inteira ja no
     primeiro bloco; o EFETIVO, nao -- e o efetivo que descreve o modelo.
     Luo et al. (2016) mostraram que numa ResNet ele cresce com sqrt(da
     profundidade) e fica muito abaixo do teorico. Aqui a pergunta e como a
     curva de crescimento difere entre convolucao, atencao global e o hibrido.

  2. EMBARALHAMENTO DE RETALHOS (de fora para dentro)
     Parte a imagem numa grade g x g, permuta os retalhos e mede quanto a
     representacao AGRUPADA muda. O agrupamento e o que torna a medida
     informativa: para um extrator puramente local com campo menor que o
     retalho, mover um retalho move suas features junto, e a media espacial
     nao muda -- delta ~ 0. Quanto maior o delta, mais a representacao depende
     do ARRANJO e nao so do conteudo.

     Confusao a controlar: embaralhar cria costuras artificiais nas bordas dos
     retalhos, e o numero de costuras cresce com g. Por isso cada grade roda em
     dois modos com o MESMO numero de costuras:
         global  permuta livre entre os g^2 retalhos  (destroi o arranjo)
         local   permuta so dentro de vizinhancas 2x2 (preserva o arranjo grosso)
     A diferenca entre os dois isola a sensibilidade ao arranjo do artefato de
     borda, que afeta os dois por igual.

Tudo em extrator CONGELADO, sem treino: sem variancia de semente nem de
particao.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import multitarefa as MT
from representacao import transformacao
from fluxo_residual import sub_blocos, canais_por_ultimo

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILIAS = ["resnet50", "maxvit_t", "vit_b16"]


def carrega(caminhos, tf, tam=512, transformar=None, cinza=True):
    """Lote (N,3,tam,tam) ja normalizado, opcionalmente perturbado.

    cinza=False mantem a cor. E o controle para a objecao de que converter
    tudo para escala de cinza -- feito para que a cor nao confunda a
    comparacao com a mamografia -- possa penalizar de forma desigual os
    modelos treinados com linguagem sobre imagem colorida, como o CLIP.
    As perturbacoes (mascara, embaralhamento) operam em 2D e so valem em cinza.
    """
    lote = []
    for q in caminhos:
        if not cinza:
            lote.append(tf(Image.open(q).convert("RGB")
                           .resize((tam, tam), Image.Resampling.BILINEAR)))
            continue
        im = Image.open(q).convert("L").resize((tam, tam), Image.Resampling.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        if transformar is not None:
            a = transformar(a)
        lote.append(tf(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)
                                       ).convert("RGB")))
    return torch.stack(lote)


# ---------------------------------------------------------------------------
# 1. Campo receptivo efetivo
# ---------------------------------------------------------------------------

def _unidade_central(saida, mod, n_prefixo):
    """Soma dos canais da posicao central -- um escalar por imagem.

    Tres formatos convivem nestes modelos e confundi-los nao da erro, da um
    numero errado com a forma certa:
      (N,C,H,W)  MBConv, ResNet
      (N,H,W,C)  atencao do MaxViT (PartitionAttentionCl, channels-last)
      (N,L,C)    fichas do ViT, com L = prefixo (CLS) + grade
    """
    if saida.dim() == 4:
        if canais_por_ultimo(mod):                       # (N,H,W,C)
            h, w = saida.shape[1], saida.shape[2]
            return saida[:, h // 2, w // 2, :].sum()
        h, w = saida.shape[2], saida.shape[3]            # (N,C,H,W)
        return saida[:, :, h // 2, w // 2].sum()
    g = int(round((saida.shape[1] - n_prefixo) ** 0.5))  # (N,L,C)
    t = saida[:, n_prefixo:, :].reshape(saida.shape[0], g, g, -1)
    return t[:, g // 2, g // 2, :].sum()


def _perfil(G, tam):
    """De um mapa de gradiente (H,W) normalizado para as estatisticas de alcance.

    raio_50 / raio_90  distancia do centro que acumula 50% / 90% da massa.
                       Referencia: um gradiente UNIFORME na imagem toda da
                       raio_50 ~ 205 px e raio_90 ~ 280 px em 512 px, entao e
                       esse o teto, nao 256.
    area_90            fracao dos pixels necessaria para acumular 90% da massa,
                       ordenando por magnitude. Mede CONCENTRACAO sem supor que
                       o campo esteja centrado.
    """
    h, w = G.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2).ravel()
    g = G.ravel()
    o = np.argsort(r)
    acum = np.cumsum(g[o])
    rs = r[o]
    raio = {f"raio_{int(q*100)}": float(rs[np.searchsorted(acum, q)])
            for q in (0.5, 0.9)}
    gs = np.sort(g)[::-1]
    a90 = int(np.searchsorted(np.cumsum(gs), 0.9)) + 1
    raio["area_90"] = a90 / g.size
    return raio


def campo_receptivo(backbone, caminhos, tam=512, bs=2):
    """Campo receptivo efetivo de cada sub-bloco, em pixels da entrada."""
    modelo, _, norm = MT.monta_backbone(backbone, pre=True, treinavel=False)
    modelo = modelo.to(DEV).eval()
    interno = modelo.m if hasattr(modelo, "m") else modelo
    blocos = sub_blocos(backbone, modelo if backbone.startswith("resnet") else interno)

    guarda = {}
    ganchos = [mod.register_forward_hook(
        lambda m, e, s, n=n: guarda.__setitem__(n, s if torch.is_tensor(s) else s[0]))
        for n, _, mod in blocos]

    n_pref = int(getattr(interno, "num_prefix_tokens", 0) or 0)
    tf = transformacao(norm)
    soma = {n: np.zeros((tam, tam), dtype=np.float64) for n, _, _ in blocos}
    vistas = 0
    for i in range(0, len(caminhos), bs):
        x = carrega(caminhos[i:i + bs], tf, tam).to(DEV).requires_grad_(True)
        guarda.clear()
        modelo(x)
        for j, (n, _, mod) in enumerate(blocos):
            if n not in guarda:
                continue
            alvo = _unidade_central(guarda[n], mod, n_pref)
            # retain_graph ate o ultimo bloco: um backward por bloco sobre o
            # mesmo grafo e muito mais barato que um forward por bloco.
            g, = torch.autograd.grad(alvo, x, retain_graph=(j < len(blocos) - 1))
            a = g.abs().sum(1).double().cpu().numpy()      # soma os 3 canais
            a = a / np.maximum(a.sum(axis=(1, 2), keepdims=True), 1e-30)
            soma[n] += a.sum(0)
        vistas += x.shape[0]
        del x, g
        torch.cuda.empty_cache()
    for h in ganchos:
        h.remove()

    linhas = []
    for n, mec, _ in blocos:
        if soma[n].sum() == 0:
            continue
        G = soma[n] / soma[n].sum()
        linhas.append(dict(backbone=backbone, bloco=n, mecanismo=mec,
                           n_imagens=vistas, **_perfil(G, tam)))
    del modelo
    torch.cuda.empty_cache()
    return linhas


# ---------------------------------------------------------------------------
# 2. Embaralhamento de retalhos
# ---------------------------------------------------------------------------

class Embaralhador:
    """Permuta os retalhos de uma grade g x g, em modo global ou local.

    Guarda o proprio gerador e e reinicializado a cada chamada, entao as tres
    familias veem EXATAMENTE as mesmas permutacoes nas mesmas imagens -- a
    comparacao entre elas nao carrega ruido de sorteio.
    """

    def __init__(self, g, modo, semente=0):
        self.g, self.modo, self.rng = g, modo, np.random.default_rng(semente)

    def _permuta(self):
        g = self.g
        idx = np.arange(g * g).reshape(g, g)
        if self.modo == "global":
            return self.rng.permutation(idx.ravel()).reshape(g, g)
        # local: permuta so dentro de cada vizinhanca 2x2 de retalhos
        fora = idx.copy()
        for i in range(0, g, 2):
            for j in range(0, g, 2):
                b = idx[i:i + 2, j:j + 2].ravel()
                fora[i:i + 2, j:j + 2] = self.rng.permutation(b).reshape(2, 2)
        return fora

    def __call__(self, a):
        g, (h, w) = self.g, a.shape
        ph, pw = h // g, w // g
        r = a[:ph * g, :pw * g].reshape(g, ph, g, pw).transpose(0, 2, 1, 3)
        p = self._permuta().ravel()
        r = r.reshape(g * g, ph, pw)[p].reshape(g, g, ph, pw)
        return r.transpose(0, 2, 1, 3).reshape(ph * g, pw * g)


def _agrupado(backbone, caminhos, tf, tam, bs, transformar=None, modelo=None):
    saida = []
    with torch.no_grad():
        for i in range(0, len(caminhos), bs):
            x = carrega(caminhos[i:i + bs], tf, tam, transformar).to(DEV)
            o = modelo(x).float()
            saida.append((o.mean(dim=(2, 3)) if o.dim() == 4 else o.mean(1)).cpu())
    return torch.cat(saida)


def cka_gram(X, Y):
    X = (X - X.mean(0)).double(); Y = (Y - Y.mean(0)).double()
    K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / (K.norm() * L.norm()).clamp_min(1e-12))


def embaralhamento(backbone, caminhos, grades=(4, 8, 16, 32), tam=512, bs=8):
    modelo, _, norm = MT.monta_backbone(backbone, pre=True, treinavel=False)
    modelo = modelo.to(DEV).eval()
    tf = transformacao(norm)

    base = _agrupado(backbone, caminhos, tf, tam, bs, None, modelo)
    nb = base.norm(dim=1).clamp_min(1e-8)
    linhas = []
    for g in grades:
        for modo in ("global", "local"):
            p = _agrupado(backbone, caminhos, tf, tam, bs,
                          Embaralhador(g, modo, semente=g * 10 + (modo == "local")),
                          modelo)
            linhas.append(dict(backbone=backbone, grade=g, retalho_px=tam // g,
                               modo=modo,
                               delta=float(((p - base).norm(dim=1) / nb).mean()),
                               cka=cka_gram(base, p)))
            print(f"   {backbone:10} g={g:2d} ({tam//g:3d} px) {modo:7} "
                  f"delta {linhas[-1]['delta']:.3f}  CKA {linhas[-1]['cka']:.3f}",
                  flush=True)
    del modelo
    torch.cuda.empty_cache()
    return linhas


# ---------------------------------------------------------------------------

def imagens(d, n):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() == ".png")[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amostras", default="data/amostras")
    p.add_argument("--bancos", default="mamografia,dtd,pet,sun397")
    p.add_argument("--out", default="resultados")
    p.add_argument("--n-campo", type=int, default=16)
    p.add_argument("--n-emb", type=int, default=96)
    a = p.parse_args()

    saida = Path(a.out); saida.mkdir(parents=True, exist_ok=True)
    campos, embs = [], []
    for banco in [b for b in a.bancos.split(",") if b]:
        d = Path(a.amostras) / banco
        if not d.exists():
            print(f"!! {banco}: {d} nao existe, pulando", flush=True)
            continue
        print(f"\n{'=' * 70}\nBANCO: {banco}\n{'=' * 70}", flush=True)

        print("\n1. CAMPO RECEPTIVO EFETIVO", flush=True)
        qs = imagens(d, a.n_campo)
        for bb in FAMILIAS:
            l = campo_receptivo(bb, qs)
            for r in l:
                r["banco"] = banco
            campos += l
            g = pd.DataFrame(l)
            for mec, s in g.groupby("mecanismo", sort=False):
                print(f"   {bb:10} {mec:24} n={len(s):2d}  raio50 "
                      f"{s.raio_50.min():5.1f}->{s.raio_50.max():5.1f} px", flush=True)
            pd.DataFrame(campos).to_csv(saida / "campo_receptivo.csv", index=False)

        print("\n2. EMBARALHAMENTO DE RETALHOS", flush=True)
        qs = imagens(d, a.n_emb)
        for bb in FAMILIAS:
            l = embaralhamento(bb, qs)
            for r in l:
                r["banco"] = banco
            embs += l
            pd.DataFrame(embs).to_csv(saida / "embaralhamento.csv", index=False)
        print(f"\n-> parcial gravado", flush=True)

    print("\nconcluido", flush=True)


if __name__ == "__main__":
    main()
