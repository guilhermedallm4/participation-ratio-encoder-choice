#!/usr/bin/env python
"""Quanto cada mecanismo MOVE a representacao, ao longo da profundidade.

As cinco medidas de representacao.py olham a saida final do extrator. Esta
olha o caminho ate ela. Toda arquitetura moderna e um fluxo residual:

    x_{n+1} = x_n + F(x_n)

onde F e uma convolucao (ResNet), uma atencao em janela (MaxViT attn_block),
uma atencao em grade (MaxViT attn_grid) ou uma atencao global (ViT). Medir
||F(x)|| / ||x|| diz que fracao da representacao cada mecanismo REESCREVE, e
CKA(x, x+F(x)) diz quanto da geometria ele PRESERVA.

E a resposta direta a duas perguntas:
  - o residuo da ResNet preserva ou reescreve a informacao espacial?
  - as atencoes local e global do MaxViT competem ou se dividem o trabalho?
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import numpy as np
from PIL import Image
import multitarefa as MT
from representacao import transformacao

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sub_blocos(backbone, modelo):
    """Lista (nome, mecanismo, modulo) dos sub-blocos residuais, em ordem."""
    fora = []
    if backbone.startswith("resnet"):
        r = modelo[0] if isinstance(modelo, torch.nn.Sequential) else modelo
        # nn.Sequential(*children[:-2]) -> indices 4..7 sao layer1..layer4
        for e, i in enumerate((4, 5, 6, 7), start=1):
            for j, bloco in enumerate(modelo[i]):
                fora.append((f"e{e}.b{j}", "conv residual", bloco))
    elif backbone.startswith("maxvit"):
        m = modelo.m if hasattr(modelo, "m") else modelo
        for e, est in enumerate(m.stages, start=1):
            for j, bloco in enumerate(est.blocks):
                fora.append((f"e{e}.b{j}.conv", "conv (MBConv)", bloco.conv))
                fora.append((f"e{e}.b{j}.local", "atencao janela 7x7", bloco.attn_block))
                fora.append((f"e{e}.b{j}.global", "atencao em grade", bloco.attn_grid))
    elif backbone.startswith("vit_"):
        m = modelo.m if hasattr(modelo, "m") else modelo
        # Hookar bloco.attn seria ERRADO: sua entrada e norm1(x), nao o fluxo
        # residual x, e sua saida nao inclui a soma. A diferenca medida nao
        # seria a atualizacao residual. O Block inteiro faz
        # x = x + attn(norm1(x)); x = x + mlp(norm2(x)) -- entra fluxo, sai
        # fluxo, exatamente como PartitionAttentionCl do MaxViT (que tambem
        # embute seu proprio MLP) e como o Bottleneck da ResNet.
        for j, bloco in enumerate(m.blocks):
            fora.append((f"b{j}", "atencao global + MLP", bloco))
    return fora


def cka_gram(X, Y):
    """CKA linear na forma de Gram: O(n^2) em vez de O(d^2).

    ||X^T Y||_F^2 = tr(K L) com K = XX^T, L = YY^T, entao
    CKA = tr(KL) / (||K||_F ||L||_F). Com n=64 amostras e d~10^6 features,
    a forma feature-space alocaria uma matriz d x d; esta aloca n x n.
    """
    X = (X - X.mean(0)).double(); Y = (Y - Y.mean(0)).double()
    K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / (K.norm() * L.norm()).clamp_min(1e-12))


def canais_por_ultimo(mod):
    """O modulo entrega (N,H,W,C) em vez de (N,C,H,W)?

    O timm roda a atencao do MaxViT em channels-last (PartitionAttentionCl, o
    sufixo Cl), enquanto o MBConv do mesmo bloco roda em NCHW. Os dois
    formatos convivem dentro do mesmo modelo, entao quem for tirar media
    espacial precisa saber qual e qual: mediar (2,3) num tensor NHWC mediaria
    largura e CANAL, devolvendo um vetor indexado por ALTURA -- um objeto sem
    sentido, que ainda assim tem a forma certa e passa despercebido.
    """
    return type(mod).__name__.endswith("Cl")


def resume(t, nhwc=False):
    """(N,C,H,W), (N,H,W,C) ou (N,L,C) -> (N,C) por media espacial.

    Guardar o mapa inteiro de 16 sub-blocos estoura a memoria; a geometria de
    similaridade entre imagens, que e o que o CKA mede, sobrevive a media
    espacial -- e e assim que representacao.py ja calcula o CKA final.
    """
    if t.dim() == 4:
        return t.mean(dim=(1, 2)) if nhwc else t.mean(dim=(2, 3))
    if t.dim() == 3:
        return t.mean(dim=1)
    return t.reshape(t.shape[0], -1)


def mede(backbone, caminhos, tam=512, bs=8, detalhe=False):
    modelo, canais, norm = MT.monta_backbone(backbone, pre=True, treinavel=False)
    modelo = modelo.to(DEV).eval()
    interno = modelo.m if hasattr(modelo, "m") else modelo
    blocos = sub_blocos(backbone, modelo if backbone.startswith("resnet") else interno)

    reg = {n: {"mec": mec, "ent": [], "sai": [], "frac": [], "forma_ok": True}
           for n, mec, _ in blocos}
    ramo = {}          # nome -> norma do ramo residual do lote corrente
    ganchos = []

    # O Bottleneck da ResNet faz out = relu(identity + F(x)): o ReLU vem DEPOIS
    # da soma, entao (saida - entrada) NAO e o ramo residual, inclui o corte do
    # ReLU. No ViT e no MaxViT, que sao PreNorm com soma pura, (saida - entrada)
    # E exatamente o ramo. Para medir a mesma grandeza nas tres, capturamos F
    # diretamente na ultima BN do ramo da ResNet.
    def faz_ramo(nome):
        def g(mod, entrada, saida):
            ramo[nome] = saida.detach().flatten(1).norm(dim=1).cpu()
        return g

    if backbone.startswith("resnet"):
        for n, _, mod in blocos:
            ganchos.append(mod.bn3.register_forward_hook(faz_ramo(n)))

    def faz(nome):
        def g(mod, entrada, saida):
            cl = canais_por_ultimo(mod)
            e = entrada[0].detach()
            s = saida.detach() if torch.is_tensor(saida) else saida[0].detach()
            if e.shape != s.shape:      # muda de forma: nao e passo residual puro
                reg[nome]["forma_ok"] = False
                return
            # fracao reescrita = ||ramo residual|| / ||fluxo de entrada||
            pe = e.flatten(1); ps = s.flatten(1)
            ne = pe.norm(dim=1).clamp_min(1e-8).cpu()
            fluxo = (ps - pe).norm(dim=1).cpu() / ne    # quanto o FLUXO muda
            nr = ramo.get(nome)                          # ResNet: F na bn3
            branch = (nr / ne) if nr is not None else fluxo
            reg[nome]["frac"].append(branch)
            reg[nome].setdefault("fluxo", []).append(fluxo)
            reg[nome]["ent"].append(resume(e, cl).float().cpu())
            reg[nome]["sai"].append(resume(s, cl).float().cpu())
        return g

    for n, _, mod in blocos:
        ganchos.append(mod.register_forward_hook(faz(n)))

    tf = transformacao(norm)
    with torch.no_grad():
        for i in range(0, len(caminhos), bs):
            lote = []
            for q in caminhos[i:i + bs]:
                im = Image.open(q).convert("L").resize((tam, tam), Image.Resampling.BILINEAR)
                lote.append(tf(im.convert("RGB")))
            modelo(torch.stack(lote).to(DEV))
    for h in ganchos:
        h.remove()

    linhas, por_imagem = [], {}
    for n, mec, _ in blocos:
        if not reg[n]["forma_ok"] or not reg[n]["frac"]:
            continue
        # guarda o ramo IMAGEM A IMAGEM, para que o intervalo possa ser
        # calculado reamostrando imagens -- a media ja agregada nao permite
        por_imagem[n] = (mec, torch.cat(reg[n]["frac"]).numpy())
        E = torch.cat(reg[n]["ent"]); S = torch.cat(reg[n]["sai"])
        linhas.append(dict(backbone=backbone, bloco=n, mecanismo=mec,
                           ramo=float(torch.cat(reg[n]["frac"]).mean()),
                           fluxo=float(torch.cat(reg[n]["fluxo"]).mean()),
                           cka_preserva=cka_gram(E, S)))
    del modelo
    torch.cuda.empty_cache()
    return (linhas, por_imagem) if detalhe else linhas


def main():
    p = argparse.ArgumentParser(description="fluxo residual e atencao")
    p.add_argument("--imagens", default="../data/inbreast_png_896")
    p.add_argument("--csv", default="../data/inbreast_images.csv")
    p.add_argument("--out", default="../data/representacao")
    p.add_argument("--n", type=int, default=64)
    a = p.parse_args()

    d = pd.read_csv(a.csv).sample(a.n, random_state=0)
    caminhos = [Path(a.imagens) / f for f in d.processed_image_filename]
    print(f"{len(caminhos)} mamografias | {DEV}\n")

    todas = []
    for b in ("resnet50", "maxvit_t", "vit_b16"):
        print(f"-- {b} --")
        linhas = mede(b, caminhos)
        todas += linhas
        g = pd.DataFrame(linhas)
        for mec, sub in g.groupby("mecanismo", sort=False):
            print(f"   {mec:22} n={len(sub):3d}  ramo {sub.ramo.mean():.3f}"
                  f"  fluxo {sub.fluxo.mean():.3f}  CKA {sub.cka_preserva.mean():.3f}")
    df = pd.DataFrame(todas)
    saida = Path(a.out); saida.mkdir(parents=True, exist_ok=True)
    df.to_csv(saida / "fluxo_residual.csv", index=False)
    print(f"\ngravado em {saida}/fluxo_residual.csv  ({len(df)} sub-blocos)")


if __name__ == "__main__":
    main()
