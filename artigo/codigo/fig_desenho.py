#!/usr/bin/env python
"""Figura 1: o desenho em cruz, que e o argumento do artigo em uma imagem.

Dezessete extratores em dois bracos que se cruzam num checkpoint compartilhado.
O braco VERTICAL varia a arquitetura dentro de um mesmo regime supervisionado; o
HORIZONTAL varia o pre-treino com a arquitetura ViT-B/16 fixa ate o ultimo
parametro. As duas correlacoes do artigo sao medidas nesses dois bracos e dao
resultados opostos -- e e isso que a figura precisa deixar obvio antes que o
leitor chegue na tabela.

Largura 12,2 cm (bloco de texto do ESANN), impressa em P&B.
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from exato import salva

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

LARGURA_CM = 12.2
POL = LARGURA_CM / 2.54
TINTA, TINTA2 = "#0b0b0b", "#52514e"
AZUL, LARANJA, VERDE, BORDA = "#2a78d6", "#eb6834", "#1baf7a", "#d8d7d3"

ARQ = ["ResNet-18", "ResNet-50", "ConvNeXt-T", "EfficientNet-B3", "Swin-T",
       "MaxViT-T", "MaxViT-S", "ViT-S/16", "ViT-B/16"]
PRE = ["superv.", "superv.\nIN-21k", "superv.\nIN-1k", "DeiT3", "DINO", "MAE",
       "CLIP\nOpenAI", "CLIP\nLAION", "SigLIP"]
CRUZ = len(ARQ) - 1          # a linha do ViT-B/16
COL = 0                      # a coluna supervisionada

TEXTO = {
    "en": dict(arq="architecture arm\n(shared supervised regime)",
               pre="pretraining arm (architecture fixed: ViT-B/16)",
               comp="shared checkpoint",
               pr=["superv.", "superv.\nIN-21k", "superv.\nIN-1k", "DeiT3",
                   "DINO", "MAE", "CLIP\nOpenAI", "CLIP\nLAION", "SigLIP"]),
    "pt": dict(arq="braço da arquitetura\n(mesmo regime supervisionado)",
               pre="braço do pré-treino (arquitetura fixa: ViT-B/16)",
               comp="checkpoint compartilhado",
               pr=PRE),
}


def main(idioma="en"):
    T = TEXTO[idioma]
    plt.rcParams.update({"font.family": "serif", "savefig.bbox": "tight",
                         "savefig.pad_inches": .015, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(POL, POL * 0.46))
    ax.set_axis_off()
    n = len(ARQ)

    for c, rot in enumerate(T["pr"]):
        ax.text(c, n - .35, rot, ha="center", va="bottom", fontsize=5.6,
                color=TINTA, linespacing=1.1)

    for i, nome in enumerate(ARQ):
        y = n - 1 - i
        ax.text(-1.45, y, nome, ha="right", va="center", fontsize=6.4, color=TINTA)
        for c in range(len(PRE)):
            na_cruz = (i == CRUZ) or (c == COL)
            if not na_cruz:
                ax.scatter([c], [y], s=5, c=BORDA, linewidths=0, zorder=3)
                continue
            cor = VERDE if (i == CRUZ and c == COL) else (LARANJA if i == CRUZ else AZUL)
            ax.scatter([c], [y], s=44, c=cor, linewidths=0, zorder=4)

    yv = n - 1 - CRUZ
    ax.add_patch(FancyBboxPatch((-.40, yv - .34), len(PRE) - .20, .68,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                facecolor="none", edgecolor=LARANJA, lw=.9, zorder=2))
    ax.text((len(PRE) - 1) / 2, yv - 1.08, T["pre"], ha="center", va="top",
            fontsize=6.2, color=LARANJA)

    ax.add_patch(FancyBboxPatch((COL - .40, yv + .54), .80, n - 1.08,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                facecolor="none", edgecolor=AZUL, lw=.9, zorder=2))
    ax.text(COL - 1.12, (n + .6) / 2, T["arq"], ha="center", va="center",
            fontsize=6.2, color=AZUL, rotation=90, linespacing=1.2)

    # o rotulo do checkpoint compartilhado vai para a esquerda: embaixo ele
    # cairia sobre o rotulo do braco do pre-treino
    ax.annotate(T["comp"], xy=(COL, yv), xytext=(-4.5, yv - .78),
                fontsize=6.0, color=VERDE, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=VERDE, lw=.7,
                                connectionstyle="arc3,rad=0.22", shrinkA=2,
                                shrinkB=4))

    ax.set_xlim(-5.1, len(PRE) - .3); ax.set_ylim(-1.9, n + .75)
    nome = "fig1_desenho" + ("" if idioma == "en" else f"_{idioma}") + ".pdf"
    w, h = salva(fig, Path(__file__).parent.parent / "figuras" / nome)
    print(f"-> figuras/{nome}  {w*2.54:.1f} x {h*2.54:.1f} cm")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--idioma", default="en", choices=["en", "pt"])
    main(a.parse_args().idioma)
