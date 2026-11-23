#!/usr/bin/env python
"""Figuras do campo receptivo efetivo e do embaralhamento de retalhos.

Identidade NUNCA so por cor: cada arquitetura tem cor, marcador e tracejado
proprios, porque a figura vai para um artigo que pode ser impresso em P&B.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta categorica validada (slots 1-3, modo claro, --pairs all):
# pior par CVD dE 9,2 e visao normal dE 24,0. O aviso de contraste do agua
# (2,74:1) e coberto pelos rotulos diretos, como manda a regra de alivio.
COR = {"resnet50": "#2a78d6", "maxvit_t": "#eb6834", "vit_b16": "#1baf7a"}
MARCA = {"resnet50": "o", "maxvit_t": "s", "vit_b16": "^"}
TRACO = {"resnet50": "-", "maxvit_t": "--", "vit_b16": ":"}
ROT = {"resnet50": "ResNet-50", "maxvit_t": "MaxViT-T", "vit_b16": "ViT-B/16"}
ROT_BANCO = {"mamografia": "mamografia", "dtd": "DTD (textura)",
             "pet": "Pet (objeto)", "sun397": "SUN397 (cena)"}
TINTA, TINTA2 = "#0b0b0b", "#52514e"
ORDEM = ["mamografia", "dtd", "pet", "sun397"]

def _uniforme(q, tam=512):
    """Raio que um gradiente UNIFORME acumularia, no mesmo grid da medida.

    E o teto da escala, e nao e 256: 256 px e a distancia ate a borda, nao a
    mediana da massa. Para q=0,5 o disco cabe no quadrado e vale
    sqrt(0,5 * tam^2 / pi) = 204,3 px; para q=0,9 ele ja e cortado pelos
    cantos, entao o valor sai do mesmo cumulativo radial usado em _perfil.
    """
    yy, xx = np.ogrid[:tam, :tam]
    r = np.sort(np.sqrt((yy - (tam - 1) / 2) ** 2 + (xx - (tam - 1) / 2) ** 2).ravel())
    return float(r[int(q * r.size)])


UNIFORME_50, UNIFORME_90 = _uniforme(0.5), _uniforme(0.9)


def estilo():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5,
        "axes.edgecolor": "#b8b7b2", "axes.linewidth": 0.6,
        "axes.labelcolor": TINTA, "axes.titlesize": 8,
        "xtick.color": TINTA2, "ytick.color": TINTA2,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "hatch.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 7,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _recessivo(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_campo(cr, saida):
    """A: crescimento do campo com a profundidade, por banco.
       B: o MaxViT aberto em conv / atencao local / atencao em grade."""
    fig = plt.figure(figsize=(7.1, 4.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], hspace=0.95, wspace=0.22,
                          top=0.86)

    # --- A: saida de bloco de cada familia ---------------------------------
    # Para o MaxViT o "bloco" e o MaxxVitBlock inteiro, cuja saida e a atencao
    # em grade; comparar com o sub-bloco conv seria comparar meio bloco com o
    # bloco inteiro das outras duas.
    saida_bloco = cr[(cr.backbone != "maxvit_t") |
                     (cr.mecanismo == "atencao em grade")]
    # Os rotulos diretos do ultimo painel se sobrepoem quando duas curvas
    # terminam perto; empurra cada um para a sua faixa.
    desvio = {"vit_b16": 5, "maxvit_t": -6, "resnet50": 0}
    for k, banco in enumerate(ORDEM):
        ax = fig.add_subplot(gs[0, k])
        ax.axhline(UNIFORME_50, color="#b8b7b2", lw=0.8, ls=(0, (4, 3)), zorder=1)
        if k == 0:
            ax.text(0.03, UNIFORME_50 * 1.1, "gradiente uniforme", fontsize=5.6,
                    color=TINTA2, va="bottom")
        for bb in ROT:
            s_ = saida_bloco[(saida_bloco.banco == banco) &
                             (saida_bloco.backbone == bb)]
            if s_.empty:
                continue
            x = np.linspace(0, 1, len(s_))
            ax.plot(x, s_.raio_50.to_numpy(), TRACO[bb], color=COR[bb], lw=1.4,
                    marker=MARCA[bb], ms=2.8, mew=0, zorder=3,
                    label=ROT[bb] if k == 0 else None)
            if k == len(ORDEM) - 1:          # rotulo direto no ultimo painel
                ax.annotate(ROT[bb], (1.0, s_.raio_50.iloc[-1]),
                            xytext=(4, desvio[bb]), textcoords="offset points",
                            fontsize=6.2, color=TINTA2, va="center")
        ax.set_yscale("log"); ax.set_ylim(2.5, 430)
        ax.set_yticks([4, 16, 64, 256]); ax.set_yticklabels(["4", "16", "64", "256"])
        ax.set_xlim(-0.05, 1.05); ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "½", "1"])
        ax.set_xlabel("profundidade", labelpad=1.5, fontsize=6.8, color=TINTA2)
        ax.set_title(ROT_BANCO[banco], color=TINTA, pad=3)
        _recessivo(ax)
        if k:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("raio do campo efetivo (px)")
        if k == len(ORDEM) - 1:
            ax.set_xlim(-0.05, 1.34)
    maos = [plt.Line2D([], [], color=COR[b], ls=TRACO[b], marker=MARCA[b],
                       ms=3.4, mew=0, lw=1.4, label=ROT[b]) for b in ROT]
    fig.legend(handles=maos, loc="upper center", ncol=3, handlelength=2.0,
               bbox_to_anchor=(0.5, 1.005), columnspacing=2.2)

    # --- B: o bloco do MaxViT por dentro -----------------------------------
    # Duas medidas porque elas discordam, e a discordancia E o achado: o nucleo
    # do campo (raio 50) continua estreito enquanto a cauda (raio 90) ja cobre
    # a imagem inteira. A atencao em grade nao desloca o campo, acrescenta uma
    # cauda longa a um nucleo que segue local.
    sub = cr[(cr.backbone == "maxvit_t") & (cr.banco == "mamografia")]
    estagios = sub.bloco.str.split(".").str[:2].str.join(".").unique()
    # A hachura do mecanismo do meio e o alivio para impressao P&B: em escala
    # de cinza o laranja e o cinza ficam a quase a mesma luminancia.
    ordem_mec = [("conv (MBConv)", "#9a9994", "", "conv (MBConv)"),
                 ("atencao janela 7x7", "#eb6834", "///", "atenção em janela 7×7"),
                 ("atencao em grade", "#2a78d6", "", "atenção em grade")]
    larg, base = 0.26, np.arange(len(estagios))
    rot_bloco = [e.replace("e", "").replace(".b", "·") for e in estagios]
    for c, (col, titulo) in enumerate([("raio_50", "núcleo do campo (raio 50%)"),
                                       ("raio_90", "cauda do campo (raio 90%)")]):
        ax = fig.add_subplot(gs[1, 2 * c:2 * c + 2])
        for i, (mec, cor, hach, rot) in enumerate(ordem_mec):
            v = [sub[(sub.bloco.str.startswith(e + ".")) &
                     (sub.mecanismo == mec)][col].mean() for e in estagios]
            # 0,93 da largura deixa a folga de superficie entre barras vizinhas
            ax.bar(base + (i - 1) * larg, v, larg * 0.93, color=cor,
                   label=rot if c == 0 else None, zorder=3, linewidth=0,
                   hatch=hach, edgecolor="white")
        ax.axhline(UNIFORME_50 if col == "raio_50" else UNIFORME_90,
                   color="#b8b7b2", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_yscale("log"); ax.set_ylim(3, 600)
        ax.set_yticks([4, 16, 64, 256]); ax.set_yticklabels(["4", "16", "64", "256"])
        ax.set_xticks(base); ax.set_xticklabels(rot_bloco, fontsize=6.2)
        ax.set_xlabel("estágio · bloco", labelpad=1.5, fontsize=6.8, color=TINTA2)
        ax.set_title(titulo, color=TINTA, pad=3)
        _recessivo(ax)
        if c == 0:
            ax.set_ylabel("raio (px)")
            maos_b, rots_b = ax.get_legend_handles_labels()
    fig.text(0.5, 0.445, "Dentro do bloco do MaxViT (mamografia)", ha="center",
             fontsize=8, color=TINTA)
    fig.legend(maos_b, rots_b, loc="upper center", ncol=3, handlelength=1.1,
               handleheight=0.8, fontsize=6.6, bbox_to_anchor=(0.5, 0.443),
               columnspacing=1.8)
    fig.savefig(saida, dpi=300)
    plt.close(fig)
    print(f"-> {saida}")


def fig_embaralho(em, saida):
    """Sensibilidade ao ARRANJO: delta(global) - delta(local), por tamanho de
    retalho. A subtracao e o controle -- os dois modos criam o mesmo numero de
    costuras artificiais, so o global destroi o arranjo grosso."""
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.95), sharey=True)
    for k, (banco, ax) in enumerate(zip(ORDEM, axs)):
        for bb in ROT:
            s = em[(em.banco == banco) & (em.backbone == bb)]
            if s.empty:
                continue
            p = s.pivot_table(index="retalho_px", columns="modo", values="delta")
            p = p.sort_index(ascending=False)
            y = (p["global"] - p["local"]).to_numpy()
            ax.plot(p.index.to_numpy(), y, TRACO[bb], color=COR[bb], lw=1.4,
                    marker=MARCA[bb], ms=3.4, mew=0, zorder=3,
                    label=ROT[bb] if k == 0 else None)
            if k == len(ORDEM) - 1:
                ax.annotate(ROT[bb], (p.index[-1], y[-1]), xytext=(3, 0),
                            textcoords="offset points", fontsize=6.2,
                            color=TINTA2, va="center")
        ax.axhline(0, color="#b8b7b2", lw=0.7, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 64, 32, 16])
        ax.set_xticklabels(["128", "64", "32", "16"])
        ax.invert_xaxis()
        ax.set_title(ROT_BANCO[banco], color=TINTA, pad=3)
        ax.set_xlabel("retalho (px)")
        _recessivo(ax)
    axs[-1].set_xlim(axs[-1].get_xlim()[0], 11)
    axs[0].set_ylabel("sensibilidade ao arranjo\n(Δ global − Δ local)")
    axs[0].legend(loc="upper left", handlelength=1.8)
    fig.savefig(saida, dpi=300)
    plt.close(fig)
    print(f"-> {saida}")


def main():
    estilo()
    r = Path("resultados"); f = Path("figuras"); f.mkdir(exist_ok=True)
    cr = pd.read_csv(r / "campo_receptivo.csv")
    em = pd.read_csv(r / "embaralhamento.csv")
    fig_campo(cr, f / "campo_receptivo.pdf")
    fig_embaralho(em, f / "embaralhamento.pdf")


if __name__ == "__main__":
    main()
