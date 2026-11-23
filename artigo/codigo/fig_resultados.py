#!/usr/bin/env python
"""Figura 2: os tres resultados que sobreviveram aos controles.

(a) Quanto cada medida responde ao DOMINIO. Os marcadores mostram o mesmo
    numero calculado em conjuntos de 6, 12 e 17 extratores: a ORDEM entre as
    medidas e estavel, as porcentagens oscilam por um fator de 2 a 3. Quem
    citar um desses numeros precisa citar o conjunto que o produziu.
(b) Os nove ViT-B/16, arquitetura identica. As tres familias de pre-treino se
    SOBREPOEM: a faixa dos quatro supervisionados cobre quase todos os outros.
    E por isso que "objetivo" nao e um fator utilizavel aqui.
(c) Razao de participacao contra sonda, nos dois bracos do desenho. Nulo entre
    arquiteturas, presente entre pre-treinos.

Largura 12,2 cm, impressa em P&B: tudo sai tambem por marcador ou hachura.
"""
import argparse, json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from exato import salva

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

LARGURA_CM = 12.2
POL = LARGURA_CM / 2.54
TINTA, TINTA2 = "#0b0b0b", "#52514e"
AZUL, LARANJA, VERDE, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#9a9994"

BANCOS = ["mamografia", "dtd", "pet", "sun397"]
ARQ = ["resnet18", "resnet50", "convnext_t", "efficientnet_b3", "swin_t",
       "maxvit_t", "maxvit_s", "vit_s16", "vit_b16"]
OBJ = ["vit_b16", "vit_b16_sup21k", "vit_b16_sup1k", "vit_b16_deit3",
       "vit_b16_dino", "vit_b16_mae", "vit_b16_clip", "vit_b16_clip2b",
       "vit_b16_siglip"]
TODOS = ARQ + OBJ[1:]
SEIS = ["resnet50", "convnext_t", "swin_t", "maxvit_t", "vit_s16", "vit_b16"]
DOZE = ARQ + ["vit_b16_dino", "vit_b16_mae", "vit_b16_clip"]
FAM = {e: ("superv." if e in OBJ[:4] else "auto-sup." if e in OBJ[4:6] else "ling.")
       for e in OBJ}

TEXTO = {
    "en": dict(med=["participation\nratio", "ERF\nradius", "residual\nflow"],
               frac="domain share of log-variance",
               conj=["6 encoders", "12 encoders", "17 encoders"],
               fam=["sup.", "self-s.", "lang."],
               fluxo=r"residual flow $\|F(x)\|/\|x\|$",
               faixa="range of the 4 supervised",
               ta="(a) response to the domain", tb="(b) nine ViT-B/16",
               tc="(c) where the association comes from",
               rho=r"Spearman $\rho$ (95\% CI)",
               s1="all 17", s2="without MAE", s3="without MAE, CLIP",
               s4="9 pretrainings", s5="9 architectures"),
    "pt": dict(med=["razão de\nparticipação", "raio do\ncampo", "fluxo\nresidual"],
               frac="fatia do domínio na variância",
               conj=["6 extratores", "12 extratores", "17 extratores"],
               fam=["sup.", "auto-s.", "ling."],
               fluxo=r"fluxo residual $\|F(x)\|/\|x\|$",
               faixa="faixa dos 4 supervisionados",
               ta="(a) resposta ao domínio", tb="(b) nove ViT-B/16",
               tc="(c) de onde vem a associação",
               rho=r"$\rho$ de Spearman (IC 95\%)",
               s1="todos os 17", s2="sem o MAE", s3="sem MAE, CLIP",
               s4="9 pré-treinos", s5="9 arquiteturas"),
}


def estilo():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 6.4,
        "axes.edgecolor": "#b8b7b2", "axes.linewidth": .5,
        "axes.labelcolor": TINTA, "axes.titlesize": 6.6,
        "xtick.color": TINTA2, "ytick.color": TINTA2,
        "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "grid.color": "#e6e5e1", "grid.linewidth": .4, "hatch.linewidth": .45,
        "legend.frameon": False, "legend.fontsize": 5.4,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": .015})


def recessivo(ax, eixo="y"):
    ax.grid(True, axis=eixo, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def dec(M):
    L = np.log(M); g = L.mean()
    vm = L.mean(1).var(ddof=0); vd = L.mean(0).var(ddof=0)
    vi = ((L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + g) ** 2).mean()
    t = vm + vd + vi
    return np.array([vm, vd, vi]) / t


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0)


def main(idioma="en"):
    T = TEXTO[idioma]; estilo()
    r = Path(__file__).parent.parent.parent / "resultados"
    t = json.loads((r / "testes.json").read_text())
    sd = json.loads((r / "sonda.json").read_text())
    geo = {(x["backbone"], x["banco"]): x
           for bl in ("ic", "arquiteturas", "leva2", "leva3") for x in t.get(bl, [])
           if "banco" in x}
    fl = {}
    for bl in ("fluxo2", "fluxo3"):
        for x in t.get(bl, []):
            for mec, v in x["mec"].items():
                fl.setdefault(x["backbone"], {})[x["banco"]] = v

    fig, axs = plt.subplots(1, 3, figsize=(POL, POL * 0.36),
                        gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    fig.subplots_adjust(wspace=.52)

    # --- (a) resposta ao dominio, em tres conjuntos -----------------------
    ax = axs[0]
    cps = [lambda e, b: geo[(e, b)]["posto"][0],
           lambda e, b: geo[(e, b)]["raio_px"][0]]
    conj = [SEIS, DOZE, TODOS]
    y = np.arange(3)[::-1]
    marcas, cores = ["o", "s", "D"], [CINZA, LARANJA, AZUL]
    for k, f in enumerate(cps):
        for j, es in enumerate(conj):
            d = dec(np.array([[f(e, b) for b in BANCOS] for e in es]))[1]
            ax.scatter([d], [y[k]], marker=marcas[j], s=16, c=cores[j],
                       linewidths=0, zorder=4,
                       label=T["conj"][j] if k == 0 else None)
    dfl = dec(np.array([[fl[e][b][0] for b in BANCOS] for e in OBJ if e in fl]))[1]
    ax.scatter([dfl], [y[2]], marker="D", s=16, c=AZUL, linewidths=0, zorder=4)
    for k in range(3):
        ax.plot([0, .72], [y[k], y[k]], color="#e6e5e1", lw=.6, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(T["med"], fontsize=5.8, linespacing=1.1)
    ax.set_xlim(-.03, .75); ax.set_xticks([0, .25, .5])
    ax.set_xticklabels(["0", "¼", "½"])
    ax.set_ylim(-1.35, 2.5)
    ax.set_xlabel(T["frac"], fontsize=5.9, color=TINTA2, labelpad=1)
    ax.set_title(T["ta"], color=TINTA, pad=3)
    recessivo(ax, "x")
    ax.legend(loc="lower center", ncol=1, handlelength=.7, labelspacing=.22,
              fontsize=5.3, borderpad=.15, handletextpad=.3)

    # --- (b) os nove ViT-B/16: as familias se sobrepoem -------------------
    ax = axs[1]
    fams = ["superv.", "auto-sup.", "ling."]
    sup = [np.mean([fl[e][b][0] for b in BANCOS]) for e in OBJ
           if e in fl and FAM[e] == "superv."]
    ax.axhspan(min(sup), max(sup), color=AZUL, alpha=.13, zorder=1,
               label=T["faixa"])
    rng = np.random.default_rng(0)
    for i, fm in enumerate(fams):
        es = [e for e in OBJ if e in fl and FAM[e] == fm]
        v = [np.mean([fl[e][b][0] for b in BANCOS]) for e in es]
        ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, s=16,
                   marker=["o", "s", "^"][i], c=[AZUL, LARANJA, VERDE][i],
                   linewidths=0, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels(T["fam"], fontsize=5.8)
    ax.set_xlim(-.55, 2.55); ax.set_ylim(.52, 1.02)
    ax.set_ylabel(T["fluxo"], fontsize=6.0)
    ax.set_title(T["tb"], color=TINTA, pad=3)
    recessivo(ax)
    ax.legend(loc="upper left", handlelength=.9, fontsize=5.3, borderpad=.15)

    # --- (c) sensibilidade: de onde vem a associacao ----------------------
    # Um grafico de floresta em vez do diagrama de dispersao: a pergunta que
    # importa deixou de ser "existe associacao" e passou a ser "ela sobrevive
    # a retirada do checkpoint colapsado". O IC vem de reamostrar EXTRATORES,
    # nao pares, porque o extrator e a unidade de agrupamento.
    ax = axs[2]
    SUB = [(TODOS, T["s1"], True), ([e for e in TODOS if e != "vit_b16_mae"],
            T["s2"], False),
           ([e for e in TODOS if e not in ("vit_b16_mae", "vit_b16_clip",
                                           "vit_b16_clip2b")], T["s3"], False),
           (OBJ, T["s4"], True), (ARQ, T["s5"], False)]
    # O intervalo e JACKKNIFE deixando um extrator de fora, na escala de
    # Fisher -- nao mais o percentil com reposicao. O percentil terminava em
    # +0,01 na linha "sem o MAE", em que a permutacao da p = 0,075: um
    # intervalo que exclui zero ao lado de um teste que nao rejeita. Nao era
    # vies de duplicata (a media das reamostras coincide com o rho observado);
    # era uma ponta nao resolvivel nessa precisao com k = 16. O jackknife e o
    # companheiro natural de uma analise sobre RETIRAR extratores e devolve
    # coerencia nas seis linhas de uma vez.
    for i, (es, rot, com_mae) in enumerate(SUB):
        G = [z([geo[(e, b)]["posto"][0] for e in es]) for b in BANCOS]
        P = [z([sd[f"{b}|{e}"]["valor"] for e in es]) for b in BANCOS]
        rho = spearmanr(np.concatenate(G), np.concatenate(P)).statistic
        k = len(es)

        def _rho(idx, G=G, P=P):
            return spearmanr(np.concatenate([g[idx] for g in G]),
                             np.concatenate([q[idx] for q in P])).statistic

        fz = lambda r: np.arctanh(np.clip(r, -0.999999, 0.999999))   # noqa: E731
        tudo = np.arange(k)
        z_i = np.array([fz(_rho(np.delete(tudo, j))) for j in range(k)])
        se = np.sqrt((k - 1) / k * ((z_i - z_i.mean()) ** 2).sum())
        y = len(SUB) - 1 - i
        lo = np.tanh(fz(rho) - 1.96 * se)
        hi = np.tanh(fz(rho) + 1.96 * se)
        cor = LARANJA if com_mae else CINZA
        ax.plot([lo, hi], [y, y], color=cor, lw=1.1, zorder=3,
                solid_capstyle="butt")
        ax.scatter([rho], [y], s=15, marker="o" if com_mae else "s", c=cor,
                   linewidths=0, zorder=4)
        ax.text(1.02, y, rot, fontsize=5.5, color=TINTA, va="center", ha="left")
    ax.axvline(0, color="#b8b7b2", lw=.6, zorder=1)
    ax.set_yticks([]); ax.set_ylim(-.7, len(SUB) - .3)
    ax.set_xlim(-.45, 1.0); ax.set_xticks([0, .5])
    ax.set_xticklabels(["0", "0,5" if idioma == "pt" else "0.5"])
    ax.set_xlabel(T["rho"], fontsize=5.9, color=TINTA2, labelpad=1)
    ax.set_title(T["tc"], color=TINTA, pad=3)
    ax.grid(True, axis="x", zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    nome = "fig2_resultados" + ("" if idioma == "en" else f"_{idioma}") + ".pdf"
    w, h = salva(fig, Path(__file__).parent.parent / "figuras" / nome)
    print(f"-> figuras/{nome}  {w*2.54:.1f} x {h*2.54:.1f} cm")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--idioma", default="en", choices=["en", "pt"])
    main(a.parse_args().idioma)
