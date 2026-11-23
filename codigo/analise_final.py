#!/usr/bin/env python
"""Analise final: estatistica corrigida e o teste receita-contra-objetivo.

DUAS CORRECOES em relacao a versao anterior.

1. PERMUTACAO NO NIVEL DO EXTRATOR. Antes os rotulos eram embaralhados
   independentemente em cada dominio, o que destroi a estrutura de medidas
   repetidas: o mesmo extrator aparece nos quatro dominios e seus desvios sao
   correlacionados entre eles. Esse nulo e "aleatorio demais" e torna o teste
   anticonservador. Aqui UMA permutacao de extratores e aplicada aos quatro
   dominios, preservando o perfil de cada um. O preco e poder: com k
   extratores ha k! permutacoes, e o menor p alcancavel e ~1/k!.

2. A medida chama-se RAZAO DE PARTICIPACAO. A expressao (sum L)^2 / sum L^2 e
   a razao de participacao; o "posto efetivo" de Roy e Vetterli (2007) e
   exp(-sum p_i log p_i), que e outra coisa. Calculamos as duas, para que a
   escolha de nome deixe de ser uma suposicao.

E O TESTE NOVO: se "objetivo" fosse so um rotulo para um pacote confundido
(objetivo + corpus + receita), quatro checkpoints SUPERVISIONADOS de receitas
e corpora diferentes deveriam se espalhar tanto quanto objetivos distintos.
Medimos isso.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

BANCOS = ["mamografia", "dtd", "pet", "sun397"]
ARQ = ["resnet18", "resnet50", "convnext_t", "efficientnet_b3", "swin_t",
       "maxvit_t", "maxvit_s", "vit_s16", "vit_b16"]
OBJ = ["vit_b16", "vit_b16_sup21k", "vit_b16_sup1k", "vit_b16_deit3",
       "vit_b16_dino", "vit_b16_mae", "vit_b16_clip", "vit_b16_clip2b",
       "vit_b16_siglip"]
FAMILIA = {"vit_b16": "supervisionado", "vit_b16_sup21k": "supervisionado",
           "vit_b16_sup1k": "supervisionado", "vit_b16_deit3": "supervisionado",
           "vit_b16_dino": "auto-supervis.", "vit_b16_mae": "auto-supervis.",
           "vit_b16_clip": "linguagem", "vit_b16_clip2b": "linguagem",
           "vit_b16_siglip": "linguagem"}
ROT = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "convnext_t": "ConvNeXt-T",
       "efficientnet_b3": "EfficientNet-B3", "swin_t": "Swin-T",
       "maxvit_t": "MaxViT-T", "maxvit_s": "MaxViT-S", "vit_s16": "ViT-S/16",
       "vit_b16": "ViT-B/16 sup. 21k$\\to$1k", "vit_b16_sup21k": "ViT-B/16 sup. 21k",
       "vit_b16_sup1k": "ViT-B/16 sup. 1k", "vit_b16_deit3": "ViT-B/16 DeiT3",
       "vit_b16_dino": "ViT-B/16 DINO", "vit_b16_mae": "ViT-B/16 MAE",
       "vit_b16_clip": "ViT-B/16 CLIP-OAI", "vit_b16_clip2b": "ViT-B/16 CLIP-2B",
       "vit_b16_siglip": "ViT-B/16 SigLIP"}
TODOS = ARQ + OBJ[1:]


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0)


def dec(M):
    L = np.log(M); g = L.mean()
    vm = L.mean(1).var(ddof=0); vd = L.mean(0).var(ddof=0)
    vi = ((L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + g) ** 2).mean()
    t = vm + vd + vi
    return np.array([vm, vd, vi]) / t


def perm_extrator(G, P, n=20000, semente=0):
    """UMA permutacao de extratores aplicada aos quatro dominios."""
    gx = np.concatenate(G)
    obs = spearmanr(gx, np.concatenate(P)).statistic
    rng = np.random.default_rng(semente)
    k = len(G[0])
    nulo = np.array([abs(spearmanr(gx, np.concatenate([p[q] for p in P])).statistic)
                     for q in (rng.permutation(k) for _ in range(n))])
    return obs, float((nulo >= abs(obs)).mean()), float(np.percentile(nulo, 95))


def main():
    r = Path("resultados")
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
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    faltam = [k for k in [(e, b) for e in TODOS for b in BANCOS] if k not in geo]
    p(f"# Números finais\n\nGrade geometria: {len(geo)} pares"
      f"{'' if not faltam else '  FALTAM: ' + str(faltam[:6])}")
    p(f"Sonda: {len(sd)} pares | fluxo: {len(fl)} extratores\n")

    # ---- 1. decomposicao -------------------------------------------------
    p("## 1. Fatia da variância em log (modelo × domínio)\n")
    p("| medida | conjunto | modelo | domínio | interação |")
    p("|---|---|---|---|---|")
    for rot, f, conjuntos in [
            ("razão de participação", lambda e, b: geo[(e, b)]["posto"][0],
             [("17 extratores", TODOS), ("9 arquiteturas", ARQ), ("9 objetivos", OBJ)]),
            ("raio do campo", lambda e, b: geo[(e, b)]["raio_px"][0],
             [("17 extratores", TODOS), ("9 arquiteturas", ARQ), ("9 objetivos", OBJ)])]:
        for nome, es in conjuntos:
            d = dec(np.array([[f(e, b) for b in BANCOS] for e in es]))
            p(f"| {rot} | {nome} | {d[0]:.1%} | {d[1]:.1%} | {d[2]:.1%} |")
    d = dec(np.array([[fl[e][b][0] for b in BANCOS] for e in OBJ if e in fl]))
    p(f"| fluxo residual | 9 objetivos | {d[0]:.1%} | {d[1]:.1%} | {d[2]:.1%} |\n")

    # ---- 2. receita contra objetivo --------------------------------------
    p("## 2. Receita ou objetivo? (os 9 ViT-B/16, arquitetura idêntica)\n")
    p("| checkpoint | família | fluxo residual | razão particip. (mamog.) |")
    p("|---|---|---|---|")
    for e in OBJ:
        if e not in fl:
            continue
        fm = np.mean([fl[e][b][0] for b in BANCOS])
        p(f"| {ROT[e]} | {FAMILIA[e]} | {fm:.3f} | "
          f"{geo[(e, 'mamografia')]['posto'][0]:.1f} |")
    sup = [e for e in OBJ if FAMILIA[e] == "supervisionado" and e in fl]
    outros = [e for e in OBJ if FAMILIA[e] != "supervisionado" and e in fl]
    fs = np.array([np.mean([fl[e][b][0] for b in BANCOS]) for e in sup])
    fo = np.array([np.mean([fl[e][b][0] for b in BANCOS]) for e in outros])
    p(f"\n- amplitude DENTRO dos 4 supervisionados: {fs.max()-fs.min():.3f} "
      f"({fs.min():.3f}–{fs.max():.3f})")
    p(f"- amplitude entre TODOS os 9 objetivos: "
      f"{max(fs.max(),fo.max())-min(fs.min(),fo.min()):.3f}")
    # Decomposicao por familia nas DUAS ponderacoes. O parecer pediu a formula
    # da segunda, que antes so existia como numero digitado no artigo.
    mm = np.array([np.mean([np.log(fl[e][b][0]) for b in BANCOS])
                   for e in OBJ if e in fl])
    fam = np.array([FAMILIA[e] for e in OBJ if e in fl])
    mf = {f: mm[fam == f].mean() for f in set(fam)}
    mfe = np.array([mf[f] for f in fam])
    ve, vd = ((mfe - mm.mean()) ** 2).mean(), ((mm - mfe) ** 2).mean()
    p(f"- decomposicao, peso por CHECKPOINT: entre {100*ve/(ve+vd):.1f} %, "
      f"dentro {100*vd/(ve+vd):.1f} %")
    fs = sorted(set(fam)); mw = np.mean([mf[f] for f in fs])
    ve2 = np.mean([(mf[f] - mw) ** 2 for f in fs])
    vd2 = np.mean([np.mean((mm[fam == f] - mf[f]) ** 2) for f in fs])
    p(f"- decomposicao, peso por FAMILIA (1/F cada): entre "
      f"{100*ve2/(ve2+vd2):.1f} %, dentro {100*vd2/(ve2+vd2):.1f} %")
    if "vit_b16_clip" in fl and "vit_b16_clip2b" in fl:
        a = np.mean([fl["vit_b16_clip"][b][0] for b in BANCOS])
        b_ = np.mean([fl["vit_b16_clip2b"][b][0] for b in BANCOS])
        p(f"- MESMO objetivo, corpora diferentes (CLIP OpenAI × LAION-2B): "
          f"{a:.3f} × {b_:.3f}, diferença {abs(a-b_):.3f}")
    p("")

    # ---- 3. correlacoes ---------------------------------------------------
    p("## 3. Geometria × sonda — permutação NO NÍVEL DO EXTRATOR\n")
    p("| medida | conjunto | k | rho | p | \\|rho\\| detectável |")
    p("|---|---|---|---|---|---|")
    medidas = [("razão de participação", lambda e, b: geo[(e, b)]["posto"][0]),
               ("raio do campo", lambda e, b: geo[(e, b)]["raio_px"][0]),
               ("largura (controle)", lambda e, b: geo[(e, b)]["canais"])]
    for rot, f in medidas:
        for nome, es in [("17 extratores", TODOS), ("9 arquiteturas", ARQ),
                         ("9 objetivos", OBJ)]:
            es = [e for e in es if all(f"{b}|{e}" in sd for b in BANCOS)]
            if len(es) < 4:
                continue
            vals = [[f(e, b) for e in es] for b in BANCOS]
            # um preditor constante (ex.: a largura nos 9 ViT-B/16, todos com
            # 768 canais) tem variancia zero: o z() dividiria por zero e o
            # teste devolveria nan com p=0. Nao ha o que correlacionar.
            if any(np.std(v) == 0 for v in vals):
                p(f"| {rot} | {nome} | {len(es)} | — | — | constante |")
                continue
            G = [z(v) for v in vals]
            P = [z([sd[f"{b}|{e}"]["valor"] for e in es]) for b in BANCOS]
            rho, pv, det = perm_extrator(G, P)
            p(f"| {rot} | {nome} | {len(es)} | {rho:+.3f} | {pv:.3f} | {det:.2f} |")
    p("")

    # ---- 4. linhas da tabela do artigo ------------------------------------
    p("## 4. Linhas LaTeX da Tabela 1\n```")
    NAT = BANCOS[1:]
    for e in TODOS:
        if (e, "mamografia") not in geo or f"mamografia|{e}" not in sd:
            continue
        pm = geo[(e, "mamografia")]["posto"][0]
        pn = np.mean([geo[(e, b)]["posto"][0] for b in NAT])
        rm = geo[(e, "mamografia")]["raio_px"][0]
        rn = np.mean([geo[(e, b)]["raio_px"][0] for b in NAT])
        sm = sd[f"mamografia|{e}"]["valor"]
        sn = np.mean([sd[f"{b}|{e}"]["valor"] for b in NAT])
        p(f"{ROT[e]} & {pm:.1f} & {pn:.1f} & {rm:.0f} & {rn:.0f} "
          f"& {sm:.3f} & {sn:.3f} \\\\")
    p("```\n")

    # ---- 5. controle RGB ---------------------------------------------------
    if "rgb" in t:
        p("## 5. Controle RGB (sem conversão para cinza)\n")
        rgb = {(x["backbone"], x["banco"]): x["posto"][0] for x in t["rgb"]}
        p("| checkpoint | cinza (mamog.) | RGB (mamog.) | cinza (nat.) | RGB (nat.) |")
        p("|---|---|---|---|---|")
        for e in OBJ:
            if (e, "mamografia") not in rgb:
                continue
            p(f"| {ROT[e]} | {geo[(e,'mamografia')]['posto'][0]:.1f} | "
              f"{rgb[(e,'mamografia')]:.1f} | "
              f"{np.mean([geo[(e,b)]['posto'][0] for b in NAT]):.1f} | "
              f"{np.mean([rgb[(e,b)] for b in NAT]):.1f} |")
        p("")

    Path("resultados/NUMEROS.md").write_text("\n".join(L) + "\n")
    print("\n-> resultados/NUMEROS.md", flush=True)


if __name__ == "__main__":
    main()
