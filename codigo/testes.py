#!/usr/bin/env python
"""Os cinco controles que separam afirmacao de observacao.

As medidas anteriores (posto efetivo, campo receptivo, fluxo residual,
embaralhamento) foram tomadas UMA vez, em tres arquiteturas que diferem em
quatro coisas ao mesmo tempo: mecanismo, corpus de pre-treino, resolucao
nativa e numero de parametros. Cada teste aqui ataca um desses problemas.

  ic            Intervalo por reamostragem de imagens. Sem piso de ruido,
                "1,1x contra 1,5x" e observacao, nao afirmacao -- que e
                exatamente a licao do estudo de mamografia.
  resolucao     O ViT-B/16 foi pre-treinado a 224 px e nos o rodamos a 512,
                com interpolacao das codificacoes de posicao. Rodar os tres
                tamanhos separa "atencao global" de "extrapolacao de posicao".
  fundo         Mamografia tem muito fundo vazio. Aplicar as MASCARAS REAIS
                das mamografias sobre imagem natural separa "o dominio e
                estreito" de "nao ha para onde olhar".
  arquiteturas  ConvNeXt-T (convolucao, receita moderna), Swin-T (hierarquico
                com atencao so local) e ViT-S/16 (atencao global, IN-1k).
                Levam n de 3 para 6 e quebram o laco mecanismo-receita.
  ablacao       Trocar a atencao em grade do MaxViT por identidade. E o unico
                teste causal: nada mais muda.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import multitarefa as MT
from representacao import transformacao, posto_efetivo
from campo_e_embaralho import carrega, Embaralhador, cka_gram, imagens


class Deslocado:
    """Embaralha numa grade deslocada de `d` pixels.

    Controle de ALINHAMENTO. Com retalhos de 16 px numa imagem de 512, a grade
    de embaralhamento coincide com a grade de fichas de um ViT-B/16 -- cada
    retalho e exatamente uma ficha. Deslocar meia ficha faz todo retalho
    atravessar duas fichas. Se a sensibilidade ao arranjo despencar com o
    deslocamento, a medida estava lendo a coincidencia das grades, nao a
    dependencia do arranjo.

    O rolamento e desfeito depois, entao o conteudo volta ao lugar; so a
    particao muda. A emenda que o rolamento cria nas bordas existe nos dois
    modos (global e local) e cancela na diferenca.
    """

    def __init__(self, emb, d):
        self.emb, self.d = emb, d

    def reinicia(self):
        if hasattr(self.emb, "reinicia"):
            self.emb.reinicia()

    def __call__(self, a):
        d = self.d
        return np.roll(self.emb(np.roll(a, (-d, -d), (0, 1))), (d, d), (0, 1))
from fluxo_residual import canais_por_ultimo

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 400
GRADES = (4, 8, 16, 32)


# ---------------------------------------------------------------------------
# Reamostragem
# ---------------------------------------------------------------------------

def ic_sub(amostras, estat, m=None, n_boot=N_BOOT, semente=0):
    """Intervalo por SUBAMOSTRAGEM sem reposicao, para estatisticas espectrais.

    O posto efetivo nao pode ser reamostrado com reposicao: uma imagem
    duplicada nao acrescenta direcao nenhuma a covariancia, entao toda
    reamostra tem posto menor que a amostra original e o intervalo nem sequer
    contem o ponto. Subamostrar sem reposicao evita a duplicacao, ao preco de
    medir em m < n -- e o posto efetivo CRESCE com n. Por isso o ponto tambem
    e relatado em m: ponto e intervalo ficam na mesma escala, e todas as
    comparacoes entre arquiteturas e bancos usam o mesmo m.
    """
    rng = np.random.default_rng(semente)
    n = len(amostras)
    m = m or max(2, n // 2)
    v = np.array([estat(rng.permutation(n)[:m]) for _ in range(n_boot)])
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(np.percentile(v, 2.5)),
            float(np.percentile(v, 97.5)), m)


def ic(amostras, estat, n_boot=N_BOOT, semente=0):
    """(ponto, inferior, superior) por reamostragem de IMAGENS com reposicao.

    A unidade reamostrada e a imagem porque e ela que varia entre execucoes
    desta medida -- nao ha semente de treino nem particao, os extratores estao
    congelados. O percentil e o de 95%.
    """
    rng = np.random.default_rng(semente)
    n = len(amostras)
    v = [estat(rng.integers(0, n, n)) for _ in range(n_boot)]
    v = np.array([x for x in v if np.isfinite(x)])
    return (float(estat(np.arange(n))), float(np.percentile(v, 2.5)),
            float(np.percentile(v, 97.5)))


# ---------------------------------------------------------------------------
# Primitivas por imagem (guardam o suficiente para reamostrar depois)
# ---------------------------------------------------------------------------

#: Quando preenchido, _modelo devolve ESTE par (modelo, transformacao) em vez
#: de montar pelo registro. E assim que a ablacao entra: o resto do codigo nao
#: precisa saber que o modelo foi modificado.
SUBSTITUTO = None


def reinicia(t):
    """Zera o contador de uma transformacao com estado, antes de cada passagem.

    AplicaMascara cicla pelas mascaras na ordem em que as imagens chegam. Se o
    contador nao for zerado entre passagens, a MESMA imagem recebe mascaras
    diferentes na passagem base e na embaralhada, e o delta mediria a troca de
    mascara em vez do embaralhamento.
    """
    if hasattr(t, "reinicia"):
        t.reinicia()
    return t


def _modelo(alias):
    if SUBSTITUTO is not None:
        return SUBSTITUTO
    m, _, norm = MT.monta_backbone(alias, pre=True, treinavel=False)
    return m.to(DEV).eval(), transformacao(norm)


def pecas_mapa(alias, caminhos, tam, bs=8, transformar=None, modelo=None,
               tf=None, cinza=True):
    """Por imagem: media espacial (C,), variancia entre posicoes (C,).

    Guardar esses dois vetores em vez do mapa inteiro deixa o posto efetivo e a
    dispersao espacial serem recalculados mil vezes de graca, e reproduz as
    duas medidas exatamente -- nenhuma aproximacao.
    """
    prop = modelo is None and SUBSTITUTO is None
    if modelo is None:
        modelo, tf = _modelo(alias)
    reinicia(transformar)
    mp, vp = [], []
    with torch.no_grad():
        for i in range(0, len(caminhos), bs):
            x = carrega(caminhos[i:i + bs], tf, tam, transformar, cinza).to(DEV)
            o = modelo(x).float()
            v = o.flatten(2)                      # (N, C, HW)
            mp.append(v.mean(2).cpu()); vp.append(v.var(2).cpu())
    if prop:
        del modelo; torch.cuda.empty_cache()
    return torch.cat(mp), torch.cat(vp)


def campo_final(alias, caminhos, tam, bs=2, transformar=None):
    """Mapa de gradiente da unidade central da SAIDA, por imagem, normalizado.

    So a saida final, nao bloco a bloco: e o numero que entra nas tabelas, e
    guardar um mapa por imagem permite reamostrar o estimador exato (perfil da
    media dos mapas), em vez da media dos perfis, que nao e a mesma coisa.
    """
    modelo, tf = _modelo(alias)
    reinicia(transformar)
    saida = []
    for i in range(0, len(caminhos), bs):
        x = carrega(caminhos[i:i + bs], tf, tam, transformar).to(DEV).requires_grad_(True)
        o = modelo(x)
        h, w = o.shape[2], o.shape[3]
        g, = torch.autograd.grad(o[:, :, h // 2, w // 2].sum(), x)
        a = g.abs().sum(1).double()
        a = a / a.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        saida.append(a.float().cpu())
        del x, g, o
    torch.cuda.empty_cache()
    if SUBSTITUTO is None:
        del modelo
    return torch.cat(saida).numpy()


def perfil_raio(G, q=0.5):
    """Raio que acumula a fracao q da massa de um mapa (H,W) ja normalizado."""
    h, w = G.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2).ravel()
    o = np.argsort(r)
    return float(r[o][np.searchsorted(np.cumsum(G.ravel()[o]), q)])


class Composto:
    """Aplica a preparacao da imagem e so depois o embaralhamento."""

    def __init__(self, antes, depois):
        self.antes, self.depois = antes, depois

    def reinicia(self):
        reinicia(self.antes); reinicia(self.depois)

    def __call__(self, a):
        return self.depois(self.antes(a) if self.antes is not None else a)


def deltas_embaralho(alias, caminhos, tam, bs=8, grades=GRADES, transformar=None):
    """Delta POR IMAGEM para cada (grade, modo). Permite reamostrar o delta."""
    modelo, tf = _modelo(alias)
    base, _ = pecas_mapa(alias, caminhos, tam, bs, transformar, modelo, tf)
    nb = base.norm(dim=1).clamp_min(1e-8)
    fora = {}
    for g in grades:
        for modo in ("global", "local"):
            emb = Embaralhador(g, modo, semente=g * 10 + (modo == "local"))
            p, _ = pecas_mapa(alias, caminhos, tam, bs,
                              Composto(transformar, emb), modelo, tf)
            fora[(g, modo)] = ((p - base).norm(dim=1) / nb).numpy()
    if SUBSTITUTO is None:
        del modelo
    torch.cuda.empty_cache()
    return fora, base


# ---------------------------------------------------------------------------
# O perfil completo de um (arquitetura, conjunto de imagens, resolucao)
# ---------------------------------------------------------------------------

def _disp(mp, vp, idx):
    entre_pos = vp[idx].mean(0).mean()
    entre_img = mp[idx].var(0).mean()
    return float(entre_pos / (entre_pos + entre_img))


def perfil(alias, caminhos, tam=512, rotulo="", transformar=None,
           n_campo=32, n_emb=96, modelo_ablado=None):
    mp, vp = pecas_mapa(alias, caminhos, tam, transformar=transformar)
    *posto, m_posto = ic_sub(mp, lambda i: posto_efetivo(mp[i]))
    posto_n = float(posto_efetivo(mp))        # em n cheio, para continuidade
    disp = ic(mp, lambda i: _disp(mp, vp, i))

    G = campo_final(alias, caminhos[:n_campo], tam, transformar=transformar)
    # o estimador e o perfil da MEDIA dos mapas; reamostrar os mapas o reproduz
    raio = ic(G, lambda i: perfil_raio(G[i].mean(0)))

    d, _ = deltas_embaralho(alias, caminhos[:n_emb], tam, transformar=transformar)
    arranjo = {}
    for g in GRADES:
        dg, dl = d[(g, "global")], d[(g, "local")]
        arranjo[g] = ic(dg, lambda i: float(dg[i].mean() - dl[i].mean()))

    r = dict(backbone=alias, rotulo=rotulo, tam=tam, canais=int(mp.shape[1]),
             n_imagens=len(caminhos), m_posto=m_posto, posto_n_cheio=posto_n,
             posto=posto, posto_frac=tuple(x / mp.shape[1] for x in posto),
             dispersao=disp, raio_px=raio,
             raio_frac=tuple(x / tam for x in raio),
             arranjo={str(g): arranjo[g] for g in GRADES})
    print(f"   {alias:11} {rotulo:22} posto {posto[0]:5.1f} [{posto[1]:.1f};{posto[2]:.1f}]"
          f" (n {posto_n:5.1f})"
          f"  raio {raio[0]:5.1f} [{raio[1]:.1f};{raio[2]:.1f}] px"
          f"  arranjo16 {arranjo[32][0]:+.3f} [{arranjo[32][1]:+.3f};{arranjo[32][2]:+.3f}]",
          flush=True)
    return r


def fluxo_ic(alias, caminhos, tam=512):
    """Fracao do fluxo que cada MECANISMO reescreve, com intervalo.

    Os blocos de um mesmo mecanismo sao promediados dentro de cada imagem
    antes de reamostrar, porque a unidade de reamostragem e a imagem, nao o
    bloco: blocos da mesma imagem nao sao independentes.
    """
    from fluxo_residual import mede
    _, det = mede(alias, caminhos, tam, detalhe=True)
    por_mec = {}
    for nome, (mec, v) in det.items():
        por_mec.setdefault(mec, []).append(v)
    fora = {}
    for mec, vs in por_mec.items():
        M = np.stack(vs)                       # (blocos, imagens)
        m = M.mean(0)                          # media dos blocos, por imagem
        fora[mec] = ic(m, lambda i: float(m[i].mean()))
        print(f"   {alias:11} {mec:24} ramo {fora[mec][0]:.3f} "
              f"[{fora[mec][1]:.3f};{fora[mec][2]:.3f}]", flush=True)
    return fora


# ---------------------------------------------------------------------------
# Controle de fundo
# ---------------------------------------------------------------------------

def mascaras(caminhos, tam=512, limiar=0.06):
    """Mascaras binarias de primeiro plano das mamografias.

    Limiar simples sobre a intensidade: numa FFDM o fundo e ar, praticamente
    zero, e a mama e tudo acima. Nao precisa ser segmentacao boa -- precisa ser
    a MESMA geometria de vazio que o modelo ve na mamografia.
    """
    fora = []
    for q in caminhos:
        a = np.asarray(Image.open(q).convert("L").resize((tam, tam),
                       Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        fora.append(a > limiar)
    return fora


class AplicaMascara:
    """Zera a imagem fora da mascara, ciclando pelo conjunto de mascaras."""

    def __init__(self, ms, semente=0):
        self.ms = ms
        self.ordem = np.random.default_rng(semente).permutation(len(ms))
        self.k = 0

    def reinicia(self):
        self.k = 0

    def __call__(self, a):
        m = self.ms[self.ordem[self.k % len(self.ms)]]
        self.k += 1
        return a * m


# ---------------------------------------------------------------------------

def ablada_grade(alias="maxvit_t"):
    """MaxViT com attn_grid trocado por identidade, tudo o mais igual."""
    m, _, norm = MT.monta_backbone(alias, pre=True, treinavel=False)
    n = 0
    for est in m.m.stages:
        for b in est.blocks:
            b.attn_grid = torch.nn.Identity(); n += 1
    print(f"   (atencao em grade removida de {n} blocos)", flush=True)
    return m.to(DEV).eval(), transformacao(norm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testes", default="ic,resolucao,fundo,arquiteturas,ablacao")
    p.add_argument("--amostras", default="data/amostras")
    p.add_argument("--out", default="resultados")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    saida = Path(a.out); saida.mkdir(parents=True, exist_ok=True)
    quais = [t for t in a.testes.split(",") if t]
    arq = saida / "testes.json"
    tudo = json.loads(arq.read_text()) if arq.exists() else {}

    def grava(chave, linhas):
        tudo[chave] = linhas
        arq.write_text(json.dumps(tudo, indent=1, default=float))
        print(f"-> {chave} gravado", flush=True)

    bancos = ["mamografia", "dtd", "pet", "sun397"]
    qs = {b: imagens(Path(a.amostras) / b, a.n) for b in bancos}

    if "ic" in quais:
        print("\n== 1. INTERVALO POR REAMOSTRAGEM (3 arquiteturas x 4 bancos) ==", flush=True)
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb in ("resnet50", "maxvit_t", "vit_b16"):
                r = perfil(bb, qs[b], 512, b); r["banco"] = b; L.append(r)
            grava("ic", L)

    if "resolucao" in quais:
        print("\n== 2. RESOLUCAO: o ViT a 224, 384 e 512 ==", flush=True)
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb, tams in (("vit_b16", (224, 384, 512)), ("vit_s16", (224, 512))):
                for t in tams:
                    r = perfil(bb, qs[b], t, f"{b} @{t}"); r["banco"] = b; L.append(r)
            grava("resolucao", L)

    if "fundo" in quais:
        print("\n== 3. CONTROLE DE FUNDO ==", flush=True)
        ms = mascaras(qs["mamografia"][:a.n])
        frac = float(np.mean([m.mean() for m in ms]))
        print(f"   fracao media de primeiro plano na mamografia: {frac:.3f}", flush=True)
        L = [{"fracao_primeiro_plano_mamografia": frac}]
        for b in ("dtd", "pet", "sun397"):
            print(f"  -- {b} com mascara de mamografia --", flush=True)
            for bb in ("resnet50", "maxvit_t", "vit_b16"):
                r = perfil(bb, qs[b], 512, f"{b} mascarado",
                           transformar=AplicaMascara(ms))
                r["banco"] = b; L.append(r)
            grava("fundo", L)

    if "arquiteturas" in quais:
        print("\n== 4. TRES ARQUITETURAS DE CONTROLE ==", flush=True)
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb in ("convnext_t", "swin_t", "vit_s16"):
                r = perfil(bb, qs[b], 512, b); r["banco"] = b; L.append(r)
            grava("arquiteturas", L)

    if "fluxo" in quais:
        print("\n== 6. FLUXO RESIDUAL COM INTERVALO ==", flush=True)
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb in ("resnet50", "maxvit_t", "vit_b16"):
                L.append(dict(banco=b, backbone=bb,
                              mec=fluxo_ic(bb, qs[b][:64])))
            grava("fluxo", L)

    if "alinhamento" in quais:
        print("\n== 7. CONTROLE DE ALINHAMENTO DA GRADE ==", flush=True)
        L = []
        for b in ("mamografia", "pet"):
            print(f"  -- {b} --", flush=True)
            for bb in ("resnet50", "maxvit_t", "vit_b16", "convnext_t",
                       "swin_t", "vit_s16"):
                mod, tf = _modelo(bb)
                base, _ = pecas_mapa(bb, qs[b][:96], 512, 8, None, mod, tf)
                nb = base.norm(dim=1).clamp_min(1e-8)
                linha = dict(banco=b, backbone=bb)
                for d in (0, 2, 8):
                    dd = {}
                    for modo in ("global", "local"):
                        e = Embaralhador(32, modo, semente=320 + (modo == "local"))
                        t = e if d == 0 else Deslocado(e, d)
                        pm, _ = pecas_mapa(bb, qs[b][:96], 512, 8, t, mod, tf)
                        dd[modo] = ((pm - base).norm(dim=1) / nb).numpy()
                    g, l = dd["global"], dd["local"]
                    linha[f"desloc{d}"] = ic(g, lambda i: float(g[i].mean() - l[i].mean()))
                    print(f"   {bb:11} {b:11} desloc {d:2d} px  "
                          f"arranjo {linha[f'desloc{d}'][0]:+.3f} "
                          f"[{linha[f'desloc{d}'][1]:+.3f};{linha[f'desloc{d}'][2]:+.3f}]",
                          flush=True)
                del mod; torch.cuda.empty_cache()
                L.append(linha)
            grava("alinhamento", L)

    if "leva2" in quais:
        print("\n== 8. SEGUNDA LEVA DE EXTRATORES ==", flush=True)
        novos = ["vit_b16_dino", "vit_b16_mae", "vit_b16_clip", "maxvit_s",
                 "efficientnet_b3", "resnet18"]
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb in novos:
                r = perfil(bb, qs[b], 512, b); r["banco"] = b; L.append(r)
            grava("leva2", L)

    if "fluxo2" in quais:
        # o fluxo residual seria "da arquitetura" -- mas os quatro ViT-B/16 tem
        # a MESMA arquitetura e objetivos de pre-treino diferentes. Se a fracao
        # reescrita mudar entre eles, o fator nao e a arquitetura e sim a
        # arquitetura MAIS o objetivo, e a afirmacao do artigo precisa mudar.
        print("\n== 9. FLUXO RESIDUAL POR OBJETIVO DE PRE-TREINO ==", flush=True)
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb in ("vit_b16", "vit_b16_dino", "vit_b16_mae", "vit_b16_clip"):
                L.append(dict(banco=b, backbone=bb,
                              mec=fluxo_ic(bb, qs[b][:64])))
            grava("fluxo2", L)

    if "leva3" in quais:
        print("\n== 10. NOVE OBJETIVOS SOBRE O MESMO ViT-B/16 ==", flush=True)
        novos = ["vit_b16_sup21k", "vit_b16_sup1k", "vit_b16_deit3",
                 "vit_b16_clip2b", "vit_b16_siglip"]
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb in novos:
                r = perfil(bb, qs[b], 512, b); r["banco"] = b; L.append(r)
            grava("leva3", L)

    if "fluxo3" in quais:
        print("\n== 11. FLUXO RESIDUAL NOS NOVE OBJETIVOS ==", flush=True)
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb in ("vit_b16", "vit_b16_sup21k", "vit_b16_sup1k",
                       "vit_b16_deit3", "vit_b16_dino", "vit_b16_mae",
                       "vit_b16_clip", "vit_b16_clip2b", "vit_b16_siglip"):
                L.append(dict(banco=b, backbone=bb, mec=fluxo_ic(bb, qs[b][:64])))
            grava("fluxo3", L)

    if "rgb" in quais:
        # controle de cor: so posto efetivo, nos nove ViT-B/16, sem converter
        print("\n== 12. CONTROLE RGB (sem conversao para cinza) ==", flush=True)
        L = []
        for b in bancos:
            print(f"  -- {b} --", flush=True)
            for bb in ("vit_b16", "vit_b16_sup21k", "vit_b16_sup1k",
                       "vit_b16_deit3", "vit_b16_dino", "vit_b16_mae",
                       "vit_b16_clip", "vit_b16_clip2b", "vit_b16_siglip"):
                mp, _ = pecas_mapa(bb, qs[b], 512, cinza=False)
                *po, m = ic_sub(mp, lambda i: posto_efetivo(mp[i]))
                L.append(dict(banco=b, backbone=bb, posto=po, m_posto=m))
                print(f"   {bb:16} {b:11} posto RGB {po[0]:5.1f} "
                      f"[{po[1]:.1f};{po[2]:.1f}]", flush=True)
            grava("rgb", L)

    if "ablacao" in quais:
        print("\n== 5. ABLACAO DA ATENCAO EM GRADE DO MAXVIT ==", flush=True)
        # NAO usar "import testes as EU; EU.SUBSTITUTO = ...": rodando como
        # __main__, esse import cria um SEGUNDO objeto de modulo, e a
        # atribuicao nao e vista por _modelo, que le o global deste. O
        # sintoma e silencioso -- a ablacao simplesmente nao acontece e os
        # numeros saem identicos aos do modelo intacto.
        global SUBSTITUTO
        L = []
        SUBSTITUTO = ablada_grade()
        try:
            for b in bancos:
                print(f"  -- {b} --", flush=True)
                r = perfil("maxvit_t", qs[b], 512, f"{b} sem grade")
                r["banco"] = b; r["variante"] = "sem_grade"; L.append(r)
                grava("ablacao", L)
        finally:
            SUBSTITUTO = None

    print("\nconcluido", flush=True)


if __name__ == "__main__":
    main()
