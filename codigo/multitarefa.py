#!/usr/bin/env python3
"""Classificacao da imagem + localizacao da lesao, no mesmo modelo.

    imagem -> backbone -+-> cabeca de classe  -> benigno / suspeito
                        +-> cabeca de caixa   -> (cx, cy, w, h)

Existe porque o dev_harness.py nao serve a esta pergunta: la o backbone e
congelado e as features sao extraidas uma vez, o que pressupoe que o backbone
nao muda. A ablacao congelado x ajustado precisa que ele possa receber
gradiente, e a caixa precisa do mapa espacial, que o pooling global destroi.

O que este modulo NAO muda em relacao ao dev_harness: agrupamento por
paciente, epoca e limiar escolhidos so na validacao, teste avaliado uma vez.

REGRA DA ABLACAO: os dois bracos de cada par passam por este mesmo codigo, com
os mesmos lotes, os mesmos aumentos, as mesmas epocas e a mesma semente. A
unica diferenca e requires_grad no backbone -- e o passo proprio dele, que so
existe quando ele treina.

ESTATISTICA DA BATCHNORM: congelar requires_grad NAO congela a BatchNorm, que
continua atualizando media e variancia em modo train(). O repositorio ja
levantou isso (commit "BUG: congelar requires_grad nao congela o BatchNorm").
Aqui o backbone congelado fica sempre em eval(), e no ajuste fino as
estatisticas tambem ficam congeladas por padrao: com lote 16 elas oscilam e
deslocam as features a cada passo.
"""
import copy
import json
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models as M
from torchvision import transforms

warnings.filterwarnings("ignore")

MAX_CAIXAS = 8

#: Cada backbone declara em que escala foi pre-treinado. Aplicar a normalizacao
#: de um a outro mede o desencontro, nao o backbone.
NORM_IMAGENET = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
NORM_MEIO = transforms.Normalize([0.5] * 3, [0.5] * 3)
#: O CLIP foi pre-treinado com estatisticas proprias, nem ImageNet nem meio.
#: Usar a errada mediria o desencontro de escala, nao o extrator.
NORM_CLIP = transforms.Normalize([0.4815, 0.4578, 0.4082],
                                 [0.2686, 0.2613, 0.2758])


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class TimmMapa(nn.Module):
    """forward_features do timm no contrato (N, C, H, W).

    Transformer puro devolve tokens (N, 1+H*W, C): os de prefixo saem e o
    resto volta a ser grade. MaxViT ja devolve mapa.
    """

    def __init__(self, m, tokens, entradas, nome, nhwc=False):
        super().__init__()
        self.m, self.tokens, self.entradas, self.nome = m, tokens, entradas, nome
        #: o Swin do timm entrega (N,H,W,C); precisa so de uma permutacao, nao
        #: do desmonte de fichas que o ViT exige.
        self.nhwc = nhwc

    def forward(self, x):
        if self.entradas and x.shape[-1] not in self.entradas:
            raise ValueError(f"{self.nome} aceita entrada de {self.entradas} px "
                             f"(janela fixa); recebeu {tuple(x.shape[-2:])}")
        o = self.m.forward_features(x)
        if self.nhwc:
            return o.permute(0, 3, 1, 2).contiguous()
        if not self.tokens:
            return o
        n = getattr(self.m, "num_prefix_tokens", 1)
        ph, pw = self.m.patch_embed.patch_size
        h, w = x.shape[-2] // ph, x.shape[-1] // pw
        return o[:, n:, :].transpose(1, 2).reshape(x.shape[0], -1, h, w)


#: Media que cada normalizacao registrada implica. O confronto com o
#: pretrained_cfg do timm impede pedir um extrator com a escala de outro, o que
#: mediria o desencontro em vez do extrator.
MEDIA_DE = {"meio": (0.5, 0.5, 0.5), "imagenet": (0.485, 0.456, 0.406),
            "clip": (0.4815, 0.4578, 0.4082)}


def _timm(nome, norm="meio", **kw):
    def fn(pre=True):
        try:
            import timm
        except ImportError:
            raise SystemExit("este experimento precisa do timm: pip install timm")
        m = timm.create_model(nome, pretrained=pre, num_classes=0, **kw)
        cfg = getattr(m, "pretrained_cfg", {}) or {}
        esperada = MEDIA_DE[norm]
        media = tuple(round(float(v), 3) for v in cfg.get("mean", esperada))
        if media != tuple(round(v, 3) for v in esperada):
            raise SystemExit(f"{nome}: timm declara media {media}, mas o registro usa "
                             f"normalizacao {norm!r} ({esperada}) -- corrija o registro")
        return m
    return fn


def _tv_resnet(fn, pesos, canais):
    def build(pre=True):
        m = fn(weights=pesos if pre else None)
        return nn.Sequential(*list(m.children())[:-2])
    return build, canais


#: nome -> (fabrica, canais, stride, normalizacao, entradas aceitas)
BACKBONES = {
    "resnet50": (lambda pre=True: nn.Sequential(*list(
        M.resnet50(weights=M.ResNet50_Weights.DEFAULT if pre else None).children())[:-2]),
        2048, 32, "imagenet", None),
    "resnet18": (lambda pre=True: nn.Sequential(*list(
        M.resnet18(weights=M.ResNet18_Weights.DEFAULT if pre else None).children())[:-2]),
        512, 32, "imagenet", None),
    "maxvit_t": (lambda pre=True: TimmMapa(
        _timm("maxvit_tiny_tf_512.in1k")(pre), False, (512,), "maxvit_t"),
        512, 32, "meio", (512,)),
    "vit_b16": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_224.augreg2_in21k_ft_in1k", dynamic_img_size=True)(pre),
        True, None, "vit_b16"),
        768, 16, "meio", None),

    # --- ViT-H: I-JEPA (auto-supervisionado) e o controle supervisionado ---
    # 632 M de parametros. Com 209 imagens de treino o regime esperado e
    # CONGELADO; o braco ajustado existe para medir isso, nao por aposta.
    #
    # O par h14@448 e o contraste de uma variavel so: mesma arquitetura
    # (ViT-H/14), mesmo corpus de pre-treino (ImageNet-21k/22k), muda so o
    # OBJETIVO -- predicao em espaco latente contra classificacao supervisionada.
    # Patch 14 exige entrada multipla de 14, e 448 e a que serve (32x32 fichas).
    "ijepa_h16": (lambda pre=True: TimmMapa(
        _timm("vit_huge_patch16_gap_448.in1k_ijepa", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "ijepa_h16"),
        1280, 16, "imagenet", None),

    "ijepa_h14": (lambda pre=True: TimmMapa(
        _timm("vit_huge_patch14_gap_224.in22k_ijepa", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "ijepa_h14"),
        1280, 14, "imagenet", None),

    "vit_h14_sup": (lambda pre=True: TimmMapa(
        _timm("vit_huge_patch14_224.orig_in21k", norm="meio",
              dynamic_img_size=True)(pre), True, None, "vit_h14_sup"),
        1280, 14, "meio", None),

    # --- segunda leva: quebra eixos que a primeira nao cobria -----------------
    # Os tres primeiros sao o MESMO ViT-B/16, mudando so o OBJETIVO de
    # pre-treino -- supervisionado, auto-supervisionado (dois objetivos) e
    # linguagem. E o contraste de uma variavel so que faltava: ate aqui os seis
    # extratores eram todos supervisionados, e "arquitetura" podia estar
    # confundida com "forma de supervisao".
    "vit_b16_dino": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_224.dino", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "vit_b16_dino"),
        768, 16, "imagenet", None),

    "vit_b16_mae": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_224.mae", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "vit_b16_mae"),
        768, 16, "imagenet", None),

    # Dois casos acrescentados para testar a afirmacao de colapso fora do MAE.
    # BEiT-3 no checkpoint ".pt": pre-treino de modelagem mascarada PURO, sem
    # o ajuste fino que os demais beit do timm trazem (sufixo "_ft_"). Mesma
    # arquitetura, mesmas 1025 fichas a 512 px, mesma normalizacao: entra no
    # braco do pre-treino sem nenhum ajuste de protocolo.
    "vit_b16_beit3": (lambda pre=True: TimmMapa(
        _timm("beit3_base_patch16_224.pt", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "vit_b16_beit3"),
        768, 16, "imagenet", None),

    # ViT-B/16 SEM TREINO NENHUM. Nao e um candidato a colapso -- e o
    # referencial de "representacao que se deve rejeitar", com resposta
    # conhecida na sonda. Se a razao de participacao NAO o sinalizar, isso
    # delimita a afirmacao: ela flagra colapso de modelagem mascarada, nao
    # representacao ruim em geral. O resultado interessa nos dois sentidos.
    "vit_b16_aleatorio": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_224.augreg_in21k_ft_in1k", norm="meio",
              dynamic_img_size=True)(False), True, None, "vit_b16_aleatorio"),
        768, 16, "meio", None),

    "vit_b16_clip": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_clip_224.openai", norm="clip",
              dynamic_img_size=True)(pre), True, None, "vit_b16_clip"),
        768, 16, "clip", None),

    # MaxViT maior: a anomalia do posto e da arquitetura ou do tamanho tiny?
    "maxvit_s": (lambda pre=True: TimmMapa(
        _timm("maxvit_small_tf_512.in1k")(pre), False, (512,), "maxvit_s"),
        768, 32, "meio", (512,)),

    # convolucao de outra familia: separavel em profundidade, squeeze-excite,
    # escalonamento composto -- nem ResNet nem ConvNeXt
    "efficientnet_b3": (lambda pre=True: TimmMapa(
        _timm("efficientnet_b3.ra2_in1k", norm="imagenet")(pre), False, None,
        "efficientnet_b3"),
        1536, 32, "imagenet", None),

    # --- terceira leva: NOVE objetivos sobre o MESMO ViT-B/16 ----------------
    # O desenho responde a objecao de que "objetivo" seria rotulo para um
    # pacote confundido (objetivo + corpus + receita). Aqui ha:
    #   4 SUPERVISIONADOS de receitas e corpora diferentes (augreg2 21k->1k,
    #     orig 21k, augreg 1k, DeiT3 1k) -- se a receita fosse o fator, eles
    #     se espalhariam tanto quanto os objetivos distintos;
    #   2 AUTO-SUPERVISIONADOS de objetivos diferentes (DINO, MAE);
    #   3 COM LINGUAGEM, dois deles o MESMO objetivo em corpora diferentes
    #     (CLIP OpenAI contra CLIP LAION-2B) -- isola corpus de objetivo.
    "vit_b16_sup21k": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_224.orig_in21k", dynamic_img_size=True)(pre),
        True, None, "vit_b16_sup21k"), 768, 16, "meio", None),

    "vit_b16_sup1k": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_224.augreg_in1k", dynamic_img_size=True)(pre),
        True, None, "vit_b16_sup1k"), 768, 16, "meio", None),

    "vit_b16_deit3": (lambda pre=True: TimmMapa(
        _timm("deit3_base_patch16_224.fb_in1k", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "vit_b16_deit3"),
        768, 16, "imagenet", None),

    "vit_b16_clip2b": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_clip_224.laion2b", norm="clip",
              dynamic_img_size=True)(pre), True, None, "vit_b16_clip2b"),
        768, 16, "clip", None),

    "vit_b16_siglip": (lambda pre=True: TimmMapa(
        _timm("vit_base_patch16_siglip_224.webli", dynamic_img_size=True)(pre),
        True, None, "vit_b16_siglip"), 768, 16, "meio", None),

    # --- controles do estudo de explicabilidade -------------------------------
    # Cada um quebra UM confundidor da comparacao original ResNet/MaxViT/ViT:
    #   convnext_t  convolucao com receita moderna  -> o comportamento da ResNet
    #               e da convolucao ou da receita de 2015?
    #   swin_t      hierarquico com atencao SO em janela local, sem grade
    #               -> a invariancia ao arranjo do MaxViT vem da hierarquia ou
    #                  da atencao em grade?
    #   vit_s16     mesma familia do ViT-B, mas IN-1k e 22 M em vez de IN-21k e
    #               88 M -> o comportamento do ViT e da atencao global ou do
    #               corpus dez vezes maior?
    "convnext_t": (lambda pre=True: TimmMapa(
        _timm("convnext_tiny.fb_in1k", norm="imagenet")(pre), False, None, "convnext_t"),
        768, 32, "imagenet", None),

    "swin_t": (lambda pre=True: TimmMapa(
        _timm("swin_tiny_patch4_window7_224.ms_in1k", norm="imagenet",
              img_size=512)(pre), False, (512,), "swin_t", nhwc=True),
        768, 32, "imagenet", (512,)),

    "vit_s16": (lambda pre=True: TimmMapa(
        _timm("vit_small_patch16_224.augreg_in1k", dynamic_img_size=True)(pre),
        True, None, "vit_s16"),
        384, 16, "meio", None),
}


def _trunca(m, n_blocos):
    """Fica com os n primeiros blocos do transformer e descarta o resto.

    Poda de profundidade sem retreino nenhum: e so encurtar a lista de blocos.
    O forward_features do timm aplica a norma final depois deles, entao a saida
    continua normalizada. Cada bloco do ViT-H tem 19,7 M de parametros, entao
    truncar em 12 leva de 632 M para 238 M.

    Motivacao alem da contagem: as camadas finais de um transformer se
    especializam na tarefa de pre-treino, e as intermediarias carregam
    estrutura visual mais geral. Para um dominio distante da ImageNet, truncar
    pode transferir melhor, nao so mais barato.
    """
    if n_blocos > len(m.blocks):
        raise SystemExit(f"pedido {n_blocos} blocos, o modelo tem {len(m.blocks)}")
    m.blocks = nn.Sequential(*list(m.blocks)[:n_blocos])
    return m


def _registra_ijepa_truncado(n_blocos):
    """Acrescenta ijepa_h16_dN ao registro, com N blocos dos 32."""
    alias = f"ijepa_h16_d{n_blocos}"
    BACKBONES[alias] = (
        lambda pre=True, _n=n_blocos, _a=alias: TimmMapa(
            _trunca(_timm("vit_huge_patch16_gap_448.in1k_ijepa", norm="imagenet",
                          dynamic_img_size=True)(pre), _n),
            True, None, _a),
        1280, 16, "imagenet", None)


for _n in (8, 12, 16, 24):
    _registra_ijepa_truncado(_n)


def monta_backbone(nome, pre=True, treinavel=False):
    if nome not in BACKBONES:
        raise SystemExit(f"backbone desconhecido: {nome}. Ha: {sorted(BACKBONES)}")
    fab, canais, _, norm, _ = BACKBONES[nome]
    m = fab(pre)
    for p in m.parameters():
        p.requires_grad = bool(treinavel)
    return (m.train() if treinavel else m.eval()), canais, norm


_NORMS = {"imagenet": NORM_IMAGENET, "meio": NORM_MEIO, "clip": NORM_CLIP}


def normalizador(nome):
    return _NORMS[BACKBONES[nome][3]]


# ---------------------------------------------------------------------------
# Cabecas
# ---------------------------------------------------------------------------

class CabecaClasse(nn.Module):
    """Pooling global, uma camada oculta e duas saidas.

    Mesma forma da cabeca do dev_harness (dim -> k -> 2), para que a compara-
    cao com aquela bateria seja de backbone e regime, nao de cabeca.
    """

    def __init__(self, canais, k=6, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(canais, k), nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(k, 2))

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class CabecaCaixa(nn.Module):
    """Reduz canais, reamostra para grade fixa e regride a caixa.

    A grade fixa torna o numero de parametros independente da resolucao, e o
    flatten preserva a posicao de cada celula -- e isso que permite regredir
    coordenada, o que um pooling global nao permitiria.
    """

    def __init__(self, canais, grade=8, reduz=32, oculta=256, dropout=0.1):
        super().__init__()
        self.reduz = nn.Sequential(
            nn.Conv2d(canais, reduz, 1), nn.GELU(),
            nn.Conv2d(reduz, reduz, 3, padding=1), nn.GELU())
        self.grade = grade
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(reduz * grade * grade, oculta), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(oculta, 4))

    def forward(self, x):
        return torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(self.reduz(x), self.grade)))


class ReducaoPCA(nn.Module):
    """Projecao linear FIXA, ajustada por PCA nas features de TREINO.

    Existe para resolver um defeito medido: com o extrator congelado, a cabeca
    recebe 1280 features e o treino tem 209 imagens. Com mais features que
    amostras, uma camada linear separa o treino por acaso e nao generaliza --
    foi o que derrubou os bracos de ViT-H congelado para a AUC do acaso.

    Projetar para k dimensoes com k < n_amostras remove esse regime. A projecao
    e ajustada SO no treino e depois congelada; ajusta-la no conjunto inteiro
    seria vazamento, do mesmo tipo que o dev_harness evita ao calcular media e
    desvio das features apenas no treino do fold.

    O branqueamento (dividir por sqrt da variancia) deixa as componentes na
    mesma escala, que e o equivalente da padronizacao do dev_harness.
    """

    def __init__(self, componentes, media, escala):
        super().__init__()
        self.register_buffer("W", torch.as_tensor(componentes, dtype=torch.float32))
        self.register_buffer("mu", torch.as_tensor(media, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(escala, dtype=torch.float32))
        self.n_saida = self.W.shape[0]

    def forward(self, x):
        x = x - self.mu.view(1, -1, 1, 1)
        x = torch.einsum("nchw,kc->nkhw", x, self.W)
        return x / self.sd.view(1, -1, 1, 1).clamp_min(1e-6)


def ajusta_pca(backbone, carregador, k, dev, max_vetores=200_000):
    """Ajusta a PCA sobre as features de treino, uma posicao espacial por vetor.

    Cada celula do mapa vira uma amostra de dimensao C. Assim a mesma projecao
    serve as duas cabecas: a de classe, que agrupa depois, e a de caixa, que
    precisa do mapa inteiro.
    """
    from sklearn.decomposition import PCA

    backbone.eval()
    blocos = []
    with torch.no_grad():
        for b in carregador:
            m = backbone(b["img"].to(dev))
            blocos.append(m.permute(0, 2, 3, 1).reshape(-1, m.shape[1]).float().cpu().numpy())
    X = np.concatenate(blocos, 0)

    if len(X) > max_vetores:
        idx = np.random.RandomState(0).choice(len(X), max_vetores, replace=False)
        X = X[idx]
    if k > min(X.shape):
        raise SystemExit(f"PCA de {k} componentes exige mais que {min(X.shape)} dimensoes")

    p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(X)
    explicada = float(p.explained_variance_ratio_.sum())
    print(f"    PCA {X.shape[1]} -> {k} | {len(X)} vetores | "
          f"variancia explicada {explicada:.1%}", flush=True)
    return ReducaoPCA(p.components_, p.mean_, np.sqrt(p.explained_variance_)), explicada


class Multitarefa(nn.Module):
    def __init__(self, backbone, canais, congelado, k=6, bn_congelada=True, reducao=None,
                 grade_caixa=8, dropout_classe=0.0):
        super().__init__()
        self.backbone, self.congelado, self.bn_congelada = backbone, congelado, bn_congelada
        self.reducao = reducao
        if reducao is not None:
            canais = reducao.n_saida
        self.classe = CabecaClasse(canais, k, dropout=dropout_classe)
        self.caixa = CabecaCaixa(canais, grade=grade_caixa)

    def train(self, modo=True):
        super().train(modo)
        if self.congelado:
            self.backbone.eval()
        elif modo and self.bn_congelada:
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        if self.congelado:
            self.backbone.eval()
            with torch.no_grad():
                mapa = self.backbone(x)
                if self.reducao is not None:
                    mapa = self.reducao(mapa)
        else:
            mapa = self.backbone(x)
            if self.reducao is not None:
                mapa = self.reducao(mapa)
        return self.classe(mapa), self.caixa(mapa)

    def grupos(self, lr, backbone_lr, wd):
        bb = [p for n, p in self.named_parameters()
              if n.startswith("backbone.") and p.requires_grad]
        cab = [p for n, p in self.named_parameters()
               if not n.startswith(("backbone.", "reducao.")) and p.requires_grad]
        g = [dict(params=cab, lr=lr, weight_decay=wd, name="cabecas")]
        if bb:
            g.append(dict(params=bb, lr=backbone_lr, weight_decay=wd, name="backbone"))
        return g

    def estado_treinavel(self):
        t = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in t}

    def n_treinaveis(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Caixas: perda e metricas
# ---------------------------------------------------------------------------

def para_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def perda_giou(pred, alvo, eps=1e-7):
    """1 - GIoU. GIoU porque IoU puro tem gradiente nulo quando nao ha
    sobreposicao, e no inicio do treino nunca ha."""
    a, b = para_xyxy(pred), para_xyxy(alvo)
    lt, rb = torch.maximum(a[..., :2], b[..., :2]), torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda z: (z[..., 2] - z[..., 0]).clamp(min=0) * (z[..., 3] - z[..., 1]).clamp(min=0)
    uniao = area(a) + area(b) - inter
    iou = inter / (uniao + eps)
    LT, RB = torch.minimum(a[..., :2], b[..., :2]), torch.maximum(a[..., 2:], b[..., 2:])
    env = (RB - LT).clamp(min=0).prod(-1)
    return 1.0 - (iou - (env - uniao) / (env + eps))


def perda_caixa(pred, alvo, mascara, w_l1=5.0, w_giou=2.0):
    """So nas imagens com lesao anotada: sem lesao nao ha alvo de posicao."""
    m = mascara.bool()
    if not m.any():
        return pred.sum() * 0.0
    p, t = pred[m], alvo[m]
    return (w_l1 * F.l1_loss(p, t, reduction="none").sum(-1) + w_giou * perda_giou(p, t)).mean()


def _iou_np(pred, gt):
    """IoU entre pred (N,4) e gt (N,K,4) -> (N,K), tudo em (cx,cy,w,h)."""
    f = lambda b: np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                            b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)
    p, g = f(pred)[:, None, :], f(gt)
    lt, rb = np.maximum(p[..., :2], g[..., :2]), np.minimum(p[..., 2:], g[..., 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda b: np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    return inter / np.maximum(area(p) + area(g) - inter, 1e-7)


def iou_por_imagem(pred, gt, valido):
    """Melhor IoU por imagem (nan onde nao ha lesao).

    Uma caixa prevista contra possivelmente varias lesoes: vale a de maior
    IoU -- acertar qualquer lesao conta.
    """
    pred = np.asarray(pred, float).reshape(-1, 4)
    gt, valido = np.asarray(gt, float), np.asarray(valido, bool)
    ious = _iou_np(pred, gt)
    ious[~valido] = -1.0
    melhor = ious.max(1)
    melhor[~valido.any(1)] = np.nan
    return melhor


def metricas_caixa(pred, gt, valido):
    """IoU medio, acertos por limiar e centro dentro da lesao.

    centro_na_lesao e o criterio classico da deteccao de massa em mamografia:
    o centro previsto cai dentro de alguma lesao anotada.
    """
    pred = np.asarray(pred, float).reshape(-1, 4)
    gt, valido = np.asarray(gt, float), np.asarray(valido, bool)
    tem = valido.any(1)
    if tem.sum() == 0:
        return dict(iou=np.nan, iou_mediana=np.nan, acerto50=np.nan,
                    acerto25=np.nan, centro=np.nan, n_lesao=0)
    p, g, v = pred[tem], gt[tem], valido[tem]
    melhor = iou_por_imagem(p, g, v)
    f = lambda b: np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                            b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)
    gx = f(g)
    c = p[:, :2][:, None, :]
    dentro = ((c >= gx[..., :2]) & (c <= gx[..., 2:])).all(-1) & v
    return dict(iou=float(melhor.mean()), iou_mediana=float(np.median(melhor)),
                acerto50=float((melhor >= 0.5).mean()), acerto25=float((melhor >= 0.25).mean()),
                centro=float(dentro.any(1).mean()), n_lesao=int(tem.sum()))


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def le_caixas(txt):
    if txt is None or (isinstance(txt, float) and np.isnan(txt)) or txt == "":
        return np.zeros((0, 4), np.float32)
    it = json.loads(txt) if isinstance(txt, str) else list(txt)
    if not it:
        return np.zeros((0, 4), np.float32)
    return np.asarray([[b["cx"], b["cy"], b["w"], b["h"]] for b in it], np.float32)


class Mamografias(Dataset):
    """PNGs com rotulo e caixas. Aumento geometrico move a caixa junto.

    O aumento nao consulta o rotulo: P(girada|maligno) = P(girada|benigno).
    Esta e a correcao #2 do relatorio de auditoria, e vale igual aqui.
    """

    def __init__(self, d, imgs, tam, norm, aumenta=False):
        self.idx_global = d.index.to_numpy()
        self.d = d.reset_index(drop=True)
        self.imgs, self.tam, self.aumenta = imgs, tam, aumenta
        self.tf = transforms.Compose([transforms.ToTensor(), norm])
        self.caixas = [le_caixas(c) for c in self.d.caixas]

    def __len__(self):
        return len(self.d)

    @staticmethod
    def _zoom(img, cx, escala, ox, oy):
        w, h = img.size
        cw, ch = int(w * escala), int(h * escala)
        x0, y0 = int((w - cw) * ox), int((h - ch) * oy)
        if len(cx):
            xa, xb = (cx[:, 0] - cx[:, 2] / 2) * w, (cx[:, 0] + cx[:, 2] / 2) * w
            ya, yb = (cx[:, 1] - cx[:, 3] / 2) * h, (cx[:, 1] + cx[:, 3] / 2) * h
            if not ((xa >= x0) & (xb <= x0 + cw) & (ya >= y0) & (yb <= y0 + ch)).all():
                return img, cx          # cortaria a lesao: desiste
        rec = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.Resampling.BILINEAR)
        if len(cx):
            cx = cx.copy()
            cx[:, 0] = (cx[:, 0] * w - x0) / cw
            cx[:, 1] = (cx[:, 1] * h - y0) / ch
            cx[:, 2] /= escala
            cx[:, 3] /= escala
        return rec, cx

    def _aumenta(self, img, cx):
        s = torch.rand(6).tolist()
        if s[0] < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(cx):
                cx = cx.copy()
                cx[:, 0] = 1.0 - cx[:, 0]
        if s[1] < 0.5:
            img, cx = self._zoom(img, cx, 0.85 + 0.15 * s[2], s[3], s[4])
        if s[5] < 0.8:
            img = ImageEnhance.Brightness(img).enhance(0.9 + 0.2 * torch.rand(1).item())
            img = ImageEnhance.Contrast(img).enhance(0.9 + 0.2 * torch.rand(1).item())
        return img, cx

    def __getitem__(self, i):
        r = self.d.iloc[i]
        img = Image.open(self.imgs / r.processed_image_filename).convert("L")
        cx = self.caixas[i]
        if self.aumenta:
            img, cx = self._aumenta(img, cx)
        if img.size != (self.tam, self.tam):
            img = img.resize((self.tam, self.tam), Image.Resampling.BILINEAR)

        gt = np.zeros((MAX_CAIXAS, 4), np.float32)
        val = np.zeros(MAX_CAIXAS, bool)
        k = min(len(cx), MAX_CAIXAS)
        if k:
            gt[:k], val[:k] = np.clip(cx[:k], 0, 1), True
        return dict(img=self.tf(img.convert("RGB")),
                    y=torch.tensor(int(r.classification)),
                    caixa=torch.from_numpy(gt[0]),
                    tem=torch.tensor(bool(val[0])),
                    gt=torch.from_numpy(gt), val=torch.from_numpy(val),
                    idx=torch.tensor(int(self.idx_global[i])))


def carregadores(d, imgs, indices, cfg, seed):
    norm = normalizador(cfg["backbone"])
    ger = torch.Generator().manual_seed(seed)
    out = {}
    for nome, ix in indices.items():
        tr = nome == "treino"
        ds = Mamografias(d.iloc[ix], imgs, cfg["tam"], norm, aumenta=tr and cfg["aumenta"])
        out[nome] = DataLoader(ds, batch_size=cfg["lote"], shuffle=tr,
                               generator=ger if tr else None,
                               drop_last=tr and len(ds) % cfg["lote"] == 1,
                               num_workers=cfg.get("workers", 4),
                               pin_memory=torch.cuda.is_available(),
                               persistent_workers=cfg.get("workers", 4) > 0)
    return out


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def pesos_classe(y, dev):
    c = np.bincount(np.asarray(y, int), minlength=2).astype(float)
    c[c == 0] = 1.0
    return torch.tensor(c.sum() / (2.0 * c), dtype=torch.float32, device=dev)


def _escalonador(opt, epocas, passos, aquece):
    total = max(1, epocas * passos)
    aq = min(total - 1, max(0, aquece * passos))

    def f(p):
        if p < aq:
            return (p + 1) / max(1, aq)
        t = (p - aq) / max(1, total - aq)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def uma_passagem(modelo, dl, crit, opt, sch, dev, cfg, treinando):
    """Uma passagem pelos dados.

    ``cfg["atualiza"]`` decide quando o otimizador anda:

    ``"lote"``
        um passo por mini-lote, o padrao do treino estocastico.
    ``"epoca"``
        os gradientes de todos os mini-lotes sao ACUMULADOS e o otimizador da
        UM passo ao fim da epoca. O gradiente resultante e o do conjunto de
        treino inteiro, sem ruido de amostragem -- e o mini-lote deixa de ter
        efeito sobre a matematica, virando so um botao de memoria. Cada perda
        e dividida pelo numero de lotes para que a soma seja a media do
        conjunto, e nao a soma dela.
    """
    modelo.train() if treinando else modelo.eval()
    por_epoca = treinando and cfg.get("atualiza", "lote") == "epoca"
    n_lotes = max(1, len(dl))
    soma = dict(perda=0.0, cls=0.0, cx=0.0)
    n = 0
    Y, S, P, G, V, I = [], [], [], [], [], []

    if por_epoca and opt is not None:
        opt.zero_grad(set_to_none=True)

    for b in dl:
        x = b["img"].to(dev, non_blocking=True)
        y, cx, tem = b["y"].to(dev), b["caixa"].to(dev), b["tem"].to(dev)
        with torch.set_grad_enabled(treinando):
            amp = cfg.get("amp", True) and dev.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logit, caixa = modelo(x)
            logit, caixa = logit.float(), caixa.float()
            lc = crit(logit, y)
            lb = perda_caixa(caixa, cx, tem)
            perda = lc + cfg["peso_caixa"] * lb
        if treinando:
            if por_epoca:
                (perda / n_lotes).backward()          # acumula, nao anda
            else:
                opt.zero_grad(set_to_none=True)
                perda.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                if sch is not None:
                    sch.step()
        k = len(x)
        n += k
        soma["perda"] += float(perda) * k
        soma["cls"] += float(lc) * k
        soma["cx"] += float(lb) * k
        Y.append(y.cpu().numpy())
        S.append(torch.softmax(logit, 1)[:, 1].detach().cpu().numpy())
        P.append(caixa.detach().cpu().numpy())
        G.append(b["gt"].numpy())
        V.append(b["val"].numpy())
        I.append(b["idx"].numpy())

    if por_epoca and opt is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if sch is not None:
            sch.step()

    pred = dict(y=np.concatenate(Y), score=np.concatenate(S), caixa=np.concatenate(P),
                gt=np.concatenate(G), val=np.concatenate(V), idx=np.concatenate(I))
    return {k: v / max(n, 1) for k, v in soma.items()}, pred


def resume(pred, perdas, limiar=0.5):
    m = dict(perda=perdas["perda"], perda_cls=perdas["cls"], perda_caixa=perdas["cx"])
    y, s = pred["y"], pred["score"]
    m["auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    m["bal"] = float(balanced_accuracy_score(y, (s > limiar).astype(int)))
    m.update(metricas_caixa(pred["caixa"], pred["gt"], pred["val"]))
    return m


def melhor_limiar(y, s):
    """Limiar por acuracia balanceada, como no dev_harness."""
    if len(np.unique(y)) < 2:
        return 0.5
    cand = np.unique(np.round(s, 3))
    return float(cand[int(np.argmax([balanced_accuracy_score(y, (s > c).astype(int))
                                     for c in cand]))])


def escore_selecao(m, criterio):
    a = 0.0 if np.isnan(m["auc"]) else m["auc"]
    i = 0.0 if np.isnan(m["iou"]) else m["iou"]
    return {"auc": a, "iou": i, "auc+iou": a + i}[criterio]


def treina(modelo, dls, y_treino, cfg, dev, avalia_teste=True, log=None):
    """Treina; epoca e limiar saem SO da validacao; teste avaliado uma vez."""
    modelo = modelo.to(dev)
    crit = nn.CrossEntropyLoss(weight=pesos_classe(y_treino, dev))
    opt = torch.optim.AdamW(modelo.grupos(cfg["lr"], cfg["backbone_lr"], cfg["wd"]))
    # Com atualizacao por epoca ha UM passo por epoca, entao o escalonador conta
    # epocas; com atualizacao por lote ele conta lotes.
    passos = 1 if cfg.get("atualiza", "lote") == "epoca" else len(dls["treino"])
    sch = _escalonador(opt, cfg["epocas"], passos, cfg["aquece"])

    hist = []
    melhor = dict(escore=-float("inf"), auc=np.nan, epoca=-1, estado=None,
                  limiar=0.5, val=None)

    for ep in range(1, cfg["epocas"] + 1):
        pt, prt = uma_passagem(modelo, dls["treino"], crit, opt, sch, dev, cfg, True)
        mt = resume(prt, pt)
        pv, prv = uma_passagem(modelo, dls["validacao"], crit, None, None, dev, cfg, False)
        mv = resume(prv, pv)
        mv["escore"] = escore_selecao(mv, cfg["criterio"])
        hist.append(dict(epoca=ep, fase="treino", **mt))
        hist.append(dict(epoca=ep, fase="validacao", **mv))
        if mv["escore"] > melhor["escore"]:
            melhor = dict(escore=mv["escore"], auc=mv["auc"], epoca=ep,
                          estado=copy.deepcopy(modelo.estado_treinavel()),
                          limiar=melhor_limiar(prv["y"], prv["score"]), val=prv)
        if log:
            log(ep, mt, mv)

    if melhor["estado"] is not None:
        modelo.load_state_dict(melhor["estado"], strict=False)

    teste = prte = None
    if avalia_teste:
        pe, prte = uma_passagem(modelo, dls["teste"], crit, None, None, dev, cfg, False)
        teste = resume(prte, pe, limiar=melhor["limiar"])
        teste["epoca_escolhida"] = melhor["epoca"]

    return dict(hist=hist, teste=teste, epoca=melhor["epoca"], val_auc=melhor["auc"],
                limiar=melhor["limiar"], pred=dict(validacao=melhor["val"], teste=prte))


# ---------------------------------------------------------------------------
# Linhas de base
# ---------------------------------------------------------------------------

def caixa_media(d, indices):
    """Caixa media do treino: a linha de base obrigatoria da localizacao.

    Um modelo que so aprendeu "lesao costuma ficar no meio" reproduz isto. IoU
    que nao supera esta caixa nao e evidencia de ter localizado coisa alguma --
    e o analogo, para a caixa, das linhas de base triviais do relatorio.
    """
    sub = d.iloc[indices]
    sub = sub[sub.tem_lesao]
    if not len(sub):
        return None
    return np.array([sub.cx.mean(), sub.cy.mean(), sub.w.mean(), sub.h.mean()], float)


def alvos_caixa(d, indices):
    """(gt, valido) no formato que metricas_caixa espera: (N, MAX_CAIXAS, 4)."""
    sub = d.iloc[indices]
    gt = np.zeros((len(sub), MAX_CAIXAS, 4), np.float32)
    val = np.zeros((len(sub), MAX_CAIXAS), bool)
    for i, txt in enumerate(sub.caixas):
        c = le_caixas(txt)
        k = min(len(c), MAX_CAIXAS)
        if k:
            gt[i, :k], val[i, :k] = np.clip(c[:k], 0, 1), True
    return gt, val


def avalia_caixa_fixa(caixa, d, indices):
    gt, val = alvos_caixa(d, indices)
    return metricas_caixa(np.tile(caixa, (len(indices), 1)), gt, val)


# ---------------------------------------------------------------------------
# Incerteza
# ---------------------------------------------------------------------------

def bootstrap_agrupado(fn, grupos, n_boot=2000, seed=0, alpha=0.05):
    """IC percentil reamostrando PACIENTES, nao imagens.

    Reamostrar imagens trata as 4 vistas de um paciente como independentes e
    estreita o intervalo artificialmente. E a mesma regra do
    cluster_bootstrap_ci do nn_fixed.py.
    """
    rng = np.random.RandomState(seed)
    unicos = np.unique(grupos)
    pos = {g: np.where(grupos == g)[0] for g in unicos}
    vals = []
    for _ in range(n_boot):
        esc = rng.choice(unicos, size=len(unicos), replace=True)
        ix = np.concatenate([pos[g] for g in esc])
        v = fn(ix)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def ic_auc(y, s, grupos, **kw):
    def f(ix):
        if len(np.unique(y[ix])) < 2:
            return None
        return roc_auc_score(y[ix], s[ix])
    return bootstrap_agrupado(f, grupos, **kw)


def ic_iou(ious, grupos, **kw):
    def f(ix):
        v = ious[ix]
        v = v[~np.isnan(v)]
        return v.mean() if len(v) else None
    return bootstrap_agrupado(f, grupos, **kw)


def ic_diferenca(fn_a, fn_b, grupos, n_boot=2000, seed=0, alpha=0.05):
    """IC da diferenca PAREADA: os dois bracos nas mesmas pacientes reamostradas."""
    rng = np.random.RandomState(seed)
    unicos = np.unique(grupos)
    pos = {g: np.where(grupos == g)[0] for g in unicos}
    difs = []
    for _ in range(n_boot):
        esc = rng.choice(unicos, size=len(unicos), replace=True)
        ix = np.concatenate([pos[g] for g in esc])
        a, b = fn_a(ix), fn_b(ix)
        if a is not None and b is not None and not (np.isnan(a) or np.isnan(b)):
            difs.append(a - b)
    if len(difs) < 20:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(difs)),
            float(np.percentile(difs, 100 * alpha / 2)),
            float(np.percentile(difs, 100 * (1 - alpha / 2))))
