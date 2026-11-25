#!/usr/bin/env python3
"""Image classification + lesion localisation, in the same model.

    image -> backbone -+-> class head -> benign / suspicious
                       +-> box head   -> (cx, cy, w, h)

It exists because dev_harness.py does not serve this question: there the
backbone is frozen and the features are extracted once, which presupposes that
the backbone does not change. The frozen-versus-fine-tuned ablation needs it to
be able to receive gradient, and the box needs the spatial map, which global
pooling destroys.

What this module does NOT change relative to dev_harness: grouping by patient,
epoch and threshold chosen on the validation set only, test evaluated once.

ABLATION RULE: both arms of each pair go through this same code, with the same
batches, the same augmentations, the same epochs and the same seed. The only
difference is requires_grad on the backbone -- and its own optimiser group,
which exists only when it trains.

BATCHNORM STATISTICS: freezing requires_grad does NOT freeze BatchNorm, which
keeps updating its running mean and variance in train() mode. The repository
already raised this (commit "BUG: freezing requires_grad does not freeze
BatchNorm"). Here the frozen backbone stays in eval() throughout, and during
fine-tuning the statistics are frozen by default too: with a batch of 16 they
fluctuate and shift the features at every step.

NOTE ON EXTERNAL COLUMN NAMES. A few DataFrame columns below keep their
original Portuguese spelling (`caixas`, `tem_lesao`). They are the column names
of the INbreast derivative tables produced by an earlier project, which are not
redistributed here; renaming them in this code would simply fail to read those
files.
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

MAX_BOXES = 8

#: Each backbone declares the scale it was pre-trained on. Applying one's
#: normalisation to another measures the mismatch, not the backbone.
NORM_IMAGENET = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
NORM_HALF = transforms.Normalize([0.5] * 3, [0.5] * 3)
#: CLIP was pre-trained with its own statistics, neither ImageNet nor half.
#: Using the wrong one would measure the scale mismatch, not the encoder.
NORM_CLIP = transforms.Normalize([0.4815, 0.4578, 0.4082],
                                 [0.2686, 0.2613, 0.2758])


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class TimmMap(nn.Module):
    """timm's forward_features under the (N, C, H, W) contract.

    A pure transformer returns tokens (N, 1+H*W, C): the prefix ones are
    dropped and the rest goes back to a grid. MaxViT already returns a map.
    """

    def __init__(self, m, tokens, inputs, name, nhwc=False):
        super().__init__()
        self.m, self.tokens, self.inputs, self.name = m, tokens, inputs, name
        #: timm's Swin returns (N,H,W,C); it needs only a permutation, not the
        #: token disassembly the ViT requires.
        self.nhwc = nhwc

    def forward(self, x):
        if self.inputs and x.shape[-1] not in self.inputs:
            raise ValueError(f"{self.name} accepts an input of {self.inputs} px "
                             f"(fixed window); got {tuple(x.shape[-2:])}")
        o = self.m.forward_features(x)
        if self.nhwc:
            return o.permute(0, 3, 1, 2).contiguous()
        if not self.tokens:
            return o
        n = getattr(self.m, "num_prefix_tokens", 1)
        ph, pw = self.m.patch_embed.patch_size
        h, w = x.shape[-2] // ph, x.shape[-1] // pw
        return o[:, n:, :].transpose(1, 2).reshape(x.shape[0], -1, h, w)


#: The mean each registered normalisation implies. Checking it against timm's
#: pretrained_cfg prevents asking for one encoder with another's scale, which
#: would measure the mismatch instead of the encoder.
MEAN_OF = {"half": (0.5, 0.5, 0.5), "imagenet": (0.485, 0.456, 0.406),
           "clip": (0.4815, 0.4578, 0.4082)}


def _timm(name, norm="half", **kw):
    def fn(pre=True):
        try:
            import timm
        except ImportError:
            raise SystemExit("this experiment needs timm: pip install timm")
        m = timm.create_model(name, pretrained=pre, num_classes=0, **kw)
        cfg = getattr(m, "pretrained_cfg", {}) or {}
        expected = MEAN_OF[norm]
        mean = tuple(round(float(v), 3) for v in cfg.get("mean", expected))
        if mean != tuple(round(v, 3) for v in expected):
            raise SystemExit(
                f"{name}: timm declares mean {mean}, but the registry uses "
                f"normalisation {norm!r} ({expected}) -- fix the registry")
        return m
    return fn


def _tv_resnet(fn, weights, channels):
    def build(pre=True):
        m = fn(weights=weights if pre else None)
        return nn.Sequential(*list(m.children())[:-2])
    return build, channels


#: name -> (factory, channels, stride, normalisation, accepted inputs)
BACKBONES = {
    "resnet50": (lambda pre=True: nn.Sequential(*list(
        M.resnet50(weights=M.ResNet50_Weights.DEFAULT if pre else None
                   ).children())[:-2]),
        2048, 32, "imagenet", None),
    "resnet18": (lambda pre=True: nn.Sequential(*list(
        M.resnet18(weights=M.ResNet18_Weights.DEFAULT if pre else None
                   ).children())[:-2]),
        512, 32, "imagenet", None),
    "maxvit_t": (lambda pre=True: TimmMap(
        _timm("maxvit_tiny_tf_512.in1k")(pre), False, (512,), "maxvit_t"),
        512, 32, "half", (512,)),
    "vit_b16": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_224.augreg2_in21k_ft_in1k",
              dynamic_img_size=True)(pre), True, None, "vit_b16"),
        768, 16, "half", None),

    # --- ViT-H: I-JEPA (self-supervised) and the supervised control --------
    # 632 M parameters. With 209 training images the expected regime is
    # FROZEN; the fine-tuned arm exists to measure that, not as a bet.
    #
    # The h14@448 pair is a one-variable contrast: same architecture
    # (ViT-H/14), same pre-training corpus (ImageNet-21k/22k), only the
    # OBJECTIVE changes -- prediction in representation space against
    # supervised classification. Patch 14 requires an input that is a multiple
    # of 14, and 448 is the one that fits (32x32 tokens).
    "ijepa_h16": (lambda pre=True: TimmMap(
        _timm("vit_huge_patch16_gap_448.in1k_ijepa", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "ijepa_h16"),
        1280, 16, "imagenet", None),

    "ijepa_h14": (lambda pre=True: TimmMap(
        _timm("vit_huge_patch14_gap_224.in22k_ijepa", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "ijepa_h14"),
        1280, 14, "imagenet", None),

    "vit_h14_sup": (lambda pre=True: TimmMap(
        _timm("vit_huge_patch14_224.orig_in21k", norm="half",
              dynamic_img_size=True)(pre), True, None, "vit_h14_sup"),
        1280, 14, "half", None),

    # --- second batch: breaks axes the first one did not cover -------------
    # The first three are the SAME ViT-B/16, changing only the pre-training
    # OBJECTIVE -- supervised, self-supervised (two objectives) and language.
    # It is the one-variable contrast that was missing: until here all six
    # encoders were supervised, and "architecture" could have been confounded
    # with "form of supervision".
    "vit_b16_dino": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_224.dino", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "vit_b16_dino"),
        768, 16, "imagenet", None),

    "vit_b16_mae": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_224.mae", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "vit_b16_mae"),
        768, 16, "imagenet", None),

    # Two cases added to test the collapse claim away from MAE.
    # BEiT-3 at the ".pt" checkpoint: PURE masked-modelling pre-training,
    # without the fine-tuning the other timm beit checkpoints carry (the "_ft_"
    # suffix). Same architecture, same 1025 tokens at 512 px, same
    # normalisation: it enters the pre-training arm with no protocol change.
    "vit_b16_beit3": (lambda pre=True: TimmMap(
        _timm("beit3_base_patch16_224.pt", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "vit_b16_beit3"),
        768, 16, "imagenet", None),

    # ViT-B/16 WITH NO TRAINING AT ALL. Not a collapse candidate -- it is the
    # reference point for "a representation one should reject", with a known
    # probe outcome. If the participation ratio does NOT flag it, that bounds
    # the claim: it flags masked-modelling collapse, not poor representations
    # in general. The result is informative either way.
    "vit_b16_random": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_224.augreg_in21k_ft_in1k", norm="half",
              dynamic_img_size=True)(False), True, None, "vit_b16_random"),
        768, 16, "half", None),

    "vit_b16_clip": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_clip_224.openai", norm="clip",
              dynamic_img_size=True)(pre), True, None, "vit_b16_clip"),
        768, 16, "clip", None),

    # A larger MaxViT: is the rank anomaly the architecture's or the tiny
    # size's?
    "maxvit_s": (lambda pre=True: TimmMap(
        _timm("maxvit_small_tf_512.in1k")(pre), False, (512,), "maxvit_s"),
        768, 32, "half", (512,)),

    # convolution from another family: depthwise separable, squeeze-excite,
    # compound scaling -- neither ResNet nor ConvNeXt
    "efficientnet_b3": (lambda pre=True: TimmMap(
        _timm("efficientnet_b3.ra2_in1k", norm="imagenet")(pre), False, None,
        "efficientnet_b3"),
        1536, 32, "imagenet", None),

    # --- third batch: NINE objectives on the SAME ViT-B/16 -----------------
    # The design answers the objection that "objective" might be a label for a
    # confounded bundle (objective + corpus + recipe). Here there are:
    #   4 SUPERVISED ones from different recipes and corpora (augreg2 21k->1k,
    #     orig 21k, augreg 1k, DeiT3 1k) -- if the recipe were the factor, they
    #     would spread as much as distinct objectives do;
    #   2 SELF-SUPERVISED ones with different objectives (DINO, MAE);
    #   3 WITH LANGUAGE, two of them the SAME objective on different corpora
    #     (CLIP OpenAI against CLIP LAION-2B) -- which isolates corpus from
    #     objective.
    "vit_b16_sup21k": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_224.orig_in21k", dynamic_img_size=True)(pre),
        True, None, "vit_b16_sup21k"), 768, 16, "half", None),

    "vit_b16_sup1k": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_224.augreg_in1k", dynamic_img_size=True)(pre),
        True, None, "vit_b16_sup1k"), 768, 16, "half", None),

    "vit_b16_deit3": (lambda pre=True: TimmMap(
        _timm("deit3_base_patch16_224.fb_in1k", norm="imagenet",
              dynamic_img_size=True)(pre), True, None, "vit_b16_deit3"),
        768, 16, "imagenet", None),

    "vit_b16_clip2b": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_clip_224.laion2b", norm="clip",
              dynamic_img_size=True)(pre), True, None, "vit_b16_clip2b"),
        768, 16, "clip", None),

    "vit_b16_siglip": (lambda pre=True: TimmMap(
        _timm("vit_base_patch16_siglip_224.webli", dynamic_img_size=True)(pre),
        True, None, "vit_b16_siglip"), 768, 16, "half", None),

    # --- controls of the explainability study ------------------------------
    # Each one breaks ONE confounder of the original ResNet/MaxViT/ViT
    # comparison:
    #   convnext_t  convolution with a modern recipe -> is the ResNet's
    #               behaviour the convolution's or the 2015 recipe's?
    #   swin_t      hierarchical with LOCAL window attention only, no grid
    #               -> does MaxViT's invariance to arrangement come from the
    #                  hierarchy or from the grid attention?
    #   vit_s16     same family as ViT-B, but IN-1k and 22 M instead of IN-21k
    #               and 88 M -> is the ViT's behaviour global attention's or
    #               the ten-times-larger corpus's?
    "convnext_t": (lambda pre=True: TimmMap(
        _timm("convnext_tiny.fb_in1k", norm="imagenet")(pre), False, None,
        "convnext_t"),
        768, 32, "imagenet", None),

    "swin_t": (lambda pre=True: TimmMap(
        _timm("swin_tiny_patch4_window7_224.ms_in1k", norm="imagenet",
              img_size=512)(pre), False, (512,), "swin_t", nhwc=True),
        768, 32, "imagenet", (512,)),

    "vit_s16": (lambda pre=True: TimmMap(
        _timm("vit_small_patch16_224.augreg_in1k", dynamic_img_size=True)(pre),
        True, None, "vit_s16"),
        384, 16, "half", None),
}


def _truncate(m, n_blocks):
    """Keep the first n transformer blocks and discard the rest.

    Depth pruning with no retraining at all: it just shortens the block list.
    timm's forward_features applies the final norm after them, so the output
    stays normalised. Each ViT-H block has 19.7 M parameters, so truncating at
    12 takes it from 632 M to 238 M.

    A motivation beyond the parameter count: the last layers of a transformer
    specialise in the pre-training task, while the intermediate ones carry more
    general visual structure. For a domain far from ImageNet, truncating may
    transfer better, not merely cheaper.
    """
    if n_blocks > len(m.blocks):
        raise SystemExit(f"asked for {n_blocks} blocks, the model has "
                         f"{len(m.blocks)}")
    m.blocks = nn.Sequential(*list(m.blocks)[:n_blocks])
    return m


def _register_truncated_ijepa(n_blocks):
    """Add ijepa_h16_dN to the registry, with N of the 32 blocks."""
    alias = f"ijepa_h16_d{n_blocks}"
    BACKBONES[alias] = (
        lambda pre=True, _n=n_blocks, _a=alias: TimmMap(
            _truncate(_timm("vit_huge_patch16_gap_448.in1k_ijepa",
                            norm="imagenet", dynamic_img_size=True)(pre), _n),
            True, None, _a),
        1280, 16, "imagenet", None)


for _n in (8, 12, 16, 24):
    _register_truncated_ijepa(_n)


def build_backbone(name, pre=True, trainable=False):
    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone: {name}. "
                         f"Available: {sorted(BACKBONES)}")
    factory, channels, _, norm, _ = BACKBONES[name]
    m = factory(pre)
    for p in m.parameters():
        p.requires_grad = bool(trainable)
    return (m.train() if trainable else m.eval()), channels, norm


_NORMS = {"imagenet": NORM_IMAGENET, "half": NORM_HALF, "clip": NORM_CLIP}


def normalizer(name):
    return _NORMS[BACKBONES[name][3]]


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class ClassHead(nn.Module):
    """Global pooling, one hidden layer and two outputs.

    The same shape as the dev_harness head (dim -> k -> 2), so that the
    comparison with that battery is about backbone and regime, not about the
    head.
    """

    def __init__(self, channels, k=6, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, k), nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(k, 2))

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class BoxHead(nn.Module):
    """Reduce channels, resample to a fixed grid and regress the box.

    The fixed grid makes the parameter count independent of the resolution, and
    the flatten preserves each cell's position -- which is what allows
    regressing a coordinate, something global pooling would not allow.
    """

    def __init__(self, channels, grid=8, reduce=32, hidden=256, dropout=0.1):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, reduce, 1), nn.GELU(),
            nn.Conv2d(reduce, reduce, 3, padding=1), nn.GELU())
        self.grid = grid
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(reduce * grid * grid, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 4))

    def forward(self, x):
        return torch.sigmoid(self.mlp(
            F.adaptive_avg_pool2d(self.reduce(x), self.grid)))


class PCAReduction(nn.Module):
    """A FIXED linear projection, fitted by PCA on the TRAINING features.

    It exists to fix a measured defect: with the encoder frozen, the head
    receives 1280 features and training has 209 images. With more features than
    samples, a linear layer separates the training set by chance and does not
    generalise -- that is what dropped the frozen ViT-H arms to chance AUC.

    Projecting to k dimensions with k < n_samples removes that regime. The
    projection is fitted on the TRAINING set only and then frozen; fitting it
    on the whole set would be leakage, of the same kind dev_harness avoids by
    computing the feature mean and standard deviation only on the fold's
    training part.

    The whitening (dividing by the square root of the variance) puts the
    components on the same scale, which is the equivalent of dev_harness's
    standardisation.
    """

    def __init__(self, components, mean, scale):
        super().__init__()
        self.register_buffer("W", torch.as_tensor(components,
                                                  dtype=torch.float32))
        self.register_buffer("mu", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(scale, dtype=torch.float32))
        self.n_out = self.W.shape[0]

    def forward(self, x):
        x = x - self.mu.view(1, -1, 1, 1)
        x = torch.einsum("nchw,kc->nkhw", x, self.W)
        return x / self.sd.view(1, -1, 1, 1).clamp_min(1e-6)


def fit_pca(backbone, loader, k, dev, max_vectors=200_000):
    """Fit the PCA on the training features, one spatial position per vector.

    Every cell of the map becomes a sample of dimension C. That way the same
    projection serves both heads: the class head, which pools afterwards, and
    the box head, which needs the whole map.
    """
    from sklearn.decomposition import PCA

    backbone.eval()
    blocks = []
    with torch.no_grad():
        for b in loader:
            m = backbone(b["img"].to(dev))
            blocks.append(m.permute(0, 2, 3, 1).reshape(
                -1, m.shape[1]).float().cpu().numpy())
    X = np.concatenate(blocks, 0)

    if len(X) > max_vectors:
        idx = np.random.RandomState(0).choice(len(X), max_vectors,
                                              replace=False)
        X = X[idx]
    if k > min(X.shape):
        raise SystemExit(f"a PCA with {k} components needs more than "
                         f"{min(X.shape)} dimensions")

    p = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(X)
    explained = float(p.explained_variance_ratio_.sum())
    print(f"    PCA {X.shape[1]} -> {k} | {len(X)} vectors | "
          f"explained variance {explained:.1%}", flush=True)
    return (PCAReduction(p.components_, p.mean_,
                         np.sqrt(p.explained_variance_)), explained)


class Multitask(nn.Module):
    def __init__(self, backbone, channels, frozen, k=6, bn_frozen=True,
                 reduction=None, box_grid=8, class_dropout=0.0):
        super().__init__()
        self.backbone, self.frozen, self.bn_frozen = backbone, frozen, bn_frozen
        self.reduction = reduction
        if reduction is not None:
            channels = reduction.n_out
        self.cls = ClassHead(channels, k, dropout=class_dropout)
        self.box = BoxHead(channels, grid=box_grid)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        elif mode and self.bn_frozen:
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        if self.frozen:
            self.backbone.eval()
            with torch.no_grad():
                fmap = self.backbone(x)
                if self.reduction is not None:
                    fmap = self.reduction(fmap)
        else:
            fmap = self.backbone(x)
            if self.reduction is not None:
                fmap = self.reduction(fmap)
        return self.cls(fmap), self.box(fmap)

    def groups(self, lr, backbone_lr, wd):
        bb = [p for n, p in self.named_parameters()
              if n.startswith("backbone.") and p.requires_grad]
        heads = [p for n, p in self.named_parameters()
                 if not n.startswith(("backbone.", "reduction."))
                 and p.requires_grad]
        g = [dict(params=heads, lr=lr, weight_decay=wd, name="heads")]
        if bb:
            g.append(dict(params=bb, lr=backbone_lr, weight_decay=wd,
                          name="backbone"))
        return g

    def trainable_state(self):
        t = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in t}

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Boxes: loss and metrics
# ---------------------------------------------------------------------------

def to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou_loss(pred, target, eps=1e-7):
    """1 - GIoU. GIoU because plain IoU has zero gradient when there is no
    overlap, and early in training there never is."""
    a, b = to_xyxy(pred), to_xyxy(target)
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda z: ((z[..., 2] - z[..., 0]).clamp(min=0)
                      * (z[..., 3] - z[..., 1]).clamp(min=0))   # noqa: E731
    union = area(a) + area(b) - inter
    iou = inter / (union + eps)
    LT = torch.minimum(a[..., :2], b[..., :2])
    RB = torch.maximum(a[..., 2:], b[..., 2:])
    env = (RB - LT).clamp(min=0).prod(-1)
    return 1.0 - (iou - (env - union) / (env + eps))


def box_loss(pred, target, mask, w_l1=5.0, w_giou=2.0):
    """Only on images with an annotated lesion: no lesion, no position target."""
    m = mask.bool()
    if not m.any():
        return pred.sum() * 0.0
    p, t = pred[m], target[m]
    return (w_l1 * F.l1_loss(p, t, reduction="none").sum(-1)
            + w_giou * giou_loss(p, t)).mean()


def _iou_np(pred, gt):
    """IoU between pred (N,4) and gt (N,K,4) -> (N,K), all in (cx,cy,w,h)."""
    f = lambda b: np.stack([b[..., 0] - b[..., 2] / 2,      # noqa: E731
                            b[..., 1] - b[..., 3] / 2,
                            b[..., 0] + b[..., 2] / 2,
                            b[..., 1] + b[..., 3] / 2], -1)
    p, g = f(pred)[:, None, :], f(gt)
    lt, rb = np.maximum(p[..., :2], g[..., :2]), np.minimum(p[..., 2:], g[..., 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda b: (np.clip(b[..., 2] - b[..., 0], 0, None)    # noqa: E731
                      * np.clip(b[..., 3] - b[..., 1], 0, None))
    return inter / np.maximum(area(p) + area(g) - inter, 1e-7)


def iou_per_image(pred, gt, valid):
    """Best IoU per image (nan where there is no lesion).

    One predicted box against possibly several lesions: the highest IoU
    counts -- hitting any lesion counts.
    """
    pred = np.asarray(pred, float).reshape(-1, 4)
    gt, valid = np.asarray(gt, float), np.asarray(valid, bool)
    ious = _iou_np(pred, gt)
    ious[~valid] = -1.0
    best = ious.max(1)
    best[~valid.any(1)] = np.nan
    return best


def box_metrics(pred, gt, valid):
    """Mean IoU, hits per threshold and centre inside the lesion.

    centre_in_lesion is the classic criterion for mass detection in
    mammography: the predicted centre falls inside some annotated lesion.
    """
    pred = np.asarray(pred, float).reshape(-1, 4)
    gt, valid = np.asarray(gt, float), np.asarray(valid, bool)
    has = valid.any(1)
    if has.sum() == 0:
        return dict(iou=np.nan, iou_median=np.nan, hit50=np.nan,
                    hit25=np.nan, centre=np.nan, n_lesion=0)
    p, g, v = pred[has], gt[has], valid[has]
    best = iou_per_image(p, g, v)
    f = lambda b: np.stack([b[..., 0] - b[..., 2] / 2,      # noqa: E731
                            b[..., 1] - b[..., 3] / 2,
                            b[..., 0] + b[..., 2] / 2,
                            b[..., 1] + b[..., 3] / 2], -1)
    gx = f(g)
    c = p[:, :2][:, None, :]
    inside = ((c >= gx[..., :2]) & (c <= gx[..., 2:])).all(-1) & v
    return dict(iou=float(best.mean()), iou_median=float(np.median(best)),
                hit50=float((best >= 0.5).mean()),
                hit25=float((best >= 0.25).mean()),
                centre=float(inside.any(1).mean()), n_lesion=int(has.sum()))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_boxes(txt):
    if txt is None or (isinstance(txt, float) and np.isnan(txt)) or txt == "":
        return np.zeros((0, 4), np.float32)
    it = json.loads(txt) if isinstance(txt, str) else list(txt)
    if not it:
        return np.zeros((0, 4), np.float32)
    return np.asarray([[b["cx"], b["cy"], b["w"], b["h"]] for b in it],
                      np.float32)


class Mammograms(Dataset):
    """PNGs with a label and boxes. Geometric augmentation moves the box too.

    The augmentation does not consult the label: P(flipped|malignant) =
    P(flipped|benign). This is correction #2 of the audit report, and it holds
    here just the same.
    """

    def __init__(self, d, imgs, size, norm, augment=False):
        self.global_idx = d.index.to_numpy()
        self.d = d.reset_index(drop=True)
        self.imgs, self.size, self.augment = imgs, size, augment
        self.tf = transforms.Compose([transforms.ToTensor(), norm])
        # `caixas` is a column of the external INbreast table (see the module
        # docstring); its name cannot be translated here.
        self.boxes = [read_boxes(c) for c in self.d.caixas]

    def __len__(self):
        return len(self.d)

    @staticmethod
    def _zoom(img, cx, scale, ox, oy):
        w, h = img.size
        cw, ch = int(w * scale), int(h * scale)
        x0, y0 = int((w - cw) * ox), int((h - ch) * oy)
        if len(cx):
            xa, xb = (cx[:, 0] - cx[:, 2] / 2) * w, (cx[:, 0] + cx[:, 2] / 2) * w
            ya, yb = (cx[:, 1] - cx[:, 3] / 2) * h, (cx[:, 1] + cx[:, 3] / 2) * h
            if not ((xa >= x0) & (xb <= x0 + cw)
                    & (ya >= y0) & (yb <= y0 + ch)).all():
                return img, cx          # would clip the lesion: give up
        rec = img.crop((x0, y0, x0 + cw, y0 + ch)).resize(
            (w, h), Image.Resampling.BILINEAR)
        if len(cx):
            cx = cx.copy()
            cx[:, 0] = (cx[:, 0] * w - x0) / cw
            cx[:, 1] = (cx[:, 1] * h - y0) / ch
            cx[:, 2] /= scale
            cx[:, 3] /= scale
        return rec, cx

    def _augment(self, img, cx):
        s = torch.rand(6).tolist()
        if s[0] < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(cx):
                cx = cx.copy()
                cx[:, 0] = 1.0 - cx[:, 0]
        if s[1] < 0.5:
            img, cx = self._zoom(img, cx, 0.85 + 0.15 * s[2], s[3], s[4])
        if s[5] < 0.8:
            img = ImageEnhance.Brightness(img).enhance(
                0.9 + 0.2 * torch.rand(1).item())
            img = ImageEnhance.Contrast(img).enhance(
                0.9 + 0.2 * torch.rand(1).item())
        return img, cx

    def __getitem__(self, i):
        r = self.d.iloc[i]
        img = Image.open(self.imgs / r.processed_image_filename).convert("L")
        cx = self.boxes[i]
        if self.augment:
            img, cx = self._augment(img, cx)
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)

        gt = np.zeros((MAX_BOXES, 4), np.float32)
        val = np.zeros(MAX_BOXES, bool)
        k = min(len(cx), MAX_BOXES)
        if k:
            gt[:k], val[:k] = np.clip(cx[:k], 0, 1), True
        return dict(img=self.tf(img.convert("RGB")),
                    y=torch.tensor(int(r.classification)),
                    box=torch.from_numpy(gt[0]),
                    has=torch.tensor(bool(val[0])),
                    gt=torch.from_numpy(gt), val=torch.from_numpy(val),
                    idx=torch.tensor(int(self.global_idx[i])))


def loaders(d, imgs, indices, cfg, seed):
    norm = normalizer(cfg["backbone"])
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, ix in indices.items():
        tr = name == "train"
        ds = Mammograms(d.iloc[ix], imgs, cfg["size"], norm,
                        augment=tr and cfg["augment"])
        out[name] = DataLoader(ds, batch_size=cfg["batch"], shuffle=tr,
                               generator=gen if tr else None,
                               drop_last=tr and len(ds) % cfg["batch"] == 1,
                               num_workers=cfg.get("workers", 4),
                               pin_memory=torch.cuda.is_available(),
                               persistent_workers=cfg.get("workers", 4) > 0)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def class_weights(y, dev):
    c = np.bincount(np.asarray(y, int), minlength=2).astype(float)
    c[c == 0] = 1.0
    return torch.tensor(c.sum() / (2.0 * c), dtype=torch.float32, device=dev)


def _scheduler(opt, epochs, steps, warmup):
    total = max(1, epochs * steps)
    wu = min(total - 1, max(0, warmup * steps))

    def f(p):
        if p < wu:
            return (p + 1) / max(1, wu)
        t = (p - wu) / max(1, total - wu)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def one_pass(model, dl, crit, opt, sch, dev, cfg, training):
    """One pass over the data.

    ``cfg["update"]`` decides when the optimiser steps:

    ``"batch"``
        one step per mini-batch, the default of stochastic training.
    ``"epoch"``
        the gradients of all mini-batches are ACCUMULATED and the optimiser
        takes ONE step at the end of the epoch. The resulting gradient is that
        of the whole training set, with no sampling noise -- and the mini-batch
        stops having any effect on the mathematics, becoming just a memory
        knob. Each loss is divided by the number of batches so that the sum is
        the mean over the set, not its sum.
    """
    model.train() if training else model.eval()
    per_epoch = training and cfg.get("update", "batch") == "epoch"
    n_batches = max(1, len(dl))
    total = dict(loss=0.0, cls=0.0, box=0.0)
    n = 0
    Y, S, P, G, V, I = [], [], [], [], [], []

    if per_epoch and opt is not None:
        opt.zero_grad(set_to_none=True)

    for b in dl:
        x = b["img"].to(dev, non_blocking=True)
        y, cx, has = b["y"].to(dev), b["box"].to(dev), b["has"].to(dev)
        with torch.set_grad_enabled(training):
            amp = cfg.get("amp", True) and dev.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logit, box = model(x)
            logit, box = logit.float(), box.float()
            lc = crit(logit, y)
            lb = box_loss(box, cx, has)
            loss = lc + cfg["box_weight"] * lb
        if training:
            if per_epoch:
                (loss / n_batches).backward()        # accumulate, do not step
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                if sch is not None:
                    sch.step()
        k = len(x)
        n += k
        total["loss"] += float(loss) * k
        total["cls"] += float(lc) * k
        total["box"] += float(lb) * k
        Y.append(y.cpu().numpy())
        S.append(torch.softmax(logit, 1)[:, 1].detach().cpu().numpy())
        P.append(box.detach().cpu().numpy())
        G.append(b["gt"].numpy())
        V.append(b["val"].numpy())
        I.append(b["idx"].numpy())

    if per_epoch and opt is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if sch is not None:
            sch.step()

    pred = dict(y=np.concatenate(Y), score=np.concatenate(S),
                box=np.concatenate(P), gt=np.concatenate(G),
                val=np.concatenate(V), idx=np.concatenate(I))
    return {k: v / max(n, 1) for k, v in total.items()}, pred


def summarise(pred, losses, threshold=0.5):
    m = dict(loss=losses["loss"], cls_loss=losses["cls"],
             box_loss=losses["box"])
    y, s = pred["y"], pred["score"]
    m["auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    m["bal"] = float(balanced_accuracy_score(y, (s > threshold).astype(int)))
    m.update(box_metrics(pred["box"], pred["gt"], pred["val"]))
    return m


def best_threshold(y, s):
    """Threshold by balanced accuracy, as in dev_harness."""
    if len(np.unique(y)) < 2:
        return 0.5
    cand = np.unique(np.round(s, 3))
    return float(cand[int(np.argmax(
        [balanced_accuracy_score(y, (s > c).astype(int)) for c in cand]))])


def selection_score(m, criterion):
    a = 0.0 if np.isnan(m["auc"]) else m["auc"]
    i = 0.0 if np.isnan(m["iou"]) else m["iou"]
    return {"auc": a, "iou": i, "auc+iou": a + i}[criterion]


def train(model, dls, y_train, cfg, dev, eval_test=True, log=None):
    """Train; epoch and threshold come ONLY from validation; test evaluated
    once."""
    model = model.to(dev)
    crit = nn.CrossEntropyLoss(weight=class_weights(y_train, dev))
    opt = torch.optim.AdamW(model.groups(cfg["lr"], cfg["backbone_lr"],
                                         cfg["wd"]))
    # With a per-epoch update there is ONE step per epoch, so the scheduler
    # counts epochs; with a per-batch update it counts batches.
    steps = 1 if cfg.get("update", "batch") == "epoch" else len(dls["train"])
    sch = _scheduler(opt, cfg["epochs"], steps, cfg["warmup"])

    hist = []
    best = dict(score=-float("inf"), auc=np.nan, epoch=-1, state=None,
                threshold=0.5, val=None)

    for ep in range(1, cfg["epochs"] + 1):
        pt, prt = one_pass(model, dls["train"], crit, opt, sch, dev, cfg, True)
        mt = summarise(prt, pt)
        pv, prv = one_pass(model, dls["val"], crit, None, None, dev, cfg, False)
        mv = summarise(prv, pv)
        mv["score"] = selection_score(mv, cfg["criterion"])
        hist.append(dict(epoch=ep, phase="train", **mt))
        hist.append(dict(epoch=ep, phase="val", **mv))
        if mv["score"] > best["score"]:
            best = dict(score=mv["score"], auc=mv["auc"], epoch=ep,
                        state=copy.deepcopy(model.trainable_state()),
                        threshold=best_threshold(prv["y"], prv["score"]),
                        val=prv)
        if log:
            log(ep, mt, mv)

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)

    test = prte = None
    if eval_test:
        pe, prte = one_pass(model, dls["test"], crit, None, None, dev, cfg,
                            False)
        test = summarise(prte, pe, threshold=best["threshold"])
        test["chosen_epoch"] = best["epoch"]

    return dict(hist=hist, test=test, epoch=best["epoch"],
                val_auc=best["auc"], threshold=best["threshold"],
                pred=dict(val=best["val"], test=prte))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def mean_box(d, indices):
    """The training set's mean box: the mandatory baseline for localisation.

    A model that only learned "the lesion is usually in the middle" reproduces
    this. An IoU that does not beat this box is no evidence of having localised
    anything -- it is the box analogue of the trivial baselines in the report.
    """
    sub = d.iloc[indices]
    # `tem_lesao` is a column of the external INbreast table (see the module
    # docstring); its name cannot be translated here.
    sub = sub[sub.tem_lesao]
    if not len(sub):
        return None
    return np.array([sub.cx.mean(), sub.cy.mean(), sub.w.mean(),
                     sub.h.mean()], float)


def box_targets(d, indices):
    """(gt, valid) in the format box_metrics expects: (N, MAX_BOXES, 4)."""
    sub = d.iloc[indices]
    gt = np.zeros((len(sub), MAX_BOXES, 4), np.float32)
    val = np.zeros((len(sub), MAX_BOXES), bool)
    for i, txt in enumerate(sub.caixas):
        c = read_boxes(txt)
        k = min(len(c), MAX_BOXES)
        if k:
            gt[i, :k], val[i, :k] = np.clip(c[:k], 0, 1), True
    return gt, val


def eval_fixed_box(box, d, indices):
    gt, val = box_targets(d, indices)
    return box_metrics(np.tile(box, (len(indices), 1)), gt, val)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def clustered_bootstrap(fn, groups, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI resampling PATIENTS, not images.

    Resampling images treats a patient's 4 views as independent and narrows the
    interval artificially. It is the same rule as cluster_bootstrap_ci in
    nn_fixed.py.
    """
    rng = np.random.RandomState(seed)
    unique = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in unique}
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([pos[g] for g in chosen])
        v = fn(ix)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def ci_auc(y, s, groups, **kw):
    def f(ix):
        if len(np.unique(y[ix])) < 2:
            return None
        return roc_auc_score(y[ix], s[ix])
    return clustered_bootstrap(f, groups, **kw)


def ci_iou(ious, groups, **kw):
    def f(ix):
        v = ious[ix]
        v = v[~np.isnan(v)]
        return v.mean() if len(v) else None
    return clustered_bootstrap(f, groups, **kw)


def ci_difference(fn_a, fn_b, groups, n_boot=2000, seed=0, alpha=0.05):
    """CI of the PAIRED difference: both arms on the same resampled patients."""
    rng = np.random.RandomState(seed)
    unique = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in unique}
    diffs = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([pos[g] for g in chosen])
        a, b = fn_a(ix), fn_b(ix)
        if a is not None and b is not None and not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    if len(diffs) < 20:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(diffs)),
            float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))))
