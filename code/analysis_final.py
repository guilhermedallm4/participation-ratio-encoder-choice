#!/usr/bin/env python
"""Final analysis: corrected statistics and the recipe-versus-objective test.

TWO CORRECTIONS relative to the earlier version.

1. PERMUTATION AT THE ENCODER LEVEL. Labels used to be shuffled independently
   within each domain, which destroys the repeated-measures structure: the same
   encoder appears in all four domains and its deviations are correlated across
   them. That null is "too random" and makes the test anticonservative. Here ONE
   permutation of encoders is applied to all four domains, preserving each
   encoder's profile. The price is power: with k encoders there are k!
   permutations, so the smallest reachable p is about 1/k!.

2. THE MEASURE IS CALLED THE PARTICIPATION RATIO. The expression
   (sum L)^2 / sum L^2 is the participation ratio; the "effective rank" of Roy
   and Vetterli (2007) is exp(-sum p_i log p_i), which is a different quantity.
   We compute both, so that the choice of name stops being an assumption.

AND THE NEW TEST: if "objective" were merely a label for a confounded bundle
(objective + corpus + recipe), then four SUPERVISED checkpoints from different
recipes and corpora ought to spread as much as distinct objectives do. We
measure that.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

DATASETS = ["mammography", "dtd", "pet", "sun397"]
ARCH = ["resnet18", "resnet50", "convnext_t", "efficientnet_b3", "swin_t",
        "maxvit_t", "maxvit_s", "vit_s16", "vit_b16"]
OBJECTIVE = ["vit_b16", "vit_b16_sup21k", "vit_b16_sup1k", "vit_b16_deit3",
             "vit_b16_dino", "vit_b16_mae", "vit_b16_clip", "vit_b16_clip2b",
             "vit_b16_siglip"]
FAMILY = {"vit_b16": "supervised", "vit_b16_sup21k": "supervised",
          "vit_b16_sup1k": "supervised", "vit_b16_deit3": "supervised",
          "vit_b16_dino": "self-supervised", "vit_b16_mae": "self-supervised",
          "vit_b16_clip": "language", "vit_b16_clip2b": "language",
          "vit_b16_siglip": "language"}
LABEL = {"resnet18": "ResNet-18", "resnet50": "ResNet-50",
         "convnext_t": "ConvNeXt-T", "efficientnet_b3": "EfficientNet-B3",
         "swin_t": "Swin-T", "maxvit_t": "MaxViT-T", "maxvit_s": "MaxViT-S",
         "vit_s16": "ViT-S/16", "vit_b16": "ViT-B/16 sup. 21k$\\to$1k",
         "vit_b16_sup21k": "ViT-B/16 sup. 21k",
         "vit_b16_sup1k": "ViT-B/16 sup. 1k", "vit_b16_deit3": "ViT-B/16 DeiT3",
         "vit_b16_dino": "ViT-B/16 DINO", "vit_b16_mae": "ViT-B/16 MAE",
         "vit_b16_clip": "ViT-B/16 CLIP-OAI",
         "vit_b16_clip2b": "ViT-B/16 CLIP-2B",
         "vit_b16_siglip": "ViT-B/16 SigLIP"}
ALL = ARCH + OBJECTIVE[1:]


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0)


def dec(M):
    L = np.log(M); g = L.mean()
    v_model = L.mean(1).var(ddof=0); v_domain = L.mean(0).var(ddof=0)
    v_inter = ((L - L.mean(1, keepdims=True)
                - L.mean(0, keepdims=True) + g) ** 2).mean()
    t = v_model + v_domain + v_inter
    return np.array([v_model, v_domain, v_inter]) / t


def perm_encoder(G, P, n=20000, seed=0):
    """ONE permutation of encoders applied to all four domains."""
    gx = np.concatenate(G)
    obs = spearmanr(gx, np.concatenate(P)).statistic
    rng = np.random.default_rng(seed)
    k = len(G[0])
    null = np.array([abs(spearmanr(gx, np.concatenate([p[q] for p in P])).statistic)
                     for q in (rng.permutation(k) for _ in range(n))])
    return obs, float((null >= abs(obs)).mean()), float(np.percentile(null, 95))


def main():
    r = Path("results")
    t = json.loads((r / "measures.json").read_text())
    pb = json.loads((r / "probe.json").read_text())
    geo = {(x["backbone"], x["dataset"]): x
           for bl in ("ci", "architectures", "batch2", "batch3")
           for x in t.get(bl, []) if "dataset" in x}
    fl = {}
    for bl in ("flow2", "flow3"):
        for x in t.get(bl, []):
            for mech, v in x["mech"].items():
                fl.setdefault(x["backbone"], {})[x["dataset"]] = v
    L = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)

    missing = [k for k in [(e, b) for e in ALL for b in DATASETS] if k not in geo]
    p(f"# Final numbers\n\nGeometry grid: {len(geo)} pairs"
      f"{'' if not missing else '  MISSING: ' + str(missing[:6])}")
    p(f"Probe: {len(pb)} pairs | residual flow: {len(fl)} encoders\n")

    # ---- 1. decomposition -------------------------------------------------
    p("## 1. Share of the variance in log (model x domain)\n")
    p("| measure | set | model | domain | interaction |")
    p("|---|---|---|---|---|")
    for lab, f, sets in [
            ("participation ratio", lambda e, b: geo[(e, b)]["rank"][0],
             [("17 encoders", ALL), ("9 architectures", ARCH),
              ("9 objectives", OBJECTIVE)]),
            ("field radius", lambda e, b: geo[(e, b)]["radius_px"][0],
             [("17 encoders", ALL), ("9 architectures", ARCH),
              ("9 objectives", OBJECTIVE)])]:
        for name, es in sets:
            d = dec(np.array([[f(e, b) for b in DATASETS] for e in es]))
            p(f"| {lab} | {name} | {d[0]:.1%} | {d[1]:.1%} | {d[2]:.1%} |")
    d = dec(np.array([[fl[e][b][0] for b in DATASETS]
                      for e in OBJECTIVE if e in fl]))
    p(f"| residual flow | 9 objectives | {d[0]:.1%} | {d[1]:.1%} | {d[2]:.1%} |\n")

    # ---- 2. recipe versus objective ---------------------------------------
    p("## 2. Recipe or objective? (the 9 ViT-B/16, identical architecture)\n")
    p("| checkpoint | family | residual flow | particip. ratio (mammog.) |")
    p("|---|---|---|---|")
    for e in OBJECTIVE:
        if e not in fl:
            continue
        fm = np.mean([fl[e][b][0] for b in DATASETS])
        p(f"| {LABEL[e]} | {FAMILY[e]} | {fm:.3f} | "
          f"{geo[(e, 'mammography')]['rank'][0]:.1f} |")
    sup = [e for e in OBJECTIVE if FAMILY[e] == "supervised" and e in fl]
    others = [e for e in OBJECTIVE if FAMILY[e] != "supervised" and e in fl]
    fs = np.array([np.mean([fl[e][b][0] for b in DATASETS]) for e in sup])
    fo = np.array([np.mean([fl[e][b][0] for b in DATASETS]) for e in others])
    p(f"\n- spread WITHIN the 4 supervised: {fs.max()-fs.min():.3f} "
      f"({fs.min():.3f}-{fs.max():.3f})")
    p(f"- spread across ALL 9 objectives: "
      f"{max(fs.max(),fo.max())-min(fs.min(),fo.min()):.3f}")
    # Family decomposition under BOTH weightings. The review asked for the
    # formula behind the second one, which used to exist only as a number
    # typed into the paper.
    mm = np.array([np.mean([np.log(fl[e][b][0]) for b in DATASETS])
                   for e in OBJECTIVE if e in fl])
    fam = np.array([FAMILY[e] for e in OBJECTIVE if e in fl])
    mf = {f: mm[fam == f].mean() for f in set(fam)}
    mfe = np.array([mf[f] for f in fam])
    v_between, v_within = ((mfe - mm.mean()) ** 2).mean(), ((mm - mfe) ** 2).mean()
    p(f"- decomposition, weighted per CHECKPOINT: between "
      f"{100*v_between/(v_between+v_within):.1f} %, "
      f"within {100*v_within/(v_between+v_within):.1f} %")
    fs = sorted(set(fam)); mw = np.mean([mf[f] for f in fs])
    v_between2 = np.mean([(mf[f] - mw) ** 2 for f in fs])
    v_within2 = np.mean([np.mean((mm[fam == f] - mf[f]) ** 2) for f in fs])
    p(f"- decomposition, weighted per FAMILY (1/F each): between "
      f"{100*v_between2/(v_between2+v_within2):.1f} %, "
      f"within {100*v_within2/(v_between2+v_within2):.1f} %")
    if "vit_b16_clip" in fl and "vit_b16_clip2b" in fl:
        a = np.mean([fl["vit_b16_clip"][b][0] for b in DATASETS])
        b_ = np.mean([fl["vit_b16_clip2b"][b][0] for b in DATASETS])
        p(f"- SAME objective, different corpora (CLIP OpenAI x LAION-2B): "
          f"{a:.3f} x {b_:.3f}, difference {abs(a-b_):.3f}")
    p("")

    # ---- 3. correlations ---------------------------------------------------
    p("## 3. Geometry x probe - permutation AT THE ENCODER LEVEL\n")
    p("| measure | set | k | rho | p | detectable \\|rho\\| |")
    p("|---|---|---|---|---|---|")
    measures = [("participation ratio", lambda e, b: geo[(e, b)]["rank"][0]),
                ("field radius", lambda e, b: geo[(e, b)]["radius_px"][0]),
                ("width (control)", lambda e, b: geo[(e, b)]["channels"])]
    for lab, f in measures:
        for name, es in [("17 encoders", ALL), ("9 architectures", ARCH),
                         ("9 objectives", OBJECTIVE)]:
            es = [e for e in es if all(f"{b}|{e}" in pb for b in DATASETS)]
            if len(es) < 4:
                continue
            vals = [[f(e, b) for e in es] for b in DATASETS]
            # A constant predictor (for instance the width across the 9
            # ViT-B/16, all with 768 channels) has zero variance: z() would
            # divide by zero and the test would return nan with p=0. There is
            # nothing to correlate.
            if any(np.std(v) == 0 for v in vals):
                p(f"| {lab} | {name} | {len(es)} | - | - | constant |")
                continue
            G = [z(v) for v in vals]
            P = [z([pb[f"{b}|{e}"]["value"] for e in es]) for b in DATASETS]
            rho, pv, det = perm_encoder(G, P)
            p(f"| {lab} | {name} | {len(es)} | {rho:+.3f} | {pv:.3f} | {det:.2f} |")
    p("")

    # ---- 4. the paper's table rows ----------------------------------------
    p("## 4. LaTeX rows for Table 1\n```")
    NAT = DATASETS[1:]
    for e in ALL:
        if (e, "mammography") not in geo or f"mammography|{e}" not in pb:
            continue
        pm = geo[(e, "mammography")]["rank"][0]
        pn = np.mean([geo[(e, b)]["rank"][0] for b in NAT])
        rm = geo[(e, "mammography")]["radius_px"][0]
        rn = np.mean([geo[(e, b)]["radius_px"][0] for b in NAT])
        sm = pb[f"mammography|{e}"]["value"]
        sn = np.mean([pb[f"{b}|{e}"]["value"] for b in NAT])
        p(f"{LABEL[e]} & {pm:.1f} & {pn:.1f} & {rm:.0f} & {rn:.0f} "
          f"& {sm:.3f} & {sn:.3f} \\\\")
    p("```\n")

    # ---- 5. RGB control ----------------------------------------------------
    if "rgb" in t:
        p("## 5. RGB control (no conversion to grayscale)\n")
        rgb = {(x["backbone"], x["dataset"]): x["rank"][0] for x in t["rgb"]}
        p("| checkpoint | gray (mammog.) | RGB (mammog.) | "
          "gray (nat.) | RGB (nat.) |")
        p("|---|---|---|---|---|")
        for e in OBJECTIVE:
            if (e, "mammography") not in rgb:
                continue
            p(f"| {LABEL[e]} | {geo[(e,'mammography')]['rank'][0]:.1f} | "
              f"{rgb[(e,'mammography')]:.1f} | "
              f"{np.mean([geo[(e,b)]['rank'][0] for b in NAT]):.1f} | "
              f"{np.mean([rgb[(e,b)] for b in NAT]):.1f} |")
        p("")

    Path("results/NUMBERS.md").write_text("\n".join(L) + "\n")
    print("\n-> results/NUMBERS.md", flush=True)


if __name__ == "__main__":
    main()
