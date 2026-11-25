# Pre-registration of the held-out encoder prediction

This file exists so that the prediction behind the test in Section 3.2 of the
paper is **auditable**, rather than merely asserted. A reviewer asked for
exactly this.

## Evidence that the prediction precedes the measurement

| file | contents | last modified |
|---|---|---|
| `codigo/segundo_colapso.py` | the prediction, in the module docstring | **2026-11-05 15:12:37 −03** |
| `resultados/segundo_colapso.json` | the first recorded measurement | 2026-11-05 16:55:50 −03 |
| `resultados/log_segundo.txt` | the run log | (same run) |

The script holding the prediction is 1 h 43 min older than the first recorded
result. Neither held-out encoder appears in `testes.json` or `sonda.json`, the
files the paper's seventeen encoders come from.

### Provenance note

The timestamps above are those of the original Portuguese files, recorded
before this repository was translated to English. Those files were renamed in
commit `ea7bdd9`'s successor:

| original | current |
|---|---|
| `codigo/segundo_colapso.py` | `code/second_collapse.py` |
| `resultados/segundo_colapso.json` | `results/second_collapse.json` |
| `resultados/log_segundo.txt` | `logs/log_second.txt` |
| `resultados/testes.json` | `results/measures.json` |
| `resultados/sonda.json` | `results/probe.json` |

A rename rewrites the modification time, so the current files no longer carry
it. The pre-translation copies, with the docstring quoted below, remain in this
repository's history at commit `5589989`; `git show 5589989:codigo/segundo_colapso.py`
retrieves the original.

## The prediction, quoted verbatim from the docstring

> `vit_b16_beit3`   BEiT-3 base, ".pt" checkpoint: PURE masked modelling,
> without the fine-tuning the other timm beit checkpoints carry.
> Same architecture, same 1025 tokens at 512 px.
> **PREDICTION: participation ratio in MAE's range (2-5) and
> probe among the worst in the set.**
>
> `vit_b16_random`  ViT-B/16 with no training at all. NOT a collapse
> candidate: it is the reference point for "a representation
> one should reject", with a known probe outcome. If the ratio
> does NOT flag it, the claim is bounded -- it flags masked-modelling
> collapse, not poor representations in general.
> Informative either way.

(The alias was `vit_b16_aleatorio` in the original; only the name changed.)

## Outcome, against the prediction

The prediction for BEiT-3 **failed in three of the four domains**: the
participation ratio came out at 17.7–22.8 on DTD, Pet and SUN397, outside the
predicted 2–5 range, while the probe landed among the worst, as predicted. Only
on mammography (2.24) did both halves of the prediction hold.

The prediction for the untrained ViT-B/16 held in all four domains.

## How "among the worst" is reported

A held-out encoder is reported by **how many of the seventeen reference
encoders probe worse than it**, not by a rank inside a pool of eighteen. The
point of holding an encoder out is that it is not part of the set; folding it
back in to compute its own rank would undo that separation, and it would invent
a denominator the paper never defines. On DTD, Pet and SUN397 only MAE probes
worse than BEiT-3; on mammography three of the seventeen do.

## Operational flag rule

An encoder is **flagged** in a domain when its participation ratio, measured by
the same protocol (`ci_sub`, $m = 100$ over 200 images), falls **below the
lowest value among the non-collapsed reference encoders in that domain** — that
is, below the minimum over the seventeen excluding MAE: 4.93 on mammography,
9.78 on DTD, 11.35 on Pet and 10.43 on SUN397.

The rule is evaluated domain by domain. "Collapse" is therefore an operational
property of the geometry measured **in that domain**, not an intrinsic,
global property of the checkpoint.
