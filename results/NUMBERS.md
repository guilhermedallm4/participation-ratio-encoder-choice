# Final numbers

Geometry grid: 68 pairs
Probe: 68 pairs | residual flow: 9 encoders

## 1. Share of the variance in log (model x domain)

| measure | set | model | domain | interaction |
|---|---|---|---|---|
| participation ratio | 17 encoders | 48.4% | 43.0% | 8.6% |
| participation ratio | 9 architectures | 35.8% | 52.3% | 12.0% |
| participation ratio | 9 objectives | 51.4% | 44.3% | 4.3% |
| field radius | 17 encoders | 81.0% | 12.0% | 7.0% |
| field radius | 9 architectures | 88.1% | 5.8% | 6.2% |
| field radius | 9 objectives | 59.7% | 36.2% | 4.1% |
| residual flow | 9 objectives | 98.0% | 0.5% | 1.5% |

## 2. Recipe or objective? (the 9 ViT-B/16, identical architecture)

| checkpoint | family | residual flow | particip. ratio (mammog.) |
|---|---|---|---|
| ViT-B/16 sup. 21k$\to$1k | supervised | 0.815 | 12.1 |
| ViT-B/16 sup. 21k | supervised | 0.760 | 6.0 |
| ViT-B/16 sup. 1k | supervised | 0.596 | 7.9 |
| ViT-B/16 DeiT3 | supervised | 0.633 | 10.7 |
| ViT-B/16 DINO | self-supervised | 0.587 | 7.3 |
| ViT-B/16 MAE | self-supervised | 0.649 | 1.8 |
| ViT-B/16 CLIP-OAI | language | 0.730 | 5.4 |
| ViT-B/16 CLIP-2B | language | 0.957 | 4.9 |
| ViT-B/16 SigLIP | language | 0.793 | 5.4 |

- spread WITHIN the 4 supervised: 0.219 (0.596-0.815)
- spread across ALL 9 objectives: 0.370
- decomposition, weighted per CHECKPOINT: between 48.7 %, within 51.3 %
- decomposition, weighted per FAMILY (1/F each): between 56.3 %, within 43.7 %
- SAME objective, different corpora (CLIP OpenAI x LAION-2B): 0.730 x 0.957, difference 0.227

## 3. Geometry x probe - permutation AT THE ENCODER LEVEL

| measure | set | k | rho | p | detectable \|rho\| |
|---|---|---|---|---|---|
| participation ratio | 17 encoders | 17 | +0.410 | 0.007 | 0.31 |
| participation ratio | 9 architectures | 9 | +0.109 | 0.662 | 0.44 |
| participation ratio | 9 objectives | 9 | +0.406 | 0.088 | 0.46 |
| field radius | 17 encoders | 17 | +0.198 | 0.256 | 0.33 |
| field radius | 9 architectures | 9 | +0.101 | 0.718 | 0.47 |
| field radius | 9 objectives | 9 | +0.559 | 0.009 | 0.47 |
| width (control) | 17 encoders | 17 | +0.170 | 0.347 | 0.34 |
| width (control) | 9 architectures | 9 | +0.309 | 0.249 | 0.48 |
| width (control) | 9 objectives | 9 | - | - | constant |

## 4. LaTeX rows for Table 1
```
ResNet-18 & 10.5 & 21.1 & 87 & 92 & 0.619 & 0.659 \\
ResNet-50 & 11.1 & 28.4 & 72 & 73 & 0.681 & 0.699 \\
ConvNeXt-T & 11.2 & 27.4 & 90 & 106 & 0.687 & 0.752 \\
EfficientNet-B3 & 10.4 & 30.9 & 71 & 108 & 0.702 & 0.778 \\
Swin-T & 11.5 & 30.7 & 162 & 141 & 0.635 & 0.760 \\
MaxViT-T & 11.8 & 13.2 & 111 & 143 & 0.689 & 0.755 \\
MaxViT-S & 7.5 & 10.5 & 110 & 149 & 0.681 & 0.768 \\
ViT-S/16 & 8.4 & 31.9 & 167 & 189 & 0.618 & 0.717 \\
ViT-B/16 sup. 21k$\to$1k & 12.1 & 35.0 & 162 & 202 & 0.577 & 0.755 \\
ViT-B/16 sup. 21k & 6.0 & 35.5 & 165 & 200 & 0.631 & 0.787 \\
ViT-B/16 sup. 1k & 7.9 & 33.7 & 145 & 202 & 0.654 & 0.737 \\
ViT-B/16 DeiT3 & 10.7 & 23.2 & 112 & 157 & 0.550 & 0.739 \\
ViT-B/16 DINO & 7.3 & 27.7 & 92 & 128 & 0.597 & 0.665 \\
ViT-B/16 MAE & 1.8 & 3.5 & 110 & 131 & 0.533 & 0.447 \\
ViT-B/16 CLIP-OAI & 5.4 & 21.7 & 115 & 161 & 0.621 & 0.674 \\
ViT-B/16 CLIP-2B & 4.9 & 20.8 & 108 & 158 & 0.577 & 0.710 \\
ViT-B/16 SigLIP & 5.4 & 24.4 & 102 & 155 & 0.586 & 0.738 \\
```

## 5. RGB control (no conversion to grayscale)

| checkpoint | gray (mammog.) | RGB (mammog.) | gray (nat.) | RGB (nat.) |
|---|---|---|---|---|
| ViT-B/16 sup. 21k$\to$1k | 12.1 | 12.1 | 35.0 | 36.7 |
| ViT-B/16 sup. 21k | 6.0 | 6.0 | 35.5 | 37.0 |
| ViT-B/16 sup. 1k | 7.9 | 7.9 | 33.7 | 39.1 |
| ViT-B/16 DeiT3 | 10.7 | 10.7 | 23.2 | 24.1 |
| ViT-B/16 DINO | 7.3 | 7.3 | 27.7 | 28.6 |
| ViT-B/16 MAE | 1.8 | 1.8 | 3.5 | 2.7 |
| ViT-B/16 CLIP-OAI | 5.4 | 5.4 | 21.7 | 23.4 |
| ViT-B/16 CLIP-2B | 4.9 | 4.9 | 20.8 | 21.7 |
| ViT-B/16 SigLIP | 5.4 | 5.4 | 24.4 | 27.0 |

