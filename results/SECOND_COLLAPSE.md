# Prediction test: a second collapse

## 1. Where the two of them land

| encoder | mammography | dtd | pet | sun397 |
|---|---|---|---|---|
| **ViT-B/16 BEiT-3 (.pt)** - PR | 2.24 | 22.83 | 20.36 | 17.69 |
| ViT-B/16 BEiT-3 (.pt) - probe | 0.577 | 0.419 | 0.427 | 0.771 |
| **ViT-B/16 untrained** - PR | 1.27 | 1.55 | 1.67 | 1.64 |
| ViT-B/16 untrained - probe | 0.499 | 0.075 | 0.060 | 0.191 |
| **ViT-B/16 MAE** - PR | 1.81 | 4.20 | 3.32 | 3.11 |
| ViT-B/16 MAE - probe | 0.533 | 0.391 | 0.322 | 0.627 |

Reference, the seventeen of the paper (PR):

| domain | minimum without MAE | median | maximum |
|---|---|---|---|
| mammography | 4.93 | 9.41 | 12.12 |
| dtd | 9.78 | 31.90 | 55.35 |
| pet | 11.35 | 26.97 | 37.88 |
| sun397 | 10.43 | 20.31 | 25.46 |

## 2. The prediction

Criterion, the same as the paper's: *flagged* = PR below the minimum
over the remaining encoders in that domain, by a margin that is not
measurement noise.

| encoder | domain | PR | minimum of the others | ratio | flagged? | encoders probing worse |
|---|---|---|---|---|---|---|
| ViT-B/16 BEiT-3 (.pt) | mammography | 2.24 | 4.93 | 2.2x | **yes** | 3 of 17 |
| ViT-B/16 BEiT-3 (.pt) | dtd | 22.83 | 9.78 | 0.4x | no | 1 of 17 |
| ViT-B/16 BEiT-3 (.pt) | pet | 20.36 | 11.35 | 0.6x | no | 1 of 17 |
| ViT-B/16 BEiT-3 (.pt) | sun397 | 17.69 | 10.43 | 0.6x | no | 1 of 17 |
| ViT-B/16 untrained | mammography | 1.27 | 4.93 | 3.9x | **yes** | 0 of 17 |
| ViT-B/16 untrained | dtd | 1.55 | 9.78 | 6.3x | **yes** | 0 of 17 |
| ViT-B/16 untrained | pet | 1.67 | 11.35 | 6.8x | **yes** | 0 of 17 |
| ViT-B/16 untrained | sun397 | 1.64 | 10.43 | 6.4x | **yes** | 0 of 17 |
| ViT-B/16 MAE | mammography | 1.81 | 4.93 | 2.7x | **yes** | 0 of 16 |
| ViT-B/16 MAE | dtd | 4.20 | 9.78 | 2.3x | **yes** | 0 of 16 |
| ViT-B/16 MAE | pet | 3.32 | 11.35 | 3.4x | **yes** | 0 of 16 |
| ViT-B/16 MAE | sun397 | 3.11 | 10.43 | 3.4x | **yes** | 0 of 16 |

## 3. If they joined the set (transparency, this is not the test)

| set | k | rho | p |
|---|---|---|---|
| the 17 of the paper | 17 | +0.410 | 0.007 |
| 17 + BEiT-3 | 18 | +0.469 | 0.002 |
| 17 + both new ones | 19 | +0.532 | 0.000 |
| 17 + BEiT-3, without MAE | 17 | +0.362 | 0.018 |

