# Review item 1 - is the participation ratio capped by n = 200?

## 1. Distance to the rank ceiling

The centred covariance of n images has rank at most n-1. With 200
images, no estimated PR can exceed **199**.

- largest PR observed across the 68 pairs: **77.1** (ViT-B/16 sup. 21k, dtd) = **38.7 %** of the ceiling
- median of the 68: 22.4 (11.3 % of the ceiling)
- smallest: 1.8 (ViT-B/16 MAE, mammography)

## 2. Does the order among encoders survive a smaller n?

Spearman between PR(m) and PR(200), across the 17 encoders, within
each domain.

| domain | m=50 | m=100 | m=150 |
|---|---|---|---|
| mammography | 0.995 | 0.998 | 0.998 |
| dtd | 0.990 | 1.000 | 1.000 |
| pet | 0.993 | 1.000 | 1.000 |
| sun397 | 0.995 | 0.995 | 0.998 |

### Magnitude: how much PR grows with n (median over the 68 pairs)

| m | PR(m)/PR(200), median | min | max |
|---|---|---|---|
| 50 | 0.742 | 0.456 | 0.995 |
| 100 | 0.897 | 0.722 | 0.991 |
| 150 | 0.963 | 0.888 | 1.009 |
| 200 | 1.000 | 1.000 | 1.000 |

### The collapsed checkpoint, on its own

| encoder | domain | PR(50) | PR(100) | PR(150) | PR(200) |
|---|---|---|---|---|---|
| ViT-B/16 MAE | mammography | 1.81 | 1.79 | 1.81 | 1.81 |
| ViT-B/16 MAE | dtd | 4.11 | 4.16 | 4.27 | 4.28 |
| ViT-B/16 MAE | pet | 3.28 | 3.31 | 3.26 | 3.34 |
| ViT-B/16 MAE | sun397 | 3.04 | 3.08 | 3.15 | 3.13 |
| ViT-B/16 sup. 1k | mammography | 7.54 | 7.91 | 8.04 | 8.17 |
| ViT-B/16 sup. 1k | dtd | 31.17 | 45.92 | 54.70 | 59.78 |
| ViT-B/16 sup. 1k | pet | 25.33 | 34.41 | 39.14 | 41.92 |
| ViT-B/16 sup. 1k | sun397 | 17.09 | 20.78 | 22.14 | 23.04 |

## 3. The main result with PR estimated from half the images

| set | k | rho with PR(200) | p | rho with PR(100) | p | rho with PR(50) | p |
|---|---|---|---|---|---|---|---|
| all 17 | 17 | +0.409 | 0.007 | +0.409 | 0.007 | +0.418 | 0.006 |
| without MAE | 16 | +0.267 | 0.083 | +0.272 | 0.078 | +0.281 | 0.066 |
| without MAE and the two CLIP | 14 | +0.198 | 0.245 | +0.209 | 0.220 | +0.227 | 0.189 |
| 9 pre-trainings | 9 | +0.409 | 0.087 | +0.406 | 0.092 | +0.422 | 0.079 |
| 9 architectures | 9 | +0.106 | 0.676 | +0.108 | 0.669 | +0.107 | 0.675 |

