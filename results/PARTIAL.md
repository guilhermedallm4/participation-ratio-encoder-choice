# Review item 2.2 - width control, and the corrected rank ceiling

## 0. The ceiling, now with the correct m

The published point of the participation ratio comes from `ic_sub`,
which subsamples **without replacement at m = n/2 = 100**. The rank
ceiling is therefore **99**, not 199. The earlier check used 199 and
is corrected here.

- largest published PR (m = 100): **55.4** (ViT-B/16 sup. 21k, dtd) = **55.9 % of the ceiling of 99**
- median of the 68: 20.1 (20.3 %)
- above 50 % of the ceiling: 1 of the 68 pairs
- for reference, with all 200 (ceiling 199): max 77.1 (38.7 %), median 22.4

## 1. Width, actually controlled for

| set | k | rho(PR, probe) | rho(width, probe) | partial rho (PR, probe; given width) | p perm |
|---|---|---|---|---|---|
| all 17 | 17 | +0.410 | +0.170 | **+0.392** | 0.012 |
| without MAE | 16 | +0.273 | +0.189 | **+0.243** | 0.113 |
| without MAE and the two CLIP | 14 | +0.211 | +0.217 | **+0.173** | 0.327 |
| 9 architectures | 9 | +0.109 | +0.309 | **+0.049** | 0.848 |
| 9 pre-trainings | 9 | +0.406 | - | - | width constant (768 channels): nothing to control for |

## 2. The same thing as a regression

Probe = b0 + b1*PR + b2*width, everything standardised within the domain.

| set | k | b1 (PR) | p perm | b2 (width) |
|---|---|---|---|---|
| all 17 | 17 | +0.572 | 0.003 | +0.016 |
| without MAE | 16 | +0.248 | 0.121 | +0.061 |
| without MAE and the two CLIP | 14 | +0.153 | 0.405 | +0.077 |
| 9 architectures | 9 | +0.021 | 0.937 | +0.139 |

