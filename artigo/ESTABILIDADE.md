# Item 1 — a razão de participação está limitada por n = 200?

## 1. Distância até o teto de posto

A covariância centrada de n imagens tem posto no máximo n−1. Com 200
imagens, nenhuma PR estimada pode passar de **199**.

- maior PR observada nos 68 pares: **77.1** (ViT-B/16 sup. 21k, dtd) = **38.7 %** do teto
- mediana das 68: 22.4 (11.3 % do teto)
- menor: 1.8 (ViT-B/16 MAE, mamografia)

## 2. A ordem entre extratores sobrevive à redução de n?

Spearman entre PR(m) e PR(200), nos 17 extratores, dentro de cada domínio.

| domínio | m=50 | m=100 | m=150 |
|---|---|---|---|
| mamografia | 0.995 | 0.998 | 0.998 |
| dtd | 0.990 | 1.000 | 1.000 |
| pet | 0.993 | 1.000 | 1.000 |
| sun397 | 0.995 | 0.995 | 0.998 |

### Magnitude: quanto a PR cresce com n (mediana entre os 68 pares)

| m | PR(m)/PR(200), mediana | mín | máx |
|---|---|---|---|
| 50 | 0.742 | 0.456 | 0.995 |
| 100 | 0.897 | 0.722 | 0.991 |
| 150 | 0.963 | 0.888 | 1.009 |
| 200 | 1.000 | 1.000 | 1.000 |

### O checkpoint colapsado, separado

| extrator | domínio | PR(50) | PR(100) | PR(150) | PR(200) |
|---|---|---|---|---|---|
| ViT-B/16 MAE | mamografia | 1.81 | 1.79 | 1.81 | 1.81 |
| ViT-B/16 MAE | dtd | 4.11 | 4.16 | 4.27 | 4.28 |
| ViT-B/16 MAE | pet | 3.28 | 3.31 | 3.26 | 3.34 |
| ViT-B/16 MAE | sun397 | 3.04 | 3.08 | 3.15 | 3.13 |
| ViT-B/16 sup. 1k | mamografia | 7.54 | 7.91 | 8.04 | 8.17 |
| ViT-B/16 sup. 1k | dtd | 31.17 | 45.92 | 54.70 | 59.78 |
| ViT-B/16 sup. 1k | pet | 25.33 | 34.41 | 39.14 | 41.92 |
| ViT-B/16 sup. 1k | sun397 | 17.09 | 20.78 | 22.14 | 23.04 |

## 3. O resultado principal com a PR estimada em metade das imagens

| conjunto | k | ρ com PR(200) | p | ρ com PR(100) | p | ρ com PR(50) | p |
|---|---|---|---|---|---|---|---|
| todos os 17 | 17 | +0.409 | 0.007 | +0.409 | 0.007 | +0.418 | 0.006 |
| sem o MAE | 16 | +0.267 | 0.083 | +0.272 | 0.078 | +0.281 | 0.066 |
| sem MAE e os dois CLIP | 14 | +0.198 | 0.245 | +0.209 | 0.220 | +0.227 | 0.189 |
| 9 pré-treinos | 9 | +0.409 | 0.087 | +0.406 | 0.092 | +0.422 | 0.079 |
| 9 arquiteturas | 9 | +0.106 | 0.676 | +0.108 | 0.669 | +0.107 | 0.675 |

