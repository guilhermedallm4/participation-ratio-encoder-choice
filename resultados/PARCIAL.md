# Item 2.2 — controle de largura, e o teto de posto corrigido

## 0. O teto, agora com o m correto

O ponto publicado da razão de participação vem de `ic_sub`, que
subamostra **sem reposição em m = n/2 = 100**. O teto de posto é
portanto **99**, não 199. A checagem anterior usou 199 e está corrigida.

- maior PR publicada (m = 100): **55.4** (ViT-B/16 sup. 21k, dtd) = **55.9 % do teto de 99**
- mediana das 68: 20.1 (20.3 %)
- acima de 50 % do teto: 1 dos 68 pares
- para referência, com os 200 (teto 199): máx 77.1 (38.7 %), mediana 22.4

## 1. A largura, controlada de verdade

| conjunto | k | ρ(PR, sonda) | ρ(largura, sonda) | ρ parcial (PR·sonda | largura) | p perm |
|---|---|---|---|---|---|
| todos os 17 | 17 | +0.410 | +0.170 | **+0.392** | 0.012 |
| sem o MAE | 16 | +0.273 | +0.189 | **+0.243** | 0.113 |
| sem MAE e os dois CLIP | 14 | +0.211 | +0.217 | **+0.173** | 0.327 |
| 9 arquiteturas | 9 | +0.109 | +0.309 | **+0.049** | 0.848 |
| 9 pré-treinos | 9 | +0.406 | — | — | largura constante (768 canais): nada a controlar |

## 2. A mesma coisa como regressão

Sonda = b0 + b1·PR + b2·largura, tudo padronizado dentro do domínio.

| conjunto | k | b1 (PR) | p perm | b2 (largura) |
|---|---|---|---|---|
| todos os 17 | 17 | +0.572 | 0.003 | +0.016 |
| sem o MAE | 16 | +0.248 | 0.121 | +0.061 |
| sem MAE e os dois CLIP | 14 | +0.153 | 0.405 | +0.077 |
| 9 arquiteturas | 9 | +0.021 | 0.937 | +0.139 |

