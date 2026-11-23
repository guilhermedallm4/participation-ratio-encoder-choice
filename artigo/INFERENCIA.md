# Item 2 — intervalo contra valor-p

| conjunto | k | rho | p_perm | limiar | IC com reposição | IC jackknife | média do bootstrap |
|---|---|---|---|---|---|---|---|
| todos os 17 | 17 | +0.410 | 0.007 | 0.31 | [+0.12; +0.68] | [+0.07; +0.67] | +0.411 |
| sem o MAE | 16 | +0.273 | 0.075 | 0.30 | [+0.01; +0.50] | [-0.01; +0.51] | +0.271 |
| sem MAE e os dois CLIP | 14 | +0.211 | 0.223 | 0.33 | [-0.07; +0.48] | [-0.10; +0.48] | +0.205 |
| 9 pré-treinos | 9 | +0.406 | 0.088 | 0.46 | [-0.11; +0.76] | [-0.18; +0.78] | +0.360 |
| 9 pré-treinos, sem MAE | 8 | +0.262 | 0.273 | 0.43 | [-0.17; +0.59] | [-0.16; +0.60] | +0.244 |
| 9 arquiteturas | 9 | +0.109 | 0.662 | 0.44 | [-0.27; +0.55] | [-0.34; +0.51] | +0.120 |

## Leitura

- `limiar` é o percentil 95 de |rho| sob permutação de extratores: a
  decisão coerente com p_perm é comparar |rho| contra ele.
- `IC com reposição` é o do artigo atual. `média do bootstrap` acima do
  `rho` observado é a assinatura do viés de duplicata numa estatística
  de posto: o extrator sorteado duas vezes entra com pares
  perfeitamente concordantes consigo mesmo.
- `IC jackknife` deixa um extrator de fora de cada vez, sem repetição,
  e é o procedimento compatível com a análise de sensibilidade.
