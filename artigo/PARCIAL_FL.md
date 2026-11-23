# Correlação parcial: inferência condicional por Freedman–Lane

O parecer final está certo. O nulo anterior permutava razão e largura
**juntas** contra a sonda, o que destrói também a associação
largura–sonda. A hipótese nula condicional correta preserva essa
associação e embaralha apenas a parte da sonda que a largura não
explica. É o que Freedman–Lane faz, e é o que está abaixo.

Permutação no nível do extrator, a mesma nos quatro domínios, 20000 sorteios.

| conjunto | k | ρ parcial | **p Freedman–Lane** | p do nulo antigo | b1 | p F–L de b1 |
|---|---|---|---|---|---|---|
| todos os 17 | 17 | **+0.392** | **0.010** | 0.012 | +0.392 | 0.009 |
| sem o MAE | 16 | **+0.243** | **0.113** | 0.113 | +0.244 | 0.108 |
| sem MAE e os dois CLIP | 14 | **+0.173** | **0.320** | 0.327 | +0.173 | 0.310 |
| 9 arquiteturas | 9 | **+0.049** | **0.847** | 0.848 | +0.047 | 0.842 |
| 9 pré-treinos | 9 | — | — | — | — | largura constante (768 canais): nada a controlar |

O ponto estimado da parcial não muda --- ele nunca dependeu do nulo.
O que muda é o $p$, e é ele que o parecer questionou.

