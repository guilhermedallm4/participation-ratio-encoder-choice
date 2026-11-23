# Teste de predição: um segundo colapso

## 1. Onde os dois caem

| extrator | mamografia | dtd | pet | sun397 |
|---|---|---|---|---|
| **ViT-B/16 sem treino** — PR | 1.27 | 1.55 | 1.67 | 1.64 |
| ViT-B/16 sem treino — sonda | 0.499 | 0.075 | 0.060 | 0.191 |
| **ViT-B/16 BEiT-3 (.pt)** — PR | 2.24 | 22.83 | 20.36 | 17.69 |
| ViT-B/16 BEiT-3 (.pt) — sonda | 0.577 | 0.419 | 0.427 | 0.771 |
| **ViT-B/16 MAE** — PR | 1.81 | 4.20 | 3.32 | 3.11 |
| ViT-B/16 MAE — sonda | 0.533 | 0.391 | 0.322 | 0.627 |

Referência, os 17 do artigo (PR):

| domínio | mínimo sem MAE | mediana | máximo |
|---|---|---|---|
| mamografia | 4.93 | 9.41 | 12.12 |
| dtd | 9.78 | 31.90 | 55.35 |
| pet | 11.35 | 26.97 | 37.88 |
| sun397 | 10.43 | 20.31 | 25.46 |

## 2. A predição

Critério, o mesmo do artigo: *flagrado* = PR abaixo do mínimo dos demais
extratores naquele domínio, por uma margem que não seja de medida.

| extrator | domínio | PR | mínimo dos outros | razão | flagrado? | posto na sonda |
|---|---|---|---|---|---|---|
| ViT-B/16 sem treino | mamografia | 1.27 | 4.93 | 3.9× | **sim** | 1º de 18 |
| ViT-B/16 sem treino | dtd | 1.55 | 9.78 | 6.3× | **sim** | 1º de 18 |
| ViT-B/16 sem treino | pet | 1.67 | 11.35 | 6.8× | **sim** | 1º de 18 |
| ViT-B/16 sem treino | sun397 | 1.64 | 10.43 | 6.4× | **sim** | 1º de 18 |
| ViT-B/16 BEiT-3 (.pt) | mamografia | 2.24 | 4.93 | 2.2× | **sim** | 4º de 18 |
| ViT-B/16 BEiT-3 (.pt) | dtd | 22.83 | 9.78 | 0.4× | não | 2º de 18 |
| ViT-B/16 BEiT-3 (.pt) | pet | 20.36 | 11.35 | 0.6× | não | 2º de 18 |
| ViT-B/16 BEiT-3 (.pt) | sun397 | 17.69 | 10.43 | 0.6× | não | 2º de 18 |
| ViT-B/16 MAE | mamografia | 1.81 | 4.93 | 2.7× | **sim** | 1º de 17 |
| ViT-B/16 MAE | dtd | 4.20 | 9.78 | 2.3× | **sim** | 1º de 17 |
| ViT-B/16 MAE | pet | 3.32 | 11.35 | 3.4× | **sim** | 1º de 17 |
| ViT-B/16 MAE | sun397 | 3.11 | 10.43 | 3.4× | **sim** | 1º de 17 |

## 3. Se entrassem no conjunto (transparência, não é o teste)

| conjunto | k | ρ | p |
|---|---|---|---|
| os 17 do artigo | 17 | +0.410 | 0.007 |
| 17 + BEiT-3 | 18 | +0.469 | 0.002 |
| 17 + os dois novos | 19 | +0.532 | 0.001 |
| 17 + BEiT-3, sem o MAE | 17 | +0.362 | 0.018 |

