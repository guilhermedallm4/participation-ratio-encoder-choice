# Registro prévio da predição — teste dos extratores retidos

Este arquivo existe para que a predição do teste de §3.2 seja **auditável**, e
não apenas afirmada no texto. O parecer pediu exatamente isso.

## Evidência de que a predição antecede a medição

| arquivo | conteúdo | modificado em |
|---|---|---|
| `codigo/segundo_colapso.py` | a predição, no *docstring* do módulo | **2026-11-05 15:12:37 −03** |
| `resultados/segundo_colapso.json` | a primeira medição gravada | 2026-11-05 16:55:50 −03 |
| `resultados/log_segundo.txt` | o log da execução | (mesma execução) |

O script que contém a predição é 1 h 43 min anterior ao primeiro resultado
gravado. Nenhum dos dois extratores retidos aparece em `testes.json` ou em
`sonda.json`, os arquivos de onde saem os dezessete do artigo.

## A predição, copiada literalmente do docstring

> `vit_b16_beit3`      BEiT-3 base, checkpoint ".pt": modelagem mascarada PURA,
> sem o ajuste fino que os demais beit do timm carregam.
> Mesma arquitetura, mesmas 1025 fichas a 512 px.
> **PREDICAO: razao de participacao na faixa do MAE (2-5) e
> sonda entre as piores do conjunto.**
>
> `vit_b16_aleatorio`  ViT-B/16 sem treino nenhum. NAO e candidato a colapso: e
> o referencial de "representacao que se deve rejeitar",
> com resposta conhecida na sonda. Se a razao NAO o
> sinalizar, a afirmacao fica delimitada -- ela flagra
> colapso de modelagem mascarada, nao representacao ruim em
> geral. Interessa nos dois sentidos.

## Resultado, contra a predição

A predição para o BEiT-3 **falhou em três dos quatro domínios**: a razão de
participação ficou em 17,7–22,8 em DTD, Pet e SUN397, fora da faixa 2–5
predita, enquanto a sonda ficou entre as piores, como predito. Só na mamografia
(2,24) a predição se cumpriu nas duas metades.

A predição para o ViT-B/16 sem treino cumpriu-se nos quatro domínios.

## Regra operacional de sinalização

Um extrator é **sinalizado** num domínio quando sua razão de participação,
medida pelo mesmo protocolo (`ic_sub`, $m = 100$ sobre 200 imagens), fica
**abaixo do menor valor observado entre os extratores de referência não
colapsados naquele domínio** — isto é, abaixo do mínimo dos dezessete excluído
o MAE: 4,93 na mamografia, 9,78 no DTD, 11,35 no Pet e 10,43 no SUN397.

A regra é avaliada domínio a domínio. "Colapso" é, portanto, uma propriedade
operacional da geometria medida **naquele domínio**, não uma propriedade
intrínseca e global do checkpoint.
