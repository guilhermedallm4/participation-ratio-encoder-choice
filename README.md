# A razão de participação sinaliza algumas representações degeneradas, mas não ordena com segurança extratores sadios

Código, resultados e fonte do artigo submetido ao **ESANN**.

A geometria da representação congelada costuma ser apresentada como uma forma
de escolher um extrator sem usar rótulos, com base na premissa de que uma
medida correlacionada com a acurácia final também serviria para ordenar
candidatos. Este trabalho testa essa premissa em dezessete extratores
congelados avaliados em quatro domínios.

**Resultado principal.** A razão de participação se correlaciona com a acurácia
de uma sonda linear (ρ = 0,41, p = 0,007), mas a correlação vem dos checkpoints
de dimensionalidade extrema: cai para 0,27 sem o MAE, para 0,21 sem também os
dois CLIP, e fica em 0,11 entre as nove arquiteturas. Um checkpoint retido, com
a predição registrada antes da medição, mostra onde a premissa falha: o BEiT-3
está entre os piores dos dezoito na sonda, mas a razão só o sinaliza em um dos
quatro domínios.

## Organização

| diretório | conteúdo |
|---|---|
| `codigo/` | medidas geométricas, sonda linear, inferência e figuras |
| `artigo/` | fonte LaTeX (pt e en), classe ESANN, figuras e relatórios |
| `resultados/` | saídas agregadas em JSON, CSV e Markdown |
| `figuras/` | figuras de diagnóstico que não entraram no artigo |
| `logs/` | log das execuções, para rastreabilidade |

### Pontos de entrada

- `executar_tudo.sh` — pipeline completo, retomável
- `codigo/baixa_bancos.py` — reconstrói a amostra de imagens
- `codigo/testes.py` — razão de participação, campo receptivo, fluxo residual
- `codigo/sonda.py` — sonda linear, 5 dobras externas, 3 partições
- `codigo/analise_final.py` — correlações e permutação no nível do extrator
- `codigo/parcial_fl.py` — correlação parcial com inferência de Freedman–Lane
- `codigo/segundo_colapso.py` — teste com os extratores retidos

### Resultados que sustentam afirmações do artigo

| arquivo | afirmação |
|---|---|
| `resultados/NUMEROS.md` | tabela completa por checkpoint |
| `resultados/PREDICAO_REGISTRADA.md` | **registro prévio** da predição do teste retido, com a regra de sinalização e os horários dos arquivos |
| `resultados/SEGUNDO_COLAPSO.md` | resultado do teste com extratores retidos |
| `resultados/PARCIAL_FL.md` | correlação parcial por Freedman–Lane |
| `resultados/ESTABILIDADE.md` | estabilidade da razão em 50, 100, 150 e 200 imagens |
| `resultados/INFERENCIA.md` | permutação, jackknife e intervalos |

## Dados: o que não está aqui, e por quê

O diretório `data/` **não é versionado**.

- **Mamografia (INbreast).** O banco só pode ser obtido mediante cadastro junto
  aos autores e não pode ser redistribuído. A planilha original do INbreast
  contém identificação de paciente não anonimizada; nada derivado dela que
  permita reidentificação está neste repositório. Os rótulos de classe e o
  agrupamento por paciente usados pela sonda são lidos de um arquivo local,
  fora deste repositório (ver `codigo/sonda.py`).
- **DTD, Pet e SUN397.** Bancos públicos com licença própria, não
  redistribuídos aqui.
- **`resultados/pr_vetores/`.** Vetores de características, 39 MB, dos quais 17
  arquivos derivam das mamografias. Ficam de fora por precaução.

Para reconstruir a amostra, use `codigo/baixa_bancos.py`, que sorteia as
imagens em rodízio entre as classes com semente fixa. A mamografia exige o
INbreast já obtido junto aos autores.

## Ambiente

Python com PyTorch, `timm`, scikit-learn, SciPy, NumPy, pandas e matplotlib.
Os identificadores exatos dos checkpoints estão em `codigo/multitarefa.py`.

## Citação

Em avaliação. Até a publicação, cite este repositório.
