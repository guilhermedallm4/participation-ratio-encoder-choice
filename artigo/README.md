# Geometria de representação descreve um extrator congelado, mas não serve para escolhê-lo

Rascunho para o ESANN 2027 (prazo 18/11/2026, 6 páginas incluindo figuras e
referências).

## Os dois arquivos

| arquivo | língua | papel |
|---|---|---|
| `artigo-pt.tex` | português | **versão de trabalho** |
| `artigo-en.tex` | inglês | versão de submissão; o ESANN só aceita inglês |

Mesmo conteúdo, mesma estrutura (20 referências nas duas). Mantenha as duas em
sincronia ou decida qual é a fonte antes de editar.

## Compilando

```
pdflatex artigo-pt.tex && pdflatex artigo-pt.tex
```

`esannV2.cls` é a classe oficial, de <https://www.esann.org/node/5>, sem
modificação. Desvios em relação ao `esannV2.tex` de exemplo: `utf8` no lugar de
`latin1`; `graphicx` puro no lugar de `[dvips]`, porque as figuras são PDF e o
caminho é pdflatex; e, só na versão em português, `babel`+`fontenc` para a
hifenização (se conflitar com a classe, remova as duas linhas — o
`\AtBeginDocument` logo abaixo já fixa `\tablename` e `\abstractname`).

**Ainda não compilado.** Não há LaTeX nesta máquina e o download do tectonic
segue bloqueado pelo sandbox. Conferir na primeira compilação: as 6 páginas e
as margens de 5,2/4,4 cm, que o ESANN deixa como responsabilidade do autor.

Estimativa: **~6,7 páginas nas duas línguas** — acima do limite. A estimativa
agora vem de `codigo/paginas.py`, que quebra o texto na largura real de 12,2 cm
com as métricas do `cmr10.ttf`, em vez de extrapolar contagem de caracteres.
Ele **não hifeniza**, e o TeX hifeniza, então o número real deve ficar uns 3 a
4 % abaixo: **~6,4 páginas**. Ainda sobra cortar ~0,4.

As correções obrigatórias do segundo parecer custaram ~0,9 página; ~0,35 já
foi recuperada enxugando prosa. **Decisão em aberto: as figuras ficam por
enquanto.** Opções de corte, em ordem de quanto custam em conteúdo:

| corte | ganho | o que se perde |
|---|---|---|
| Figura 1 (desenho em cruz) | 0,39 pág | só a ilustração; a introdução já descreve o desenho em duas frases |
| 6 citações de arquitetura | 0,26 pág | ResNet, Swin, ConvNeXt, EfficientNet, MaxViT, DeiT3 ficam sem referência |
| detalhe do item 7 para cá | 0,15 pág | pré-processamento e validação da sonda saem do artigo |
| seção 3.1 vira parágrafo | 0,20 pág | o resultado "ordem estável, magnitudes não" perde destaque |

A tabela de dados por checkpoint **já foi removida** por espaço; ela está no
repositório, como o parecer admitiu que fosse feito.

As figuras já são gravadas na largura exata de 12,2 cm (`codigo/exato.py`), em
duas passagens, para que `width=\textwidth` não as amplie por fatores
diferentes — sem isso a tipografia das duas saía em tamanhos distintos.

## Antes de submeter

- [ ] **`\author`**: o nome completo e o nome exato da unidade ainda são um
      `TODO` nos dois `.tex`. Só você pode preencher.
- [ ] compilar e conferir as 6 páginas e as margens.

## A tese

Dezessete extratores congelados em quatro domínios, num **desenho em cruz**:
nove arquiteturas com o pré-treino supervisionado fixo, e nove pré-treinos
sobre o mesmo ViT-B/16.

**1. A razão de participação detecta colapso, não ordena qualidade.**

| conjunto | k | ρ | p | IC 95% |
|---|---|---|---|---|
| todos os 17 | 17 | +0,41 | **0,007** | [+0,12; +0,68] |
| sem o MAE | 16 | +0,27 | 0,075 | [+0,02; +0,50] |
| sem MAE e os dois CLIP | 14 | +0,21 | 0,223 | [−0,07; +0,48] |
| 9 pré-treinos | 9 | +0,41 | 0,088 | [−0,11; +0,76] |
| 9 arquiteturas | 9 | +0,11 | 0,662 | [−0,26; +0,56] |

A associação existe no conjunto cheio e **vem dos checkpoints colapsados**.
Entre extratores de dimensionalidade usual a medida não ordena. Controle de
largura em canais: +0,17 (p = 0,35).

**2. "Objetivo de pré-treino" não é um fator sem réplicas.** Os quatro
ViT-B/16 supervisionados — objetivo genuinamente igual — espalham o fluxo
residual por 0,219, **59 % da amplitude total**; dois CLIP com objetivo
idêntico e corpora diferentes, por 0,227 (61 %). A decomposição por família dá
48,7/51,3 por checkpoint e 56,3/43,7 por família, então o artigo **não** se
apoia nela: apoia-se nas duas comparações sem ambiguidade.

**3. A ordem das medidas por sensibilidade ao domínio é estável** (razão de
participação 43 % > raio 12 % > fluxo 0,5 %) em conjuntos de 6, 12 e 17
extratores; as porcentagens oscilam por fator de 2–3.

### Protocolo de extração

Idêntico nas dezessete: **média espacial global do último mapa**. Nos ViTs as
fichas de prefixo são descartadas e as espaciais remontadas em grade — o token
CLS nunca é usado.

### Testes múltiplos

Nove testes na seção 3.2; só o primeiro era pré-especificado, sem correção de
multiplicidade. A associação raio × sonda no braço do pré-treino (ρ = +0,56,
p = 0,009) está marcada como **exploratória**.

## Figuras

Geradas, 12,2 cm (bloco de texto do ESANN), as duas línguas da mesma fonte:

```
python codigo/fig_desenho.py --idioma pt    # e --idioma en
python codigo/fig_resultados.py --idioma pt
```

As atas são impressas em P&B: toda distinção sai por traço, marcador ou
hachura além da cor.

## Respostas ao segundo parecer

### Item 1 — a razão de participação está limitada por n = 200?

`codigo/estabilidade_pr.py` extrai os vetores dos 17 extratores nos 4 domínios
numa passagem e guarda em `resultados/pr_vetores/`; `analise_estabilidade.py`
lê e responde. Saída em `resultados/ESTABILIDADE.md`.

A objeção é correta no fato — a covariância de n imagens tem posto ≤ n−1, logo
com 200 imagens nenhuma PR passa de 199 — mas não morde:

- maior PR dos 68 pares: **77,1** (ViT-B/16 sup. 21k em DTD) = **38,7 % do teto**;
  mediana 22,4 (11,3 %);
- a **magnitude** depende mesmo de n: mediana de PR(50)/PR(200) = **0,74**;
- a **ordem** não: Spearman entre PR(50) e PR(200) fica em **0,990–1,000** nos
  quatro domínios;
- a tabela de sensibilidade inteira é indistinguível com metade das imagens
  (ρ = +0,41, p = 0,007 com 200, 100 ou 50);
- o checkpoint colapsado é o mais estável de todos: **1,81 com 50 imagens e
  1,81 com 200**.

Vale registrar o que isso faz com a tese: o sinal de colapso é estável ao
tamanho da amostra e as magnitudes dos extratores sadios não são. É o mesmo
contraste da seção 3.1, agora num segundo eixo.

### Item 2 — o IC excluía zero onde p > 0,05

`codigo/inferencia.py`, saída em `resultados/INFERENCIA.md`.

A primeira hipótese foi **testada e descartada**: supúnhamos viés de duplicata
no bootstrap com reposição (um extrator sorteado duas vezes entra com pares
concordantes consigo mesmo). A média das reamostras coincide com o ρ observado
até a terceira casa, então esse viés não existe aqui.

O que restou é menor: o intervalo percentil terminava em +0,01 numa linha com
p = 0,075. O artigo passou a usar **jackknife deixando um extrator de fora, na
escala de Fisher** — procedimento companheiro de uma análise que trata de
retirar extratores —, e as **seis linhas ficam coerentes de uma vez**: o
intervalo contém zero exatamente onde p > 0,05. A Fig. 2(c) foi regerada com o
mesmo procedimento.

### Itens 3 e 4 — especificação das medidas

O fluxo residual foi medido **só nos nove ViT-B/16**, o que dissolve a objeção
sobre equiparar blocos entre famílias. O artigo agora diz isso, define a
unidade (o bloco pré-norma inteiro), o critério de exclusão (passos em que a
forma muda) e a ordem de promediação (blocos dentro da imagem, depois imagens).

Corrigido de passagem um erro do texto anterior: a seção 3.1 atribuía os 0,5 %
de variância do fluxo "aos dezessete extratores", e o fluxo tem nove.

O campo receptivo ganhou o protocolo completo: soma dos canais na posição
central do mapa já no formato (N,C,H,W), gradiente em valor absoluto somado nos
canais de entrada, normalizado por imagem, raio sobre o mapa **médio** entre
imagens, e a nota de que o raio vive em pixels da entrada, que é o que torna
grades de 32×32 e 16×16 comparáveis.

### Item 3.5 — referências de 2026 conferidas

As quatro foram verificadas e existem, com os identificadores corretos:

| chave | arXiv | confere |
|---|---|---|
| `Hemato` | 2608.25148 | sim |
| `SEMASIA` | 2605.09485 | sim |
| `GlobalGeo` | 2602.03282 | sim |
| `Fase` | 2606.08132 | sim, mas **removida** do artigo por espaço |

## Respostas ao terceiro parecer

**Atenção: o terceiro parecer foi emitido sobre a versão anterior.** O próprio
revisor diz que não identificou alterações. Das oito passagens que ele cita
literalmente, as oito já estavam corrigidas quando o parecer chegou — fluxo
residual, campo receptivo, "59 %" no resumo, a justificativa da reamostragem,
o IC `[+0,02; +0,50]`, o título antigo, "show it does not" e "width
controlled". Conferir com `grep` antes de reagir a esse parecer.

O que ele traz de **novo** está abaixo.

### 2.2 — a largura NÃO estava controlada (prioridade máxima)

Ele está certo, e era um furo real: o artigo reportava ρ(PR, sonda) = +0,41 ao
lado de ρ(largura, sonda) = +0,17 e chamava o segundo de "controle". Reportar a
correlação do confundidor com o desfecho não controla nada.

`codigo/parcial.py` faz o controle de verdade, em duas formas, com a mesma
permutação no nível do extrator (PR e largura viajam juntas, porque são
propriedades do mesmo extrator). Saída em `resultados/PARCIAL.md`:

| conjunto | ρ(PR, sonda) | ρ(largura, sonda) | ρ parcial | p |
|---|---|---|---|---|
| todos os 17 | +0,410 | +0,170 | **+0,392** | 0,012 |
| sem o MAE | +0,273 | +0,189 | +0,243 | 0,113 |
| sem MAE e os dois CLIP | +0,211 | +0,217 | +0,173 | 0,327 |
| 9 arquiteturas | +0,109 | +0,309 | +0,049 | 0,848 |
| 9 pré-treinos | +0,406 | — | — | largura constante (768) |

Na regressão `sonda = b0 + b1·PR + b2·largura`: b1 = +0,572 (p = 0,003) contra
b2 = +0,016. **A associação sobrevive e a largura não a explica**, e o padrão de
sensibilidade se mantém inteiro depois do controle.

### 2.3 — o teto de posto estava errado por um fator de dois

Aqui ele pegou um erro **meu**, introduzido na rodada anterior. O ponto
publicado da PR sai de `ic_sub`, que subamostra sem reposição em m = n/2 = 100.
O teto é portanto **99**, não 199. A versão anterior deste README e do artigo
dizia "máximo 77,1 = 38,7 % do teto de 199". O número certo é:

- maior PR publicada: **55,4** = **56 % do teto de 99**, e é o único dos 68
  pares acima da metade; mediana 20,1 (20 %).

O teto está mais perto do que eu havia dito, e o artigo agora diz isso. O que
sustenta a conclusão não é a folga até o teto, e sim a invariância: a ordem
entre extratores não muda (ρ = 0,990–1,000 entre PR(50) e PR(200)), a tabela de
sensibilidade inteira é a mesma com metade das imagens, e o checkpoint
colapsado está em 1,8 — duas ordens de grandeza abaixo de qualquer efeito de
teto.

### 2.1 — definição operacional de colapso

Acrescentada, e explicitamente **posterior à medição**: a razão do MAE é
1,8–4,2, de três a dez vezes menor que a do extrator seguinte em qualquer
domínio. É designação sobre este conjunto, não critério geral — validar um
detector exigiria mais de um checkpoint colapsado. O título já usava *flags*.

### Itens menores, todos aplicados

| item | o que mudou |
|---|---|
| 4 contagens de classe | eram três para quatro domínios; agora 2, 47, 37 e 22 |
| n do SUN397 | 384, não 387, depois de descartar classes com < 5 exemplos |
| teto de 204,3 px | agora com a derivação: πr² = 512²/2 |
| conjuntos de 6/12/17 | são as levas de medição acumuladas, não uma seleção |
| ponderação por família | fórmula no artigo e calculada em `analise_final.py` |
| reamostragem com reposição | justificativa corrigida: desloca o espectro, não "fica sempre abaixo" |
| resumo | restringido ao conjunto avaliado, e com o controle de largura |

## Reprodução

Os scripts aqui são os que geram as figuras. A bateria completa está no
diretório do estudo (`../codigo/`):

| script | o que faz |
|---|---|
| `representacao.py` | posto efetivo, dispersão, CKA, bandas de frequência |
| `campo_e_embaralho.py` | campo receptivo efetivo e embaralhamento de retalhos |
| `fluxo_residual.py` | fração do fluxo que cada mecanismo reescreve |
| `testes.py` | os sete controles (intervalos, resolução, fundo, arquiteturas, ablação, fluxo com IC, alinhamento) |
| `sonda.py` | sonda linear nos doze, agrupada por paciente na mamografia |
| `analise_final.py` | decomposições, permutação por extrator, receita × objetivo |
| `estabilidade_pr.py` | PR em 50/100/150/200 imagens, uma passagem de extração |
| `analise_estabilidade.py` | teto de posto, estabilidade de ordem, conclusão com metade das imagens |
| `inferencia.py` | IC jackknife × percentil × permutação, lado a lado |
| `parcial.py` | Spearman parcial e regressão controlando a largura; teto de posto no m certo |
| `artigo/codigo/paginas.py` | estimativa de páginas por quebra de linha real |

## O que fica em aberto

- A sonda é linear e sobre características congeladas; não acompanha ajuste fino.
- A ablação não retreinou o modelo: é o MaxViT com um mecanismo amputado.
- Só um dos seis extratores tem atenção em grade, então a afirmação é sobre o
  MaxViT-T, não sobre uma classe.
- 387 imagens e até 47 classes deixam a sonda fraca em valor absoluto; o que
  sustenta as conclusões é a comparação pareada, não cada ponto.

## Dados

O INbreast exige cadastro junto aos autores e não é redistribuível. DTD,
Oxford-IIIT Pet e SUN397 são públicos.

## Rodada 4 — reenquadramento após o teste do checkpoint retido

Experimento novo, fora do conjunto dos dezessete, com a predição registrada
antes da medição (`codigo/segundo_colapso.py`, `codigo/analise_segundo.py`,
saída em `resultados/SEGUNDO_COLAPSO.md`):

| extrator retido | mamografia | DTD | Pet | SUN397 |
|---|---|---|---|---|
| BEiT-3 base (`.pt`) — PR | **2,24** (flagrado) | 22,83 | 20,36 | 17,69 |
| BEiT-3 base — sonda | 0,577 (3º–5º pior de 18) | 0,419 (2º pior) | 0,427 (2º pior) | 0,771 (2º pior) |
| ViT-B/16 sem treino — PR | **1,27** | **1,55** | **1,67** | **1,64** |
| ViT-B/16 sem treino — sonda | 0,499 | 0,075 | 0,060 | 0,191 |

A predição (PR na faixa do MAE nos quatro domínios) **falhou em três dos
quatro**: o BEiT-3 sonda entre os piores sem que a razão de participação o
acuse. Isso reforça a metade negativa do artigo e delimita a positiva.

Efeito colateral útil: o BEiT-3 é concordante em posto nos quatro domínios,
então com ele no lugar do MAE a correlação fica em ρ = +0,362 (p = 0,018),
contra +0,273 (p = 0,075) sem nenhum dos dois — a associação deixa de depender
de um único checkpoint.

O modelo sem treino **não** entra em nenhuma correlação do artigo: é ponto de
alavancagem extremo e não é um checkpoint pré-treinado. Só serve como controle
de degeneração conhecida.

### O que mudou nos dois .tex

1. **Resumo** — o resultado principal passa a ser declarado como negativo
   (*não ordena extratores não colapsados*); a frase sobre "sinalizar
   dimensionalidade atipicamente baixa" sai e entra o teste retido.
2. **Introdução, contribuições** — reordenadas com a negativa à frente.
3. **§3.2** — o parágrafo de "colapsado" passa a declarar a **circularidade**
   do critério (colapso *é* PR baixa, logo não é falsificável dentro do
   conjunto), seguido do teste retido e da delimitação ("flagra degeneração
   *no domínio em que se mede*; não é garantia").
4. **§3.2, multiplicidade** — o teste retido é registrado como pré-especificado.
5. **§4** — a recomendação é reescrita como mais negativa que positiva.
6. **§4, limitações** — o teste retido cobre um checkpoint colapsado por vez e
   não substitui um conjunto deles.
7. **Bibliografia** — acrescentado `\bibitem{BEiT3}` (Wang et al., CVPR 2023).

### Erro corrigido de passagem

O texto dizia *"o MAE sonda entre 0,447 e 0,533"*. Os dois números são de
colunas diferentes de `resultados/NUMEROS.md`: 0,533 é a AUC na mamografia e
0,447 é a **média** das acurácias dos três domínios naturais — não é uma faixa.
A faixa real é 0,322–0,627, e o texto agora dá os quatro valores.

### Custo em páginas

~7,03 → **~7,70** páginas estimadas (limite 6). O reenquadramento custou
~0,65 página. O corte continua pendente de decisão.

## Rodada 5 — as cinco prioridades do parecer (Weak Accept → tendência a aceitar)

O parecer confirmou as correções das rodadas 1–4 e listou cinco prioridades.
Todas aplicadas nos dois idiomas.

**1. Moderar a conclusão estatística.** O revisor está certo: os IC chegam a
+0,51 no limite superior, então "o PR não ordena" transforma ausência de
evidência em evidência de ausência. Resumo, introdução, §3.2 e §4 passaram a
dizer **"não encontramos capacidade de ordenação"**, com um parágrafo novo em
§3.2 declarando o que os números *não* estabelecem e registrando que não
fizemos teste de equivalência contra um efeito mínimo relevante. §4 acrescenta
que a recomendação prática não exige que a capacidade seja nula.

**2. Braço de arquiteturas.** O revisor está certo e os tags confirmam:
`convnext_tiny.fb_in1k`, `efficientnet_b3.ra2_in1k`, `swin_tiny.ms_in1k`,
`vit_small.augreg_in1k`, ResNets da torchvision com pesos `V2`, MaxViT `tf`
nativos de 512 px. São receitas e fornecedores diferentes. Trocado em resumo,
introdução, legenda da Fig. 1 e §2.1: *"nove arquiteturas sob um mesmo **regime**
de pré-treino supervisionado na ImageNet"*, com a ressalva explícita de que o
que está fixo é o regime, não a receita.

**3. Validação com o BEiT-3.** §3.2 passou a trazer a **regra operacional de
sinalização** com os quatro limiares (4,93 / 9,78 / 11,35 / 10,43) e a dizer
que "colapso" é propriedade da geometria *naquele domínio*. O ρ = +0,36 está
rotulado como **análise secundária, não pré-registrada**, e o texto admite que
dois extratores retidos não estimam sensibilidade nem especificidade. O
registro prévio virou arquivo auditável: `resultados/PREDICAO_REGISTRADA.md`,
com a predição literal e os mtimes (`segundo_colapso.py` às 15:12:37 contra o
primeiro resultado às 16:55:50 do mesmo dia).

**4. Dados e treino.** "Nothing is trained" já tinha sido corrigido na rodada
anterior; adotada a redação mais clara do revisor. O 387/384 passou a ser
afirmado uma vez só, sem a contradição.

**5. Reprodutibilidade.** Acrescentados: interpolação dos *embeddings*
posicionais nos ViT a 512 px (`dynamic_img_size` do timm) e os MaxViT como
checkpoints nativos; sementes 0, 1 e 2 como três repetições completas da
validação cruzada; 400 subamostras no `ic_sub` e 40 por degrau na escada de
estabilidade; amostragem em rodízio entre classes com semente fixa e a
distribuição por classe (8–9 no DTD, 5–16 no Pet, 3–24 no SUN397), com a
observação de que isso importa porque a covariância global mistura variação
intra e interclasse; nota de rodapé reservada para a **URL do repositório**;
e as referências de DTD (Cimpoi 2014), Pet (Parkhi 2012) e SUN397 (Xiao 2010).

Também atendidas as ressalvas de 1.1 e 1.2: como a parcial de Spearman é
calculada e de onde vem seu *p* (permutação no nível do extrator, razão e
largura permutadas juntas), que os coeficientes da regressão são padronizados
dentro do domínio, que a parcial não era pré-especificada, e que o 0,990–1,000
é uma comparação por domínio sobre médias de 40 subamostras, não uma
distribuição de repetições.

### Pendências

- **URL do repositório** — nota de rodapé com `TODO` nos dois .tex.
- **`\author`** — ainda com `TODO`.
- **Título** — ~~mantido o atual~~ **aplicado o sugerido pelo revisor**:
  *"Participation ratio flags some degenerate representations, but does not
  reliably rank healthy encoders"* / *"A razão de participação sinaliza algumas
  representações degeneradas, mas não ordena com segurança extratores sadios"*.
  Fica alinhado com a moderação feita no corpo na mesma rodada.
- **Páginas**: ~7,79 → **~9,0** estimadas, contra o limite de 6.

## Rodada 6 — parecer final (Weak Accept, tendência a aceitar)

Três pontos, todos atendidos exceto o que depende do autor.

**1. Inferência da correlação parcial — o revisor estava certo.** O nulo
anterior permutava razão de participação e largura **juntas** contra a sonda,
o que destrói também a associação largura–sonda. A hipótese nula condicional
correta ("a razão nada acrescenta depois da largura") deve *preservar* essa
associação. Trocado por **Freedman–Lane** (`codigo/parcial_fl.py`,
`resultados/PARCIAL_FL.md`): regride-se a sonda na largura, permutam-se só os
resíduos — no nível do extrator, mesma permutação nos quatro domínios — e
remonta-se a sonda nula somando o ajuste intacto.

| conjunto | k | ρ parcial | p Freedman–Lane | p do nulo antigo |
|---|---|---|---|---|
| todos os 17 | 17 | +0,392 | **0,010** | 0,012 |
| sem o MAE | 16 | +0,243 | 0,113 | 0,113 |
| sem MAE e os dois CLIP | 14 | +0,173 | 0,320 | 0,327 |
| 9 arquiteturas | 9 | +0,049 | 0,847 | 0,848 |

O coeficiente da regressão na escala padronizada do artigo: b1 = +0,572,
**p = 0,003** por Freedman–Lane (era 0,003 pelo nulo antigo). A conclusão não
muda; a inferência agora responde à pergunta certa. O `p = 0,012` virou
`p = 0,010` em §3.2 dos dois .tex, com a justificativa reescrita.

**2. Rótulo da Figura 1.** `(pretraining fixed)` → `(shared supervised regime)`
e `(pré-treino fixo)` → `(mesmo regime supervisionado)` em
`codigo/fig_desenho.py`; as duas figuras foram regeradas. A contradição entre a
arte e o texto desapareceu.

**3. Redação.** `we find no ordering ability` → `we find no reliable evidence of
ordering ability` (e o equivalente em português) nos três lugares: resumo,
introdução e §3.2.

### O único ponto que não posso fechar

A **URL do repositório** segue como placeholder (`repository url --- fill in
before submission`). O revisor a apontou como "o problema mais evidente do PDF
atual", porque o texto apoia nela os identificadores dos checkpoints, as
sementes, o protocolo de extração e o registro prévio da predição. Exige criar
o repositório público e colar o endereço. Idem o `\author`.

## Rodada 7 — corte para 6 páginas e comentários da coautoria

O artigo saiu de **~9,1 páginas estimadas para 5,85 (EN) e 6,10 (PT)**. O
estimador não hifeniza, então é limite superior por 3–4 %: o inglês deve sair
em ~5,6 páginas reais.

### Comentários da Cecilia, aplicados

1. **Primeiro parágrafo da introdução** — reescrito em torno da *promessa* da
   geometria (sem rótulos, um passo adiante por extrator, escolher sem rodar a
   avaliação incerta) e fechando com a pergunta central explicitada: uma medida
   *correlacionada* com desempenho serve para *ordenar* candidatos? O artigo é
   agora apresentado como um teste dessa premissa.
2. **O desenho em cruz** — passou a explicar o que ficaria ambíguo sem ele:
   "um aparente efeito de arquitetura pode ser a receita de pré-treino que cada
   família traz consigo".
3. **Contribuições sem abrir na negativa** — "Localizamos onde a associação de
   fato mora" em vez de "o resultado principal é negativo".
4. **O intervalo do PR** (§2.3) — explicitado que são os percentis 2,5 e 97,5
   dos 400 valores, sem ajuste, e que ele estima o quanto a razão se move entre
   subconjuntos de 100 imagens *desta* amostra, não a incerteza sobre um valor
   populacional, que exigiria um modelo de como a amostra surgiu.
5. **§3.2 começa pelo achado** — "A associação enfraquece bruscamente quando os
   checkpoints extremos saem", e só depois a sensibilidade.
6. **Sem preparação defensiva** — o parágrafo da incerteza agora começa direto
   em "Os intervalos continuam compatíveis com associação moderada"; saiu o
   "é preciso dizer o que esses números não estabelecem".
7. **Discussão abre pela mensagem científica** — "Uma representação pode ter
   dimensionalidade global sadia e ainda assim ser ruim para a tarefa".

### O que foi cortado, e por quê

| corte | páginas | razão |
|---|---|---|
| **§3.3, "objetivo não é fator sem réplicas"** | ~0,55 | Segunda alegação, com medida própria (fluxo residual) e conjunto próprio (9 checkpoints). Alvo de um artigo separado. O achado de fatia de variância foi preservado em §3.1. |
| Figura 1 (o desenho) | 0,39 | O desenho passou a ser explicado em prosa, como a Cecilia pediu no comentário 2. |
| Figura 2 (resultados) | 0,33 | O painel (c) duplicava a Tabela 1; (a) são três números do texto; (b) pertencia à §3.3. |
| Definição do campo receptivo | ~0,15 | Produz só um nulo. **O resultado continua relatado** (+0,20, p = 0,26) e a definição foi para o repositório. |
| Bibliografia em formato enxuto | 0,34 | 23 → 21 itens, venues abreviados. |
| Prosa comprimida em todas as seções | ~0,6 | — |

Nada medido foi suprimido do relato: os nulos do campo receptivo e o achado
exploratório (ρ = +0,56) continuam no texto. **O material da §3.3 permanece
íntegro em `resultados/NUMEROS.md`, `codigo/analise_final.py` e nas figuras
`figuras/fig2_resultados*.pdf`**, que seguem no repositório embora não sejam
mais referenciadas pelo artigo.

### Estado

- EN ~5,85 pág · PT ~6,10 pág · limite 6 · nenhuma figura, uma tabela
- Pendentes: **URL do repositório** e **`\author`**, ambos com `TODO` visível
- Nunca compilado neste ambiente (tectonic bloqueado)

## Rodada 8 — linguagem mais leve

Pedido: texto menos carregado de terminologia. Páginas: EN 5,85 → **5,94**;
PT 6,10 → 6,26 (o PT não vai para a conferência, então pode folgar).

### Sobre "latent space"

**Não trocado, de propósito.** O que se mede é a covariância dos mapas de
características do último bloco depois da média espacial. Chamar isso de
"espaço latente" importa a conotação de variável latente com prior, de VAE, que
não se aplica — ficaria mais vago e menos exato ao mesmo tempo. `representation`
já é a palavra leve e correta, e é a que o texto usa.

### O que foi efetivamente trocado

| antes | agora |
|---|---|
| *the share of log-variance attributable to the domain* | *how much of a measure's spread comes from the dataset rather than the encoder* |
| *an effective-receptive-field radius (gradient of the summed channels at the central position)* | *how far a central output position looks back into the image (a gradient-based radius in input pixels)* |
| *a residual flow, ‖F(x)‖/‖x‖ per pre-norm block* | *how much each transformer block changes what it receives, ‖F(x)‖/‖x‖* |
| *less sensitive to the spectral tail* | *less swayed by the smallest eigenvalues* |
| *Metrics are out-of-fold* | *Scores come from the held-out fold* |
| *an over-random null and an anti-conservative test* | *the null too random and the test too easy to pass* |
| *jackknife on the Fisher scale* | *jackknife, taken after Fisher's z-transform* |
| *the conditional null is "the ratio adds nothing beyond width"* | *the hypothesis to rule out is that the ratio adds nothing once width is known* |
| *Being rank-concordant throughout* | *Ranking low on both measures in every domain* |
| *"Collapsed" is operational and posterior to measurement* | *We call a checkpoint collapsed only after looking* |
| *cannot estimate sensitivity or specificity* | *cannot pin down how often the measure is right* |
| *a diagnostic for degeneration, not a proxy for representational capacity* | *tells you when a representation has gone degenerate; it does not tell you how good one is* |
| *a variance share of this kind must come with the set of models* | *a number like "43 % of the spread comes from the dataset" must travel with the set of models* |
| *first-order global statistics, and miss compositional structure* | *every measure here looks at the representation as a whole and so cannot see how its parts combine* |

### O que foi mantido pesado, e por quê

`participation ratio`, `linear probe`, `encoder`, `Freedman--Lane`,
`permutation`, `jackknife`. São os nomes que a área usa; trocá-los por
paráfrases tornaria o artigo mais difícil de localizar e daria a impressão de
que o autor não conhece a literatura. A razão de participação já vem glosada na
primeira aparição ("how many directions it actually uses"), que é onde a leveza
importa.

### Margem

O inglês ficou a **0,06 página do limite**. O estimador não hifeniza, então o
real deve sair em ~5,7, mas não há folga para acrescentar nada sem cortar algo.
