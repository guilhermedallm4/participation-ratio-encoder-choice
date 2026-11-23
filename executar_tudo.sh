#!/bin/bash
# Pipeline completo, pensado para sobreviver a queda da conexao.
#
# Ordem: medidas novas -> sonda nos 17 -> analise corrigida -> figuras -> zip.
# Cada etapa grava incrementalmente, entao uma interrupcao perde no maximo a
# etapa corrente; relancar o script retoma de onde parou, porque testes.py e
# sonda.py pulam chaves ja gravadas.
set -u
cd /home/guilhermelima/quantico/explicabilidade
PY=~/quantico/venv-mamografia/bin/python
MARCA() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

MARCA "1/6 geometria dos 5 checkpoints novos"
$PY -u codigo/testes.py --testes leva3 --n 200 --out resultados

MARCA "2/6 fluxo residual nos 9 objetivos"
$PY -u codigo/testes.py --testes fluxo3 --n 200 --out resultados

MARCA "3/6 controle RGB"
$PY -u codigo/testes.py --testes rgb --n 200 --out resultados

MARCA "4/6 sonda linear nos 17 extratores, 3 particoes"
$PY -u codigo/sonda.py --n 387

MARCA "5/6 analise final (permutacao por extrator, receita x objetivo)"
$PY -u codigo/analise_final.py

MARCA "6/6 figuras e pacote"
cd artigo
for L in pt en; do
  $PY codigo/fig_desenho.py --idioma $L || echo "!! fig_desenho $L falhou"
done
cd ..
rm -f artigo-esann.zip
zip -rq artigo-esann.zip artigo -x '*/__pycache__/*'

MARCA "CONCLUIDO"
echo "Resumo em resultados/NUMEROS.md"
