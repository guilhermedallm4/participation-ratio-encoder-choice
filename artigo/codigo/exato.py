#!/usr/bin/env python
"""Grava a figura com a largura final EXATA, em duas passagens.

Com bbox_inches="tight" o matplotlib recorta a margem branca, entao o PDF sai
mais estreito que a figsize pedida -- e por fatores diferentes em cada figura.
Como o .tex usa width=\\textwidth, cada uma e AMPLIADA por um fator proprio, e
as fontes, escolhidas em pontos, acabam em tamanhos diferentes nas duas. O
leitor ve duas figuras com tipografia desencontrada.

A correcao: gravar, medir o que saiu, reescalar a figsize pela razao e gravar
de novo. Duas passagens bastam porque o recorte e proporcional.
"""
ALVO_CM = 12.2


def salva(fig, caminho, alvo_cm=ALVO_CM, passagens=5):
    import pymupdf
    pol = alvo_cm / 2.54
    for _ in range(passagens):
        fig.savefig(caminho, dpi=300)
        d = pymupdf.open(caminho)
        larg = d[0].rect.width / 72
        alt = d[0].rect.height / 72
        d.close()
        if abs(larg - pol) < 0.01:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * pol / larg, h * pol / larg)
    return larg, alt
