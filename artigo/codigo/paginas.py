#!/usr/bin/env python
"""Estimador de paginas para a classe ESANN, por quebra de linha de verdade.

Nao ha LaTeX nesta maquina e o download do tectonic esta bloqueado, entao a
alternativa a esta simulacao e um palpite por contagem de caracteres -- que foi
o que produziu as estimativas anteriores e nao serve para decidir cortes.

A classe fixa o bloco de texto em 12,2 x 19,2 cm e carrega size10.clo
(\\@ptsize = 0), logo corpo de 10 pt com entrelinha de 12 pt: 45 linhas por
pagina. A fonte e a Computer Modern Roman padrao, e o matplotlib distribui
cmr10.ttf -- as larguras dos glifos sao portanto MEDIDAS, nao supostas.

O que isto simula: paragrafos de texto corrido, quebrados avidamente na
largura real. O que NAO simula: hifenizacao (o TeX quebra palavras e ganha
espaco, entao esta contagem e um limite SUPERIOR por alguns pontos percentuais),
ajuste fino de espacamento e flutuacao de figuras. Os ambientes de figura e
tabela entram pela altura declarada, nao pelo texto.
"""
import re, sys
from pathlib import Path

from fontTools.ttLib import TTFont
import matplotlib

PT_POR_CM = 28.3465
LARG_PT = 12.2 * PT_POR_CM          # 345,8 pt
LINHAS_POR_PAG = 19.2 * PT_POR_CM / 12.0

_f = TTFont(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "cmr10.ttf")
_upm = _f["head"].unitsPerEm
_hmtx = _f["hmtx"]
_cmap = _f.getBestCmap()


def larg(ch, tam=10.0):
    g = _cmap.get(ord(ch))
    if g is None:
        g = _cmap.get(ord("n"))
    return _hmtx[g][0] / _upm * tam


LARG_CACHE = {}


def larg_txt(t, tam=10.0):
    s = 0.0
    for ch in t:
        k = (ch, tam)
        if k not in LARG_CACHE:
            LARG_CACHE[k] = larg(ch, tam)
        s += LARG_CACHE[k]
    return s


def visivel(t):
    """Texto que o leitor ve, a partir do fonte LaTeX."""
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)
    t = re.sub(r"\\(emph|textbf|textit|textsc|text|mathrm)\{([^{}]*)\}", r"\2", t)
    t = re.sub(r"\\cite\{[^}]*\}", "[00]", t)                 # [1] ~ 3 glifos
    t = re.sub(r"\\ref\{[^}]*\}", "0", t)
    t = re.sub(r"\$[^$]*\$", lambda m: "n" * max(1, len(m.group(0)) // 3), t)
    t = re.sub(r"\\paragraph\{([^}]*)\}", r"\1 ", t)
    t = re.sub(r"\\[a-zA-Z@]+\s*", " ", t)
    t = t.replace("~", " ").replace("{", "").replace("}", "")
    t = t.replace("---", "\u2014").replace("--", "\u2013")
    t = t.replace("\\,", " ").replace("\\%", "%")
    return re.sub(r"[ \t]+", " ", t)


def linhas_de(par, tam=10.0, recuo=0.0):
    """Quebra avida de um paragrafo na largura do bloco."""
    pal = par.split()
    if not pal:
        return 0
    n, x, esp = 1, recuo, larg_txt(" ", tam)
    for p in pal:
        w = larg_txt(p, tam)
        if x > recuo and x + esp + w > LARG_PT:
            n += 1; x = w
        else:
            x += (esp if x > recuo else 0) + w
    return n


def mede(arq):
    s = Path(arq).read_text(encoding="utf-8")
    corpo = s.split("\\maketitle", 1)[1]
    corpo, bib = corpo.split("\\begin{footnotesize}", 1)

    linhas = 0.0
    relato = []

    # --- blocos de altura fixa, medidos ou declarados ---------------------
    # fig1 e fig2 sao gravadas em 12,2 cm de largura; alturas reais dos PDFs.
    import subprocess
    def altura_cm(pdf):
        try:
            from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401
            import re as _re
            d = Path(arq).parent / pdf
            t = d.read_bytes()[:4000].decode("latin-1", "ignore")
            m = _re.search(r"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", t)
            return float(m.group(2)) / PT_POR_CM if m else 4.2
        except Exception:
            return 4.2

    for amb, extra_cm in (("figure", None), ("table", None)):
        for m in re.finditer(r"\\begin\{" + amb + r"\}(.*?)\\end\{" + amb + r"\}",
                             corpo, re.S):
            bloco = m.group(1)
            if amb == "figure":
                g = re.search(r"\{(figuras/[^}]+)\}", bloco)
                h = altura_cm(g.group(1)) if g else 4.2
            else:
                h = 0.42 * (bloco.count("\\\\") + 3)      # linhas da tabela
            leg = re.search(r"\\caption\{(.*?)\}\s*(\\label|\Z)", bloco, re.S)
            l_leg = linhas_de(visivel(leg.group(1)), 9.0) if leg else 0
            custo = h * PT_POR_CM / 12.0 + l_leg * 0.75 + 2.0
            linhas += custo
            relato.append((amb, round(custo, 1)))
            corpo = corpo.replace(m.group(0), "")

    # --- resumo (menor, com recuo nos dois lados) -------------------------
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", corpo, re.S)
    if m:
        global LARG_PT
        guarda = LARG_PT; LARG_PT = guarda - 2 * 0.5 * PT_POR_CM
        n = linhas_de(visivel(m.group(1)), 9.0)
        LARG_PT = guarda
        linhas += n * 0.75 + 3
        relato.append(("resumo", round(n * 0.75 + 3, 1)))
        corpo = corpo.replace(m.group(0), "")

    # --- titulo, autor, maketitle -----------------------------------------
    cab = 10.0
    linhas += cab; relato.append(("título+autor", cab))

    # --- texto corrido -----------------------------------------------------
    corrido = 0.0
    for par in re.split(r"\n\s*\n", corpo):
        par = par.strip()
        if not par:
            continue
        if re.match(r"\\(sub)*section\*?\{", par):
            corrido += linhas_de(visivel(par), 11.0) + 2.0
            continue
        v = visivel(par).strip()
        if not v:
            continue
        corrido += linhas_de(v, 10.0, recuo=15.0)
    linhas += corrido
    relato.append(("texto corrido", round(corrido, 1)))

    # --- bibliografia (footnotesize = 8 pt, entrelinha 9,5) ----------------
    nb = 0.0
    for it in re.split(r"\\bibitem", bib)[1:]:
        it = re.sub(r"^\{[^}]*\}", "", it.strip())
        guarda = LARG_PT
        nb += linhas_de(visivel(it), 8.0) * (9.5 / 12.0)
        LARG_PT = guarda
    linhas += nb + 2
    relato.append(("bibliografia", round(nb + 2, 1)))

    return linhas / LINHAS_POR_PAG, relato


if __name__ == "__main__":
    for arq in (sys.argv[1:] or ["artigo-en.tex", "artigo-pt.tex"]):
        pag, rel = mede(arq)
        print(f"\n{arq}: ~{pag:.2f} páginas  (limite 6)")
        for k, v in rel:
            print(f"    {k:16} {v:7.1f} linhas = {v/LINHAS_POR_PAG:.2f} pág")
