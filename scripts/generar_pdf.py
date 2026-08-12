"""Genera entrega/informe_tecnico.pdf desde docs/informe_tecnico.md (§1.4).

Sin pandoc/LaTeX: markdown (python-markdown) -> HTML con CSS de imprenta ->
Chrome headless (--print-to-pdf). Verifica el límite de 8 páginas con pdfinfo
y avisa si quedan huecos [COMPLETAR].

La nota de fuente del principio del .md (el blockquote "> Fuente de este
documento...") es interna y se elimina del PDF.

Uso:  python scripts/generar_pdf.py [--estricto]
      --estricto : falla (exit != 0) si quedan [COMPLETAR] o si excede 8 págs.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUENTE = ROOT / "docs" / "informe_tecnico.md"
SALIDA = ROOT / "entrega" / "informe_tecnico.pdf"
MAX_PAGINAS = 8

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body  { font-family: "DejaVu Serif", Georgia, serif; font-size: 10.5pt;
        line-height: 1.45; color: #111; max-width: 100%; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 14pt 0 6pt; border-bottom: 1px solid #999;
     padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 10pt 0 4pt; }
p  { margin: 5pt 0; text-align: justify; }
ul, ol { margin: 5pt 0 5pt 18pt; padding: 0; }
li { margin: 2pt 0; }
table { border-collapse: collapse; margin: 7pt 0; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #888; padding: 3pt 6pt; text-align: left;
         vertical-align: top; }
th { background: #eee; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt;
       background: #f2f2f2; padding: 0 2pt; }
pre  { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt;
       background: #f6f6f6; border: 1px solid #ccc; padding: 6pt;
       overflow: hidden; line-height: 1.25; }
pre code { background: none; padding: 0; }
blockquote { margin: 5pt 0 5pt 12pt; padding-left: 8pt;
             border-left: 3px solid #bbb; color: #333; }
hr { border: none; border-top: 1px solid #bbb; margin: 10pt 0; }
strong { font-weight: bold; }
"""


def _chrome() -> str:
    for nombre in ("google-chrome-stable", "google-chrome", "chromium"):
        ruta = shutil.which(nombre)
        if ruta:
            return ruta
    print("[ERROR] no se encontró Chrome/Chromium para imprimir el PDF.")
    sys.exit(1)


def _quitar_nota_interna(md: str) -> str:
    """Elimina el blockquote inicial "> Fuente de este documento..." (interno)."""
    lineas = md.split("\n")
    filtradas, saltando = [], False
    for linea in lineas:
        if linea.startswith("> Fuente de este documento"):
            saltando = True
        if saltando:
            if linea.startswith(">") or not linea.strip():
                continue
            saltando = False
        filtradas.append(linea)
    return "\n".join(filtradas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estricto", action="store_true",
                        help="falla si quedan [COMPLETAR] o excede 8 páginas")
    args = parser.parse_args()

    try:
        import markdown
    except ImportError:
        print("[ERROR] falta python-markdown (pip install markdown).")
        sys.exit(1)

    md = _quitar_nota_interna(FUENTE.read_text(encoding="utf-8"))
    huecos = re.findall(r"\[COMPLETAR[^\]]*\]", md)

    cuerpo = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{cuerpo}</body></html>"
    )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", encoding="utf-8", delete=False
    ) as f:
        f.write(html)
        ruta_html = f.name

    subprocess.run(
        [
            _chrome(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={SALIDA}", f"file://{ruta_html}",
        ],
        check=True,
        capture_output=True,
    )
    Path(ruta_html).unlink(missing_ok=True)

    paginas = None
    if shutil.which("pdfinfo"):
        info = subprocess.run(
            ["pdfinfo", str(SALIDA)], capture_output=True, text=True, check=True
        ).stdout
        m = re.search(r"Pages:\s+(\d+)", info)
        paginas = int(m.group(1)) if m else None

    print(f"PDF generado: {SALIDA}")
    fallo = False
    if paginas is not None:
        estado = "OK" if paginas <= MAX_PAGINAS else f"EXCEDE el máximo de {MAX_PAGINAS}"
        print(f"  páginas: {paginas}  [{estado}]")
        fallo |= paginas > MAX_PAGINAS
    if huecos:
        print(f"  [AVISO] quedan {len(huecos)} huecos [COMPLETAR]:")
        for h in huecos:
            print(f"    - {h[:70]}")
        fallo |= args.estricto
    if args.estricto and fallo:
        sys.exit(1)


if __name__ == "__main__":
    main()
