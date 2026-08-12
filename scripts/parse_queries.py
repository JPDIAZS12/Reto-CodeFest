"""Extrae las 50 preguntas del PDF de ADL y las guarda como queries.jsonl.

El PDF tiene el formato:  qNNN <texto de la pregunta, puede ocupar varias líneas>

Uso:
    python scripts/parse_queries.py "C:\\ruta\\Extracto_Preguntas_50_v2.pdf"
    python scripts/parse_queries.py "<pdf>" --out queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import QUERIES_FILE


def parse_queries_pdf(pdf_path: Path) -> list[dict]:
    """Devuelve [{'query_id': 'q001', 'query': '...'}, ...] desde el PDF."""
    texto = "".join(page.get_text("text") for page in fitz.open(pdf_path))
    # Captura 'qNNN' seguido de todo hasta el próximo 'qNNN' o el fin del texto
    pares = re.findall(r"(q\d{3})\s+(.*?)(?=\s*q\d{3}\b|\Z)", texto, re.DOTALL)
    queries = []
    for qid, pregunta in pares:
        pregunta = re.sub(r"\s+", " ", pregunta).strip()  # colapsa saltos de línea
        queries.append({"query_id": qid, "query": pregunta})
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae queries del PDF a JSON Lines.")
    parser.add_argument("pdf", help="Ruta al PDF con las preguntas (qNNN ...).")
    parser.add_argument("--out", default=str(QUERIES_FILE),
                        help="Archivo de salida JSON Lines. Por defecto: %(default)s")
    args = parser.parse_args()

    queries = parse_queries_pdf(Path(args.pdf))
    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"{len(queries)} consultas escritas en {out}")
    # Aviso si no salieron exactamente 50
    if len(queries) != 50:
        print(f"[AVISO] se esperaban 50 consultas, se obtuvieron {len(queries)}. "
              f"Revisa el parseo del PDF.")
    # Muestra las primeras y últimas para inspección
    for q in queries[:2] + queries[-2:]:
        print(f"  {q['query_id']}: {q['query'][:70]}...")


if __name__ == "__main__":
    main()
