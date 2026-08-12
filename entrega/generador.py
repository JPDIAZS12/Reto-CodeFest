"""generador.py — produce resultados.jsonl a partir del índice y las consultas.

Entregable 4 del reto: script reproducible que lee el archivo de consultas,
usa la base vectorial y escribe entrega/resultados.jsonl.

Uso:  python entrega/generador.py
Uso:  python generador.py

Requiere haber construido antes la base:  python -m src.build_index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ES_REPO = (AQUI.parent / "config.py").exists() and (AQUI.parent / "src").is_dir()
if ES_REPO:
    sys.path.insert(0, str(AQUI))
    sys.path.insert(0, str(AQUI.parent))
else:
    sys.path.insert(0, str(AQUI.parent))
    sys.path.insert(0, str(AQUI))

from config import ENCODER_SLUG, TOP_N_DOCUMENTS, TOP_N_FRAGMENTS
from src.encode import Encoder
from src.retrieve import load_base, retrieve

INDICE_DIR = AQUI / "base_vectorial" / f"encoder_{ENCODER_SLUG}"
RESULTADOS_DEFECTO = AQUI / "resultados.jsonl"

if (AQUI / "queries.jsonl").exists():
    QUERIES_DEFECTO = AQUI / "queries.jsonl"
else:
    QUERIES_DEFECTO = AQUI.parent / "queries.jsonl"


def load_queries(path: Path = QUERIES_DEFECTO) -> list[tuple[str, str]]:
    """Lee el archivo de consultas -> lista de (query_id, texto).
    """
    queries: list[tuple[str, str]] = []
    for linea in Path(path).read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        obj = json.loads(linea)
        qid = obj.get("query_id") or obj.get("id")
        texto = obj.get("query") or obj.get("text") or obj.get("consulta")
        queries.append((qid, texto))
    return queries



def build_result_object(query_id: str, retrieved: dict) -> dict:
    """Convierte la salida de retrieve() en el objeto JSON del esquema Tabla 2.
    """
    lista_docs = []
    for indice, doc_id in enumerate(retrieved["documents"][:TOP_N_DOCUMENTS], start=1):
        lista_docs.append({"rank": indice, "doc_id": doc_id})
        
    lista_fragmentos = []
    
    for indice, fragmento in enumerate(retrieved["fragments"][:TOP_N_FRAGMENTS], start=1):
        lista_fragmentos.append({
            "rank": indice,
            "chunk_id": fragmento["chunk_id"],
            "doc_id": fragmento["doc_id"],
            "text": fragmento["text"]
        })

    dict_resultado = {
        "query_id": query_id,
        "documents": lista_docs,
        "fragments": lista_fragmentos
    }
    
    return dict_resultado


#Orquestador principal
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Genera resultados.jsonl a partir del índice y las consultas."
    )
    parser.add_argument(
        "--queries",
        default=str(QUERIES_DEFECTO),
        help="Ruta al archivo de consultas (JSON Lines). Por defecto: %(default)s",
    )
    parser.add_argument(
        "--out",
        default=str(RESULTADOS_DEFECTO),
        help="Ruta del archivo de resultados a escribir. Por defecto: %(default)s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries_path = Path(args.queries)
    out_path = Path(args.out)

    print(f"Cargando base vectorial de {INDICE_DIR} y encoder...")
    
    index, metas = load_base(INDICE_DIR)
    encoder = Encoder()

    queries = load_queries(queries_path)
    print(f"{len(queries)} consultas leídas de {queries_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as salida:
        for query_id, texto in queries:
            retrieved = retrieve(texto, index, metas, encoder)
            obj = build_result_object(query_id, retrieved)
            salida.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Listo. {len(queries)} líneas escritas en {out_path}")


if __name__ == "__main__":
    main()
