"""generador.py — produce resultados.jsonl a partir del índice y las consultas.

Entregable 4 del reto: script reproducible que lee el archivo de consultas,
usa la base vectorial y escribe entrega/resultados.jsonl con EXACTAMENTE una
línea por consulta, siguiendo el esquema de la Tabla 2 (Sección 9.3).

Uso (desde la raíz del proyecto):  python entrega/generador.py

Requiere haber construido antes la base:  python -m src.build_index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# entrega/generador.py vive dentro de entrega/; la raíz del proyecto (donde
# están config.py y src/) es el directorio padre. Lo añadimos al path para
# que los imports funcionen al ejecutar el script desde la entrega.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    QUERIES_FILE,
    RESULTADOS_FILE,
    GRAFO_DIR,
    USE_GRAPH,
    TOP_N_DOCUMENTS,
    TOP_N_FRAGMENTS,
)
from src.encode import Encoder
from src.retrieve import load_base, retrieve


def load_graph_if_available():
    """Carga el grafo de conocimiento opcional (Sección 7) si fue construido
    con `python -m src.graph`. Si no existe, la recuperación sigue siendo
    puramente vectorial (el grafo es un componente bonus, no obligatorio).
    """
    ruta = GRAFO_DIR / "grafo.graphml"
    if not USE_GRAPH or not ruta.exists():
        return None, None
    from src.graph import load_graph, build_entity_index
    print(f"Grafo de conocimiento encontrado en {ruta}, se usará como señal adicional.")
    grafo = load_graph(ruta)
    return grafo, build_entity_index(grafo)


def load_queries(path: Path = QUERIES_FILE) -> list[tuple[str, str]]:
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

    `retrieved` tiene la forma:
        {
          "documents": ["DOC-A", "DOC-B", "DOC-C", ...],          # doc_id ordenados
          "fragments": [{"chunk_id":..., "doc_id":..., "text":...}, ...]  # ordenados
        }

    Debes devolver un dict con ESTA estructura exacta:
        {
          "query_id": "q001",
          "documents": [
              {"rank": 1, "doc_id": "DOC-A"},
              {"rank": 2, "doc_id": "DOC-B"},
              {"rank": 3, "doc_id": "DOC-C"}
          ],
          "fragments": [
              {"rank": 1, "chunk_id": "...", "doc_id": "...", "text": "..."},
              ... (hasta TOP_N_FRAGMENTS)
          ]
        }

    Pistas:
      - El rank empieza en 1 (no en 0). Con un for + enumerate(..., start=1)
        o llevando un contador, según tu estilo.
      - documents: recorre retrieved["documents"][:TOP_N_DOCUMENTS] y crea un
        dict {"rank": r, "doc_id": d} por cada uno.
      - fragments: recorre retrieved["fragments"][:TOP_N_FRAGMENTS] y crea un
        dict {"rank": r, "chunk_id":..., "doc_id":..., "text":...} por cada uno.
      - Devuelve {"query_id": query_id, "documents": [...], "fragments": [...]}.
    """
    # TODO(tú): construir el objeto de resultado con ranks (esquema Tabla 2).
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


# --------------------------------------------------------------------------- #
# Orquestador principal (provisto)
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Lee los argumentos de línea de comandos.

    --queries : ruta al archivo de consultas de ADL (por defecto, el de config).
    --out     : ruta del archivo de resultados a escribir (por defecto, config).
    """
    parser = argparse.ArgumentParser(
        description="Genera resultados.jsonl a partir del índice y las consultas."
    )
    parser.add_argument(
        "--queries",
        default=str(QUERIES_FILE),
        help="Ruta al archivo de consultas (JSON Lines). Por defecto: %(default)s",
    )
    parser.add_argument(
        "--out",
        default=str(RESULTADOS_FILE),
        help="Ruta del archivo de resultados a escribir. Por defecto: %(default)s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries_path = Path(args.queries)
    out_path = Path(args.out)

    print("Cargando base vectorial y encoder...")
    index, metas = load_base()
    encoder = Encoder()
    grafo, entity_index = load_graph_if_available()

    queries = load_queries(queries_path)
    print(f"{len(queries)} consultas leídas de {queries_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as salida:
        for query_id, texto in queries:
            retrieved = retrieve(texto, index, metas, encoder, graph=grafo, entity_index=entity_index)
            obj = build_result_object(query_id, retrieved)
            salida.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Listo. {len(queries)} líneas escritas en {out_path}")


if __name__ == "__main__":
    main()
