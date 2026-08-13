"""generador.py — produce resultados.jsonl a partir del índice y las consultas.

Entregable 4 del reto: script reproducible que lee el archivo de consultas,
usa la base vectorial y escribe resultados.jsonl con EXACTAMENTE una línea por
consulta, siguiendo el esquema de la Tabla 3 (Sección 9.3).

Contrato de invocación (§1.5, Tabla 1):

    python generador.py --consultas consultas.jsonl \
                        --base-vectorial ./base_vectorial \
                        --salida resultados.jsonl

Los tres argumentos son opcionales; la invocación sin argumentos es equivalente
al comando anterior ejecutado desde la raíz del directorio de entrega.

Requiere haber construido antes la base:  python -m src.build_index
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Este script debe correr en DOS disposiciones distintas:
#   (a) dentro del repo:     <raiz>/entrega/generador.py, con config.py y src/ en <raiz>
#   (b) en el paquete final: <entrega>/generador.py, con config.py y src/ COPIADOS al lado
#
# El orden importa: dentro del repo, entrega/ contiene una COPIA de src/ que dejó
# empaquetar_entrega.py, y si esa copia ganara eclipsaría al src/ real y estarías
# ejecutando código viejo sin notarlo. Por eso, si el directorio padre parece la
# raíz del repo (tiene config.py y src/), esa gana; si no, estamos en el paquete
# entregado y manda el directorio local.
AQUI = Path(__file__).resolve().parent
ES_REPO = (AQUI.parent / "config.py").exists() and (AQUI.parent / "src").is_dir()
if ES_REPO:
    sys.path.insert(0, str(AQUI))
    sys.path.insert(0, str(AQUI.parent))
else:
    sys.path.insert(0, str(AQUI.parent))
    sys.path.insert(0, str(AQUI))

from config import (
    GRAFO_DIR,
    USE_GRAPH,
    ENCODER_SLUG,
    TOP_N_DOCUMENTS,
    TOP_N_FRAGMENTS,
)
from src.encode import Encoder
from src.retrieve import load_base, retrieve

# Rutas por defecto ANCLADAS a la ubicación de este archivo, NO importadas de
# config. config.py deriva las suyas de su propio directorio (ROOT = su
# carpeta), así que al copiarlo dentro de entrega/ apuntaría a
# entrega/entrega/base_vectorial. Anclando aquí, el paquete funciona esté donde
# esté, y coincide con la invocación desde la raíz de la entrega que exige §1.5.
BASE_VECTORIAL_DEFECTO = AQUI / "base_vectorial"
RESULTADOS_DEFECTO = AQUI / "resultados.jsonl"

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


# El contrato nombra el archivo de consultas "consultas.jsonl" en la raíz de la
# entrega. Durante el desarrollo dentro del repo se llama queries.jsonl y vive
# en la raíz del repo; se acepta cualquiera de las dos disposiciones.
_CANDIDATOS_CONSULTAS = [
    AQUI / "consultas.jsonl",
    AQUI / "queries.jsonl",
    AQUI.parent / "consultas.jsonl",
    AQUI.parent / "queries.jsonl",
]
QUERIES_DEFECTO = next(
    (c for c in _CANDIDATOS_CONSULTAS if c.exists()), _CANDIDATOS_CONSULTAS[0]
)


def load_queries(path: Path = QUERIES_DEFECTO) -> list[tuple[str, str]]:
    """Lee el archivo de consultas -> lista de (query_id, texto).
    """
    queries: list[tuple[str, str]] = []
    # split("\n") y no splitlines(): U+2028/U+2029 dentro de un texto partirían
    # el objeto JSON por la mitad (misma regla que en todo el proyecto).
    for linea in Path(path).read_text(encoding="utf-8").split("\n"):
        if not linea.strip():
            continue
        obj = json.loads(linea)
        qid = obj.get("query_id") or obj.get("id")
        texto = obj.get("query") or obj.get("text") or obj.get("consulta")
        queries.append((qid, texto))
    return queries



def build_result_object(query_id: str, retrieved: dict) -> dict:
    """Convierte la salida de retrieve() en el objeto JSON del esquema §9.3.

    `retrieved` tiene la forma:
        {
          "documents": ["DOC-A", "DOC-B", "DOC-C"],               # doc_id ordenados
          "fragments": [{"chunk_id":..., "doc_id":..., "text":...}, ...]  # ordenados
        }
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


# --------------------------------------------------------------------------- #
# Orquestador principal (provisto)
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Argumentos de línea de comandos, con los nombres EXACTOS de la Tabla 1.

    --consultas      : ruta al archivo de consultas (§10.1).
    --base-vectorial : directorio raíz de la base vectorial entregada.
    --salida         : ruta del archivo de resultados a generar (§9).
    """
    parser = argparse.ArgumentParser(
        description="Genera resultados.jsonl a partir del índice y las consultas."
    )
    parser.add_argument(
        "--consultas",
        default=str(QUERIES_DEFECTO),
        help="Ruta al archivo de consultas (JSON Lines). Por defecto: %(default)s",
    )
    parser.add_argument(
        "--base-vectorial",
        dest="base_vectorial",
        default=str(BASE_VECTORIAL_DEFECTO),
        help="Directorio raíz de la base vectorial. Por defecto: %(default)s",
    )
    parser.add_argument(
        "--salida",
        default=str(RESULTADOS_DEFECTO),
        help="Ruta del archivo de resultados a escribir. Por defecto: %(default)s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries_path = Path(args.consultas)
    out_path = Path(args.salida)
    # Todas las rutas a índices y metadata se resuelven relativas al directorio
    # recibido en --base-vectorial, como exige la Tabla 1.
    indice_dir = Path(args.base_vectorial) / f"encoder_{ENCODER_SLUG}"

    print(f"Cargando base vectorial de {indice_dir} y encoder...")
    # Ruta EXPLÍCITA: el default de load_base() sale de config.BASE_VECTORIAL_DIR,
    # que apunta mal cuando config.py está copiado dentro del paquete.
    index, metas = load_base(indice_dir)
    encoder = Encoder()
    grafo, entity_index = load_graph_if_available()

    queries = load_queries(queries_path)
    print(f"{len(queries)} consultas leídas de {queries_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("El re-ranking codifica ~46 sub-fragmentos por consulta: en CPU son "
          "unos 45 s por consulta, en GPU unos pocos segundos.", flush=True)

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as salida:
        for i, (query_id, texto) in enumerate(queries, start=1):
            retrieved = retrieve(texto, index, metas, encoder, graph=grafo, entity_index=entity_index)
            obj = build_result_object(query_id, retrieved)
            salida.write(json.dumps(obj, ensure_ascii=False) + "\n")
            salida.flush()
            transcurrido = time.time() - t0
            restante = transcurrido / i * (len(queries) - i)
            print(f"  [{i:>2}/{len(queries)}] {query_id}  "
                  f"({transcurrido/60:.1f} min transcurridos, "
                  f"~{restante/60:.0f} min restantes)", flush=True)

    print(f"Listo. {len(queries)} líneas escritas en {out_path}")


if __name__ == "__main__":
    main()
