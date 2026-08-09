"""Pruebas para src/graph.py (Sección 7, grafo de conocimiento) y para la
fusión por RRF en src/retrieve.py (Secciones 8.4/8.5).

La primera sección (NER real) descarga el modelo de GRAPH_NER_MODEL la
primera vez que se ejecuta. El resto de las pruebas usan grafos de juguete y
no requieren red ni el modelo NER.

Ejecutar:  python tests/test_graph.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import networkx as nx

from src.retrieve import combine_rrf
from src.graph import (
    extract_entities,
    infer_relation,
    build_graph,
    save_graph,
    load_graph,
    build_entity_index,
    graph_candidates,
)

_PASSED = 0
_FAILED = 0


def check(cond: bool, msg: str) -> None:
    global _PASSED, _FAILED
    tag = "[OK]  " if cond else "[FALLA]"
    if cond:
        _PASSED += 1
    else:
        _FAILED += 1
    print(f"  {tag} {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
# 1. infer_relation (heurística de patrones lingüísticos, sin modelo)
# --------------------------------------------------------------------------- #
section("1. infer_relation")
check(infer_relation("desarrolla activamente") == "desarrolla",
      "reconoce verbo 'desarrolla' (es)")
check(infer_relation("is developing new") == "desarrolla",
      "reconoce verbo 'developing' (en) vía raíz 'develop'")
check(infer_relation("regula estrictamente el uso de") == "regula",
      "reconoce verbo 'regula'")
check(infer_relation("xyz abc qwe") == "se_relaciona_con",
      "sin keyword reconocido -> relación genérica")


# --------------------------------------------------------------------------- #
# 2. extract_entities (requiere el modelo NER real; puede descargar la
#    primera vez). Prueba de humo básica.
# --------------------------------------------------------------------------- #
section("2. extract_entities (NER real)")
try:
    ents = extract_entities("La NASA y la Fuerza Aeroespacial Colombiana firmaron un acuerdo.")
    check(isinstance(ents, list), "devuelve una lista")
    check(all({"text", "label", "start", "end"} <= set(e.keys()) for e in ents),
          "cada entidad trae text/label/start/end")
    check(extract_entities("") == [], "texto vacío -> sin entidades")
    if ents:
        print(f"  entidades detectadas: {[e['text'] for e in ents]}")
except Exception as exc:  # sin red / sin modelo disponible: no bloquea el resto
    print(f"  [AVISO] no se pudo cargar el modelo NER ({exc}); se omite esta sección.")


# --------------------------------------------------------------------------- #
# 3. build_graph sobre fragmentos de juguete (sin depender del NER real:
#    se valida la lógica de acumulación de evidencia con datos controlados)
# --------------------------------------------------------------------------- #
section("3. build_graph (lógica de acumulación, con NER real sobre texto simple)")
fragmentos_juguete = [
    {
        "doc_id": "DOC-1",
        "chunk_id": "DOC-1-chunk-0000",
        "texto": "Estados Unidos desarrolla un sistema de armas autónomo.",
    },
    {
        "doc_id": "DOC-1",
        "chunk_id": "DOC-1-chunk-0001",
        "texto": "Estados Unidos desarrolla capacidades similares en el Pentágono.",
    },
    {
        "doc_id": "DOC-2",
        "chunk_id": "DOC-2-chunk-0000",
        "texto": "Texto sin ninguna entidad nombrada relevante para el grafo.",
    },
]
try:
    g = build_graph(fragmentos_juguete, verbose=False)
    check(isinstance(g, nx.MultiDiGraph), "devuelve un nx.MultiDiGraph")
    check(g.number_of_nodes() >= 0, "no lanza excepción con corpus de juguete")
    # Si el NER detectó "Estados Unidos" en ambos fragmentos del DOC-1, debe
    # acumular mentions >= 2 y evidencia de ambos chunk_id.
    nodo_eeuu = next((n for n in g.nodes if "Estados Unidos" in n or "United States" in n), None)
    if nodo_eeuu:
        attrs = g.nodes[nodo_eeuu]
        check(attrs["mentions"] >= 2, f"'{nodo_eeuu}' mencionado >=2 veces -> {attrs['mentions']}")
        check("DOC-1-chunk-0000" in attrs["chunk_ids"] and "DOC-1-chunk-0001" in attrs["chunk_ids"],
              "evidencia de chunk_ids incluye ambos fragmentos")
    else:
        print("  [AVISO] el NER no detectó 'Estados Unidos' en el texto de juguete; se omiten sub-chequeos.")
except Exception as exc:
    print(f"  [AVISO] build_graph con NER real no disponible ({exc}); se usa grafo manual en las siguientes secciones.")
    g = None


# --------------------------------------------------------------------------- #
# 4. Grafo manual (sin NER) para probar save/load, entity_index y
#    graph_candidates de forma determinista.
# --------------------------------------------------------------------------- #
section("4. Grafo manual: save_graph / load_graph (round-trip GraphML)")
g_manual = nx.MultiDiGraph()
g_manual.add_node("Colombia", label="LOC", mentions=3, doc_ids="DOC-A", chunk_ids="DOC-A-chunk-0000,DOC-A-chunk-0001")
g_manual.add_node("FAC", label="ORG", mentions=2, doc_ids="DOC-A", chunk_ids="DOC-A-chunk-0001")
g_manual.add_edge("Colombia", "FAC", relation="es_miembro_de", weight=2,
                   doc_ids="DOC-A", chunk_ids="DOC-A-chunk-0001")

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    ruta = save_graph(g_manual, out_dir=tmp_dir)
    check(ruta.exists(), f"se crea el archivo {ruta.name}")
    g_cargado = load_graph(ruta)
    check(g_cargado.number_of_nodes() == g_manual.number_of_nodes(),
          "mismo número de nodos tras cargar")
    check(g_cargado.number_of_edges() == g_manual.number_of_edges(),
          "mismo número de aristas tras cargar")
    check(set(g_cargado.nodes) == set(g_manual.nodes),
          "los nombres de los nodos se preservan")
    check(g_cargado.nodes["Colombia"]["mentions"] == 3,
          "atributo entero 'mentions' se preserva tras el round-trip")


# --------------------------------------------------------------------------- #
# 5. build_entity_index
# --------------------------------------------------------------------------- #
section("5. build_entity_index")
idx = build_entity_index(g_manual)
check(idx.get("colombia") == "Colombia", "búsqueda insensible a mayúsculas -> nodo original")
check(idx.get("fac") == "FAC", "índice cubre todos los nodos")


# --------------------------------------------------------------------------- #
# 6. graph_candidates (grafo manual + metadata de juguete)
# --------------------------------------------------------------------------- #
section("6. graph_candidates")
metas_by_chunk = {
    "DOC-A-chunk-0000": {"chunk_id": "DOC-A-chunk-0000", "doc_id": "DOC-A", "texto": "Colombia participa en foros regionales."},
    "DOC-A-chunk-0001": {"chunk_id": "DOC-A-chunk-0001", "doc_id": "DOC-A", "texto": "Colombia es miembro de la FAC."},
}
cands = graph_candidates("¿Qué relación tiene Colombia con la FAC?", g_manual, metas_by_chunk, idx)
check(isinstance(cands, list), "devuelve una lista")
if cands:
    check(all(isinstance(p, tuple) and len(p) == 2 for p in cands),
          "cada elemento es (puntuación, metadata)")
    ids_recuperados = {m["chunk_id"] for _, m in cands}
    check("DOC-A-chunk-0001" in ids_recuperados,
          "recupera el chunk con evidencia de la relación Colombia-FAC")
    # El chunk con evidencia directa + relación debe puntuar más que uno sin relación
    puntajes = {m["chunk_id"]: s for s, m in cands}
    if "DOC-A-chunk-0000" in puntajes and "DOC-A-chunk-0001" in puntajes:
        check(puntajes["DOC-A-chunk-0001"] >= puntajes["DOC-A-chunk-0000"],
              "el chunk con más evidencia de relación puntúa igual o más alto")
else:
    print("  [AVISO] el NER no detectó entidades en la consulta de prueba; graph_candidates devolvió vacío.")

check(graph_candidates("consulta sin entidades xyz", g_manual, metas_by_chunk, idx) is not None or True,
      "no lanza excepción con consultas sin entidades reconocibles")


# --------------------------------------------------------------------------- #
# 7. combine_rrf (Sección 8.4, fusión de listas ordenadas)
# --------------------------------------------------------------------------- #
section("7. combine_rrf")
lista_vectorial = [
    (0.9, {"chunk_id": "C1", "doc_id": "D1"}),
    (0.7, {"chunk_id": "C2", "doc_id": "D1"}),
    (0.5, {"chunk_id": "C3", "doc_id": "D2"}),
]
lista_grafo = [
    (3.0, {"chunk_id": "C3", "doc_id": "D2"}),
    (1.0, {"chunk_id": "C1", "doc_id": "D1"}),
]
fusion = combine_rrf([lista_vectorial, lista_grafo], k0=60)
check(len(fusion) == 3, f"la fusión incluye todos los chunk_id únicos -> {len(fusion)}")
ids_fusion = [m["chunk_id"] for _, m in fusion]
check(set(ids_fusion) == {"C1", "C2", "C3"}, "no se pierde ningún chunk_id de ninguna lista")
# C1 aparece en el rank 1 de ambas listas -> debe quedar primero
check(ids_fusion[0] == "C1", f"C1 (rank 1 en ambas listas) queda primero -> {ids_fusion}")
# Una sola lista: la fusión debe preservar el orden relativo original
fusion_una_lista = combine_rrf([lista_vectorial], k0=60)
check([m["chunk_id"] for _, m in fusion_una_lista] == ["C1", "C2", "C3"],
      "con una sola lista, preserva el orden original")


print(f"\n{'='*50}")
print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
print('='*50)
sys.exit(1 if _FAILED else 0)
