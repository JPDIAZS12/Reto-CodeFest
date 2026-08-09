"""Grafo de conocimiento (Sección 7, componente bonus).

Construcción (Sección 7.2):
    1. NER: se reconocen entidades por fragmento con un modelo encoder
       multilingüe (es/en/pt) de HuggingFace (GRAPH_NER_MODEL).
    2. RE: para cada par de entidades del mismo fragmento cuya distancia en
       caracteres es corta (~misma oración), se infiere el tipo de relación
       mediante una heurística de patrones lingüísticos (verbos clave en los
       3 idiomas); si no se reconoce ninguno se usa la relación genérica
       "se_relaciona_con".
    3. Cada tripleta (sujeto, relación, objeto) guarda como evidencia el
       doc_id y chunk_id de origen (trazabilidad, Sección 7.2 punto 3).

El grafo resultante (nx.MultiDiGraph) se exporta a GraphML (Sección 7.3 /
entregable 1) y se usa en la recuperación (Sección 8.5) para complementar la
búsqueda vectorial: las entidades de la consulta se buscan en el grafo, se
recuperan sus vecinos de primer orden y los chunks de evidencia se fusionan
con los resultados de FAISS mediante RRF (véase src/retrieve.combine_rrf).

Uso:  python -m src.graph
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx

from config import (
    BASE_VECTORIAL_DIR,
    ENCODER_SLUG,
    GRAFO_DIR,
    GRAPH_NER_MODEL,
    GRAPH_MAX_GAP_CHARS,
    GRAPH_MAX_ENTITIES_PER_FRAGMENT,
    GRAPH_MAX_EVIDENCE_PER_EDGE,
)

# --------------------------------------------------------------------------- #
# Heurística de extracción de relaciones (RE): verbos/keywords -> etiqueta.
# Se busca por subcadena (insensible a mayúsculas) en el texto entre las dos
# entidades, cubriendo español, inglés y portugués con raíces compartidas.
# --------------------------------------------------------------------------- #
_RELATION_KEYWORDS: dict[str, str] = {
    "desarroll": "desarrolla",
    "develop": "desarrolla",
    "desenvolv": "desarrolla",
    "regul": "regula",
    "financi": "financia",
    "fund": "financia",
    "coopera": "coopera_con",
    "colabor": "coopera_con",
    "lidera": "lidera",
    "lead": "lidera",
    "opera": "opera",
    "investig": "investiga",
    "invest": "investiga",
    "amenaz": "amenaza",
    "threat": "amenaza",
    "ameaç": "amenaza",
    "afect": "afecta",
    "affect": "afecta",
    "afeta": "afecta",
    "colision": "colisiona_con",
    "collide": "colisiona_con",
    "colid": "colisiona_con",
    "proh": "prohibe",
    "ban": "prohibe",
    "proib": "prohibe",
    "firm": "firma_acuerdo_con",
    "sign": "firma_acuerdo_con",
    "assin": "firma_acuerdo_con",
    "pertenc": "pertenece_a",
    "belong": "pertenece_a",
    "miembro": "es_miembro_de",
    "member": "es_miembro_de",
    "membro": "es_miembro_de",
    "aprueba": "aprueba",
    "approve": "aprueba",
    "aprova": "aprueba",
    "implement": "implementa",
    "utiliz": "utiliza",
    "usa": "utiliza",
    "use": "utiliza",
    "reduc": "reduce",
    "reduz": "reduce",
    "increment": "incrementa",
    "increase": "incrementa",
    "aument": "incrementa",
    "gener": "genera",
    "genera": "genera",
}
_RELACION_GENERICA = "se_relaciona_con"


_ner_pipeline = None


def get_ner_pipeline(model_name: str = GRAPH_NER_MODEL):
    """Carga (una sola vez, perezosamente) el pipeline de NER multilingüe."""
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        _ner_pipeline = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def _normalize_entity(text: str) -> str:
    text = text.strip(" .,;:()[]{}\"'“”«»")
    return re.sub(r"\s+", " ", text)


def extract_entities(text: str) -> list[dict]:
    """NER sobre `text`. Devuelve entidades con posición de carácter:
    [{"text":..., "label":..., "start":..., "end":...}, ...]
    """
    if not text or not text.strip():
        return []
    ner = get_ner_pipeline()
    crudas = ner(text, truncation=True)
    entidades = []
    for e in crudas:
        nombre = _normalize_entity(e["word"])
        if len(nombre) < 2:
            continue
        entidades.append({
            "text": nombre,
            "label": e["entity_group"],
            "start": int(e["start"]) if e.get("start") is not None else 0,
            "end": int(e["end"]) if e.get("end") is not None else len(nombre),
        })
    return entidades


def infer_relation(gap_text: str) -> str:
    """Heurística de patrones lingüísticos (Sección 7.2, RE): busca un verbo
    o keyword relevante en el texto entre dos entidades. Si no reconoce
    ninguno, devuelve la relación genérica de co-ocurrencia."""
    gap_lower = gap_text.lower()
    for keyword, label in _RELATION_KEYWORDS.items():
        if keyword in gap_lower:
            return label
    return _RELACION_GENERICA


# --------------------------------------------------------------------------- #
# Construcción del grafo
# --------------------------------------------------------------------------- #
def build_graph(fragments: list[dict], verbose: bool = True) -> nx.MultiDiGraph:
    """Construye el grafo de conocimiento a partir de fragmentos con metadata
    (se esperan al menos los campos doc_id, chunk_id, texto de la Tabla 1).

    Devuelve un nx.MultiDiGraph con:
      - nodos = entidades, atributos: label (tipo NER), mentions (conteo),
        doc_ids / chunk_ids (evidencia, cadenas separadas por coma).
      - aristas = relaciones, atributos: relation, weight (conteo),
        doc_ids / chunk_ids (evidencia).
    """
    node_mentions: dict[str, int] = defaultdict(int)
    node_label: dict[str, str] = {}
    node_docs: dict[str, set] = defaultdict(set)
    node_chunks: dict[str, set] = defaultdict(set)

    edge_weight: dict[tuple[str, str, str], int] = defaultdict(int)
    edge_docs: dict[tuple[str, str, str], set] = defaultdict(set)
    edge_chunks: dict[tuple[str, str, str], set] = defaultdict(set)

    total = len(fragments)
    for i, frag in enumerate(fragments):
        if verbose and total and i % 200 == 0:
            print(f"  NER: fragmento {i}/{total}")
        texto = frag.get("texto", "")
        doc_id = frag.get("doc_id", "")
        chunk_id = frag.get("chunk_id", "")
        if not texto:
            continue

        entidades = extract_entities(texto)
        if len(entidades) > GRAPH_MAX_ENTITIES_PER_FRAGMENT:
            entidades = entidades[:GRAPH_MAX_ENTITIES_PER_FRAGMENT]

        for e in entidades:
            key = e["text"]
            node_mentions[key] += 1
            node_label.setdefault(key, e["label"])
            node_docs[key].add(doc_id)
            node_chunks[key].add(chunk_id)

        for a in range(len(entidades)):
            for b in range(a + 1, len(entidades)):
                ent_a, ent_b = entidades[a], entidades[b]
                if ent_a["text"] == ent_b["text"]:
                    continue
                gap_start = min(ent_a["end"], ent_b["end"])
                gap_end = max(ent_a["start"], ent_b["start"])
                if gap_end < gap_start or (gap_end - gap_start) > GRAPH_MAX_GAP_CHARS:
                    continue
                relacion = infer_relation(texto[gap_start:gap_end])
                clave = (ent_a["text"], ent_b["text"], relacion)
                edge_weight[clave] += 1
                edge_docs[clave].add(doc_id)
                edge_chunks[clave].add(chunk_id)

    g = nx.MultiDiGraph()
    for key, mentions in node_mentions.items():
        g.add_node(
            key,
            label=node_label.get(key, ""),
            mentions=mentions,
            doc_ids=",".join(sorted(node_docs[key])[:GRAPH_MAX_EVIDENCE_PER_EDGE]),
            chunk_ids=",".join(sorted(node_chunks[key])[:GRAPH_MAX_EVIDENCE_PER_EDGE]),
        )
    for (sujeto, objeto, relacion), weight in edge_weight.items():
        clave = (sujeto, objeto, relacion)
        g.add_edge(
            sujeto,
            objeto,
            relation=relacion,
            weight=weight,
            doc_ids=",".join(sorted(edge_docs[clave])[:GRAPH_MAX_EVIDENCE_PER_EDGE]),
            chunk_ids=",".join(sorted(edge_chunks[clave])[:GRAPH_MAX_EVIDENCE_PER_EDGE]),
        )

    if verbose:
        print(f"  grafo: {g.number_of_nodes()} entidades, {g.number_of_edges()} relaciones")
    return g


def save_graph(g: nx.MultiDiGraph, out_dir: Path = GRAFO_DIR) -> Path:
    """Exporta el grafo a GraphML (formato exigido por el entregable 1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "grafo.graphml"
    nx.write_graphml(g, path)
    return path


def load_graph(path: Path | None = None) -> nx.MultiDiGraph:
    """Carga un grafo previamente exportado a GraphML."""
    if path is None:
        path = GRAFO_DIR / "grafo.graphml"
    return nx.read_graphml(path)


def build_entity_index(g: nx.MultiDiGraph) -> dict[str, str]:
    """Índice de búsqueda insensible a mayúsculas: texto normalizado -> nodo."""
    return {node.lower(): node for node in g.nodes}


# --------------------------------------------------------------------------- #
# Uso en recuperación (Sección 8.5)
# --------------------------------------------------------------------------- #
def graph_candidates(
    query: str,
    g: nx.MultiDiGraph,
    metas_by_chunk: dict[str, dict],
    entity_index: dict[str, str] | None = None,
) -> list[tuple[float, dict]]:
    """Recupera chunks candidatos a partir del grafo para una consulta.

    Pasos (Sección 8.5):
      1. NER sobre la consulta con el mismo componente usado al construir
         el grafo.
      2. Se buscan esas entidades como nodos del grafo.
      3. Se toman sus vecinos de primer orden (sucesores + predecesores).
      4. Se puntúa cada chunk de evidencia por el número de relaciones
         relevantes en las que aparece (nodos emparejados + aristas a sus
         vecinos).

    Devuelve una lista (puntuación, metadata) ordenada de mayor a menor,
    con la misma forma que src.retrieve.search(), lista para fusionarse por
    RRF con los resultados vectoriales.
    """
    if g is None or g.number_of_nodes() == 0:
        return []
    if entity_index is None:
        entity_index = build_entity_index(g)

    entidades_query = extract_entities(query)
    nodos_emparejados = []
    for e in entidades_query:
        nodo = entity_index.get(e["text"].lower())
        if nodo is not None:
            nodos_emparejados.append(nodo)

    if not nodos_emparejados:
        return []

    puntajes: dict[str, float] = defaultdict(float)

    def _sumar_evidencia(chunk_ids_str: str, peso: float) -> None:
        for cid in chunk_ids_str.split(","):
            cid = cid.strip()
            if cid:
                puntajes[cid] += peso

    for nodo in nodos_emparejados:
        # Evidencia directa: menciones de la propia entidad.
        _sumar_evidencia(g.nodes[nodo].get("chunk_ids", ""), 1.0)

        # Vecinos de primer orden (relaciones salientes y entrantes).
        for _, vecino, datos in g.out_edges(nodo, data=True):
            _sumar_evidencia(datos.get("chunk_ids", ""), 1.0)
        for vecino, _, datos in g.in_edges(nodo, data=True):
            _sumar_evidencia(datos.get("chunk_ids", ""), 1.0)

    candidatos = []
    for chunk_id, puntaje in puntajes.items():
        meta = metas_by_chunk.get(chunk_id)
        if meta is not None:
            candidatos.append((puntaje, meta))

    candidatos.sort(key=lambda par: par[0], reverse=True)
    return candidatos


# --------------------------------------------------------------------------- #
# CLI: construye el grafo a partir de la base vectorial ya indexada.
# --------------------------------------------------------------------------- #
def _load_fragments_from_base(out_dir: Path | None = None) -> list[dict]:
    if out_dir is None:
        out_dir = BASE_VECTORIAL_DIR / f"encoder_{ENCODER_SLUG}"
    lineas = (out_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lineas if l.strip()]


def main() -> None:
    print("Cargando fragmentos de la base vectorial ya construida...")
    fragmentos = _load_fragments_from_base()
    if not fragmentos:
        print("[ERROR] metadata.jsonl vacío. Ejecuta antes: python -m src.build_index")
        return
    print(f"  fragmentos: {len(fragmentos)}")

    print("Extrayendo entidades y relaciones (NER + heurística)...")
    g = build_graph(fragmentos)

    path = save_graph(g)
    print(f"Listo. Grafo guardado en {path}")


if __name__ == "__main__":
    main()
