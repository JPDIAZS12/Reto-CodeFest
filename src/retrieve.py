"""Módulo de recuperación (Secciones 8 y 9).

Dada una consulta en lenguaje natural produce los DOS niveles de resultado:
  - documents: top-3 doc_id  (se evalúa con F1@3)
  - fragments: top-10 fragmentos ≤250 palabras (se evalúa con NDCG@10)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from config import (
    BASE_VECTORIAL_DIR,
    ENCODER_SLUG,
    TOP_K_CHUNKS_SEARCH,
    TOP_N_FRAGMENTS,
    TOP_N_DOCUMENTS,
    MAX_WORDS_PER_FRAGMENT,
    DOC_AGGREGATION,
    GRAPH_RRF_K0,
    TOP_M_CHUNKS_POR_DOC,
    DEDUP_JACCARD,
    RERANK_FRAGMENTOS,
    RERANK_POOL_CHUNKS,
)
from src.encode import Encoder
from src.chunk import split_sentences


def load_base(out_dir: Path | None = None) -> tuple[faiss.Index, list[dict]]:
    """Carga index.faiss y metadata.jsonl. La línea i de metas describe el
    vector con id interno i (invariante garantizada por build_index)."""
    if out_dir is None:
        out_dir = BASE_VECTORIAL_DIR / f"encoder_{ENCODER_SLUG}"
    index = faiss.read_index(str(out_dir / "index.faiss"))
    lineas = (out_dir / "metadata.jsonl").read_text(encoding="utf-8").split("\n")
    metas = [json.loads(l) for l in lineas if l.strip()]
    return index, metas


def search(
    query: str,
    index: faiss.Index,
    metas: list[dict],
    encoder: Encoder,
    k: int = TOP_K_CHUNKS_SEARCH,
    query_embedding: np.ndarray | None = None,
) -> list[tuple[float, dict]]:
    """Busca la consulta y devuelve hasta k pares (similitud, metadata),
    ordenados de mayor a menor similitud.

    `query_embedding` permite reutilizar un embedding ya calculado de la
    consulta (forma (1, dim)) y evitar codificarla dos veces.
    """
    if query_embedding is None:
        query_embedding = encoder.encode_queries([query])
    distancias, ids = index.search(query_embedding, k)
    resultados = []
    for i in range(len(ids[0])):
        if ids[0][i] != -1:
            resultados.append((float(distancias[0][i]), metas[ids[0][i]]))
    
    return resultados
    


def combine_rrf(
    ranked_lists: list[list[tuple[float, dict]]],
    k0: int = GRAPH_RRF_K0,
) -> list[tuple[float, dict]]:
    """Reciprocal Rank Fusion (Sección 8.4, Ecuación 7).

    Combina varias listas ya ordenadas de mayor a menor puntuación (cada
    elemento (score, metadata)) en una sola lista fusionada, usando el RANGO
    de cada fragmento en cada lista en vez de su puntuación cruda. Esto es lo
    que permite tratar el pool del grafo de conocimiento (Sección 8.5) como
    "un índice adicional" pese a tener una escala de puntuación distinta a la
    similitud coseno de FAISS.

    Cada fragmento se identifica por su chunk_id.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    meta_by_id: dict[str, dict] = {}
    for lista in ranked_lists:
        for rank, (_, metadata) in enumerate(lista, start=1):
            chunk_id = metadata["chunk_id"]
            meta_by_id[chunk_id] = metadata
            rrf_scores[chunk_id] += 1.0 / (k0 + rank)

    combinados = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(score, meta_by_id[chunk_id]) for chunk_id, score in combinados]


def top_documents(
    scored: list[tuple[float, dict]],
    n: int = TOP_N_DOCUMENTS,
) -> list[str]:
    """Agrega los fragmentos por doc_id y devuelve los n mejores doc_id.
    """
    doc_scores = defaultdict(float)

    if DOC_AGGREGATION == "max":
        for similitud, metadata in scored:
            doc_id = metadata["doc_id"]
            doc_scores[doc_id] = max(doc_scores[doc_id], similitud)

    elif DOC_AGGREGATION == "sum":
        for similitud, metadata in scored:
            doc_id = metadata["doc_id"]
            doc_scores[doc_id] += similitud
            

    elif DOC_AGGREGATION == "mean":
        doc_counts = defaultdict(int)
        for similitud, metadata in scored:
            doc_id = metadata["doc_id"]
            doc_scores[doc_id] += similitud
            doc_counts[doc_id] += 1
        for doc_id in doc_scores:
            doc_scores[doc_id] /= doc_counts[doc_id]

    elif DOC_AGGREGATION == "topm":
        dict_doc_chunks = defaultdict(list)
        for similitud, metadata in scored:
            doc_id = metadata["doc_id"]
            dict_doc_chunks[doc_id].append(similitud)
        for doc_id, similitudes in dict_doc_chunks.items():
            # sorted() y no confiar en el orden del pool: la suma debe ser de
            # los m MEJORES chunks aunque `scored` llegue desordenado.
            doc_scores[doc_id] = sum(
                sorted(similitudes, reverse=True)[:TOP_M_CHUNKS_POR_DOC]
            )

    else:
        raise ValueError(f"Método de agregación a nivel documento no conocido: {DOC_AGGREGATION}")


    documentos_ordenados = sorted(doc_scores.items(), key=lambda similitud: similitud[1], reverse=True)

    lista_n_ids = []

    for i in range(min(n, len(documentos_ordenados))):
        lista_n_ids.append(documentos_ordenados[i][0])

    return lista_n_ids



def _split_by_words(text: str, max_words: int = MAX_WORDS_PER_FRAGMENT) -> list[str]:
    """Divide `text` en sub-fragmentos de a lo sumo `max_words` palabras,
    sin cortar oraciones.

    """
    sub_fragmentos = []
    buffer = []
    palabras_sub_fragmento_actual = 0
    
    if len(text.split()) <= max_words:
        return [text]
    
    for oracion in split_sentences(text):
        cantidad_palabras_oracion = len(oracion.split())
        if cantidad_palabras_oracion > max_words: 
            if buffer:
                sub_fragmento_valido = " ".join(buffer)
                sub_fragmentos.append(sub_fragmento_valido)
                buffer = [] 
                palabras_sub_fragmento_actual = 0
            palabras = oracion.split()
            for inicio in range(0, len(palabras), max_words):
                sub_fragmento_duro = " ".join(palabras[inicio: inicio + max_words])
                sub_fragmentos.append(sub_fragmento_duro)
            continue
        if palabras_sub_fragmento_actual + cantidad_palabras_oracion > max_words:
            sub_fragmento_valido = " ".join(buffer)
            sub_fragmentos.append(sub_fragmento_valido)
            buffer = [oracion]
            palabras_sub_fragmento_actual = cantidad_palabras_oracion
        else:
            buffer.append(oracion)      
            palabras_sub_fragmento_actual += cantidad_palabras_oracion    

    if buffer:
        restante = " ".join(buffer)
        sub_fragmentos.append(restante)
        
    return sub_fragmentos
                

def _es_duplicado(
    texto: str,
    textos_previos: list[str],
    umbral: float = DEDUP_JACCARD,
) -> bool:
    """True si `texto` es casi-duplicado de ALGUNO de `textos_previos`.
    """
    palabras = set(texto.lower().split())
    for prev in textos_previos:
        prev_palabras = set(prev.lower().split())
        interseccion = palabras & prev_palabras
        union = palabras | prev_palabras
        jaccard = len(interseccion) / max(len(union), 1)
        if jaccard >= umbral:
            return True
    
    return False
    
    


def _sub_candidatos(
    scored: list[tuple[float, dict]],
    max_words: int,
) -> list[dict]:
    """Divide cada chunk del pool en sub-fragmentos ≤ max_words (§9.2.1) y los
    devuelve como candidatos de salida, preservando el orden recibido.

    El chunk_id de cada candidato es el del chunk ORIGINAL del índice, como
    exige §9.2.1 (trazabilidad; varios sub-fragmentos comparten chunk_id).
    """
    candidatos: list[dict] = []
    for _, metadata in scored:
        for sub_fragmento in _split_by_words(metadata["texto"], max_words):
            candidatos.append({
                "chunk_id": metadata["chunk_id"],
                "doc_id": metadata["doc_id"],
                "text": sub_fragmento,
            })
    return candidatos


def _seleccionar_con_dedup(candidatos: list[dict], n: int) -> list[dict]:
    """Toma los primeros n candidatos que no sean casi-duplicados entre sí.

    §9.3.2 penaliza los arrays con un número distinto de n elementos, así que
    entregar n fragmentos prima sobre el filtro de casi-duplicados: si tras la
    primera pasada con dedup no se llega a n, una segunda pasada rellena con
    los mejores candidatos descartados por duplicación.
    """
    fragmentos: list[dict] = []
    textos_incluidos: list[str] = []
    descartados_por_dedup: list[dict] = []
    for candidato in candidatos:
        if _es_duplicado(candidato["text"], textos_incluidos):
            descartados_por_dedup.append(candidato)
            continue
        fragmentos.append(candidato)
        textos_incluidos.append(candidato["text"])
        if len(fragmentos) >= n:
            return fragmentos

    # Pool insuficiente tras el dedup: rellenar con los descartados (en el
    # orden de relevancia en que se acumularon).
    for candidato in descartados_por_dedup:
        if len(fragmentos) >= n:
            break
        fragmentos.append(candidato)
    return fragmentos


def top_fragments(
    scored: list[tuple[float, dict]],
    n: int = TOP_N_FRAGMENTS,
    max_words: int = MAX_WORDS_PER_FRAGMENT,
) -> list[dict]:
    """Devuelve exactamente n fragmentos de salida (si el pool lo permite),
    cada uno ≤ max_words palabras, en el orden grueso del pool: los
    sub-fragmentos heredan la posición de su chunk padre.
    """
    return _seleccionar_con_dedup(_sub_candidatos(scored, max_words), n)


def top_fragments_rerank(
    scored: list[tuple[float, dict]],
    query_embedding: np.ndarray,
    encoder: Encoder,
    n: int = TOP_N_FRAGMENTS,
    max_words: int = MAX_WORDS_PER_FRAGMENT,
    pool_chunks: int = RERANK_POOL_CHUNKS,
) -> list[dict]:
    """Como top_fragments, pero re-ordena a grano fino los sub-fragmentos de
    los `pool_chunks` mejores chunks según su PROPIA similitud con la consulta.

    Motivo: la búsqueda gruesa puntúa chunks de ~450 tokens, pero el NDCG@10 se
    juzga sobre el texto de sub-fragmentos ≤250 palabras (§10.2.1). Sin
    re-scoring, los sub-fragmentos heredan el orden del chunk padre y la mitad
    irrelevante de un buen chunk sale antes que la mitad excelente del
    siguiente.

    Cumplimiento: usa exclusivamente el MISMO encoder del índice (prefijo
    "passage: ") y operaciones vectoriales — §8.3 solo prohíbe modelos
    generativos, y §8.7 permite post-filtros que operen directamente sobre los
    vectores. El chunk_id reportado sigue siendo el del chunk original (§9.2.1).
    """
    finos = _sub_candidatos(scored[:pool_chunks], max_words)
    if finos:
        embeddings = encoder.encode_passages([c["text"] for c in finos])
        # Vectores normalizados: producto interno = similitud coseno (§8.2).
        similitudes = embeddings @ query_embedding[0]
        orden = np.argsort(-similitudes)
        finos = [finos[i] for i in orden]

    # Cola de garantía: el resto del pool en orden grueso, solo por si el dedup
    # descarta tantos candidatos finos que no se llega a n (§9.3.2).
    cola = _sub_candidatos(scored[pool_chunks:], max_words)
    return _seleccionar_con_dedup(finos + cola, n)



def retrieve(
    query: str,
    index: faiss.Index,
    metas: list[dict],
    encoder: Encoder,
    graph=None,
    entity_index: dict[str, str] | None = None,
    metas_by_chunk: dict[str, dict] | None = None,
) -> dict:
    """Devuelve {"documents": [doc_id,...], "fragments": [{chunk_id,doc_id,text},...]}.

    Si se pasa `graph` (nx.MultiDiGraph construido por src.graph.build_graph),
    se complementa la búsqueda vectorial con el pool de candidatos del grafo
    de conocimiento (Sección 8.5), fusionado mediante RRF (Sección 8.4).
    Sin `graph`, el comportamiento es idéntico al de la recuperación puramente
    vectorial.

    Los rangos (rank) y el envoltorio JSON final los añade generador.py.
    """
    # El embedding de la consulta se calcula UNA vez y se reutiliza en la
    # búsqueda gruesa y en el re-ranking fino.
    query_embedding = encoder.encode_queries([query])
    scored = search(query, index, metas, encoder, query_embedding=query_embedding)

    if graph is not None:
        from src.graph import graph_candidates
        if metas_by_chunk is None:
            metas_by_chunk = {m["chunk_id"]: m for m in metas}
        candidatos_grafo = graph_candidates(query, graph, metas_by_chunk, entity_index)
        if candidatos_grafo:
            scored = combine_rrf([scored, candidatos_grafo])

    documentos = top_documents(scored)
    if RERANK_FRAGMENTOS:
        fragmentos = top_fragments_rerank(scored, query_embedding, encoder)
    else:
        fragmentos = top_fragments(scored)

    # Garantía de esquema (§9.3.2): exactamente TOP_N_DOCUMENTS documentos.
    # Solo puede faltar si el pool entero tiene menos doc_id distintos que n
    # (corpus degenerado); se rellena determinísticamente con los primeros
    # documentos del índice aún no incluidos.
    if len(documentos) < TOP_N_DOCUMENTS:
        incluidos = set(documentos)
        for metadata in metas:
            if len(documentos) >= TOP_N_DOCUMENTS:
                break
            if metadata["doc_id"] not in incluidos:
                documentos.append(metadata["doc_id"])
                incluidos.add(metadata["doc_id"])

    return {"documents": documentos, "fragments": fragmentos}
