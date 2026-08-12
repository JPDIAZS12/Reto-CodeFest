"""Módulo de recuperación
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
    TOP_M_CHUNKS_POR_DOC,
    DEDUP_JACCARD,
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
) -> list[tuple[float, dict]]:
    """Busca la consulta y devuelve hasta k pares (similitud, metadata),
    ordenados de mayor a menor similitud.

    """
    embeddings_query = encoder.encode_queries([query]) 
    distancias, ids = index.search(embeddings_query, k)
    resultados = []
    for i in range(len(ids[0])):
        if ids[0][i] != -1:
            resultados.append((float(distancias[0][i]), metas[ids[0][i]]))
    
    return resultados
    


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
            doc_scores[doc_id] = sum(similitudes[:TOP_M_CHUNKS_POR_DOC])

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
    
    


def top_fragments(
    scored: list[tuple[float, dict]],
    n: int = TOP_N_FRAGMENTS,
    max_words: int = MAX_WORDS_PER_FRAGMENT,
) -> list[dict]:
    """Devuelve hasta n fragmentos de salida, cada uno ≤ max_words palabras.
    """
    fragmentos: list[dict] = []
    textos_incluidos: list[str] = []  
    for similitud, metadata in scored:
        for sub_fragmento in _split_by_words(metadata["texto"], max_words):
            if _es_duplicado(sub_fragmento, textos_incluidos):
                continue
            fragmentos.append({
                "chunk_id": metadata["chunk_id"],
                "doc_id": metadata["doc_id"],
                "text": sub_fragmento,
            })
            textos_incluidos.append(sub_fragmento)
            if len(fragmentos) >= n:
                return fragmentos
    return fragmentos



def retrieve(
    query: str,
    index: faiss.Index,
    metas: list[dict],
    encoder: Encoder,
) -> dict:
    """Devuelve {"documents": [doc_id,...], "fragments": [{chunk_id,doc_id,text},...]}.

    Los rangos (rank) y el envoltorio JSON final los añade generador.py.
    """
    scored = search(query, index, metas, encoder)
    documentos = top_documents(scored)
    fragmentos = top_fragments(scored)
    return {"documents": documentos, "fragments": fragmentos}
