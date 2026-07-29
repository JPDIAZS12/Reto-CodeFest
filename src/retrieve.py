"""Módulo de recuperación (Secciones 8 y 9).

Dada una consulta en lenguaje natural produce los DOS niveles de resultado:
  - documents: top-3 doc_id  (se evalúa con F1@3)
  - fragments: top-10 fragmentos ≤250 palabras (se evalúa con NDCG@10)

Flujo (Figura 2):
  consulta --(query:)--> vector --> FAISS.search (pool amplio ~50)
        --> pool de candidatos (sim, metadata)
              ├─ top_documents: max pooling por doc_id -> top-3
              └─ top_fragments: top-10 + división a ≤250 palabras (Sección 9.2.1)

Restricción (Sección 8.3): todo opera sobre vectores, puntuaciones y metadata.
Ningún modelo generativo interviene.
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
)
from src.encode import Encoder
from src.chunk import split_sentences


def load_base(out_dir: Path | None = None) -> tuple[faiss.Index, list[dict]]:
    """Carga index.faiss y metadata.jsonl. La línea i de metas describe el
    vector con id interno i (invariante garantizada por build_index)."""
    if out_dir is None:
        out_dir = BASE_VECTORIAL_DIR / f"encoder_{ENCODER_SLUG}"
    index = faiss.read_index(str(out_dir / "index.faiss"))
    lineas = (out_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
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

    Estrategia por defecto: MAX POOLING (Sección 8.6) -> la puntuación de un
    documento es la del mejor de sus fragmentos recuperados.

    Pistas:
      - Recorre `scored` y, por cada (sim, meta), acumula la puntuación del
        documento meta["doc_id"]. Con max pooling: guarda el MÁXIMO sim visto
        por doc_id (útil un dict doc_id -> mejor_sim, p.ej. con defaultdict).
      - Ordena los doc_id de mayor a menor puntuación agregada.
      - Devuelve los primeros n doc_id (solo los ids, en orden).
      - (DOC_AGGREGATION en config permite cambiar a "sum"/"mean" si quisieras
        experimentar, pero implementa primero "max".)
    """
    # TODO(tú): max pooling por doc_id y devolver los n mejores doc_id.
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# HUECO C — división de un texto en sub-fragmentos ≤ max_words
# --------------------------------------------------------------------------- #
def _split_by_words(text: str, max_words: int = MAX_WORDS_PER_FRAGMENT) -> list[str]:
    """Divide `text` en sub-fragmentos de a lo sumo `max_words` palabras,
    SIN cortar oraciones (respeta la completitud lingüística, Sección 9.2.1).

    Pistas:
      - Si el texto ya cabe (len(text.split()) <= max_words), devuelve [text].
      - Si no: obtén las oraciones con split_sentences(text) y agrúpalas de
        forma golosa (como en chunk.group_sentences, pero contando PALABRAS,
        no tokens, y SIN solapamiento):
            * acumula oraciones mientras la suma de palabras no supere max_words
            * cuando la siguiente oración se pasaría, cierra el sub-fragmento
              (" ".join del buffer) y arranca uno nuevo con esa oración.
      - Cuenta palabras con len(oracion.split()).
      - Caso borde: una sola oración con más de max_words palabras no se puede
        dividir sin cortarla. Como la evaluación DESCARTA fragmentos >250
        palabras, en ese caso haz un corte duro por palabras (text.split()
        en trozos de max_words) para no perder el fragmento. Es raro.
      - Devuelve la lista de sub-fragmentos (cada uno ≤ max_words palabras).
    """
    # TODO(tú): dividir en sub-fragmentos respetando oraciones.
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Construcción de la lista de fragmentos (provisto, usa _split_by_words)
# --------------------------------------------------------------------------- #
def top_fragments(
    scored: list[tuple[float, dict]],
    n: int = TOP_N_FRAGMENTS,
    max_words: int = MAX_WORDS_PER_FRAGMENT,
) -> list[dict]:
    """Devuelve hasta n fragmentos de salida, cada uno ≤ max_words palabras.

    Recorre los candidatos por orden de similitud; si un candidato supera
    max_words se parte en sub-fragmentos (todos conservan el chunk_id original)
    y cada sub-fragmento ocupa su propia posición hasta completar n.
    """
    fragmentos: list[dict] = []
    for _sim, meta in scored:
        for sub in _split_by_words(meta["texto"], max_words):
            fragmentos.append({
                "chunk_id": meta["chunk_id"],   # id del fragmento ORIGINAL (trazabilidad)
                "doc_id": meta["doc_id"],
                "text": sub,
            })
            if len(fragmentos) >= n:
                return fragmentos
    return fragmentos


# --------------------------------------------------------------------------- #
# Orquestador (provisto)
# --------------------------------------------------------------------------- #
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
