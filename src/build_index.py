"""Flujo completo de indexación (Sección 6): construye la base vectorial.

    corpus (data/) -> extract -> clean -> chunk -> encode -> FAISS + metadata

Produce, en entrega/base_vectorial/encoder_<slug>/:
    index.faiss     -> índice FAISS serializado con faiss.write_index()
    metadata.jsonl  -> un objeto JSON por línea, UNA por fragmento

INVARIANTE CRÍTICA (Sección 5.3 y entregable 1):
    El identificador interno que FAISS asigna a cada vector es su posición de
    inserción: 0, 1, 2, ... Por eso la línea i de metadata.jsonl debe describir
    EXACTAMENTE el fragmento cuyo vector se insertó en la posición i. Si se
    rompe ese orden, la metadata queda desalineada de los vectores y todo el
    índice es inservible. Para garantizarlo mantenemos una ÚNICA lista de
    fragmentos y la usamos tanto para codificar como para escribir la metadata.

Uso:  python -m src.build_index
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from config import (
    DATA_DIR,
    BASE_VECTORIAL_DIR,
    ENCODER_SLUG,
    EMBED_DIM,
    ENCODE_BATCH_SIZE,
)
from src.extract import iter_documents
from src.clean import clean_document, detect_language
from src.chunk import chunk_document, Fragmento
from src.encode import Encoder



def collect_fragments(root: Path, tokenizer) -> list[Fragmento]:
    """Recorre el corpus y devuelve TODOS los fragmentos en un orden estable.

    """
    fragmentos: list[Fragmento] = []
    for doc in iter_documents(root):
        doc.texto = clean_document(doc.texto)
        if not doc.texto.strip():
            continue
        idioma = detect_language(doc.texto)
        fragmentos.extend(chunk_document(doc, tokenizer, idioma=idioma))
    return fragmentos


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Crea un índice FAISS y le inserta los `embeddings` (n, dim).

    Usamos IndexFlatIP (producto interno). Como los vectores vienen
    normalizados desde encode.py, el producto interno == similitud coseno, y el índice plano da resultados exactos.
.
    """
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index



def persist(index: faiss.Index, fragmentos: list[Fragmento], out_dir: Path) -> None:
    """Guarda index.faiss y metadata.jsonl en out_dir, en orden alineado.
    """
    carpeta = out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "metadata.jsonl", "w", encoding= "utf-8") as file:
        for fragmento in fragmentos:
            linea = json.dumps(fragmento.to_dict(), ensure_ascii=False)
            file.write(linea + "\n")
            
    assert index.ntotal == len(fragmentos), "El número de vectores en el índice no coincide con la cantidad de fragmentos. Alineamiento roto."



def parse_args() -> argparse.Namespace:
    """--data: carpeta del corpus a indexar. --out: carpeta de salida."""
    parser = argparse.ArgumentParser(
        description="Construye la base vectorial (index.faiss + metadata.jsonl)."
    )
    parser.add_argument(
        "--data", default=str(DATA_DIR),
        help="Carpeta del corpus a indexar. Por defecto: %(default)s",
    )
    parser.add_argument(
        "--out", default=str(BASE_VECTORIAL_DIR / f"encoder_{ENCODER_SLUG}"),
        help="Carpeta de salida encoder_<slug>. Por defecto: %(default)s",
    )
    parser.add_argument(
        "--batch", type=int, default=ENCODE_BATCH_SIZE,
        help="Tamaño de lote al codificar. Bájalo si hay poca RAM. Por defecto: %(default)s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out)

    print("Cargando encoder.")
    enc = Encoder()
    tokenizer = enc.model.tokenizer

    print(f"Recorriendo corpus en {data_dir} ...")
    fragmentos = collect_fragments(data_dir, tokenizer)
    if not fragmentos:
        print(f"[ERROR] No se generaron fragmentos. ¿Hay documentos en {data_dir}?")
        return
    print(f"  fragmentos totales: {len(fragmentos)}")

    print(f"Codificando fragmentos (passage:) con batch={args.batch} ...")
    textos = [f.texto for f in fragmentos]
    embeddings = enc.encode_passages(textos, batch_size=args.batch)

    print("Construyendo índice FAISS ...")
    index = build_faiss_index(embeddings)

    print(f"Guardando en {out_dir} ...")
    persist(index, fragmentos, out_dir)

    print(f"Listo. {index.ntotal} vectores indexados.")


if __name__ == "__main__":
    main()
