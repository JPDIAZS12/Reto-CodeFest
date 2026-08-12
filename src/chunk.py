"""Chunking: fragmentación del texto (Sección 3 de la especificación).


Cada fragmento sale con la metadata obligatoria de la Tabla 1.
"""
from __future__ import annotations
from transformers import AutoTokenizer

import re
from dataclasses import dataclass, asdict

from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_SENTENCES,
    CHUNK_MIN_TOKENS,
    MAX_CHUNKS_POR_DOCUMENTO,
)


#Metadat
@dataclass
class Fragmento:
    doc_id: str        # documento de origen
    chunk_id: str      # id único del fragmento dentro del documento
    fuente: str        
    formato: str       
    fenomeno: int      
    posicion: int     
    num_tokens: int    # nº de tokens del fragmento
    texto: str         # texto original del fragmento
    idioma: str = ""   # campo adicional opcional

    def to_dict(self) -> dict:
        return asdict(self)



def split_sentences(text: str) -> list[str]:
    """Divide `text` en una lista de oraciones completas.
    """
    oraciones_validas = []
    expresion_regular = re.compile(r'(?<=[.!?…])\s+(?=[«"“¿¡A-ZÁÉÍÓÚÑ0-9])')
    for parrafo in text.split("\n"):
        oraciones = expresion_regular.split(parrafo)
        for oracion in oraciones:
            oracion_limpia = oracion.strip()
            if oracion_limpia:
                oraciones_validas.append(oracion_limpia)
    return oraciones_validas


def count_tokens(text: str, tokenizer) -> int:
    """Devuelve el nº de tokens de `text` según el tokenizer del encoder.
    """
    lista_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(lista_ids)


def group_sentences(
    sentences: list[str],
    tokenizer,
    max_tokens: int,
    overlap: int,
    max_chunks: int | None = None,
) -> list[str]:
    """Agrupa oraciones consecutivas en textos de chunk.

    `max_chunks` corta la generación al alcanzar ese número de chunks. Se corta
    AQUÍ y no después para no tokenizar el documento entero: un CSV de 35 MB
    tiene cientos de miles de oraciones y contarlas todas cuesta minutos de CPU
    que se tirarían a la basura.

    Devuelve: lista de textos de chunk, en orden.
    """
    chunks = []

    buffer = []          # lista de (oracion, n_tokens) del chunk en construcción
    tokens_buffer = 0    # suma de tokens de las oraciones del buffer

    for oracion in sentences:
        # Cada oración se cuenta UNA sola vez (antes se re-tokenizaba todo el
        # buffer en cada paso, costo cuadrático). Se cuenta dentro del bucle, y
        # no en una lista previa, para poder abandonar temprano con max_chunks.
        n_tokens = count_tokens(oracion, tokenizer)
        # ¿Agregar esta oración haría que el chunk actual supere el límite?
        if buffer and tokens_buffer + n_tokens > max_tokens:
            # Cerrar el chunk actual con lo acumulado
            texto_chunk = " ".join(s for s, _ in buffer)
            chunks.append(texto_chunk)
            # Solapamiento: conservar las últimas `overlap` oraciones
            if overlap > 0:
                buffer = buffer[-overlap:]
            else:
                buffer = []
            # Recalcular la suma de tokens del buffer que quedó (el de solape)
            tokens_buffer = 0
            for _, t in buffer:
                tokens_buffer += t

        # Agregar la oración actual al buffer
        buffer.append((oracion, n_tokens))
        tokens_buffer += n_tokens

        # Caso borde: una sola oración que por sí sola excede el límite no se
        # puede dividir sin cortarla; se emite como su propio chunk.
        if len(buffer) == 1 and n_tokens > max_tokens:
            chunks.append(oracion)
            buffer = []
            tokens_buffer = 0

        if max_chunks is not None and len(chunks) >= max_chunks:
            return chunks[:max_chunks]

    # Emitir cualquier oración restante en el buffer como un chunk final
    if buffer:
        texto_chunk_final = " ".join(s for s, _ in buffer)
        chunks.append(texto_chunk_final)
    if max_chunks is not None:
        return chunks[:max_chunks]
    return chunks



def chunk_document(doc, tokenizer, idioma: str = "") -> list[Fragmento]:
    """Fragmenta un Documento y arma la metadata Tabla 1.

    `doc` debe exponer: .doc_id, .fuente, .formato, .fenomeno, .texto
    """
    sentences = split_sentences(doc.texto)
    chunk_texts = group_sentences(
        sentences, tokenizer,
        max_tokens=CHUNK_MAX_TOKENS,
        overlap=CHUNK_OVERLAP_SENTENCES,
        max_chunks=MAX_CHUNKS_POR_DOCUMENTO,
    )

    fragmentos: list[Fragmento] = []
    posicion = 0
    for texto in chunk_texts:
        texto = texto.strip()
        if not texto:
            continue
        n_tok = count_tokens(texto, tokenizer)
        if n_tok < CHUNK_MIN_TOKENS:
            continue
        fragmentos.append(Fragmento(
            doc_id=doc.doc_id,
            chunk_id=f"{doc.doc_id}-chunk-{posicion:04d}",
            fuente=doc.fuente,
            formato=doc.formato,
            fenomeno=doc.fenomeno,
            posicion=posicion,
            num_tokens=n_tok,
            texto=texto,
            idioma=idioma,
        ))
        posicion += 1
    return fragmentos
