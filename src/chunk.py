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
    # TODO(tú): implementar la división en oraciones.
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
) -> list[str]:
    """Agrupa oraciones consecutivas en textos de chunk.

    Devuelve: lista de textos de chunk, en orden.
    """
    # TODO(tú): implementar el agrupamiento con solapamiento.
    chunks = []
    buffer = []
    for oracion in sentences:
        buffer.append(oracion)
        texto_buffer_actual = " ".join(buffer)
        numero_tokens_actual = count_tokens(texto_buffer_actual, tokenizer)
        if numero_tokens_actual > max_tokens:
            buffer.pop()  
            if buffer:
                texto_chunk = " ".join(buffer)
                chunks.append(texto_chunk)
                if overlap > 0:
                    buffer = buffer[-overlap:]  
                else:
                    buffer = []
                buffer.append(oracion)  
            else:
                # Caso borde: la oración sola excede el límite, emitirla como chunk
                chunks.append(oracion)
                buffer = []
    # Emitir cualquier oración restante en el buffer como un chunk final
    if buffer:
        texto_chunk_final = " ".join(buffer)
        chunks.append(texto_chunk_final)
    return chunks



def chunk_document(doc, tokenizer, idioma: str = "") -> list[Fragmento]:
    """Fragmenta un Documento (de extract.py) y arma la metadata Tabla 1.

    `doc` debe exponer: .doc_id, .fuente, .formato, .fenomeno, .texto
    """
    sentences = split_sentences(doc.texto)
    chunk_texts = group_sentences(
        sentences, tokenizer,
        max_tokens=CHUNK_MAX_TOKENS,
        overlap=CHUNK_OVERLAP_SENTENCES,
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
