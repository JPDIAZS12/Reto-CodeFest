"""Batería de pruebas para src/chunk.py.

Ejecuta:  python tests/test_chunk.py

Valida:
  1. count_tokens       -> conteo correcto con el tokenizer real del e5.
  2. split_sentences    -> divide bien, no corta en decimales/abreviaturas,
                           no deja oraciones vacías, preserva puntuación.
  3. _group_sentences   -> ningún chunk excede max_tokens (salvo oración
                           gigante), no se pierde texto, aplica solapamiento.
  4. chunk_document     -> metadata Tabla 1 correcta (ids, posiciones, tokens).
  5. Casos borde        -> oración gigante, texto vacío.

"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Permitir importar config.py y el paquete src/ desde la raíz del proyecto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer 

from config import ENCODER_NAME, CHUNK_MAX_TOKENS  
from src.chunk import (  
    split_sentences,
    count_tokens,
    group_sentences,
    chunk_document,
)

_PASSED = 0
_FAILED = 0


def check(cond: bool, msg: str) -> None:
    global _PASSED, _FAILED
    if cond:
        _PASSED += 1
        print(f"  [OK]   {msg}")
    else:
        _FAILED += 1
        print(f"  [FALLA] {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")



print(f"Cargando tokenizer '{ENCODER_NAME}' (puede descargar la primera vez)...")
TOK = AutoTokenizer.from_pretrained(ENCODER_NAME)
print("Tokenizer cargado.\n")


def test_count_tokens():
    section("1. count_tokens")
    n = count_tokens("Hola mundo", TOK)
    check(isinstance(n, int), "devuelve un entero")
    check(n > 0, f"'Hola mundo' -> {n} tokens (> 0)")
    check(count_tokens("", TOK) == 0, "cadena vacía -> 0 tokens")
    # Más texto debe dar más tokens
    corto = count_tokens("uno", TOK)
    largo = count_tokens("uno dos tres cuatro cinco seis siete", TOK)
    check(largo > corto, f"texto más largo -> más tokens ({largo} > {corto})")



def test_split_sentences():
    section("2. split_sentences")

    # Multilingüe básico
    s = split_sentences("Hola mundo. ¿Qué tal? Todo bien.")
    check(len(s) == 3, f"3 oraciones simples -> {len(s)}")
    check(all(o.strip() == o for o in s), "sin espacios sobrantes en los bordes")
    check(all(o for o in s), "sin oraciones vacías")
    check(s[0] == "Hola mundo.", "conserva el punto final de la oración")

    # No debe cortar en decimales
    s2 = split_sentences("El PIB creció 3.5 por ciento este año.")
    check(len(s2) == 1, f"no corta en decimal '3.5' -> {len(s2)} oración(es)")

    # No debe cortar en minúscula tras punto (abreviatura tipo 'etc.')
    s3 = split_sentences("Usamos IA, drones, etc. para el análisis.")
    check(len(s3) == 1, f"no corta antes de minúscula -> {len(s3)} oración(es)")

    # Inglés y portugués
    s4 = split_sentences("Space debris is a risk. Satellites collide often.")
    check(len(s4) == 2, f"inglés: 2 oraciones -> {len(s4)}")
    s5 = split_sentences("A órbita baixa está congestionada. Há muitos detritos.")
    check(len(s5) == 2, f"portugués: 2 oraciones -> {len(s5)}")

    # Párrafos separados por salto de línea no se pegan
    s6 = split_sentences("Primera idea\nSegunda idea")
    check(len(s6) == 2, f"bloques por salto de línea separados -> {len(s6)}")

    # Texto vacío
    check(split_sentences("") == [], "texto vacío -> lista vacía")
    check(split_sentences("   \n  ") == [], "solo espacios -> lista vacía")


def testgroup_sentences():
    section("3. _group_sentences")

    # Oraciones cortas y controladas; max_tokens pequeño para forzar varios chunks
    sentences = [
        "La inteligencia artificial transforma la defensa nacional.",
        "Los drones autónomos aumentan la capacidad operativa.",
        "El espacio orbital sufre congestión creciente.",
        "Los satélites obsoletos generan riesgo de colisión.",
        "América Latina enfrenta dinámicas territoriales complejas.",
    ]
    max_t = 30
    chunks = group_sentences(sentences, TOK, max_tokens=max_t, overlap=1)

    check(len(chunks) >= 2, f"genera varios chunks con max={max_t} -> {len(chunks)}")

    # Ningún chunk excede max_tokens, salvo que sea una sola oración
    ok_limite = True
    for c in chunks:
        n = count_tokens(c, TOK)
        if n > max_t and len(split_sentences(c)) > 1:
            ok_limite = False
            print(f"     chunk multi-oración excede el límite: {n} > {max_t}")
    check(ok_limite, "ningún chunk multi-oración supera max_tokens")

    # No se pierde texto: cada oración original aparece en algún chunk
    blob = " ".join(chunks)
    todas_presentes = all(o in blob for o in sentences)
    check(todas_presentes, "ninguna oración se pierde (bug del pop resuelto)")

    # Solapamiento: la última oración de un chunk reaparece en el siguiente
    hay_overlap = False
    for i in range(len(chunks) - 1):
        ult = split_sentences(chunks[i])[-1]
        if ult in chunks[i + 1]:
            hay_overlap = True
            break
    check(hay_overlap, "aplica solapamiento entre chunks consecutivos (overlap=1)")

    # overlap=0 -> sin repetición de la última oración
    chunks0 = group_sentences(sentences, TOK, max_tokens=max_t, overlap=0)
    sin_overlap = True
    for i in range(len(chunks0) - 1):
        ult = split_sentences(chunks0[i])[-1]
        if ult in chunks0[i + 1]:
            sin_overlap = False
    check(sin_overlap, "overlap=0 -> no repite oraciones entre chunks")


def test_chunk_document():
    section("4. chunk_document (metadata Tabla 1)")

    texto = (
        "La adopción de inteligencia artificial en el sector defensa avanza "
        "rápidamente. Los observatorios registran un aumento de la inversión. "
        "Persisten brechas de talento y riesgos éticos. La cooperación regional "
        "resulta clave para cerrar esas brechas."
    )
    doc = SimpleNamespace(
        doc_id="DOC-test-001",
        fuente="informe_ia.pdf",
        formato="pdf",
        fenomeno=1,
        texto=texto,
    )
    frags = chunk_document(doc, TOK, idioma="es")

    check(len(frags) >= 1, f"produce al menos un fragmento -> {len(frags)}")

    campos = ("doc_id", "chunk_id", "fuente", "formato", "fenomeno",
              "posicion", "num_tokens", "texto")
    primero = frags[0]
    check(all(hasattr(primero, c) for c in campos),
          "cada fragmento tiene los 8 campos obligatorios de la Tabla 1")

    # Posiciones consecutivas empezando en 0
    posiciones = [f.posicion for f in frags]
    check(posiciones == list(range(len(frags))),
          f"posiciones consecutivas desde 0 -> {posiciones}")

    # chunk_id con el formato esperado
    check(primero.chunk_id == "DOC-test-001-chunk-0000",
          f"formato de chunk_id correcto -> {primero.chunk_id}")

    # num_tokens coincide con el conteo real del texto
    check(primero.num_tokens == count_tokens(primero.texto, TOK),
          "num_tokens coincide con count_tokens del texto")

    # Metadata heredada del documento
    check(all(f.doc_id == "DOC-test-001" for f in frags), "doc_id heredado")
    check(all(f.fuente == "informe_ia.pdf" for f in frags), "fuente heredada")
    check(all(f.fenomeno == 1 for f in frags), "fenomeno heredado")
    check(all(f.idioma == "es" for f in frags), "idioma asignado")


def test_edge_cases():
    section("5. Casos borde")

    # Oración gigante (sin punto interno) que excede el límite -> chunk propio
    gigante = "palabra " * 80  # ~80+ tokens, una sola 'oración'
    gigante = gigante.strip() + "."
    chunks = group_sentences([gigante], TOK, max_tokens=30, overlap=1)
    check(len(chunks) == 1, f"oración gigante -> un solo chunk ({len(chunks)})")
    check(count_tokens(chunks[0], TOK) > 30,
          "la oración gigante se emite completa aunque exceda el límite")

    # Documento vacío -> sin fragmentos
    doc_vacio = SimpleNamespace(
        doc_id="DOC-vacio", fuente="x.txt", formato="txt",
        fenomeno=0, texto="   \n  ",
    )
    check(chunk_document(doc_vacio, TOK) == [], "documento vacío -> 0 fragmentos")


def demo_visual():
    section("6. Muestra visual de chunking real")
    texto = (
        "La seguridad espacial preocupa a las agencias. La órbita baja "
        "terrestre está cada vez más congestionada por satélites y restos. "
        "Los fragmentos de colisiones anteriores incrementan el riesgo. "
        "Se requieren marcos de gobernanza para la sostenibilidad orbital. "
        "Varios organismos multilaterales debaten posibles soluciones."
    )
    doc = SimpleNamespace(
        doc_id="DOC-demo", fuente="reporte_leo.pdf", formato="pdf",
        fenomeno=2, texto=texto,
    )
    for f in chunk_document(doc, TOK, idioma="es"):
        print(f"  [{f.posicion}] ({f.num_tokens} tok) {f.texto[:90]}...")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_count_tokens()
    test_split_sentences()
    testgroup_sentences()
    test_chunk_document()
    test_edge_cases()
    demo_visual()

    print(f"\n{'='*50}")
    print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
    print('='*50)
    sys.exit(1 if _FAILED else 0)
