"""Demo del pipeline real (extract -> clean -> chunk) sobre un PDF largo.

Usa el propio PDF del reto como documento de prueba realista.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")  # tildes correctas en consola Windows

from transformers import AutoTokenizer

from config import ENCODER_NAME, CHUNK_MAX_TOKENS
from src.extract import extract_document
from src.clean import clean_document, detect_language
from src.chunk import chunk_document, count_tokens, split_sentences

PDF = Path(r"C:\Users\juanp\Downloads\CODEFEST_2026-1.pdf")

print(f"Cargando tokenizer '{ENCODER_NAME}'...")
TOK = AutoTokenizer.from_pretrained(ENCODER_NAME)

# 1) Extracción
doc = extract_document(PDF, root=PDF.parent)
assert doc is not None, "no se pudo extraer el PDF"
print(f"\n=== Extracción ===")
print(f"doc_id : {doc.doc_id}")
print(f"fuente : {doc.fuente}  | formato: {doc.formato}  | fenomeno: {doc.fenomeno}")
print(f"caracteres extraídos: {len(doc.texto):,}")

# 2) Limpieza
texto_limpio = clean_document(doc.texto)
idioma = detect_language(texto_limpio)
print(f"\n=== Limpieza ===")
print(f"caracteres tras limpieza: {len(texto_limpio):,}  "
      f"(-{len(doc.texto) - len(texto_limpio):,})")
print(f"idioma detectado: {idioma}")

# 3) Chunking (sobre el texto limpio)
doc.texto = texto_limpio
frags = chunk_document(doc, TOK, idioma=idioma)

n_tokens = [f.num_tokens for f in frags]
print(f"\n=== Chunking ===")
print(f"total de oraciones : {len(split_sentences(texto_limpio)):,}")
print(f"total de fragmentos: {len(frags)}")
print(f"tokens/fragmento   : min={min(n_tokens)}  max={max(n_tokens)}  "
      f"prom={sum(n_tokens) / len(n_tokens):.1f}")

# 4) Verificaciones clave del reto
print(f"\n=== Verificaciones ===")
# Ningún chunk multi-oración debe exceder CHUNK_MAX_TOKENS
excedidos = [f for f in frags
             if f.num_tokens > CHUNK_MAX_TOKENS and len(split_sentences(f.texto)) > 1]
print(f"[{'OK' if not excedidos else 'FALLA'}] "
      f"fragmentos multi-oración sobre {CHUNK_MAX_TOKENS} tokens: {len(excedidos)}")

# Completitud lingüística: cada chunk debe terminar en signo de cierre
cierres = (".", "!", "?", "…", "\"", "»", ":", ")")
sin_cierre = [f for f in frags if not f.texto.rstrip().endswith(cierres)]
print(f"[INFO] fragmentos que no terminan en signo de cierre: {len(sin_cierre)} "
      f"(esperable algo por tablas/listas del PDF)")

# chunk_ids únicos
ids = [f.chunk_id for f in frags]
print(f"[{'OK' if len(ids) == len(set(ids)) else 'FALLA'}] chunk_ids únicos")

# 5) Muestra: primeros 3 y últimos 2 fragmentos
print(f"\n=== Muestra de fragmentos ===")
for f in frags[:3] + frags[-2:]:
    print(f"\n[pos {f.posicion}] ({f.num_tokens} tok)")
    print(f"  {f.texto[:220]}{'...' if len(f.texto) > 220 else ''}")
