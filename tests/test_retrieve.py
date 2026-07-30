"""Pruebas para src/retrieve.py.

Requiere haber corrido antes:  python -m src.build_index
Ejecutar:                        python tests/test_retrieve.py

"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config import MAX_WORDS_PER_FRAGMENT
from src.encode import Encoder
from src.retrieve import (
    load_base,
    search,
    top_documents,
    _split_by_words,
    top_fragments,
    retrieve,
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


print("Cargando base (index.faiss + metadata.jsonl) y encoder...")
index, metas = load_base()
enc = Encoder()
print(f"Base cargada: {index.ntotal} vectores.\n")


# --------------------------------------------------------------------------- #
# 1. search
# --------------------------------------------------------------------------- #
print("=== 1. search ===")
scored = search("riesgos de la basura espacial en la órbita baja", index, metas, enc)
check(len(scored) == index.ntotal,
      f"con k>ntotal devuelve ntotal resultados y salta -1 ({len(scored)})")
sims = [s for s, _ in scored]
check(sims == sorted(sims, reverse=True), "resultados ordenados por similitud descendente")
check(all(-1.0001 <= s <= 1.0001 for s in sims), "similitudes en rango coseno [-1, 1]")
check(all(isinstance(m, dict) and "doc_id" in m for _, m in scored),
      "cada resultado trae su metadata (dict con doc_id)")
# El mejor resultado debe ser del fenómeno 2 (espacio)
check(scored[0][1]["fenomeno"] == 2, "el top-1 pertenece al fenómeno 2 (espacio)")


# --------------------------------------------------------------------------- #
# 2. top_documents
# --------------------------------------------------------------------------- #
print("\n=== 2. top_documents (max pooling) ===")
docs = top_documents(scored, n=3)
check(len(docs) == 3, f"devuelve 3 doc_id -> {len(docs)}")
check(len(set(docs)) == 3, "los 3 doc_id son únicos (no repite documentos)")
# El documento top-1 debe ser uno de espacio (fenómeno 2)
top_doc_meta = next(m for _, m in scored if m["doc_id"] == docs[0])
check(top_doc_meta["fenomeno"] == 2, "el documento top-1 es del fenómeno 2 (espacio)")

juguete = [
    (0.9, {"doc_id": "A"}),
    (0.5, {"doc_id": "B"}),
    (0.7, {"doc_id": "A"}),  
    (0.8, {"doc_id": "C"}),
]
orden = top_documents(juguete, n=3)
check(orden == ["A", "C", "B"],
      f"max pooling ordena A(0.9)>C(0.8)>B(0.5) -> {orden}")


# --------------------------------------------------------------------------- #
# 3. _split_by_words
# --------------------------------------------------------------------------- #
print("\n=== 3. _split_by_words ===")
# Texto corto: no se divide
corto = "Una sola oración corta y completa."
check(_split_by_words(corto, max_words=250) == [corto], "texto corto no se divide")

# Texto con varias oraciones y límite pequeño 
parrafo = ("Primera oración con varias palabras aquí. "
           "Segunda oración también con varias palabras. "
           "Tercera oración para forzar el corte definitivo. "
           "Cuarta oración final del párrafo completo.")
subs = _split_by_words(parrafo, max_words=8)
check(all(len(s.split()) <= 8 for s in subs),
      f"ningún sub-fragmento supera 8 palabras -> tam={[len(s.split()) for s in subs]}")
check(len(subs) >= 2, f"se generó más de un sub-fragmento -> {len(subs)}")
# No se pierde texto: todas las palabras originales están presentes en orden
check(" ".join(subs).split() == parrafo.split(), "no se pierde ni reordena texto")

# Caso borde: una sola oración gigante > max_words
gigante = "palabra " * 25 
gigante = gigante.strip()
subs_g = _split_by_words(gigante, max_words=10)
check(all(len(s.split()) <= 10 for s in subs_g),
      f"corte duro respeta el límite -> tam={[len(s.split()) for s in subs_g]}")
check(" ".join(subs_g).split() == gigante.split(), "corte duro no pierde palabras")


# --------------------------------------------------------------------------- #
# 4. retrieve (end-to-end, cross-lingual)
# --------------------------------------------------------------------------- #
print("\n=== 4. retrieve end-to-end ===")
res = retrieve("What are the risks of orbital debris?", index, metas, enc)
check(set(res.keys()) == {"documents", "fragments"}, "estructura con documents y fragments")
check(len(res["documents"]) == 3, f"3 documentos -> {len(res['documents'])}")
check(all(len(f["text"].split()) <= MAX_WORDS_PER_FRAGMENT for f in res["fragments"]),
      "ningún fragmento de salida supera 250 palabras")
check(all({"chunk_id", "doc_id", "text"} <= set(f.keys()) for f in res["fragments"]),
      "cada fragmento trae chunk_id, doc_id y text")
top_meta = next(m for _, m in search("orbital debris risks", index, metas, enc)
                if m["doc_id"] == res["documents"][0])
check(top_meta["fenomeno"] == 2,
      "consulta EN recupera documento del fenómeno correcto (espacio)")

print("\n  Muestra retrieve (consulta EN sobre basura orbital):")
print(f"   documentos: {res['documents']}")
print(f"   fragmento top-1: {res['fragments'][0]['text'][:80]}...")


print(f"\n{'='*50}")
print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
print('='*50)
sys.exit(1 if _FAILED else 0)
