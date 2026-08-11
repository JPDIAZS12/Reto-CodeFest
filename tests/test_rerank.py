"""Pruebas para el re-ranking fino de sub-fragmentos (top_fragments_rerank).

No requiere el índice ni el modelo real: usa un encoder determinístico de
bolsa de palabras, así que puede correrse en cualquier máquina.
Ejecutar:  python tests/test_rerank.py
"""
from __future__ import annotations

import re
import sys
import types
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def _stub_si_falta(nombre: str, **atributos) -> None:
    """Registra un módulo falso SOLO si la dependencia pesada no está
    instalada. Con faiss/torch presentes (p. ej. en Colab) se usan los reales."""
    try:
        __import__(nombre)
    except ImportError:
        modulo = types.ModuleType(nombre)
        for clave, valor in atributos.items():
            setattr(modulo, clave, valor)
        sys.modules[nombre] = modulo


_stub_si_falta("faiss", Index=object, read_index=lambda p: None)
_stub_si_falta("sentence_transformers", SentenceTransformer=object)
_stub_si_falta("transformers", AutoTokenizer=object)

import numpy as np  # noqa: E402  (dependencia real del proyecto)

import src.retrieve as r  # noqa: E402

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


# --------------------------------------------------------------------------- #
# Encoder falso: bolsa de palabras con hashing estable -> vectores normalizados.
# La similitud coseno resultante crece con el solapamiento de vocabulario,
# que es exactamente lo que el test necesita controlar.
# --------------------------------------------------------------------------- #
DIM = 512


def _vec(texto: str) -> np.ndarray:
    v = np.zeros(DIM, dtype="float32")
    # \w+ descarta la puntuación: "espacial." y "espacial" deben ser la misma
    # palabra para que la similitud del juguete refleje el solapamiento real.
    for palabra in re.findall(r"\w+", texto.lower()):
        v[zlib.crc32(palabra.encode("utf-8")) % DIM] += 1.0
    norma = np.linalg.norm(v)
    return v / norma if norma > 0 else v


class FakeEncoder:
    def encode_queries(self, textos, batch_size=32):
        return np.stack([_vec(t) for t in textos])

    def encode_passages(self, textos, batch_size=32):
        return np.stack([_vec(t) for t in textos])


enc = FakeEncoder()

# --------------------------------------------------------------------------- #
# 1. El re-ranking reordena sub-fragmentos por su PROPIA similitud
# --------------------------------------------------------------------------- #
print("=== 1. Reordenamiento fino ===")
# Chunk A (mejor rank grueso): 1ª oración irrelevante, 2ª muy relevante.
meta_a = {
    "chunk_id": "DOC-A-chunk-0000",
    "doc_id": "DOC-A",
    "texto": ("El presupuesto anual se firma en enero. "
              "Los satélites chocan y generan basura espacial."),
}
# Chunk B (peor rank grueso): una sola oración medianamente relevante.
meta_b = {
    "chunk_id": "DOC-B-chunk-0000",
    "doc_id": "DOC-B",
    "texto": "La basura espacial preocupa a las agencias.",
}
consulta = "satélites basura espacial colisión"
q_emb = enc.encode_queries([consulta])
scored = [(0.9, meta_a), (0.8, meta_b)]

sin_rerank = r.top_fragments(scored, n=3, max_words=8)
con_rerank = r.top_fragments_rerank(scored, q_emb, enc, n=3, max_words=8,
                                    pool_chunks=2)

check([f["text"] for f in sin_rerank]
      == ["El presupuesto anual se firma en enero.",
          "Los satélites chocan y generan basura espacial.",
          "La basura espacial preocupa a las agencias."],
      "sin re-ranking: los sub-fragmentos heredan el orden del chunk padre")
check(con_rerank[0]["text"] == "Los satélites chocan y generan basura espacial.",
      "con re-ranking: el sub-fragmento más afín sube al rank 1")
check(con_rerank[-1]["text"] == "El presupuesto anual se firma en enero.",
      "con re-ranking: el sub-fragmento sin relación cae al final")
check(len(con_rerank) == 3, "devuelve los 3 candidatos disponibles")

# --------------------------------------------------------------------------- #
# 2. Trazabilidad (§9.2.1): el chunk_id es el del chunk original
# --------------------------------------------------------------------------- #
print("\n=== 2. Trazabilidad del chunk_id ===")
ids_de_a = [f["chunk_id"] for f in con_rerank if f["doc_id"] == "DOC-A"]
check(ids_de_a == ["DOC-A-chunk-0000"] * 2,
      "ambos sub-fragmentos del chunk A reportan el chunk_id original")

# --------------------------------------------------------------------------- #
# 3. Límite de palabras (§9.2)
# --------------------------------------------------------------------------- #
print("\n=== 3. Límite de palabras ===")
check(all(len(f["text"].split()) <= 8 for f in con_rerank),
      "ningún sub-fragmento supera max_words")

# --------------------------------------------------------------------------- #
# 4. Cola de garantía: chunks fuera del tramo re-rankeado siguen disponibles
# --------------------------------------------------------------------------- #
print("\n=== 4. Cola de garantía (§9.3.2) ===")
solo_uno = r.top_fragments_rerank(scored, q_emb, enc, n=3, max_words=8,
                                  pool_chunks=1)
check(len(solo_uno) == 3,
      "con pool_chunks=1 igual se llega a n usando el resto del pool")
check(solo_uno[-1]["doc_id"] == "DOC-B",
      "la cola gruesa entra después del tramo re-rankeado")

# --------------------------------------------------------------------------- #
# 5. Relleno ante casi-duplicados
# --------------------------------------------------------------------------- #
print("\n=== 5. Exactamente n pese al dedup ===")
meta_c = {
    "chunk_id": "DOC-C-chunk-0000",
    "doc_id": "DOC-C",
    "texto": ("Los satélites chocan y generan basura espacial. "
              "Los satélites chocan y generan basura espacial."),
}
duplicados = r.top_fragments_rerank([(0.9, meta_c)], q_emb, enc, n=2,
                                    max_words=8, pool_chunks=1)
check(len(duplicados) == 2,
      "rellena hasta n con los descartados por duplicación")

# --------------------------------------------------------------------------- #
# 6. retrieve end-to-end con índice falso (ambos modos)
# --------------------------------------------------------------------------- #
print("\n=== 6. retrieve end-to-end (índice falso) ===")


class FakeIndex:
    """Índice plano de juguete: coseno exacto sobre vectores normalizados."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def search(self, q_emb: np.ndarray, k: int):
        sims = self.embeddings @ q_emb[0]
        orden = np.argsort(-sims)[:k]
        return np.array([sims[orden]]), np.array([orden])


metas = [meta_a, meta_b, meta_c,
         {"chunk_id": "DOC-D-chunk-0000", "doc_id": "DOC-D",
          "texto": "Informe agrícola sobre cosechas de café en la región."}]
indice = FakeIndex(np.stack([_vec(m["texto"]) for m in metas]))

modo_original = r.RERANK_FRAGMENTOS
for modo in (True, False):
    r.RERANK_FRAGMENTOS = modo
    res = r.retrieve(consulta, indice, metas, enc)
    etiqueta = "re-ranking ON" if modo else "re-ranking OFF"
    check(set(res.keys()) == {"documents", "fragments"},
          f"[{etiqueta}] estructura con documents y fragments")
    check(len(res["documents"]) == 3,
          f"[{etiqueta}] exactamente 3 documentos")
    check(res["documents"][0] in {"DOC-A", "DOC-C"},
          f"[{etiqueta}] el documento top-1 es de los afines a la consulta")
    check(all({"chunk_id", "doc_id", "text"} <= set(f.keys())
              for f in res["fragments"]),
          f"[{etiqueta}] cada fragmento trae chunk_id, doc_id y text")
r.RERANK_FRAGMENTOS = modo_original

print(f"\n{'='*50}")
print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
print('='*50)
sys.exit(1 if _FAILED else 0)
