"""Pruebas del dedup de fragmentos (_es_duplicado + top_fragments con relleno).

Ejecuta:  python tests/test_dedup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config import DEDUP_JACCARD
from src.retrieve import _es_duplicado, top_fragments

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


print("=== 1. _es_duplicado ===")
check(_es_duplicado("hola mundo", []) is False, "sin previos -> no es duplicado")

prev = ["los satelites obsoletos generan riesgo de colision orbital"]
check(_es_duplicado("los satelites obsoletos generan riesgo de colision orbital", prev) is True,
      "texto idéntico -> duplicado")

a = "la orbita baja esta congestionada los satelites obsoletos generan riesgo de colision"
b = "los satelites obsoletos generan riesgo de colision cada impacto crea mas fragmentos"
check(_es_duplicado(a, [b]) is False,
      "solapamiento parcial por debajo del umbral -> no duplicado")

c = "uno dos tres cuatro cinco seis siete ocho nueve diez"
d = "uno dos tres cuatro cinco seis siete ocho nueve once"  # 9/11 palabras comunes ~0.82
check(_es_duplicado(c, [d]) is True, "Jaccard alto (>=0.8) -> duplicado")

# Distinto por completo -> no duplicado
check(_es_duplicado("gatos perros aves", ["motores turbinas alas"]) is False,
      "textos sin palabras comunes -> no duplicado")


previos = ["tema completamente distinto sobre economia",
           "otro tema sobre educacion rural",
           "uno dos tres cuatro cinco seis siete ocho nueve diez"]
check(_es_duplicado("uno dos tres cuatro cinco seis siete ocho nueve once", previos) is True,
      "detecta duplicado contra el 3er previo (recorre toda la lista)")


print("\n=== 2. top_fragments salta duplicados y rellena ===")
def meta(chunk_id, doc_id, texto):
    return {"chunk_id": chunk_id, "doc_id": doc_id, "texto": texto}


scored = [
    (0.90, meta("c1", "A", "uno dos tres cuatro cinco seis siete ocho nueve diez")),
    (0.89, meta("c2", "A", "uno dos tres cuatro cinco seis siete ocho nueve once")),  # dup de c1
    (0.70, meta("c3", "B", "energia solar paneles fotovoltaicos red electrica nacional")),
    (0.60, meta("c4", "C", "migracion territorio gobernanza seguridad humana regional")),
]
frags = top_fragments(scored, n=3)
textos = [f["text"] for f in frags]
check(len(frags) == 3, f"rellena hasta n=3 saltando el duplicado -> {len(frags)}")
chunk_ids = [f["chunk_id"] for f in frags]
check("c2" not in chunk_ids, "el casi-duplicado c2 fue saltado")
check(chunk_ids == ["c1", "c3", "c4"], f"quedan c1,c3,c4 (rellenó con distintos) -> {chunk_ids}")

# Sin duplicados: comportamiento normal
scored2 = [
    (0.9, meta("x1", "A", "alfa beta gamma delta")),
    (0.8, meta("x2", "B", "uno dos tres cuatro")),
]
frags2 = top_fragments(scored2, n=5)
check(len(frags2) == 2, "sin duplicados devuelve todos los candidatos distintos")


print(f"\n{'='*50}")
print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
print('='*50)
sys.exit(1 if _FAILED else 0)
