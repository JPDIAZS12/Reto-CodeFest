"""Pruebas para el índice construido por src/build_index.py.

Requiere haber corrido antes:  python -m src.build_index
Ejecuta:                        python tests/test_build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import faiss
import numpy as np

from config import BASE_VECTORIAL_DIR, ENCODER_SLUG, EMBED_DIM
from src.encode import Encoder

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


OUT = BASE_VECTORIAL_DIR / f"encoder_{ENCODER_SLUG}"
IDX = OUT / "index.faiss"
META = OUT / "metadata.jsonl"

print("=== 1. Archivos existen ===")
check(IDX.exists(), f"existe {IDX.name}")
check(META.exists(), f"existe {META.name}")

# Cargar índice y metadata
index = faiss.read_index(str(IDX))
metas = [json.loads(l) for l in META.read_text(encoding="utf-8").splitlines() if l.strip()]

print("\n=== 2. Conteo alineado ===")
check(index.ntotal == len(metas),
      f"index.ntotal ({index.ntotal}) == líneas metadata ({len(metas)})")

print("\n=== 3. Dimensión ===")
check(index.d == EMBED_DIM, f"index.d ({index.d}) == EMBED_DIM ({EMBED_DIM})")

print("\n=== 4. Campos obligatorios (Tabla 1) ===")
obligatorios = {"doc_id", "chunk_id", "fuente", "formato",
                "fenomeno", "posicion", "num_tokens", "texto"}
faltan = [i for i, m in enumerate(metas) if not obligatorios.issubset(m.keys())]
check(not faltan, f"todas las líneas tienen los 8 campos (faltan en {len(faltan)})")

print("\n=== 5. Vectores normalizados ===")
todos = index.reconstruct_n(0, index.ntotal)  
normas = np.linalg.norm(todos, axis=1)
check(np.allclose(normas, 1.0, atol=1e-3),
      f"normas ~= 1.0 (min={normas.min():.4f} max={normas.max():.4f})")

print("\n=== 6. Alineamiento semántico (vector i <-> texto de línea i) ===")
enc = Encoder()

posiciones = [0, index.ntotal // 2, index.ntotal - 1]
ok_align = True
for i in posiciones:
    vec_indice = index.reconstruct(i)                       
    vec_texto = enc.encode_passages([metas[i]["texto"]])[0]  
    sim = float(np.dot(vec_indice, vec_texto))
    print(f"  pos {i}: auto-similitud = {sim:.4f}  ({metas[i]['fuente']})")
    if sim < 0.99:
        ok_align = False
check(ok_align, "el vector i coincide con el texto de la línea i (auto-sim >= 0.99)")

print("\n=== Distribución por fenómeno / formato (informativo) ===")
from collections import Counter
print("  fenómeno:", dict(Counter(m["fenomeno"] for m in metas)))
print("  formato :", dict(Counter(m["formato"] for m in metas)))

print(f"\n{'='*50}")
print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
print('='*50)
sys.exit(1 if _FAILED else 0)
