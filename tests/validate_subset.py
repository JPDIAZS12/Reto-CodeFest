"""Validación cualitativa del índice de subconjunto con queries reales.

Requiere haber corrido:
    python -m src.build_index --data data_subset --out _scratch/subset/encoder_e5-large

Ejecuta:  python tests/validate_subset.py

Corre una consulta real por fenómeno y muestra los documentos y el fragmento
top para inspeccionar a ojo si la recuperación es coherente (tema + fenómeno).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.encode import Encoder
from src.retrieve import load_base, search, retrieve

SUBSET_DIR = ROOT / "_scratch" / "subset" / "encoder_e5-large"

# Una consulta real por fenómeno (del queries.jsonl de ADL)
CONSULTAS = [
    ("q004 (F1 IA)", "¿Qué riesgos representa la escasez de talento especializado "
                     "en inteligencia artificial para el desarrollo de capacidades de defensa?"),
    ("q026 (F2 espacio)", "¿Cuál ha sido el impacto de las pruebas antisatélite "
                          "sobre la generación de desechos orbitales?"),
    ("q033 (F3 territorio)", "¿Cómo utilizan los grupos armados ilegales el control "
                             "territorial para sustituir funciones del Estado?"),
]

print("Cargando índice de subconjunto y encoder...")
index, metas = load_base(SUBSET_DIR)
enc = Encoder()
print(f"Índice: {index.ntotal} vectores, {len(metas)} metadatos.\n")

for etiqueta, consulta in CONSULTAS:
    print(f"{'='*70}\n{etiqueta}\n  {consulta}")
    scored = search(consulta, index, metas, enc)
    res = retrieve(consulta, index, metas, enc)
    print(f"\n  Top-3 documentos:")
    for i, doc_id in enumerate(res["documents"], 1):
        meta = next(m for _, m in scored if m["doc_id"] == doc_id)
        print(f"   {i}. [F{meta['fenomeno']}] {meta['fuente']}")
    print(f"\n  Fragmento top-1 (sim={scored[0][0]:.3f}):")
    print(f"   {res['fragments'][0]['text'][:200]}...")
    print()
