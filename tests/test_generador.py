"""Valida entrega/resultados.jsonl contra el esquema de la Sección 9.3.

Requiere haber corrido antes:
    python -m src.build_index
    python entrega/generador.py
Ejecuta:  python tests/test_generador.py

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config import (
    RESULTADOS_FILE,
    TOP_N_DOCUMENTS,
    TOP_N_FRAGMENTS,
    MAX_WORDS_PER_FRAGMENT,
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


lineas = RESULTADOS_FILE.read_text(encoding="utf-8").splitlines()

print("=== 1. Formato JSON Lines ===")
objs = []
json_ok = True
for i, ln in enumerate(lineas):
    if not ln.strip():
        continue
    try:
        objs.append(json.loads(ln))
    except json.JSONDecodeError:
        json_ok = False
        print(f"     línea {i} no es JSON válido")
check(json_ok, "cada línea es un objeto JSON válido e independiente")
check(len(objs) == len(lineas), f"sin líneas vacías intercaladas ({len(objs)} objetos)")

print("\n=== 2. Esquema por consulta (Tabla 2) ===")
claves_ok = all({"query_id", "documents", "fragments"} <= set(o.keys()) for o in objs)
check(claves_ok, "todos los objetos tienen query_id, documents y fragments")

docs_ok = True
for o in objs:
    d = o["documents"]
    if len(d) != TOP_N_DOCUMENTS:
        docs_ok = False
    if [x.get("rank") for x in d] != list(range(1, TOP_N_DOCUMENTS + 1)):
        docs_ok = False
    if not all("doc_id" in x for x in d):
        docs_ok = False
check(docs_ok, f"documents: exactamente {TOP_N_DOCUMENTS}, ranks 1..{TOP_N_DOCUMENTS}, con doc_id")

# fragments: 1..TOP_N_FRAGMENTS, ranks consecutivos desde 1, campos completos
frags_ok = True
for o in objs:
    fr = o["fragments"]
    if not (1 <= len(fr) <= TOP_N_FRAGMENTS):
        frags_ok = False
    if [x.get("rank") for x in fr] != list(range(1, len(fr) + 1)):
        frags_ok = False
    if not all({"rank", "chunk_id", "doc_id", "text"} <= set(x.keys()) for x in fr):
        frags_ok = False
check(frags_ok, "fragments: ranks consecutivos desde 1 y con rank/chunk_id/doc_id/text")

print("\n=== 3. Límite de 250 palabras por fragmento ===")
overflow = []
for o in objs:
    for x in o["fragments"]:
        if len(x["text"].split()) > MAX_WORDS_PER_FRAGMENT:
            overflow.append((o["query_id"], x["rank"], len(x["text"].split())))
check(not overflow, f"ningún fragmento supera {MAX_WORDS_PER_FRAGMENT} palabras ({len(overflow)} exceden)")

print("\n=== 4. query_id presentes y sin duplicar ===")
ids = [o["query_id"] for o in objs]
check(all(ids), "todos los query_id son no nulos")
check(len(ids) == len(set(ids)), "no hay query_id duplicados")

print("\n=== Diagnóstico (informativo) ===")
n_frags = [len(o["fragments"]) for o in objs]
print(f"  fragmentos por consulta: min={min(n_frags)} max={max(n_frags)}")
if max(n_frags) < TOP_N_FRAGMENTS:
    print(f"  [AVISO] ninguna consulta llega a {TOP_N_FRAGMENTS} fragmentos: "
          f"esperado con el mini corpus; con el corpus real se completarán.")

print(f"\n{'='*50}")
print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
print('='*50)
sys.exit(1 if _FAILED else 0)
