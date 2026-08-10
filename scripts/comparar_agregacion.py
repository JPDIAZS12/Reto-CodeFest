"""Compara estrategias de agregación a nivel documento (Sección 8.6).

Para cada una de las 50 consultas corre UNA sola búsqueda y reutiliza ese pool
con todas las estrategias, así que lo caro (codificar la consulta + FAISS) se
paga una vez por consulta y no una vez por estrategia.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import src.retrieve as R
from src.encode import Encoder
from src.retrieve import load_base, search
from config import QUERIES_FILE, TOP_N_DOCUMENTS, DOC_AGGREGATIONS_VALIDAS

# Índice del subset
INDICE_POR_DEFECTO = ROOT / "_scratch" / "indice_subset" / "encoder_e5-large"

# Estrategia de referencia: la que está en producción hoy.
BASELINE = ("max", None)

RANGOS_FENOMENO = [(1, 16, 1), (17, 32, 2), (33, 50, 3)]

# (metodo, m). m solo aplica a "topm"; en el resto va None.
ESTRATEGIAS = [
    ("max", None),
    ("mean", None),
    ("sum", None),
    ("topm", 1),
    ("topm", 2),
    ("topm", 3),
    ("topm", 5),
    ("topm", 10),
]

for _metodo, _m in ESTRATEGIAS:
    if _metodo not in DOC_AGGREGATIONS_VALIDAS:
        raise ValueError(
            f"ESTRATEGIAS incluye {_metodo!r}, que no está en "
            f"DOC_AGGREGATIONS_VALIDAS ({DOC_AGGREGATIONS_VALIDAS})"
        )


def nombre(metodo: str, m: int | None) -> str:
    if m is None:
        return metodo
    return f"{metodo}(m={m})"


def parse_estrategia(texto: str) -> tuple[str, int | None]:
    """Traduce el texto del CLI (--contra) a una estrategia del barrido.
    """
    if texto.startswith("topm"):
        estrategia = ("topm", int(texto[len("topm"):]))
    else:
        estrategia = (texto, None)

    if estrategia not in ESTRATEGIAS:
        raise SystemExit(
            f"--contra {texto!r} no está en el barrido. Opciones: "
            + ", ".join(nombre(mt, m) for mt, m in ESTRATEGIAS)
        )
    return estrategia


def top3_con(scored: list[tuple[float, dict]], metodo: str, m: int | None) -> list[str]:
    """Devuelve el top-3 de doc_id que produce `scored` bajo la estrategia dada.
    """
    R.DOC_AGGREGATION = metodo
    if metodo == "topm":
        R.TOP_M_CHUNKS_POR_DOC = m
    return R.top_documents(scored, n=TOP_N_DOCUMENTS)


def evidencia_por_doc(scored: list[tuple[float, dict]]) -> dict[str, tuple[float, str]]:
    """Para cada doc_id del pool: su MEJOR similitud y el texto de ese chunk.
    """
    
    evidencias = {}
    
    for similitud, metadata in scored:
        doc_id = metadata["doc_id"]
        if doc_id not in evidencias:
            evidencias[doc_id] = (similitud, metadata["texto"])
            
    return evidencias


def discrepan(top3_a: list[str], top3_b: list[str]) -> bool:
    """True si los dos top-3 son DISTINTOS como conjunto.
    """
    
    for doc_id in top3_a:
        if doc_id not in top3_b:
            return True
    return False


def fenomeno_de_consulta(query_id: str) -> int:
    """'q034' -> 3. Usa los rangos declarados en RANGOS_FENOMENO."""
    numero = int(query_id.lstrip("q"))
    for inicio, fin, fenomeno in RANGOS_FENOMENO:
        if inicio <= numero <= fin:
            return fenomeno
    raise ValueError(f"query_id fuera de los rangos conocidos: {query_id}")


def fenomeno_por_doc(metas: list[dict]) -> dict[str, int]:
    """{doc_id: fenomeno} recorriendo TODA la metadata del índice.
    """
    mapa: dict[str, int] = {}
    for meta in metas:
        if meta["doc_id"] not in mapa:
            mapa[meta["doc_id"]] = meta["fenomeno"]
    return mapa


def coherencia_fenomeno(
    consultas: list[tuple[str, str]],
    top3_de_una_estrategia: dict[str, list[str]],
    doc_a_fenomeno: dict[str, int],
) -> tuple[int, int, int]:
    """Mide cuántos documentos devueltos son del fenómeno que toca.
    """
    consultas_con_fallo = 0
    total_documentos = 0
    aciertos = 0

    for query_id, _ in consultas:
        fenomeno_esperado = fenomeno_de_consulta(query_id)
        fallo_en_esta_consulta = False
        for doc_id in top3_de_una_estrategia[query_id]:
            if doc_id in doc_a_fenomeno:
                total_documentos += 1
                if doc_a_fenomeno[doc_id] == fenomeno_esperado:
                    aciertos += 1
                else:
                    fallo_en_esta_consulta = True
        if fallo_en_esta_consulta:
            consultas_con_fallo += 1

    return aciertos, total_documentos, consultas_con_fallo
    


def cargar_consultas(path: Path, limite: int | None) -> list[tuple[str, str]]:
    """Lee queries.jsonl -> [(query_id, texto)]. Mismo formato que generador.py."""
    consultas: list[tuple[str, str]] = []
    for linea in Path(path).read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        obj = json.loads(linea)
        consultas.append((obj["query_id"], obj["query"]))
    if limite is not None:
        return consultas[:limite]
    return consultas


def imprimir_resumen(top3s: dict[str, dict[str, list[str]]], total: int) -> None:
    """Tabla: cuántas consultas cambian de top-3 respecto al baseline."""
    etiqueta_base = nombre(*BASELINE)
    print(f"\n{'='*66}")
    print(f"RESUMEN — consultas cuyo top-3 (como conjunto) difiere de '{etiqueta_base}'")
    print(f"{'='*66}")
    print(f"  {'estrategia':<16} {'difieren':>10}   {'%':>6}")
    for metodo, m in ESTRATEGIAS:
        etiqueta = nombre(metodo, m)
        if etiqueta == etiqueta_base:
            continue
        distintas = [
            qid for qid in top3s[etiqueta_base]
            if discrepan(top3s[etiqueta_base][qid], top3s[etiqueta][qid])
        ]
        pct = 100.0 * len(distintas) / max(total, 1)
        print(f"  {etiqueta:<16} {len(distintas):>6}/{total:<4} {pct:>5.0f}%")


def imprimir_detalle(
    consultas: list[tuple[str, str]],
    top3s: dict[str, dict[str, list[str]]],
    evidencias: dict[str, dict[str, tuple[float, str]]],
    retador: tuple[str, int | None],
) -> None:
    """Detalle de las consultas donde el retador difiere del baseline."""
    base = nombre(*BASELINE)
    ret = nombre(*retador)
    print(f"\n{'='*66}")
    print(f"DETALLE — '{base}' vs '{ret}'  (solo consultas que difieren)")
    print(f"{'='*66}")

    n_mostradas = 0
    for qid, texto in consultas:
        docs_base = top3s[base][qid]
        docs_ret = top3s[ret][qid]
        if not discrepan(docs_base, docs_ret):
            continue

        n_mostradas += 1
        print(f"\n{qid}  {texto}")
        solo_base = [d for d in docs_base if d not in docs_ret]
        solo_ret = [d for d in docs_ret if d not in docs_base]
        comunes = [d for d in docs_base if d in docs_ret]

        print(f"  en ambos:      {', '.join(comunes) if comunes else '(ninguno)'}")
        for doc_id in solo_base:
            sim, frag = evidencias[qid].get(doc_id, (0.0, ""))
            print(f"  SOLO {base:<10} {doc_id}  (mejor sim {sim:.3f})")
            print(f"      {frag[:200].strip()}...")
        for doc_id in solo_ret:
            sim, frag = evidencias[qid].get(doc_id, (0.0, ""))
            print(f"  SOLO {ret:<10} {doc_id}  (mejor sim {sim:.3f})")
            print(f"      {frag[:200].strip()}...")

    if n_mostradas == 0:
        print("\n  (ninguna: las dos estrategias devuelven el mismo conjunto siempre)")
    else:
        print(f"\n  -> {n_mostradas} consultas para juzgar a mano.")


def imprimir_coherencia(
    consultas: list[tuple[str, str]],
    top3s: dict[str, dict[str, list[str]]],
    doc_a_fenomeno: dict[str, int],
) -> None:
    """Tabla de coherencia de fenómeno por estrategia (señal automática)."""
    print(f"\n{'='*66}")
    print("COHERENCIA DE FENÓMENO — documentos del tema correcto")
    print(f"{'='*66}")
    print("  Señal de diagnóstico, NO ground truth: un documento del fenómeno")
    print("  correcto puede ser irrelevante igual. Pero uno de OTRO fenómeno")
    print("  casi con seguridad está mal, y ocupa un cupo de los 3.\n")
    print(f"  {'estrategia':<16} {'docs correctos':>15}   {'consultas con fallo':>20}")
    for metodo, m in ESTRATEGIAS:
        etiqueta = nombre(metodo, m)
        aciertos, total, con_fallo = coherencia_fenomeno(
            consultas, top3s[etiqueta], doc_a_fenomeno
        )
        pct = 100.0 * aciertos / max(total, 1)
        print(f"  {etiqueta:<16} {aciertos:>6}/{total:<4} {pct:>4.0f}%"
              f"   {con_fallo:>10}/{len(consultas)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--indice", default=str(INDICE_POR_DEFECTO),
                   help="Carpeta encoder_<slug>. Por defecto: %(default)s")
    p.add_argument("--queries", default=str(QUERIES_FILE),
                   help="Archivo de consultas. Por defecto: %(default)s")
    p.add_argument("--contra", default="mean",
                   help="Estrategia retadora para el detalle: mean | sum | topm3 ... "
                        "Por defecto: %(default)s")
    p.add_argument("--limite", type=int, default=None,
                   help="Usar solo las primeras N consultas (pasada rápida).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retador = parse_estrategia(args.contra)

    print(f"Cargando índice {args.indice} ...")
    index, metas = load_base(Path(args.indice))
    print(f"  {index.ntotal} vectores.")
    print("Cargando encoder (tarda; es e5-large en CPU) ...")
    enc = Encoder()

    consultas = cargar_consultas(Path(args.queries), args.limite)
    print(f"Consultas a evaluar: {len(consultas)}\n")

    top3s: dict[str, dict[str, list[str]]] = {nombre(mt, m): {} for mt, m in ESTRATEGIAS}
    evidencias: dict[str, dict[str, tuple[float, str]]] = {}

    for i, (qid, texto) in enumerate(consultas, 1):
        # UNA búsqueda por consulta, reutilizada por todas las estrategias.
        scored = search(texto, index, metas, enc)
        evidencias[qid] = evidencia_por_doc(scored)
        for metodo, m in ESTRATEGIAS:
            top3s[nombre(metodo, m)][qid] = top3_con(scored, metodo, m)
        print(f"  [{i:>3}/{len(consultas)}] {qid}")

    imprimir_resumen(top3s, len(consultas))
    imprimir_coherencia(consultas, top3s, fenomeno_por_doc(metas))
    imprimir_detalle(consultas, top3s, evidencias, retador)

    base = nombre(*BASELINE)
    iguales_topm1 = all(
        set(top3s[base][qid]) == set(top3s["topm(m=1)"][qid]) for qid, _ in consultas
    )
    print(f"\nControl: topm(m=1) reproduce a max en las {len(consultas)} consultas -> {iguales_topm1}")


if __name__ == "__main__":
    main()
