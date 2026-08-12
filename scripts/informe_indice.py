"""Informe de sanidad del índice y de los resultados. Todo offline: no carga
el encoder ni FAISS para buscar, así que corre en segundos.

Responde las preguntas de la corrida final:
  A) ¿Se indexó todo? cobertura por fenómeno, formato e idioma; archivos perdidos.
  B) ¿El chunking respetó el límite del encoder? chunks por encima de 512 tokens.
  C) ¿El dedup rompió la salida? fragmentos repetidos o consultas con menos de 10.
  D) ¿Hay recuperación cross-lingual? idiomas de los documentos devueltos.

Uso:
    python scripts/informe_indice.py --indice entrega/base_vectorial/encoder_e5-large
    python scripts/informe_indice.py --indice <ruta> --resultados entrega/resultados.jsonl
    python scripts/informe_indice.py --indice <ruta> --logs /ruta/log_F1.txt /ruta/log_F2.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config import (
    BASE_VECTORIAL_DIR,
    ENCODER_SLUG,
    MAX_INPUT_TOKENS,
    TOP_N_FRAGMENTS,
    MAX_WORDS_PER_FRAGMENT,
)


def cargar_metadata(indice: Path) -> list[dict]:
    """Lee metadata.jsonl. split('\\n') y no splitlines(): ver retrieve.py:49."""
    crudas = (indice / "metadata.jsonl").read_text(encoding="utf-8").split("\n")
    return [json.loads(l) for l in crudas if l.strip()]


def tabla(titulo: str, conteo: Counter, total: int) -> None:
    print(f"\n  {titulo}")
    for clave, n in conteo.most_common():
        pct = 100.0 * n / max(total, 1)
        print(f"    {str(clave):<24} {n:>8}  {pct:>5.1f}%")


def informe_cobertura(metas: list[dict]) -> None:
    """A) Qué entró al índice."""
    print("=" * 66)
    print("A. COBERTURA DEL ÍNDICE")
    print("=" * 66)

    docs = {}
    for m in metas:
        if m["doc_id"] not in docs:
            docs[m["doc_id"]] = m

    print(f"\n  chunks totales     : {len(metas):,}")
    print(f"  documentos         : {len(docs):,}")
    print(f"  chunks por documento: media {len(metas)/max(len(docs),1):.1f}")

    tabla("documentos por fenómeno:", Counter(d["fenomeno"] for d in docs.values()), len(docs))
    tabla("documentos por formato:", Counter(d["formato"] for d in docs.values()), len(docs))
    tabla("documentos por idioma:", Counter(d.get("idioma", "?") for d in docs.values()), len(docs))

    # Un fenómeno ausente o casi vacío es un fallo grave: las consultas de ese
    # bloque no tendrían nada que recuperar.
    presentes = {d["fenomeno"] for d in docs.values()}
    for fen in (1, 2, 3):
        if fen not in presentes:
            print(f"\n  [X] NO hay ningún documento del fenómeno {fen}.")

    # Documentos con un solo chunk: normales en JSON cortos, sospechosos en PDF.
    chunks_por_doc = Counter(m["doc_id"] for m in metas)
    de_un_chunk = [d for d, n in chunks_por_doc.items() if n == 1]
    pdfs_de_un_chunk = [d for d in de_un_chunk if docs[d]["formato"] == "pdf"]
    print(f"\n  documentos de 1 solo chunk : {len(de_un_chunk)}")
    print(f"    de los cuales son PDF    : {len(pdfs_de_un_chunk)}  "
          f"(sospechoso: un PDF que rinde 1 chunk suele ser un PDF escaneado "
          f"sin texto extraíble)")
    for d in pdfs_de_un_chunk[:5]:
        print(f"      {docs[d]['fuente']}")


def informe_chunking(metas: list[dict]) -> None:
    """B) El chunking respetó el límite del encoder."""
    print("\n" + "=" * 66)
    print("B. CHUNKING")
    print("=" * 66)

    tokens = [m["num_tokens"] for m in metas]
    palabras = [len(m["texto"].split()) for m in metas]
    tokens.sort()
    palabras.sort()

    def pct(lista, p):
        return lista[min(int(len(lista) * p), len(lista) - 1)]

    print(f"\n  tokens/chunk   mediana {pct(tokens,0.5)}   p95 {pct(tokens,0.95)}   máx {tokens[-1]}")
    print(f"  palabras/chunk mediana {pct(palabras,0.5)}   p95 {pct(palabras,0.95)}   máx {palabras[-1]}")

    # Un chunk por encima del límite del encoder se truncó al codificar: su
    # vector no representa todo su texto.
    excedidos = [t for t in tokens if t > MAX_INPUT_TOKENS]
    if excedidos:
        print(f"\n  [X] {len(excedidos)} chunks superan {MAX_INPUT_TOKENS} tokens: "
              f"e5 los truncó y su vector NO representa el texto completo.")
    else:
        print(f"\n  [OK] ningún chunk supera los {MAX_INPUT_TOKENS} tokens del encoder.")

    # Aviso, no error: los chunks de más de 250 palabras los parte _split_by_words
    # al generar la salida, pero conviene saber cuántos son.
    largos = [p for p in palabras if p > MAX_WORDS_PER_FRAGMENT]
    print(f"  chunks de más de {MAX_WORDS_PER_FRAGMENT} palabras: {len(largos)} "
          f"({100.0*len(largos)/max(len(palabras),1):.0f}%) — se parten al generar la salida.")


def informe_logs(rutas: list[Path]) -> None:
    """Archivos que se cayeron durante la extracción, según los logs de Colab."""
    print("\n" + "=" * 66)
    print("A-bis. ARCHIVOS PERDIDOS EN LA EXTRACCIÓN (según los logs)")
    print("=" * 66)

    patron = re.compile(r"\[WARN\] error extrayendo (.+?): (.+)")
    fallos = []
    for ruta in rutas:
        if not ruta.exists():
            print(f"  (no existe: {ruta})")
            continue
        for linea in ruta.read_text(encoding="utf-8", errors="ignore").split("\n"):
            m = patron.search(linea)
            if m:
                fallos.append((m.group(1), m.group(2)))

    if not fallos:
        print("\n  [OK] ningún archivo falló en la extracción.")
        return

    print(f"\n  {len(fallos)} archivos no pudieron extraerse:")
    for motivo, n in Counter(motivo for _, motivo in fallos).most_common(8):
        print(f"    {n:>4}  {motivo[:70]}")


def informe_resultados(ruta: Path, metas: list[dict]) -> None:
    """C) y D) sobre resultados.jsonl, sin volver a buscar."""
    print("\n" + "=" * 66)
    print("C/D. RESULTADOS: DEDUP Y COBERTURA MULTILINGÜE")
    print("=" * 66)

    idioma_por_doc = {}
    for m in metas:
        if m["doc_id"] not in idioma_por_doc:
            idioma_por_doc[m["doc_id"]] = m.get("idioma", "?")

    crudas = ruta.read_text(encoding="utf-8").split("\n")
    objetos = [json.loads(l) for l in crudas if l.strip()]

    cortas = []
    con_repetidos = []
    idiomas_frag = Counter()
    idiomas_doc = Counter()
    consultas_con_otro_idioma = 0

    for obj in objetos:
        qid = obj["query_id"]
        frags = obj["fragments"]

        # C) el dedup no debe dejar la salida corta ni repetir CONTENIDO.
        # Se compara el TEXTO, no el chunk_id: un chunk de más de 250 palabras
        # se parte en varios fragmentos que heredan el mismo chunk_id del padre,
        # así que repetir chunk_id es normal (con la mediana de 253 palabras por
        # chunk, ocurre en las 50 consultas). Lo que sí sería un fallo del dedup
        # es devolver dos veces el mismo texto.
        if len(frags) != TOP_N_FRAGMENTS:
            cortas.append((qid, len(frags)))
        textos = [f["text"] for f in frags]
        if len(set(textos)) != len(textos):
            con_repetidos.append(qid)

        # D) idiomas de lo recuperado
        idiomas_de_esta = set()
        for f in frags:
            idi = idioma_por_doc.get(f["doc_id"], "?")
            idiomas_frag[idi] += 1
            idiomas_de_esta.add(idi)
        for d in obj["documents"]:
            idiomas_doc[idioma_por_doc.get(d["doc_id"], "?")] += 1
        if idiomas_de_esta - {"es", "?"}:
            consultas_con_otro_idioma += 1

    print(f"\n  consultas             : {len(objetos)}")
    if cortas:
        print(f"  [X] {len(cortas)} consultas con != {TOP_N_FRAGMENTS} fragmentos "
              f"(§9.3.2 las penaliza): {cortas[:6]}")
    else:
        print(f"  [OK] las {len(objetos)} consultas traen {TOP_N_FRAGMENTS} fragmentos.")

    if con_repetidos:
        print(f"  [X] {len(con_repetidos)} consultas con TEXTO repetido: {con_repetidos[:6]}")
    else:
        print("  [OK] ninguna consulta repite texto (el dedup hizo su trabajo).")

    tabla("idioma de los fragmentos devueltos:", idiomas_frag, sum(idiomas_frag.values()))
    tabla("idioma de los documentos devueltos:", idiomas_doc, sum(idiomas_doc.values()))

    print(f"\n  consultas que recuperaron algo en un idioma != es: "
          f"{consultas_con_otro_idioma}/{len(objetos)}")
    if consultas_con_otro_idioma == 0:
        print("    [X] NINGUNA. Las 50 consultas son en español y el corpus es ES/EN/PT:")
        print("        que jamás cruce de idioma apunta a un problema del encoder")
        print("        o a que el corpus indexado quedó solo en español.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--indice", default=str(BASE_VECTORIAL_DIR / f"encoder_{ENCODER_SLUG}"),
                   help="Carpeta encoder_<slug>. Por defecto: %(default)s")
    p.add_argument("--resultados", default=None,
                   help="resultados.jsonl a analizar (opcional).")
    p.add_argument("--logs", nargs="*", default=[],
                   help="Logs de build_index para contar archivos perdidos (opcional).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    indice = Path(args.indice)
    if not (indice / "metadata.jsonl").exists():
        raise SystemExit(f"No hay metadata.jsonl en {indice}")

    metas = cargar_metadata(indice)

    informe_cobertura(metas)
    informe_chunking(metas)

    if args.logs:
        informe_logs([Path(l) for l in args.logs])

    if args.resultados:
        informe_resultados(Path(args.resultados), metas)

    print()


if __name__ == "__main__":
    main()
