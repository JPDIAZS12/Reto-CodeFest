"""Fusiona índices FAISS parciales (una tanda por fenómeno) en uno solo.

El corpus completo son ~80.000 chunks, unas 2 horas de T4. Correrlo de una sola
vez en Colab gratis es apostar a no desconectarse, así que se indexa por tandas
(F1, F2, F3) y se fusionan aquí.

INVARIANTE QUE HAY QUE PRESERVAR (§5.3): la línea i de metadata.jsonl describe
el vector con id interno i. Al concatenar, los vectores de la parte 2 quedan en
las posiciones [n1, n1+n2), así que su metadata debe ir exactamente después de
la de la parte 1, en el mismo orden. Este script no reordena nada: concatena
vectores y metadata en el MISMO orden de las partes que le pasas.

Uso:
    python scripts/fusionar_indices.py salida/F1 salida/F2 salida/F3 \
        --out entrega/base_vectorial/encoder_e5-large
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config import BASE_VECTORIAL_DIR, ENCODER_SLUG, EMBED_DIM


def leer_parte(carpeta: Path) -> tuple[np.ndarray, list[str]]:
    """Devuelve (vectores, líneas_de_metadata) de un índice parcial.

    Los vectores se recuperan con reconstruct_n, que en un IndexFlatIP devuelve
    exactamente lo que se insertó (no hay cuantización que los degrade).
    """
    index = faiss.read_index(str(carpeta / "index.faiss"))
    # split("\n") y no splitlines(): ver retrieve.py:49.
    crudas = (carpeta / "metadata.jsonl").read_text(encoding="utf-8").split("\n")
    lineas = [l for l in crudas if l.strip()]

    if index.ntotal != len(lineas):
        raise SystemExit(
            f"{carpeta.name}: DESALINEADO antes de fusionar — {index.ntotal} vectores "
            f"y {len(lineas)} líneas de metadata. No se puede fusionar."
        )
    if index.d != EMBED_DIM:
        raise SystemExit(
            f"{carpeta.name}: dimensión {index.d}, se esperaba {EMBED_DIM}. "
            f"¿Se indexó con otro encoder?"
        )

    vectores = index.reconstruct_n(0, index.ntotal)
    return np.asarray(vectores, dtype="float32"), lineas


def verificar_ids_unicos(lineas: list[str]) -> list[str]:
    """Devuelve los chunk_id repetidos (deberían ser cero).

    Si aparecen repetidos es que las tandas se indexaron SIN --root apuntando a
    la raíz del corpus: cada tanda calculó sus doc_id relativos a su propia
    carpeta y dos fenómenos con la misma estructura interna colisionan.
    """
    vistos = set()
    repetidos = []
    for linea in lineas:
        chunk_id = json.loads(linea)["chunk_id"]
        if chunk_id in vistos:
            repetidos.append(chunk_id)
        else:
            vistos.add(chunk_id)
    return repetidos


def descubrir_partes(carpeta: Path) -> list[Path]:
    """Subcarpetas de `carpeta` que contengan un index.faiss, en orden alfabético.

    Con 20 tandas, pasarlas a mano en la línea de comandos es incómodo y en
    PowerShell el glob no se expande solo. El orden alfabético es determinista,
    que es lo único que importa: la metadata se concatena en el mismo orden.
    """
    partes = sorted(
        d for d in carpeta.iterdir()
        if d.is_dir() and (d / "index.faiss").exists()
    )
    if not partes:
        raise SystemExit(f"No hay ninguna subcarpeta con index.faiss dentro de {carpeta}")
    return partes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("partes", nargs="*",
                   help="Carpetas de los índices parciales, EN EL ORDEN deseado.")
    p.add_argument("--partes-en", default=None,
                   help="Carpeta que CONTIENE las tandas: se descubren solas, "
                        "en orden alfabético. Alternativa a listarlas una por una.")
    p.add_argument("--out", default=str(BASE_VECTORIAL_DIR / f"encoder_{ENCODER_SLUG}"),
                   help="Carpeta de salida. Por defecto: %(default)s")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    salida = Path(args.out)

    if args.partes_en is not None:
        partes = descubrir_partes(Path(args.partes_en))
        print(f"Descubiertas {len(partes)} tandas en {args.partes_en}:")
    elif args.partes:
        partes = [Path(n) for n in args.partes]
    else:
        raise SystemExit("Pasa las carpetas de las tandas, o --partes-en <carpeta>.")

    todos_vectores = []
    todas_lineas: list[str] = []

    for carpeta in partes:
        if not carpeta.exists():
            raise SystemExit(f"No existe la parte: {carpeta}")
        vectores, lineas = leer_parte(carpeta)
        print(f"  {carpeta.name:<20} {len(lineas):>8} chunks")
        todos_vectores.append(vectores)
        todas_lineas.extend(lineas)

    print("\nVerificando que no haya chunk_id repetidos entre tandas ...")
    repetidos = verificar_ids_unicos(todas_lineas)
    if repetidos:
        print(f"  [X] {len(repetidos)} chunk_id repetidos. Los primeros:")
        for c in repetidos[:5]:
            print(f"      {c}")
        raise SystemExit(
            "Las tandas colisionan. Vuelve a indexarlas pasando "
            "--root <raiz del corpus> a build_index."
        )
    print("  [OK] todos los chunk_id son únicos.")

    matriz = np.vstack(todos_vectores)
    print(f"\nFusionando {matriz.shape[0]} vectores de dimensión {matriz.shape[1]} ...")

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(matriz)

    salida.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(salida / "index.faiss"))
    with open(salida / "metadata.jsonl", "w", encoding="utf-8") as f:
        for linea in todas_lineas:
            f.write(linea + "\n")

    # La invariante, comprobada sobre lo que quedó escrito en disco.
    escrito = faiss.read_index(str(salida / "index.faiss"))
    crudas = (salida / "metadata.jsonl").read_text(encoding="utf-8").split("\n")
    n_metas = len([l for l in crudas if l.strip()])
    if escrito.ntotal != n_metas:
        raise SystemExit(
            f"[X] El índice fusionado quedó desalineado: {escrito.ntotal} vectores "
            f"vs {n_metas} líneas."
        )

    print(f"\nListo: {escrito.ntotal} vectores en {salida}")
    print("  [OK] index.faiss y metadata.jsonl alineados.")


if __name__ == "__main__":
    main()
