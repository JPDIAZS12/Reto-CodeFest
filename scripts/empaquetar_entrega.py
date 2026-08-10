"""Arma y verifica el paquete final de entrega/ (§1.4 y §9.3 de la guía).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config import (
    ENTREGA_DIR,
    ENCODER_SLUG,
    QUERIES_FILE,
    TOP_N_DOCUMENTS,
    TOP_N_FRAGMENTS,
    MAX_WORDS_PER_FRAGMENT,
)

# Lo que hay que copiar dentro de entrega/ para que sea autocontenida.
ARCHIVOS_A_COPIAR = ["config.py"]
CARPETAS_A_COPIAR = ["src"]

# Consultas: van al paquete para que generador.py corra sin argumentos.
N_CONSULTAS = 50


def copiar_dependencias(raiz: Path, entrega: Path) -> list[str]:
    """Copia config.py, src/ y queries.jsonl dentro de entrega/. Devuelve lo copiado."""
    copiados = []

    for nombre in ARCHIVOS_A_COPIAR:
        shutil.copy2(raiz / nombre, entrega / nombre)
        copiados.append(nombre)

    for nombre in CARPETAS_A_COPIAR:
        destino = entrega / nombre
        if destino.exists():
            shutil.rmtree(destino)
        shutil.copytree(raiz / nombre, destino,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        copiados.append(f"{nombre}/")

    shutil.copy2(QUERIES_FILE, entrega / "queries.jsonl")
    copiados.append("queries.jsonl")

    return copiados


def validar_estructura(entrega: Path) -> list[str]:
    """Comprueba que estén todos los entregables de §1.4. Devuelve errores."""
    errores: list[str] = []

    indice_dir = entrega / "base_vectorial" / f"encoder_{ENCODER_SLUG}"
    obligatorios = [
        entrega / "generador.py",
        entrega / "resultados.jsonl",
        entrega / "config.py",
        entrega / "src" / "retrieve.py",
        indice_dir / "index.faiss",
        indice_dir / "metadata.jsonl",
    ]
    for ruta in obligatorios:
        if not ruta.exists():
            errores.append(f"FALTA: {ruta.relative_to(entrega)}")

    if not (entrega / "informe_tecnico.pdf").exists():
        errores.append("FALTA: informe_tecnico.pdf")

    if (indice_dir / "index.faiss").exists() and (indice_dir / "metadata.jsonl").exists():
        import faiss
        index = faiss.read_index(str(indice_dir / "index.faiss"))
        lineas = (indice_dir / "metadata.jsonl").read_text(encoding="utf-8").split("\n")
        n_metas = len([l for l in lineas if l.strip()])
        if index.ntotal != n_metas:
            errores.append(
                f"DESALINEADO: index.faiss tiene {index.ntotal} vectores pero "
                f"metadata.jsonl tiene {n_metas} líneas"
            )

    return errores


def validar_resultados(path: Path) -> list[str]:
    """Valida resultados.jsonl contra el esquema de la Tabla 2.
    """
    errores = []

    lineas = []

    for i, linea in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not linea.strip():
            continue
        try:
            obj = json.loads(linea)
        except json.JSONDecodeError as e:
            errores.append(f"Línea {i+1}: JSON inválido ({e})")
            continue
        lineas.append(obj)

    if len(lineas) != N_CONSULTAS:
        errores.append(
            f"el archivo tiene {len(lineas)} objetos válidos, se esperaban {N_CONSULTAS}"
        )

    for posicion, obj in enumerate(lineas, start=1):
        esperado = f"q{posicion:03d}"         

        faltantes = [k for k in ("query_id", "documents", "fragments") if k not in obj]
        if faltantes:
            errores.append(f"objeto {posicion} ({esperado}): faltan claves {faltantes}")
            continue

        qid = obj["query_id"]

        if qid != esperado:
            errores.append(
                f"objeto {posicion}: query_id '{qid}', se esperaba '{esperado}' (deben ir en orden)"
            )

        documentos = obj["documents"]
        if not isinstance(documentos, list):
            errores.append(f"{qid}: 'documents' no es una lista")
        else:
            if len(documentos) != TOP_N_DOCUMENTS:
                errores.append(
                    f"{qid}: documents tiene {len(documentos)} elementos, "
                    f"se esperaban {TOP_N_DOCUMENTS}"
                )
            for j, doc in enumerate(documentos, start=1):
                if not isinstance(doc, dict):
                    errores.append(f"{qid}: documento {j} no es un objeto")
                    continue
                for clave in ("rank", "doc_id"):
                    if clave not in doc:
                        errores.append(f"{qid}: documento {j} sin '{clave}'")
                if doc.get("rank") != j:
                    errores.append(
                        f"{qid}: documento {j} tiene rank {doc.get('rank')}, se esperaba {j}"
                    )
        fragmentos = obj["fragments"]
        if not isinstance(fragmentos, list):
            errores.append(f"{qid}: 'fragments' no es una lista")
        else:
            if len(fragmentos) != TOP_N_FRAGMENTS:
                errores.append(
                    f"{qid}: fragments tiene {len(fragmentos)} elementos, "
                    f"se esperaban {TOP_N_FRAGMENTS}"
                )
            for j, frag in enumerate(fragmentos, start=1):
                if not isinstance(frag, dict):
                    errores.append(f"{qid}: fragmento {j} no es un objeto")
                    continue
                for clave in ("rank", "chunk_id", "doc_id", "text"):
                    if clave not in frag:
                        errores.append(f"{qid}: fragmento {j} sin '{clave}'")
                if frag.get("rank") != j:
                    errores.append(
                        f"{qid}: fragmento {j} tiene rank {frag.get('rank')}, se esperaba {j}"
                    )
        
                n_palabras = len(str(frag.get("text", "")).split())
                if n_palabras > MAX_WORDS_PER_FRAGMENT:
                    errores.append(
                        f"{qid}: fragmento {j} tiene {n_palabras} palabras "
                        f"(máx {MAX_WORDS_PER_FRAGMENT})"
                    )

    return errores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--entrega", default=str(ENTREGA_DIR),
                   help="Carpeta de entrega. Por defecto: %(default)s")
    p.add_argument("--verificar", action="store_true",
                   help="Solo verificar, sin copiar dependencias.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entrega = Path(args.entrega)

    if not entrega.exists():
        raise SystemExit(f"No existe la carpeta de entrega: {entrega}")

    if not args.verificar:
        print(f"Copiando dependencias a {entrega} ...")
        for nombre in copiar_dependencias(ROOT, entrega):
            print(f"  + {nombre}")

    print("\nVerificando estructura (§1.4) ...")
    errores_estructura = validar_estructura(entrega)
    for e in errores_estructura:
        print(f"  [X] {e}")
    if not errores_estructura:
        print("  [OK] todos los entregables presentes y alineados.")

    print("\nVerificando resultados.jsonl (§9.3) ...")
    resultados = entrega / "resultados.jsonl"
    if resultados.exists():
        errores_resultados = validar_resultados(resultados)
        for e in errores_resultados[:20]:
            print(f"  [X] {e}")
        if len(errores_resultados) > 20:
            print(f"  ... y {len(errores_resultados) - 20} errores más")
        if not errores_resultados:
            print("  [OK] 50 líneas, esquema y límite de 250 palabras correctos.")
    else:
        errores_resultados = ["FALTA resultados.jsonl"]

    total = len(errores_estructura) + len(errores_resultados)
    print(f"\n{'='*60}")
    if total == 0:
        print("PAQUETE LISTO PARA ENTREGAR")
    else:
        print(f"{total} problema(s) por resolver antes de entregar")
    print(f"{'='*60}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
