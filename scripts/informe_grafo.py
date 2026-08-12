"""Estadísticas del grafo de conocimiento para el informe técnico (§6).

Lee entrega/grafo/grafo.graphml y emite exactamente los números que pide el
hueco [COMPLETAR] del informe: nº de entidades, nº de relaciones, % de aristas
tipadas vs. genéricas, distribución de tipos de relación y top de entidades.

Uso:  python scripts/informe_grafo.py [--grafo entrega/grafo/grafo.graphml]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GRAFO_DIR  # noqa: E402

RELACION_GENERICA = "se_relaciona_con"
TOP_ENTIDADES = 10
TOP_RELACIONES = 15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grafo",
        type=Path,
        default=GRAFO_DIR / "grafo.graphml",
        help="ruta al grafo.graphml exportado por src.graph",
    )
    args = parser.parse_args()

    try:
        import networkx as nx
    except ImportError:
        print("[ERROR] networkx no está instalado (pip install networkx).")
        sys.exit(1)

    if not args.grafo.exists():
        print(f"[ERROR] no existe {args.grafo}. Ejecuta antes: python -m src.graph")
        sys.exit(1)

    g = nx.read_graphml(args.grafo)

    n_nodos = g.number_of_nodes()
    n_aristas = g.number_of_edges()
    relaciones = Counter(
        datos.get("relation", "?") for _, _, datos in g.edges(data=True)
    )
    genericas = relaciones.get(RELACION_GENERICA, 0)
    tipadas = n_aristas - genericas

    print("=" * 66)
    print("GRAFO DE CONOCIMIENTO — números para el informe (§6)")
    print("=" * 66)
    print(f"\n  entidades (nodos)      : {n_nodos:,}")
    print(f"  relaciones (aristas)   : {n_aristas:,}")
    if n_aristas:
        print(f"  aristas tipadas        : {tipadas:,}  ({100*tipadas/n_aristas:.1f}%)")
        print(f"  genéricas (co-ocurr.)  : {genericas:,}  ({100*genericas/n_aristas:.1f}%)")

    print(f"\n  tipos de relación (top {TOP_RELACIONES}):")
    for rel, n in relaciones.most_common(TOP_RELACIONES):
        print(f"    {rel:<24} {n:>8}  {100*n/max(n_aristas,1):>5.1f}%")

    print(f"\n  entidades más mencionadas (top {TOP_ENTIDADES}):")
    por_menciones = sorted(
        g.nodes(data=True),
        key=lambda par: int(par[1].get("mentions", 0)),
        reverse=True,
    )
    for nodo, datos in por_menciones[:TOP_ENTIDADES]:
        print(f"    {nodo:<40} {int(datos.get('mentions', 0)):>7} menciones"
              f"  [{datos.get('label', '?')}]")

    grados = [g.degree(n) for n in g.nodes]
    if grados:
        print(f"\n  grado medio            : {sum(grados)/len(grados):.1f}")
        print(f"  nodos aislados         : {sum(1 for d in grados if d == 0):,}")


if __name__ == "__main__":
    main()
