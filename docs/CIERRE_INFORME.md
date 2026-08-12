# Checklist de cierre del informe técnico

Cómo llenar cada hueco `[COMPLETAR]` de `docs/informe_tecnico.md` y generar el
PDF final. Todos los números salen de comandos ya escritos — nadie tiene que
calcular nada a mano.

**Requisito previo:** la corrida del corpus completo terminada, es decir:
`entrega/base_vectorial/encoder_e5-large/` (índice fusionado) y
`entrega/resultados.jsonl` (50 consultas). Para el hueco del grafo, además
`entrega/grafo/grafo.graphml`.

---

## Huecos y su comando

| # | Hueco (sección del informe) | Comando | Dónde mirar en la salida |
|---|---|---|---|
| 1 | Nombre del equipo e integrantes (encabezado) | — | lo define el equipo |
| 2 | % de chunks >512 tokens (§2, caso borde) | `python scripts/informe_indice.py --indice entrega/base_vectorial/encoder_e5-large` | sección **B. CHUNKING**, línea de chunks que superan los tokens del encoder (dividir por "chunks totales" de la sección A) |
| 3 | Confirmar `max` pooling (§5.2) | `python scripts/comparar_agregacion.py --indice entrega/base_vectorial/encoder_e5-large` | **RESUMEN**: % de consultas cuyo top-3 difiere por estrategia. Regla acordada: se mantiene `max` salvo que otra estrategia reduzca claramente los documentos del tema equivocado (revisar el DETALLE a ojo) |
| 4 | Tabla comparativa de agregación (§5.2) | el mismo comando del punto 3 | copiar la tabla del RESUMEN (estrategia / difieren / %) |
| 5 | Dimensiones del grafo (§6) | `python scripts/informe_grafo.py` | todo el bloque: entidades, relaciones, % tipadas vs. genéricas |
| 6 | Distribución de idiomas (§8, limitación 3) | comando del punto 2 | sección **A. COBERTURA**, tabla "documentos por idioma" |

Si al portar el grafo se agregan los tests (`tests/test_graph.py`), actualizar
también el conteo de casos de §7 del informe:

```
for f in tests/test_*.py; do grep -c "^\s*check(" "$f"; done | paste -sd+ | bc
```

(hoy da **119**; ese número ya está escrito en el informe).

## Generar y verificar el PDF

```
python scripts/generar_pdf.py --estricto
```

- Convierte `docs/informe_tecnico.md` → `entrega/informe_tecnico.pdf` (sin
  pandoc: markdown + Chrome headless).
- `--estricto` falla si queda algún `[COMPLETAR]` o si el PDF excede las
  **8 páginas** (§1.4). Sin la bandera solo avisa.
- Con el borrador actual el PDF ocupa **5 páginas**: hay ~3 de margen para la
  tabla de agregación y los números del grafo.

## Último paso

```
python scripts/empaquetar_entrega.py
```

Valida la entrega completa (§1.4: entregables presentes, alineación
índice↔metadata, esquema de resultados, contrato de generador.py).
