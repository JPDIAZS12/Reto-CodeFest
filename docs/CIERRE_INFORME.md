# Checklist de cierre del informe técnico

Estado: el informe ya es **entregable sin números de la corrida final**. Los
antiguos huecos `[COMPLETAR]` de chunking, agregación e idiomas se
reescribieron en forma cualitativa y verificable (cada afirmación apunta al
script que la reproduce). Queda **un solo hueco**, y solo en la versión con
grafo.

**Lo verdaderamente obligatorio (§1.4) no es del informe:** son
`entrega/resultados.jsonl` y `entrega/base_vectorial/` — salen únicamente de
la corrida en Colab (celda de cierre).

---

## Único hueco restante

| Hueco | Dónde | Cómo se resuelve |
|---|---|---|
| Dimensiones del grafo (§6) | solo en `main` (versión con grafo) | **Si el grafo llega:** `python scripts/informe_grafo.py` e insertar el bloque. **Si no llega al corte (jueves 10:00):** mergear PR #8 (`docs/informe-sin-grafo`), que elimina la sección y deja el informe en 0 huecos. |

## Mejoras opcionales (si llegan los números de Colab)

No bloquean la entrega; solo hacen el informe más contundente. Cada una
reemplaza una frase cualitativa por el número real:

| Sección | Comando | Qué copiar |
|---|---|---|
| §2 caso borde | `python scripts/informe_indice.py --indice entrega/base_vectorial/encoder_e5-large` | sección **B. CHUNKING**: % de chunks >512 tokens |
| §5.2 agregación | `python scripts/comparar_agregacion.py --indice entrega/base_vectorial/encoder_e5-large` | tabla del **RESUMEN** (estrategia / difieren / %); confirmar que `max` se mantiene |
| §8 limitación 3 | el mismo `informe_indice.py` | sección **A. COBERTURA**: documentos por idioma |

Si se agregan tests del grafo (`tests/test_graph.py`), recontar los casos de
§7 del informe:

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
  **8 páginas** (§1.4). En `main` avisará el hueco del grafo hasta que se
  resuelva; en la rama `docs/informe-sin-grafo` debe pasar limpio.

## Último paso

```
python scripts/empaquetar_entrega.py
```

Valida la entrega completa (§1.4: entregables presentes, alineación
índice↔metadata, esquema de resultados, contrato de generador.py).
