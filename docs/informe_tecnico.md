# Informe técnico — Base de conocimiento vectorial

**CODEFEST AD ASTRA 2026 · Etapa 1**
**Equipo:** Git Init — Juan Pablo Diaz, Samuel Antonio Castro, Nicolas Arce
y Wolfgang Felipe Guzman
**Fecha:** 13 de agosto de 2026

> Fuente de este documento: `docs/informe_tecnico.md`. Convertir a
> `entrega/informe_tecnico.pdf` antes de empaquetar (máx. 8 páginas, §1.4).
> Los valores marcados **[COMPLETAR]** se llenan con la corrida del corpus
> completo; el resto está verificado sobre el código entregado.

---

## 1. Arquitectura general

El sistema construye una base de conocimiento vectorial sobre el corpus
multilingüe provisto por ADL (~1.848 archivos, 3 fenómenos) y responde las 50
consultas devolviendo los 3 documentos y 10 fragmentos más relevantes por
consulta, sin intervención de ningún modelo generativo (§8.3).

```
corpus → extract → clean → chunk → encode → FAISS IndexFlatIP + metadata.jsonl
                                                      │
consulta → encode("query: ") → búsqueda (pool 200) ───┤
                                                      ├→ top-3 documentos  (F1@3)
                                                      └→ top-10 fragmentos (NDCG@10)
```

Todos los formatos del corpus se procesan: PDF (PyMuPDF, con OCR de respaldo
por página para escaneados), HTML (BeautifulSoup), JSON (parser propio que
evita duplicar cuerpo), CSV/XLSX (pandas, tolerante a fallos), TXT/MD,
imágenes (Tesseract, `spa+eng+por`) y PBF (mapbox-vector-tile). Los formatos
ruidosos (OCR, mapas) pasan filtros de calidad específicos en lugar de
descartarse. La limpieza normaliza Unicode (NFC), unifica separadores de línea
(`U+2028/U+2029` → `\n`), reunifica palabras cortadas por guión, elimina
boilerplate repetido (encabezados/pies) y detecta el idioma por documento.

Dos salvaguardas de escala: (a) **OCR de respaldo para PDFs escaneados** — 47
de los 62 PDF de Alertas Tempranas no traían capa de texto y se perdían
enteros; solo actúa en páginas sin texto; (b) **tope de 2.000 fragmentos por
documento** — cinco CSV bibliográficos de PubMed generaban ~110.000 fragmentos
(más de la mitad del índice); el tope es general y ningún documento de prosa
se le acerca (el PDF mayor ronda los 500).

---

## 2. Estrategia de chunking y justificación (§3)

**Decisión: chunking híbrido — frontera de oración + empaque goloso hasta 450
tokens + solapamiento de 1 oración.**

El texto se parte en oraciones completas; se acumulan oraciones consecutivas
mientras quepan en 450 tokens (medidos con el tokenizer del encoder); al
llegar al límite se cierra el chunk y el siguiente arranca conservando la
última oración del anterior.

**Por qué no las estrategias puras (§3.2):**

- *Una oración por chunk* produce fragmentos diminutos, con poco contexto
  semántico para el encoder, y multiplica el tamaño del índice.
- *Tamaño fijo en tokens* corta oraciones por la mitad: degrada el embedding y
  **viola el requisito obligatorio de completitud lingüística de §3.3** (los
  cortes entre chunks deben caer en límites oracionales).

El híbrido conserva unidades semánticas completas y de tamaño suficiente.

| parámetro | valor | justificación |
|---|---|---|
| `CHUNK_MAX_TOKENS` | 450 | margen de seguridad bajo el límite de 512 tokens de e5 |
| `CHUNK_OVERLAP_SENTENCES` | 1 | una idea que cruza la frontera queda representada en ambos chunks (§3.2, "semántica con superposición") |
| `CHUNK_MIN_TOKENS` | 20 | descarta residuos sin contenido (líneas sueltas, pies de tabla) |
| `MAX_CHUNKS_POR_DOCUMENTO` | 2.000 | evita que archivos tabulares gigantes dominen el índice (§1) |

**Caso borde declarado:** una oración que por sí sola excede 450 tokens se
emite como chunk propio (dividirla violaría §3.3). Si supera los 512 tokens
del encoder, e5 trunca el vector, pero **el texto completo queda en la
metadata**, de modo que el fragmento entregado está íntegro. Ocurre en texto
sin puntuación final (tablas, referencias) y es minoritario por construcción
(solo lo dispara una oración indivisible de más de 450 tokens);
`scripts/informe_indice.py` reporta su magnitud exacta sobre el índice.

Cada fragmento almacena la metadata obligatoria de la Tabla 2 (§3.4):
`doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`,
`num_tokens`, `texto`, más el campo adicional `idioma`.

---

## 3. Encoder seleccionado y criterios de elección (§4)

**Decisión: `intfloat/multilingual-e5-large`**, evaluado contra los seis
criterios de §4.3:

| criterio (§4.3) | cumplimiento |
|---|---|
| Soporte multilingüe | nativo ES/EN/PT (base XLM-RoBERTa, 100 idiomas); esencial: el corpus incluye además FR/RU/AR/ZH (UNOOSA publica en los 6 idiomas ONU) |
| Dimensionalidad | 1.024 |
| Longitud máxima | 512 tokens — condiciona `CHUNK_MAX_TOKENS = 450` |
| Benchmarks | entre los mejores encoders multilingües en las tablas de *retrieval* de MTEB/BEIR |
| Licencia | MIT |
| Eficiencia | 2,2 GB; inviable en CPU para indexar → indexación en GPU (§6 de este informe) |

Es un **encoder** (familia BERT), no un decoder, conforme a §4.2. La familia
e5 exige **prefijos de inferencia**: `"query: "` para consultas y
`"passage: "` para fragmentos; omitirlos degrada la calidad silenciosamente.
Los embeddings se producen **normalizados** (norma 1), lo que habilita la
equivalencia coseno = producto interno (§8.2).

**Decisión de NO usar un segundo encoder (§4.4).** Se evaluó y se descartó:
(1) duplica el costo de indexación, que es el cuello de botella real;
(2) duplica el tamaño de la entrega; (3) sin juicios de relevancia no hay
forma de verificar que la fusión mejore — RRF vota por rango ignorando la
magnitud, y un encoder peor puede arrastrar el ranking. La arquitectura lo
soporta (subcarpetas `encoder_<slug>/`), pero la guía lo permite, no lo exige.

---

## 4. Índice FAISS empleado (§5)

**Decisión: `IndexFlatIP` sobre vectores normalizados.**

- Producto interno sobre vectores de norma 1 ≡ **similitud coseno** (§8.2),
  sin necesidad de un índice de coseno aparte.
- **Búsqueda exacta, recall del 100%.** Los índices aproximados
  (`IndexIVFFlat`, `IndexHNSW`) cambian exactitud por velocidad — una
  compensación que solo tiene sentido a escala de millones de vectores (§5.2).
  Con ~10⁵ chunks la búsqueda exacta responde en milisegundos: aceptar pérdida
  de recall no compraría nada.

**Invariante crítica (§5.3):** la línea *i* de `metadata.jsonl` describe el
fragmento cuyo vector tiene id interno *i* en FAISS. Se garantiza usando una
única lista de fragmentos para codificar y para escribir la metadata, y se
**verifica automáticamente en tres puntos**: al construir el índice, al
fusionar índices parciales y al empaquetar la entrega (`ntotal == nº líneas`).

Persistencia según §5.4: `faiss.write_index()` → `index.faiss` +
`metadata.jsonl` (JSON Lines), directamente cargables con `faiss.read_index()`.

---

## 5. Módulo de recuperación (§8, §9)

### 5.1 Pool de candidatos

`TOP_K_CHUNKS_SEARCH = 200` (subido desde 50, con evidencia): con pool de 50
hubo consultas cuyos candidatos provenían de solo 2 documentos distintos — el
sistema no podía devolver 3 documentos. La causa es la heterogeneidad del
corpus: hay documentos que generan cientos de chunks y copan el pool. El costo
es nulo: `IndexFlatIP` recorre todos los vectores de todos modos.

### 5.2 Agregación a nivel documento (§8.6)

Se implementaron y compararon cuatro estrategias (`max`, `sum`, `mean`,
`topm`). Propiedad útil: `topm` es el **continuo** entre `max` (m=1) y `sum`
(m≥nº chunks del documento) — verificado empíricamente como control de validez
del experimento.

**Decisión: `max` pooling**, por ser
la única estrategia **neutral a la longitud del documento** en un corpus
extremadamente heterogéneo (JSON cortos vs. PDFs de cientos de páginas), donde
un sesgo sistemático con solo 3 cupos es caro. `mean` quedó descartada con
datos (peor coherencia temática, empeora al ampliar el pool). En el
subconjunto de validación, `max` obtuvo **84 % de coherencia temática frente a
77 % de `mean`**; `scripts/comparar_agregacion.py` reproduce la comparación de
las cuatro estrategias sobre las 50 consultas y el índice final.

### 5.3 Selección de fragmentos y re-ranking fino

1. Cada chunk del pool se divide en sub-fragmentos de **≤250 palabras sin
   cortar oraciones** (§9.2.1); el `chunk_id` reportado es el del chunk
   original del índice (trazabilidad, §9.2.1).
2. **Re-ranking fino:** los sub-fragmentos de los 25 mejores chunks se
   codifican con el **mismo encoder del índice** y se reordenan por su
   **propia** similitud coseno con la consulta. Motivo: el NDCG@10 se juzga
   sobre el texto del sub-fragmento entregado (§10.2.1), pero la búsqueda
   gruesa puntúa chunks de ~450 tokens; sin re-scoring, la mitad irrelevante
   de un buen chunk sale por delante de la mitad excelente del siguiente.
   *Legalidad:* §8.3 prohíbe modelos generativos — aquí solo intervienen el
   encoder del índice y operaciones vectoriales; §8.7 permite post-filtros
   sobre los vectores.
3. **Deduplicación** de casi-duplicados por similitud de Jaccard sobre
   conjuntos de palabras (umbral 0,8): el solapamiento de una oración entre
   chunks vecinos produce fragmentos casi idénticos que desperdiciarían cupos
   del top-10.
4. **Garantía de esquema por construcción:** siempre se entregan exactamente
   3 documentos y 10 fragmentos (§9.3.2 penaliza arrays incompletos). Si el
   dedup deja menos de 10, se rellena con los mejores candidatos descartados.

**Post-filtros descartados con justificación:** umbral duro de similitud
(rompería el requisito de exactamente 10 fragmentos) y filtrado por fenómeno
(las consultas no vienen etiquetadas; cualquier asignación sería inferencia
nuestra — se usa solo como señal de diagnóstico, nunca en producción).

---

## 6. Indexación a escala y reproducibilidad

**Hardware:** sin GPU local. e5-large en CPU es inviable para indexar (la
estimación superaba los días) → **indexación en Google Colab (GPU T4)**, en
**tandas por subcarpeta** con salida directa a Drive y un script de fusión que
preserva la invariante índice↔metadata. Los `doc_id` se derivan de la ruta
relativa a una raíz común (`--root`), de modo que las tandas producen
exactamente los mismos identificadores que una corrida única; el fusionador
verifica además que no haya `chunk_id` repetidos.

**Contrato de invocación (§1.5) implementado literal:** `generador.py` acepta
exactamente `--consultas`, `--base-vectorial` y `--salida` con los valores por
defecto de la Tabla 1; la invocación sin argumentos desde la raíz de la
entrega es equivalente al comando completo. La entrega es **autocontenida**
(rutas resueltas respecto al propio script) y fue verificada ejecutándola
fuera del repositorio.

**Validación automática antes de entregar** (`scripts/empaquetar_entrega.py`):
presencia de los entregables de §1.4, alineación índice↔metadata de §5.3,
esquema completo de §9.3 (50 líneas en orden `q001`–`q050`, exactamente 3
documentos y 10 fragmentos con ranks desde 1, límite de 250 palabras) y
presencia de los tres flags del contrato. Sale con código ≠ 0 si algo falla.
Cada módulo tiene además pruebas automáticas (119 casos en 8 archivos de
`tests/`).

---

## 7. Limitaciones conocidas

Decisiones informadas, no descuidos:

1. **Sin juicios de relevancia durante el reto:** ninguna decisión pudo
   optimizarse contra NDCG/F1 reales. Se usó la coherencia temática
   (proporción de documentos del fenómeno de la consulta) como señal indirecta
   y, ante empates, se prefirió el baseline defendible.
2. **Chunks que exceden 512 tokens** se truncan al codificar; el texto
   entregado está completo, el vector no lo representa entero (§2).
3. **El corpus no es estrictamente trilingüe** (FR/RU/AR/ZH presentes); e5
   los indexa por ser multilingüe. La metadata registra el `idioma` de cada
   fragmento, de modo que la distribución es auditable con
   `scripts/informe_indice.py`.
4. **La asignación de fenómeno por consulta es inferencia del equipo**
   (deducida del PDF de preguntas y verificada a mano); se usa solo como
   diagnóstico, nunca como filtro en producción.
