# Decisiones de diseño — Sistema de recuperación (Etapa 1)

Documento de trabajo para redactar el `informe_tecnico.pdf`. Recoge **qué se
decidió y por qué** en el sistema de recuperación, con los números que respaldan
cada decisión.

> **Aviso sobre los números.** Los marcados como **[SUBSET]** salen de un índice
> de prueba de 75 documentos y 7.654 chunks, que es **95% PDFs** y por tanto no
> representa el corpus real. Sirven para ilustrar el método, **no como
> resultados finales**. Los marcados **[PENDIENTE]** se llenarán tras la corrida
> del corpus completo. No los publiquen como definitivos.

La guía del reto (§1.4) exige que el informe describa: **estrategia de chunking
y su justificación**, **encoder(s) y criterios de elección**, **tipo de índice
FAISS**, y el grafo si aplica. Las secciones 3, 4 y 5 de este documento cubren
esos tres puntos; el resto es material adicional que fortalece el informe.

---

## 1. Arquitectura general

```
corpus → extract → clean → chunk → encode → FAISS IndexFlatIP + metadata.jsonl
                                                      │
consulta → encode(query:) → search (pool de 200) ─────┤
                                                      ├→ top_documents → 3 doc_id      (F1@3)
                                                      └→ top_fragments → 10 fragmentos (NDCG@10)
```

Todo el módulo de recuperación opera **exclusivamente sobre vectores,
puntuaciones y metadata**. No interviene ningún modelo generativo, conforme a la
restricción de §8.3.

---

## 2. Preprocesamiento

### 2.1 Extracción por formato

Se procesan **todos** los formatos presentes en el corpus:

| formato | herramienta | nota |
|---|---|---|
| PDF | PyMuPDF | se preserva el orden de lectura por página |
| JSON | parser propio | ver abajo |
| CSV / XLSX | pandas | cada fila → `col: valor \| col: valor` |
| HTML | BeautifulSoup + lxml | se eliminan `script`, `style`, `noscript` |
| TXT / MD | lectura directa | |
| Imágenes | Tesseract OCR (`spa+eng+por`) | con filtro de calidad |
| PBF (vector tiles) | mapbox-vector-tile | con dedup y filtro de atributos |

**Decisión: no descartar ningún formato.** La alternativa considerada era
ignorar PBF e imágenes por ser ruidosos; se optó por incorporarlos con filtros
de calidad específicos en vez de perder su contenido.

**Filtros de calidad.** El OCR y los mapas producen mucho ruido, así que cada uno
tiene un criterio de admisión:

- **OCR** (`_ocr_es_util`): se descarta el texto con menos de 8 palabras o con
  menos del 60% de caracteres alfabéticos. Un OCR fallido produce sopa de
  símbolos; ese umbral la filtra sin perder texto real.
- **PBF** (`_prop_es_util`): se conservan solo los atributos con significado
  textual. Se descartan identificadores internos y códigos administrativos
  (`fid`, `osm_id`, `b_adm2_pcode`…), valores puramente numéricos y cadenas de
  menos de 3 caracteres. Además se deduplican features con atributos idénticos,
  que en los vector tiles se repiten masivamente.

**JSON sin duplicar el cuerpo.** Los JSON del corpus traen a la vez `body_text`
(texto completo) y `body_paragraphs` (el mismo texto como lista). Concatenar
ambos duplicaba todo el cuerpo del documento e inflaba el índice con contenido
repetido. Se elige **una sola** fuente de cuerpo, con preferencia por la lista
de párrafos.

**CSV tolerante a fallos.** Los CSV reales no siempre son limpios: hay
separadores distintos de la coma y filas con más campos que la cabecera. El
parseo estricto tumbaba el archivo entero y se perdía el documento completo. Si
el parseo estándar falla, se reintenta dejando que pandas olfatee el separador y
saltando las filas rotas. Resultado: **26/26 CSV del corpus se extraen** (antes,
25/26).

### 2.2 Limpieza

- Normalización Unicode NFC y eliminación de caracteres de control.
- **Unificación de separadores de línea Unicode** (`U+2028`, `U+2029`, `U+0085`)
  a `\n`. Esto no es cosmético: esos caracteres aparecen en el texto extraído de
  PDFs reales, `str.splitlines()` los trata como salto de línea y **partían un
  objeto JSON por la mitad** al leer el `metadata.jsonl`. Por el mismo motivo,
  todo el código lee JSON Lines con `split("\n")` y nunca con `splitlines()`.
- Reunificación de palabras cortadas por guión a final de línea (`infor-\nmación`
  → `información`), frecuente en PDFs maquetados a columnas.
- **Eliminación de boilerplate**: líneas cortas (≤80 caracteres) que se repiten
  4 o más veces en el documento se consideran encabezado o pie recurrente y se
  eliminan, igual que la numeración de página. Sin esto, el índice se llena de
  chunks idénticos con el título del informe.
- Detección de idioma por documento (`langdetect`), guardada en la metadata.

---

## 3. Chunking (requisito del informe)

**Decisión: chunking híbrido — frontera de oración + empaque goloso hasta 450
tokens + solapamiento de 1 oración.**

El proceso es: partir el texto en oraciones completas, ir acumulando oraciones
consecutivas mientras quepan en 450 tokens, cerrar el chunk al llegar al límite,
y arrancar el siguiente conservando la última oración del anterior.

**Por qué no las dos estrategias puras:**

- *Una oración por chunk* produce fragmentos diminutos, con poco contexto para
  que el encoder los sitúe semánticamente, y multiplica el tamaño del índice.
- *Tamaño fijo en tokens* corta oraciones por la mitad. Además de degradar el
  embedding, incumple el **requisito de completitud lingüística de §3.3**: los
  fragmentos entregados deben ser unidades de texto completas y legibles.

El híbrido conserva lo bueno de ambas: unidades semánticas completas y de tamaño
suficiente.

**Parámetros y su justificación:**

| parámetro | valor | por qué |
|---|---|---|
| `CHUNK_MAX_TOKENS` | 450 | margen de seguridad bajo el límite de 512 tokens de e5 |
| `CHUNK_OVERLAP_SENTENCES` | 1 | una idea que cruza la frontera entre dos chunks queda representada en ambos |
| `CHUNK_MIN_TOKENS` | 20 | descarta residuos sin contenido (líneas sueltas, pies de tabla) |

### 3.1 Tope de fragmentos por documento

**Decisión: un solo documento aporta como máximo 2.000 fragmentos al índice**
(`MAX_CHUNKS_POR_DOCUMENTO`).

El corpus incluye volcados bibliográficos en CSV —listados de título y abstract
de artículos de PubMed— de hasta 35 MB. Uno solo de esos archivos generaba
**~36.000 fragmentos**, y los cinco juntos rondaban los **110.000: más de la
mitad de todo el índice, salidos de cinco archivos**.

Tres razones para acotarlo:

1. **Calidad de recuperación.** Un documento con 36.000 fragmentos tiene 36.000
   oportunidades de producir un chunk con similitud alta por azar, y compite con
   ventaja injusta en la agregación a nivel documento.
2. **Diversidad del pool.** Un documento así monopoliza los candidatos y deja
   fuera a documentos distintos — el mismo problema que ya había obligado a
   ampliar el pool de 50 a 200 (ver §6.1).
3. **Costo.** Duplicaba con creces el tiempo de indexación para incorporar
   listados bibliográficos, que son metadatos de artículos y no texto analítico
   sobre los fenómenos del reto.

El tope es **general, no una excepción para un archivo concreto**, y está fijado
donde ningún documento de prosa lo alcanza: el PDF más grande del corpus produce
unos 500 fragmentos, cuatro veces por debajo del límite. En la práctica solo
recorta tablas. Los documentos recortados **siguen presentes y siendo
recuperables**; lo que se pierde son filas adicionales de un listado, no el
contenido de un documento argumentativo.

**Caso borde documentado:** una oración que por sí sola excede los 450 tokens no
se puede dividir sin cortarla, así que se emite como su propio chunk. Si supera
los 512 tokens del encoder, e5 la trunca al codificar y el vector no representa
su parte final — aunque **el texto completo sí queda en la metadata**, de modo
que el fragmento entregado al evaluador está íntegro. Ocurre en texto sin
puntuación final: tablas, listas de referencias, PDFs con maquetación rota.
Magnitud: **[PENDIENTE]** de cuantificar sobre el corpus completo.

---

## 4. Encoder (requisito del informe)

**Decisión: `intfloat/multilingual-e5-large`.**

Evaluado contra los seis criterios de §4.3:

| criterio | cumplimiento |
|---|---|
| Soporte multilingüe | nativo en ES/EN/PT (base XLM-RoBERTa, 100 idiomas) |
| Dimensionalidad | 1024 |
| Longitud máxima | 512 tokens, que es lo que condiciona `CHUNK_MAX_TOKENS` |
| Rendimiento en benchmarks | de los mejores en las tablas de *retrieval* de MTEB/BEIR |
| Licencia | MIT |
| Eficiencia | 2,2 GB; **inviable en CPU** (ver §9) |

Es un **encoder**, no un decoder, conforme a la prohibición de §4.2.

**Prefijos obligatorios.** La familia e5 se entrenó con prefijos explícitos y los
exige en inferencia: `"query: "` para consultas y `"passage: "` para fragmentos.
Omitirlos degrada la calidad de forma silenciosa. Están en `config.py` y los
aplica `Encoder.encode`.

**Normalización.** Los embeddings se producen normalizados (norma 1), lo que
permite usar producto interno como similitud coseno (ver §5).

### 4.1 Decisión de NO usar un segundo encoder

§4.4 permite construir la base con varios encoders y §8.4 describe cómo fusionar
sus rankings (CombSUM, CombMNZ, RRF). **Se evaluó y se descartó**, por tres
razones:

1. **Duplica el costo de indexación**, que es el cuello de botella real: sin GPU
   local, cada corrida va a Colab y el corpus completo ya son varias horas.
2. **Duplica el tamaño de la entrega** (~4 KB por chunk a 1024 dimensiones).
3. **No hay forma de verificar que mejore.** Sin juicios de relevancia no se
   puede medir si la fusión sube o baja el NDCG. RRF vota por rango ignorando la
   magnitud: un encoder peor pesa igual que e5 y puede arrastrar el ranking hacia
   abajo. Adoptarlo a ciegas es una apuesta, no una mejora.

La arquitectura lo soporta si algún día se quisiera (cada encoder tendría su
subcarpeta `encoder_<slug>/`), pero la decisión es deliberada y defendible: **la
guía lo permite, no lo exige.**

---

## 5. Índice FAISS (requisito del informe)

**Decisión: `IndexFlatIP` sobre vectores normalizados.**

- **Producto interno + vectores normalizados = similitud coseno.** No hace falta
  un índice de coseno aparte; es la equivalencia matemática estándar.
- **Búsqueda exacta, recall del 100%.** `IndexFlatIP` compara la consulta contra
  todos los vectores. Los índices aproximados (`IndexIVFFlat`, `IndexHNSW`)
  cambian exactitud por velocidad, y esa es una compensación que **solo tiene
  sentido a escala de millones de vectores**. Con ~10⁵ chunks, la búsqueda exacta
  responde en milisegundos: aceptar pérdida de recall no compraría nada.

**Invariante crítica (§5.3).** El id interno que FAISS asigna a cada vector es su
posición de inserción, así que **la línea *i* de `metadata.jsonl` debe describir
el fragmento cuyo vector se insertó en la posición *i***. Si ese orden se rompe,
la metadata queda desalineada y el índice entero es inservible. Se garantiza
manteniendo una única lista de fragmentos que sirve a la vez para codificar y
para escribir la metadata, y se **verifica automáticamente** en tres puntos: al
construir el índice, al fusionar índices parciales y al empaquetar la entrega.

---

## 6. Módulo de recuperación

### 6.1 Pool de candidatos

**Decisión: `TOP_K_CHUNKS_SEARCH = 200`** (subido desde 50).

Motivo, con evidencia: con un pool de 50 chunks había consultas (q017 y q025)
cuyos 50 candidatos provenían de **solo 2 documentos distintos**. El sistema no
podía devolver 3 documentos aunque quisiera, y perdía recall garantizado en
F1@3. La causa es la distribución del corpus: hay documentos que generan cientos
de chunks (hasta 420 **[SUBSET]**) y copan el pool ellos solos. Con 200, esas
consultas ven 19 y 28 documentos distintos.

El costo es nulo: `IndexFlatIP` ya recorre todos los vectores, así que `k` solo
cambia cuántos resultados se devuelven, no el trabajo de búsqueda.

### 6.2 Agregación a nivel documento

El índice solo conoce fragmentos; los 3 documentos hay que derivarlos agrupando
el pool por `doc_id` y asignando **una** puntuación a cada documento. Se
implementaron y compararon cuatro estrategias:

| estrategia | qué premia | sesgo que introduce |
|---|---|---|
| `max` | el mejor pasaje del documento | neutral al tamaño, pero ciego a la acumulación de evidencia |
| `sum` | acumulación de evidencia | **sesgo de longitud**: los documentos largos aportan más chunks al pool |
| `mean` | consistencia de lo recuperado | favorece al documento con un único chunk afortunado |
| `topm` | acumulación **acotada** a los *m* mejores chunks | punto medio |

**Propiedad útil para el informe:** `topm` no es una cuarta estrategia
independiente, es el **continuo** entre las otras dos — con *m*=1 es exactamente
`max`, y con *m* ≥ número de chunks del documento es exactamente `sum`. El
parámetro *m* regula cuánta acumulación se permite. Se verificó empíricamente.

**Decisión actual: `max`.** Justificación:

1. Es la **única neutral a la longitud del documento**, y el corpus es
   extremadamente heterogéneo en ese eje (964 JSON cortos frente a 760 PDFs de
   cientos de páginas). Con solo 3 cupos, un sesgo sistemático es caro.
2. `mean` quedó **descartada con datos**: es la peor en coherencia temática y
   empeora al ampliar el pool.
3. La diferencia entre `max` y las demás **[SUBSET]** es de 3 consultas sobre 50,
   sobre un índice no representativo. Insuficiente para abandonar un baseline
   estándar y defendible.

Decisión final: **[PENDIENTE]** de la corrida completa. Regla acordada: se
mantiene `max` salvo que otra estrategia reduzca claramente los documentos del
tema equivocado.

### 6.3 Selección de fragmentos

1. Se recorre el pool en orden de similitud descendente.
2. Cada chunk se divide en sub-fragmentos de **≤250 palabras sin cortar
   oraciones** (§9.2.1). Los chunks de 450 tokens rondan las 300 palabras, así
   que esta división aplica a la mayoría **[SUBSET: 87%]**.
3. Se descartan los casi-duplicados (ver §7) y se rellena con el siguiente
   candidato, hasta completar 10.

---

## 7. Post-filtros

**Deduplicación de fragmentos: SÍ.** El solapamiento de una oración entre chunks
consecutivos hace que dos fragmentos vecinos compartan texto. Devolver ambos
desperdicia cupos del top-10 sin aportar información nueva y penaliza el NDCG.
Se comparan por **similitud de Jaccard sobre el conjunto de palabras**, con
umbral **0,8**, y el descartado se sustituye por el siguiente candidato del pool.

**Umbral duro de similitud: NO.** Descartar fragmentos por debajo de una
similitud mínima rompería el requisito de entregar **exactamente 10** fragmentos
(§9.3.2 penaliza los arrays con un número distinto de elementos). Es preferible
entregar un décimo fragmento mediocre que nueve.

**Filtrado por fenómeno: NO.** Las consultas no vienen etiquetadas con su
fenómeno; cualquier asignación sería una inferencia nuestra. Se usa como
**señal de diagnóstico** para evaluar estrategias, nunca como filtro en
producción.

---

## 8. Formato de salida y reproducibilidad

- `resultados.jsonl`: exactamente 50 líneas, `q001`–`q050` **en orden**, con 3
  documentos y 10 fragmentos por consulta, ranks desde 1 (§9.3).
- **La entrega es autocontenida.** `generador.py` resuelve todas sus rutas
  respecto a su propia ubicación, no respecto a `config.py` — porque `config.py`
  deriva las suyas de su propio directorio y al copiarlo dentro de `entrega/`
  apuntaría a `entrega/entrega/`. Verificado ejecutando el paquete fuera del
  repositorio, en una carpeta sin dependencias en el directorio padre.
- **Validador automático** (`scripts/empaquetar_entrega.py`): comprueba la
  presencia de los entregables de §1.4, la alineación índice↔metadata de §5.3 y
  el esquema completo de §9.3, incluido el límite de 250 palabras. Sale con
  código distinto de cero si algo falla, para usarlo como puerta antes de
  entregar.

---

## 9. Indexación a escala

**Restricción de hardware:** sin GPU local (Intel Iris Xe). e5-large en CPU es
inviable: la estimación para el corpus completo superaba los días. **Solución:
indexación en Google Colab con GPU T4.**

**Estrategia de tandas.** El corpus completo son ~133.000 chunks estimados y unas
3-4 horas de T4, por encima de lo que garantiza una sesión gratuita. Se indexa en
**20 tandas, una por subcarpeta**, con salida directa a Drive, y un script de
fusión las une preservando la alineación. Si una sesión se corta, se retoman solo
las tandas que faltan.

**Detalle técnico:** los `doc_id` se derivan de la ruta relativa a la carpeta
indexada. Al indexar por tandas, cada una los habría calculado respecto a su
propia carpeta, produciendo identificadores que podrían colisionar al fusionar.
Por eso `build_index` acepta `--root`, la raíz contra la que se calculan los ids,
de modo que las tandas producen exactamente los mismos `doc_id` que una corrida
única. El fusionador además verifica que no haya `chunk_id` repetidos.

---

## 10. Resultados medidos

**Todo lo de esta sección es [SUBSET]: 75 documentos, 7.654 chunks, 95% PDFs.**
Ilustra el método de evaluación, no el rendimiento final.

Comparación de estrategias de agregación sobre las 50 consultas reales, con pool
de 200. "Coherencia temática" = proporción de documentos devueltos que pertenecen
al fenómeno de la consulta; es una **condición necesaria**, no una medida de
relevancia.

| estrategia | documentos del tema correcto | consultas con ≥1 documento ajeno |
|---|---|---|
| `max` | 126/150 (84%) | 15/50 |
| `mean` | 115/150 (77%) | 21/50 |
| `sum` | 130/150 (87%) | 12/50 |
| `topm(m=2)` | 129/150 (86%) | 13/50 |
| `topm(m=3)` | 130/150 (87%) | 12/50 |
| `topm(m=10)` | 130/150 (87%) | 12/50 |

Controles de validez del experimento: `topm(m=1)` reprodujo `max` en las 50
consultas y `topm(m=10)` convergió a `sum`, como exige la equivalencia
matemática. Cambiar de `max` a `mean` altera el conjunto de 3 documentos en el
**80% de las consultas**: la agregación no es un ajuste marginal.

---

## 11. Limitaciones conocidas

Conviene declararlas en el informe; son decisiones informadas, no descuidos.

1. **No hay juicios de relevancia.** El ground truth no es público durante el
   reto, así que ninguna decisión pudo optimizarse contra la métrica real. Se usó
   coherencia temática como señal indirecta y, ante empates, se prefirió el
   baseline defendible.
2. **Chunks que exceden 512 tokens** se truncan al codificar (§3). El texto
   entregado está completo; el vector no lo representa entero.
3. **El corpus no es estrictamente trilingüe.** Además de ES/EN/PT aparecen
   documentos en francés, ruso, árabe y chino — UNOOSA publica en los seis
   idiomas oficiales de la ONU. e5 los indexa sin problema por ser multilingüe.
   **[PENDIENTE]** confirmar la distribución sobre el corpus completo.
4. **La asignación de fenómeno a cada consulta es una inferencia nuestra**,
   deducida del orden de los bloques en el PDF de preguntas y verificada leyendo
   las 50. No es un dato provisto por el reto.

---

## 12. Pendiente de cerrar

- Corrida de indexación del corpus completo.
- Decisión final de `DOC_AGGREGATION` (`max` vs `topm(m=3)`) con el índice real.
- Cuantificar los chunks truncados y la distribución de idiomas.
- Regenerar `resultados.jsonl` y pasar el validador de entrega.
