# Decisiones de diseño — Sistema de recuperación (Etapa 1)

Documento de trabajo para redactar el `informe_tecnico.pdf`. Recoge **qué se
decidió y por qué** en el sistema de recuperación, con los números que respaldan
cada decisión.

> **Los números de este documento son definitivos**, medidos sobre el índice
> final del corpus completo: **79.141 fragmentos de 1.785 documentos**, el
> 98,4% de los archivos del corpus.

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

**OCR de respaldo para PDFs escaneados.** Cuando una página de PDF no trae capa
de texto, se renderiza a imagen y se pasa por Tesseract. Esto no es un adorno:
**47 de los 62 PDFs de las Alertas Tempranas de la Defensoría del Pueblo son
escaneos**, y sin este respaldo se descartaban enteros — 367 MB de informes
sobre grupos armados y control territorial, la fuente más pertinente para las 18
consultas del fenómeno 3, ausentes del índice sin que nada avisara. El respaldo
actúa **página por página y solo cuando no hay texto**, así que los PDFs
normales no pagan ningún costo. Recuperó 45 documentos y llevó esa carpeta de
794 a **1.768 fragmentos**.

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
Magnitud sobre el corpus completo: 6.568 chunks (8,3%), pero concentrados casi
por entero en formatos tabulares — de 47.293 fragmentos de PDF solo uno se
truncó. Ver la limitación 11.2.2 para el desglose.

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
de chunks (hasta 2.000, el tope de §3.1) y copan el pool ellos solos. Con 200,
esas consultas ven 19 y 28 documentos distintos.

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

**Decisión: `topm` con m=3.** Se partió de `max` como baseline y se cambió con
evidencia del corpus completo:

| estrategia | documentos del tema correcto | consultas con ≥1 documento ajeno |
|---|---|---|
| `max` | 108/150 (72%) | 18/50 |
| `mean` | 96/150 (64%) | 22/50 |
| `sum` | 137/150 (91%) | 8/50 |
| **`topm(m=3)`** | **138/150 (92%)** | **7/50** |
| `topm(m=5)` | 140/150 (93%) | 6/50 |

Tres razones:

1. **La mejora es grande y estable.** De 18 a 7 consultas con documento fuera de
   tema, y el resultado se sostiene en m=2, 3, 5 y 10 — no depende de acertar el
   parámetro.
2. **Se entendió el mecanismo del fallo de `max`, no solo el síntoma.** Bajo max
   basta un único chunk bueno para ganar, así que los documentos de un solo
   fragmento compiten en igualdad con informes de cientos. El efecto era
   extremo: **un mismo artículo aparecía en 14 de las 16 consultas del fenómeno
   1**. Eso no es recuperación, es un *hub*: un documento genéricamente parecido
   a cualquier pregunta del área, que no discrimina entre ellas. Con `topm` un
   documento necesita tres fragmentos buenos, y ese artículo —que aporta un solo
   fragmento— deja de dominar.
3. **Entraron las fuentes especializadas.** Las posiciones ocupadas por fuentes
   propias de IA en defensa (Defence AI Observatory, CSET Georgetown, SIPRI)
   pasaron de 2 a 11 sobre 48.

`mean` quedó descartada con datos: es la peor de las cuatro y empeora al ampliar
el pool, porque cada documento recibe más fragmentos de similitud baja que le
hunden el promedio.

**Alcance del cambio:** afecta **solo al F1@3**. Los 10 fragmentos los elige
`top_fragments` por similitud directa, sin pasar por la agregación, así que el
NDCG@10 es idéntico con cualquiera de las cuatro estrategias.

### 6.3 Selección de fragmentos

1. Se recorre el pool en orden de similitud descendente.
2. Cada chunk se divide en sub-fragmentos de **≤250 palabras sin cortar
   oraciones** (§9.2.1). La mediana de un chunk es de 253 palabras, así que la
   división aplica al **51%** de ellos. Como consecuencia, varios fragmentos de
   una misma respuesta pueden compartir `chunk_id`: son trozos distintos del
   mismo chunk padre, no contenido repetido.
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

**Estrategia de tandas.** El corpus completo son 79.141 chunks, por encima de lo
que garantiza una sesión gratuita de GPU. Se indexó en **20 tandas, una por
subcarpeta de fuente**, con salida directa a Drive, y un script de fusión las une
preservando la alineación. La corrida principal tomó **135 minutos en T4** para
19 tandas; la vigésima carpeta resultó vacía en origen. La decisión de trocear
demostró su valor: dos sesiones se cortaron a mitad y solo hubo que repetir la
tanda en curso.

**Detalle técnico:** los `doc_id` se derivan de la ruta relativa a la carpeta
indexada. Al indexar por tandas, cada una los habría calculado respecto a su
propia carpeta, produciendo identificadores que podrían colisionar al fusionar.
Por eso `build_index` acepta `--root`, la raíz contra la que se calculan los ids,
de modo que las tandas producen exactamente los mismos `doc_id` que una corrida
única. El fusionador además verifica que no haya `chunk_id` repetidos.

---

## 10. El índice construido

| | |
|---|---|
| fragmentos | **79.141** |
| documentos | **1.785** de 1.839 archivos reales (**98,4%**) |
| chunks por documento (media) | 44,3 |
| dimensión de cada vector | 1.024 |

**Por fenómeno:** F3 49,5%, F2 25,8%, F1 24,6% de los documentos.
**Por formato:** json 51,8%, pdf 42,3%, pbf 4,1%, csv 1,5%, xlsx/img/txt 0,4%.
**Por idioma:** inglés 56,9%, español 33,8%, portugués 7,2%; el resto son
documentos en francés, chino, árabe, ruso, alemán, catalán, coreano y japonés,
casi todos de UNOOSA, que publica en los seis idiomas oficiales de la ONU.

**Validación de la salida:** las 50 consultas devuelven exactamente 10
fragmentos, ninguno supera las 250 palabras, ninguna repite texto (el dedup
funciona) y el esquema de §9.3 pasa el validador automático.

**Controles de validez del experimento de agregación:** `topm(m=1)` reprodujo
`max` en las 50 consultas y `topm(m=10)` convergió a `sum`, como exige la
equivalencia matemática descrita en §6.2. Sin esos controles, la comparación no
sería confiable.

---

## 11. Limitaciones conocidas

Conviene declararlas en el informe; son decisiones informadas, no descuidos.

### 11.1 Sesgo del encoder hacia el idioma de la consulta (la principal)

Las 50 consultas están en español. Medimos, para las 16 consultas del fenómeno 1,
la similitud máxima que alcanza cada fuente del corpus:

| fuente | similitud media | chunks | idioma |
|---|---:|---:|---|
| CEEEP | 0,861 | 83 | español |
| ILIA | 0,844 | 2.208 | español |
| DAIO | 0,834 | 1.957 | inglés |
| CSET Georgetown | 0,831 | 3.909 | inglés |
| CENIA | 0,823 | 276 | español |
| AI Index (Stanford) | 0,815 | 20.405 | inglés |
| Atlantic Council | 0,811 | 936 | inglés |

**La tabla está prácticamente ordenada por idioma**: las fuentes en español
promedian 0,843 y las inglesas 0,823, una brecha de ~0,02 en coseno constante en
las 16 consultas sin excepción.

El dato que aísla la causa: **AI Index tiene diez veces más fragmentos que ILIA y
alcanza menos similitud**. Más fragmentos son más oportunidades de puntuar alto;
si con diez veces más intentos queda por debajo, no es cuestión de tamaño ni de
cobertura temática.

**Por qué ocurre.** Los encoders multilingües proyectan todos los idiomas a un
espacio común, pero esa alineación es imperfecta: parte de la capacidad del
vector codifica *en qué idioma está escrito* el texto, no solo qué dice. Un
pasaje en español queda más cerca de una consulta en español que su equivalente
en inglés. Probablemente se suma un efecto de registro: las consultas son prosa
institucional en español, más parecida en estilo a CEEEP o ILIA que a un informe
de datos en inglés técnico.

**Consecuencia:** el fenómeno 1, cuyo corpus es 95% inglés, recibe documentos
mayoritariamente en español. Afecta a 16 de las 50 consultas; F2 y F3 no se ven
comprometidos.

**Cómo se corregiría**, en orden de eficacia: (a) un segundo encoder con mejor
alineación cross-lingual —los entrenados sobre pares de traducción alinean mucho
más fuerte que e5— fusionado por RRF, que es exactamente lo que contemplan §4.4 y
§8.4; (b) codificar la consulta también en inglés, inviable aquí porque traducir
exige un modelo generativo, prohibido por §8.3; (c) normalizar las puntuaciones
por idioma antes de ordenar, barato pero imposible de calibrar sin datos de
validación.

No se aplicó ninguna: las tres exigen o recursos de cómputo ya agotados o una
validación que sin ground truth no existe, y aplicar una corrección no validada
al entregable final habría sido una apuesta.

### 11.2 Otras limitaciones

1. **No hay juicios de relevancia.** El ground truth no es público durante el
   reto, así que ninguna decisión pudo optimizarse contra la métrica real. Se usó
   la coherencia temática como señal indirecta y, ante empates, se prefirió el
   baseline defendible.
2. **6.568 chunks (8,3%) exceden los 512 tokens** y e5 los trunca al codificar.
   El impacto real es pequeño y está acotado: **de 47.293 fragmentos de PDF,
   exactamente uno se truncó**. El problema vive en los formatos tabulares (22,7%
   de los chunks de CSV, 24,6% de los de PBF), donde cada fila es una "oración"
   sin puntuación final que el chunker no puede partir. En total queda sin
   representar el **2,8% de los tokens del corpus**, y de los chunks afectados el
   encoder alcanza a ver el 81% de media. La prosa —lo que responde las
   consultas— está intacta.
3. **La asignación de fenómeno a cada consulta es una inferencia nuestra**,
   deducida del orden de los bloques en el PDF de preguntas y verificada leyendo
   las 50. No es un dato provisto por el reto, y por eso se usó solo como
   diagnóstico y nunca como filtro. Tiene además un falso negativo conocido: el
   CEEEP está archivado bajo el fenómeno 3 pero publica sobre IA y defensa, así
   que la métrica lo penaliza injustamente.
4. **Quedaron fuera del índice 54 archivos** (1,6%): 2 JSON vacíos en origen, 5
   PDFs escaneados de CSIS y CSET cuyo OCR no se ejecutó por costo frente a
   beneficio, y algunos archivos que fallaron al extraerse.

---

## 12. Estado

**Cerrado:** índice del corpus completo construido y fusionado, agregación
decidida con evidencia, `resultados.jsonl` regenerado, y el paquete de entrega
validado — presentes los entregables de §1.4, alineación índice↔metadata de §5.3
verificada, y esquema de §9.3 conforme.

**Pendiente, a cargo del equipo:** `informe_tecnico.pdf` y el grafo de
conocimiento (componente bonus de §7).

---

## Anexo: cómo reproducir cada número de este documento

```
python scripts/informe_indice.py --indice entrega/base_vectorial/encoder_e5-large \
    --resultados entrega/resultados.jsonl      # §10 y limitación 11.2.2
python scripts/comparar_agregacion.py --indice entrega/base_vectorial/encoder_e5-large
                                               # tabla de §6.2
python scripts/empaquetar_entrega.py           # validación de §8
```
