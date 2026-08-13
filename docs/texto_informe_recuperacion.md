# Sistema de recuperación — texto para el informe técnico

> Redactado para pegar directamente en `informe_tecnico`. Cada sección es
> independiente; se pueden reordenar o recortar según el espacio disponible.

---

## Preprocesamiento y extracción de texto

El corpus provisto reúne 1.839 archivos en siete formatos distintos. Se procesaron
todos ellos: PDF mediante PyMuPDF, HTML con BeautifulSoup eliminando marcado y
scripts, JSON con un intérprete propio de campos, CSV y XLSX con pandas
convirtiendo cada fila a texto `columna: valor`, imágenes por OCR y mapas
vectoriales PBF decodificando sus atributos. La decisión de no descartar ningún
formato se tomó frente a la alternativa de ignorar los más ruidosos; se optó por
incorporarlos con filtros de calidad específicos en lugar de perder su contenido.

Tres decisiones de extracción merecen mención porque corrigen pérdidas de datos
que de otro modo habrían pasado inadvertidas:

**OCR de respaldo para PDFs escaneados.** Cuando una página de PDF no contiene
capa de texto, se renderiza a imagen y se procesa con Tesseract. La necesidad no
era teórica: 47 de los 62 PDFs de las Alertas Tempranas de la Defensoría del
Pueblo son escaneos, y sin este respaldo se descartaban íntegros. Se trata de 367
MB de informes sobre grupos armados y control territorial, la fuente más
pertinente para las consultas del tercer fenómeno. Incorporarlos elevó esa
carpeta de 794 a 1.768 fragmentos y recuperó 45 documentos. El respaldo actúa
página por página y únicamente cuando no hay texto extraíble, de modo que los
PDFs con capa de texto no incurren en ningún costo adicional.

**Selección de una única fuente de cuerpo en JSON.** Los artículos en JSON
incluyen simultáneamente los campos `body_text` y `body_paragraphs` con el mismo
contenido. Concatenar ambos habría duplicado el cuerpo de cada documento e
inflado el índice con texto repetido, de modo que se selecciona explícitamente
una sola fuente.

**Tolerancia a CSV malformados.** Varios CSV del corpus emplean separadores
distintos de la coma o contienen filas con más campos que la cabecera, lo que
hacía fallar el parseo y perder el archivo completo. Ante un fallo se reintenta
con detección automática de separador y descarte de las filas irrecuperables. El
resultado es que los 26 CSV del corpus se extraen correctamente.

En la limpieza posterior se normaliza a Unicode NFC, se eliminan caracteres de
control, se reunifican las palabras partidas por guión al final de línea —muy
frecuentes en PDFs maquetados a columnas— y se suprimen encabezados y pies
recurrentes, identificados como líneas cortas que se repiten cuatro o más veces
en un mismo documento. Se unifican además los separadores de línea Unicode
(U+2028, U+2029, U+0085) a salto de línea convencional, porque de lo contrario
rompen la lectura posterior del almacén de metadata en formato JSON Lines.

---

## Estrategia de chunking

**Decisión: fragmentación híbrida, con cortes en frontera de oración, empaque
goloso hasta 450 tokens y solapamiento de una oración entre fragmentos
consecutivos.**

El procedimiento divide el texto en oraciones completas, acumula oraciones
consecutivas mientras quepan en el límite de tokens, cierra el fragmento al
alcanzarlo y comienza el siguiente conservando la última oración del anterior.

Se descartaron las dos estrategias puras por motivos opuestos. La fragmentación
por oración individual produce unidades demasiado breves, con contexto
insuficiente para que el encoder las sitúe semánticamente, y multiplica
innecesariamente el tamaño del índice. La fragmentación por tamaño fijo en
tokens, en cambio, corta oraciones por la mitad: además de degradar la calidad
del embedding, incumple el requisito de completitud lingüística, que exige que
los fragmentos entregados sean unidades de texto completas y legibles. El
enfoque híbrido conserva la propiedad relevante de cada una: unidades semánticas
íntegras y de extensión suficiente.

El límite de 450 tokens deja margen deliberado bajo los 512 que admite el
encoder. El solapamiento de una oración asegura que una idea que cruza la
frontera entre dos fragmentos quede representada en ambos. Se descartan los
fragmentos de menos de 20 tokens por carecer de contenido aprovechable.

**Tope de fragmentos por documento.** Se limita a 2.000 la cantidad de
fragmentos que un solo documento puede aportar al índice. El corpus incluye
volcados bibliográficos en CSV de hasta 35 MB; uno solo de ellos generaba
alrededor de 36.000 fragmentos y el conjunto superaba los 110.000, más de la
mitad del índice completo procedente de cinco archivos. El tope responde a tres
razones: un documento con decenas de miles de fragmentos dispone de otras tantas
oportunidades de producir una coincidencia alta por azar y compite con ventaja
injusta en la agregación a nivel documento; monopoliza el conjunto de candidatos
y desplaza a documentos distintos; y duplicaba con creces el tiempo de
indexación para incorporar listados de metadatos bibliográficos. El límite es
general, no una excepción dirigida a archivos concretos, y está fijado donde
ningún documento de prosa lo alcanza —el PDF más extenso del corpus produce
alrededor de 500 fragmentos—, de modo que en la práctica solo afecta a
estructuras tabulares. Los documentos recortados permanecen en el índice y
siguen siendo recuperables.

**Caso límite documentado.** Una oración que por sí sola excede el límite no
puede dividirse sin cortarla, por lo que se emite como fragmento propio. Si
supera los 512 tokens del encoder, este la trunca al codificar y el vector no
representa su parte final, aunque el texto íntegro se conserva en la metadata y
el fragmento entregado está completo. Afecta a 6.568 fragmentos (8,3% del
índice), pero se concentra casi por entero en formatos tabulares: de los 47.293
fragmentos procedentes de PDF, exactamente uno resultó truncado. En conjunto
queda sin representar el 2,8% de los tokens del corpus.

---

## Selección del encoder

**Decisión: `intfloat/multilingual-e5-large`.**

Se evaluó contra los seis criterios establecidos por la especificación:

| Criterio | Cumplimiento |
|---|---|
| Soporte multilingüe | Nativo en español, inglés y portugués; base XLM-RoBERTa entrenada sobre 100 idiomas |
| Dimensionalidad | 1.024 |
| Longitud máxima de entrada | 512 tokens, magnitud que condiciona directamente la estrategia de chunking |
| Rendimiento en benchmarks | Posiciones destacadas en las tablas de recuperación de MTEB y BEIR |
| Licencia | MIT |
| Eficiencia computacional | 2,2 GB de parámetros; inviable en CPU para indexación, resuelto mediante GPU |

Se trata de un modelo encoder, conforme a la prohibición de emplear
arquitecturas decoder. La familia e5 fue entrenada con prefijos explícitos y los
requiere en inferencia (`query:` para consultas, `passage:` para fragmentos);
omitirlos degrada la calidad de forma silenciosa, por lo que se aplican de manera
sistemática. Los embeddings se producen normalizados, lo que permite emplear
producto interno como similitud coseno.

**Decisión de no incorporar un segundo encoder.** La especificación permite
construir la base con varios encoders y describe estrategias para fusionar sus
rankings. Se evaluó la opción y se descartó por tres razones: duplica el costo de
indexación, que constituía el cuello de botella real del proyecto; duplica el
tamaño de la entrega; y, de manera decisiva, no existía forma de verificar que
mejorara el resultado, dado que sin juicios de relevancia no puede medirse si la
fusión eleva o reduce las métricas. La fusión por rangos pondera por igual a
ambos encoders con independencia de su calidad, de modo que incorporar uno
inferior arrastra el ranking hacia abajo. Se optó por no adoptar a ciegas una
técnica cuyo efecto no podía medirse. La arquitectura del proyecto la soporta si
en el futuro se dispone de datos de validación.

---

## Índice vectorial

**Decisión: `IndexFlatIP` de FAISS sobre vectores normalizados.**

Con los vectores normalizados, el producto interno equivale matemáticamente a la
similitud coseno, de modo que no se requiere un índice específico para esta
métrica. `IndexFlatIP` realiza búsqueda exacta comparando la consulta contra la
totalidad de los vectores, con recuperación completa garantizada.

Se consideraron los índices aproximados que ofrece FAISS. Tanto la cuantificación
vectorial invertida como los grafos de proximidad intercambian exactitud por
velocidad, y esa compensación solo resulta ventajosa a escala de millones de
vectores. Con los 79.141 fragmentos del corpus, la búsqueda exacta responde en
milisegundos: aceptar una pérdida de recuperación no habría aportado ningún
beneficio observable.

**Invariante de alineación.** El identificador interno que FAISS asigna a cada
vector corresponde a su posición de inserción, de modo que la línea *i* del
almacén de metadata debe describir exactamente el fragmento cuyo vector ocupa la
posición *i*. Romper ese orden inutiliza el índice completo. Se garantiza
manteniendo una única lista de fragmentos que sirve simultáneamente para
codificar y para escribir la metadata, y se verifica automáticamente en tres
puntos del flujo: al construir el índice, al fusionar índices parciales y al
empaquetar la entrega.

---

## Módulo de recuperación

### Conjunto de candidatos

**Decisión: recuperar 200 fragmentos candidatos por consulta.**

El valor inicial de 50 se elevó a partir de una observación empírica: existían
consultas cuyos 50 fragmentos candidatos procedían de únicamente dos documentos
distintos, de modo que el sistema no podía devolver tres documentos aunque el
corpus los contuviera, con pérdida garantizada de recuperación a nivel documento.
La causa es la distribución del corpus, donde ciertos documentos generan cientos
de fragmentos y copan el conjunto de candidatos por sí solos. Con 200, esas
mismas consultas acceden a 19 y 28 documentos distintos respectivamente. El costo
es nulo, ya que el índice plano recorre todos los vectores en cualquier caso y el
parámetro solo determina cuántos resultados se devuelven.

### Agregación a nivel documento

El índice conoce únicamente fragmentos, de modo que los tres documentos deben
derivarse agrupando los candidatos por documento de origen y asignando una
puntuación agregada a cada uno. Se implementaron y compararon cuatro estrategias:
máximo, suma, media y suma de los *m* mejores fragmentos.

Conviene señalar una propiedad que simplifica el análisis: la suma de los *m*
mejores no constituye una cuarta estrategia independiente, sino el continuo entre
las otras dos. Con *m* igual a uno equivale exactamente al máximo, y con *m*
mayor o igual al número de fragmentos del documento equivale exactamente a la
suma. El parámetro regula cuánta acumulación de evidencia se permite. La
equivalencia se verificó empíricamente sobre las 50 consultas y se empleó como
control de validez del experimento.

**Decisión: suma de los tres mejores fragmentos por documento.**

Se partió del máximo como estrategia de referencia y se modificó a partir de la
evidencia recogida sobre el corpus completo:

| Estrategia | Documentos del fenómeno esperado | Consultas con al menos un documento ajeno |
|---|---|---|
| Máximo | 108/150 (72%) | 18/50 |
| Media | 96/150 (64%) | 22/50 |
| Suma | 137/150 (91%) | 8/50 |
| **Suma de los 3 mejores** | **138/150 (92%)** | **7/50** |
| Suma de los 5 mejores | 140/150 (93%) | 6/50 |

La mejora es sustancial y estable: las consultas que incorporan algún documento
ajeno al fenómeno se reducen de 18 a 7, y el resultado se mantiene para valores
de *m* entre 2 y 10, de modo que no depende de acertar con el parámetro.

Más relevante que la cifra es el mecanismo del fallo identificado. Bajo la
estrategia de máximo basta un único fragmento afortunado para que un documento
puntúe alto, de modo que los documentos que aportan un solo fragmento compiten en
igualdad de condiciones con informes de centenares. El efecto observado era
extremo: un mismo artículo aparecía en catorce de las dieciséis consultas del
primer fenómeno. Eso no constituye recuperación sino un artefacto de
concentración: un documento genéricamente próximo a cualquier pregunta del área,
incapaz de discriminar entre ellas. Al exigir tres fragmentos buenos, ese
documento deja de dominar y entran en su lugar las fuentes especializadas: las
posiciones ocupadas por publicaciones propias del dominio de inteligencia
artificial aplicada a defensa pasaron de 2 a 11 sobre 48.

La estrategia de media quedó descartada con datos: es la peor de las cuatro y
empeora al ampliar el conjunto de candidatos, dado que cada documento recibe más
fragmentos de similitud baja que reducen su promedio.

Es importante precisar el alcance del cambio: afecta exclusivamente a la métrica
de documentos. Los diez fragmentos se seleccionan por similitud directa sin pasar
por la agregación, de modo que la métrica de fragmentos es idéntica bajo
cualquiera de las cuatro estrategias.

### Selección y re-ordenamiento de fragmentos

Los fragmentos entregados se obtienen recorriendo los candidatos por similitud
descendente y dividiendo cada uno en sub-fragmentos de 250 palabras como máximo
sin cortar oraciones. Dado que la mediana de un fragmento del índice es de 253
palabras, la división se aplica al 51% de ellos. Como consecuencia, varios
fragmentos de una misma respuesta pueden compartir identificador de fragmento
original: son porciones distintas del mismo texto, no contenido duplicado.

**Re-ordenamiento fino.** La búsqueda gruesa puntúa fragmentos de
aproximadamente 450 tokens, mientras que la evaluación de fragmentos se realiza
sobre el texto de los sub-fragmentos entregados. Sin un nuevo cálculo de
similitud, los sub-fragmentos heredan la posición de su fragmento padre y la
mitad menos pertinente de un buen fragmento puede anteponerse a la mitad
excelente del siguiente. Por ello, los sub-fragmentos procedentes de los 25
mejores candidatos se codifican con el mismo encoder del índice y se reordenan
según su propia similitud con la consulta. La operación emplea exclusivamente
vectores y el encoder ya presente, sin intervención de ningún modelo generativo.

La profundidad del conjunto re-ordenado se verificó empíricamente: en once de
trece consultas analizadas, al menos uno de los diez fragmentos finales procede
de un candidato situado más allá de la decimoquinta posición, y en dos casos
procede exactamente de la vigésimo quinta. El re-ordenamiento aprovecha por tanto
la profundidad completa del conjunto examinado.

---

## Post-filtros

**Deduplicación de fragmentos.** El solapamiento de una oración entre fragmentos
consecutivos hace que dos fragmentos vecinos compartan texto. Devolver ambos
consumiría posiciones del resultado sin aportar información nueva y penalizaría
la métrica. Se comparan mediante similitud de Jaccard sobre el conjunto de
palabras con umbral de 0,8, y el fragmento descartado se sustituye por el
siguiente candidato disponible. En la entrega final ninguna de las 50 consultas
devuelve texto repetido.

**Umbral mínimo de similitud: descartado.** Eliminar fragmentos por debajo de una
similitud mínima habría impedido garantizar exactamente diez fragmentos por
consulta, cantidad que la especificación exige y cuya violación se penaliza. Se
prefiere entregar un décimo fragmento mediocre antes que nueve.

**Filtrado por fenómeno: descartado.** Las consultas no vienen etiquetadas con su
fenómeno temático; cualquier asignación constituiría una inferencia propia. Se
empleó como señal de diagnóstico para evaluar estrategias, nunca como filtro en
el sistema entregado.

---

## Reproducibilidad de la entrega

El paquete entregado es autocontenido. El script generador resuelve todas sus
rutas respecto a su propia ubicación en lugar de importarlas de la configuración,
puesto que esta última las deriva de su propio directorio y al copiarla dentro
del paquete apuntaría a una ruta inexistente. Se verificó ejecutando el paquete
fuera del repositorio de desarrollo, en un directorio sin dependencias
accesibles, comprobando que la invocación sin argumentos produce el archivo de
resultados correcto.

Se dispone además de un validador automático que comprueba la presencia de todos
los entregables, la alineación entre índice y metadata, y la conformidad completa
del archivo de resultados con el esquema exigido, incluido el límite de palabras
por fragmento. Devuelve código de error si algo no se cumple, de modo que puede
emplearse como verificación previa a la entrega.

**Nota sobre el tiempo de ejecución.** El re-ordenamiento fino codifica
alrededor de 46 sub-fragmentos por consulta. En un equipo con GPU el proceso
completo tarda pocos minutos; en CPU se aproxima a los cuarenta. El script
informa del tiempo transcurrido y estimado restante tras cada consulta.

---

## Limitaciones conocidas

### Afinidad del encoder por el idioma de la consulta

Se identificó y cuantificó la principal limitación del sistema. Para las
dieciséis consultas del primer fenómeno se midió la similitud máxima que alcanza
cada fuente del corpus:

| Fuente | Similitud media | Fragmentos | Idioma |
|---|---|---|---|
| CEEEP | 0,861 | 83 | Español |
| ILIA | 0,844 | 2.208 | Español |
| DAIO | 0,834 | 1.957 | Inglés |
| CSET Georgetown | 0,831 | 3.909 | Inglés |
| CENIA | 0,823 | 276 | Español |
| AI Index (Stanford) | 0,815 | 20.405 | Inglés |
| Atlantic Council | 0,811 | 936 | Inglés |

La ordenación coincide casi exactamente con el idioma: las fuentes en español
promedian 0,843 frente a 0,823 de las fuentes en inglés, una diferencia constante
en las dieciséis consultas.

El dato que aísla la causa es que el AI Index dispone de diez veces más
fragmentos que ILIA y alcanza una similitud inferior. Un mayor número de
fragmentos supone más oportunidades de obtener una coincidencia alta; si aun así
queda por debajo, la explicación no reside en el tamaño ni en la cobertura
temática.

El comportamiento se confirmó de manera simétrica formulando la misma pregunta en
ambos idiomas: la versión en inglés devuelve tres documentos en inglés y la
versión en español, tres en español. No se trata por tanto de una preferencia por
un idioma concreto, sino de afinidad por el idioma de la consulta.

La explicación es que los encoders multilingües proyectan todos los idiomas a un
espacio común mediante una alineación imperfecta: parte de la capacidad del
vector codifica el idioma del texto y no únicamente su significado. Probablemente
se suma un efecto de registro, dado que las consultas comparten estilo con las
fuentes en español más que con informes técnicos en inglés.

La consecuencia práctica es que el primer fenómeno, cuyo corpus es
mayoritariamente anglófono, recibe documentos predominantemente en español.
Afecta a dieciséis de las cincuenta consultas; los otros dos fenómenos no se ven
comprometidos.

Las vías de corrección, por orden de eficacia, serían: incorporar un segundo
encoder con mejor alineación entre idiomas —los entrenados sobre pares de
traducción la logran considerablemente más estrecha— y fusionar ambos rankings;
codificar la consulta también en inglés, alternativa inviable en este contexto
porque la traducción exigiría un modelo generativo; o normalizar las puntuaciones
por idioma antes de ordenar, opción económica pero imposible de calibrar sin
datos de validación. No se aplicó ninguna: las tres requieren recursos de cómputo
ya agotados o una validación inexistente, y aplicar una corrección no verificada
al sistema final habría constituido una apuesta.

### Otras limitaciones

**Ausencia de juicios de relevancia.** El conjunto de referencia no es público
durante la competencia, de modo que ninguna decisión pudo optimizarse contra la
métrica real. Se empleó la coherencia temática como señal indirecta y, ante
resultados equivalentes, se prefirió la opción más defendible.

**Cobertura del corpus.** Se indexaron 1.785 de los 1.839 archivos del corpus, un
98,4%. Los 54 restantes corresponden a archivos vacíos en origen, cinco PDFs
escaneados cuyo procesamiento por OCR se descartó por relación entre costo y
beneficio, y algunos archivos que fallaron en la extracción.

**Inferencia del fenómeno por consulta.** La asignación de cada consulta a su
fenómeno temático se dedujo del orden de los bloques en el documento de preguntas
y se verificó leyendo las cincuenta. No constituye un dato provisto por la
organización, razón por la cual se empleó únicamente como diagnóstico. Presenta
además un falso negativo conocido: una de las fuentes está archivada bajo el
tercer fenómeno pero publica sobre inteligencia artificial y defensa, de modo que
la métrica la penaliza indebidamente.

---

## Cifras de la base construida

| Magnitud | Valor |
|---|---|
| Fragmentos indexados | 79.141 |
| Documentos | 1.785 |
| Cobertura del corpus | 98,4% |
| Fragmentos por documento (media) | 44,3 |
| Dimensión del vector | 1.024 |
| Distribución por fenómeno | 49,5% / 25,8% / 24,6% |
| Distribución por idioma | Inglés 56,9%, español 33,8%, portugués 7,2% |
| Distribución por formato | JSON 51,8%, PDF 42,3%, PBF 4,1%, CSV 1,5%, resto 0,4% |
