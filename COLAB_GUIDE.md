# Guía: construir el índice del CORPUS COMPLETO en Google Colab (GPU)

Estimación: ~130.000 chunks, ~3,5 horas de T4 en total. Por eso se corre en
**tres tandas (F1, F2, F3)** y se fusionan después: si una se cae, se repite
solo esa. Las salidas se escriben **directamente en Drive**, no en `/content/`,
para que una desconexión no se lleve el trabajo hecho.

---

## Paso 0 — Preparar y subir a Drive

En tu PC, comprime **por separado**:

- `codigo.zip` → `src/`, `config.py`, `requirements.txt`
- `F1.zip`, `F2.zip`, `F3.zip` → una por carpeta de fenómeno del corpus

> **Ojo con el zip.** `Compress-Archive` de Windows escribe las rutas con `\` y
> en Linux aparecen archivos planos llamados `src\encode.py` en vez de la
> carpeta `src/`. Usa **7-Zip** (clic derecho → 7-Zip → Añadir al archivo).

Sube los cuatro a una carpeta de Drive, por ejemplo `Mi unidad/codefest/`.
Son ~3 GB: deja que terminen de subir antes de seguir.

---

## Paso 1 — Notebook con GPU

**Archivo → Nuevo notebook** → **Entorno de ejecución → Cambiar tipo de entorno**
→ Acelerador: **GPU (T4)** → Guardar.

---

### Celda 1 — Verificar la GPU

```python
import torch
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NINGUNA")
```

Debe decir `True` y `Tesla T4`. Si dice `False`, no quedó en GPU: repite el Paso 1.

---

### Celda 2 — Montar Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Si da `ValueError: Mountpoint must not already contain files`**, es que algo
creó carpetas en `/content/drive` antes de montar (típicamente, correr la Celda
3 primero). Prueba `drive.mount('/content/drive', force_remount=True)` y si no,
en este orden exacto:

```python
!fusermount -u /content/drive 2>/dev/null   # 1. desmontar por si quedó colgado
!rm -rf /content/drive                      # 2. borrar el directorio local
from google.colab import drive
drive.mount('/content/drive')               # 3. montar limpio
```

⚠️ Nunca hagas `rm -rf /content/drive` con el Drive montado: borrarías archivos
reales de tu Google Drive. Por eso el `fusermount -u` va primero.

---

### Celda 3 — Descomprimir código y corpus

```python
import zipfile, os, time

BASE    = '/content/drive/MyDrive/codefest'   # ajusta si usaste otro nombre
PROY    = '/content/proyecto'
CORPUS  = '/content/corpus'
SALIDA  = f'{BASE}/salida'                    # ¡en Drive, sobrevive a desconexiones!

# Guarda: si Drive no está montado, os.makedirs crearía este árbol en el disco
# local de la VM, el trabajo NO quedaría respaldado y además dejaría ocupado el
# mountpoint (que es lo que provoca el error de la Celda 2).
assert os.path.isdir('/content/drive/MyDrive'), \
    "Drive no está montado: vuelve a la Celda 2 antes de seguir."

os.makedirs(PROY, exist_ok=True)
os.makedirs(CORPUS, exist_ok=True)
os.makedirs(SALIDA, exist_ok=True)

t0 = time.time()
with zipfile.ZipFile(f'{BASE}/codigo.zip') as z:
    z.extractall(PROY)
for nombre in ('F1', 'F2', 'F3'):
    with zipfile.ZipFile(f'{BASE}/{nombre}.zip') as z:
        z.extractall(CORPUS)
print(f"descomprimido en {time.time()-t0:.0f}s")

print("proyecto:", sorted(os.listdir(PROY)))    # src, config.py, requirements.txt
print("corpus  :", sorted(os.listdir(CORPUS)))  # F1_..., F2_..., F3_...
```

`corpus` **debe** listar las tres carpetas con sus nombres originales
(`F1_IA_y_Capacidades_Estrategicas`, etc.): `_fenomeno_from_path` deduce el
fenómeno de ese nombre. Si ves `src\encode.py` como archivo plano, el zip se
hizo con `Compress-Archive`: rehazlo con 7-Zip.

---

### Celda 4 — Dependencias

```python
!apt-get -qq install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-por
!pip install -q sentence-transformers faiss-cpu langdetect pymupdf pandas openpyxl \
                pytesseract pillow mapbox-vector-tile
```

NO instalar `torch`: Colab ya trae uno con CUDA y pisarlo rompe la GPU.
`faiss-cpu` está bien — lo pesado (codificar) va en GPU vía torch; FAISS solo
guarda vectores. Sin `tesseract-ocr-spa` y `-por`, el OCR de imágenes falla en
silencio y pierdes esos documentos.

---

### Celda 5 — Prueba de humo (1 minuto, ANTES de las 2 horas)

```python
%cd /content/proyecto
import glob, os

def tam(d):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(d) for f in fs)

# La MÁS PEQUEÑA de verdad (con sorted(...)[0] tomabas la primera alfabética,
# que resulta ser AI_Index_Stanford: 412 MB y media hora de prueba de humo).
candidatas = [d for d in glob.glob(f'{CORPUS}/F1_*/*') if os.path.isdir(d)]
sub = min((d for d in candidatas if tam(d) > 1e6), key=tam)   # >1 MB, para que no sea trivial
print(f"probando con: {sub}  ({tam(sub)/1e6:.1f} MB)")

!python -m src.build_index --data "{sub}" --root {CORPUS} \
    --out /content/prueba --batch 64
```

Debe terminar con `Listo. NNN vectores indexados.` Si falla aquí —una
dependencia, un formato, una ruta— lo descubres ahora y no a los 40 minutos.

---

### Celda 6 — Listar las tandas (una por subcarpeta)

El corpus son ~130.000 chunks, ~3,5 h de T4. Tres tandas grandes son frágiles:
si el runtime se cae a la hora y media, pierdes esa tanda entera. Una tanda por
subcarpeta (20 en total) hace el proceso **resumible**.

```python
%cd /content/proyecto
import os, glob, time

def tiene_archivos(d):
    return sum(len(fs) for _, _, fs in os.walk(d)) > 0

partes = []
for fen in sorted(glob.glob(f'{CORPUS}/F*_*')):
    if not os.path.isdir(fen):
        continue
    for d in sorted(glob.glob(f'{fen}/*')):
        if os.path.isdir(d) and tiene_archivos(d):
            partes.append((os.path.basename(fen)[:2], d))

print(f"{len(partes)} tandas:")
for fen, d in partes:
    print(f"  {fen}  {os.path.basename(d)}")
```

### Celda 7 — Correr las tandas (resumible: sáltala y repítela sin miedo)

```python
for fen, d in partes:
    nombre = f"{fen}__{os.path.basename(d)}"
    out = f"{SALIDA}/{nombre}"
    if os.path.exists(f"{out}/index.faiss"):
        print(f"[ya está] {nombre}")
        continue
    print(f"\n===== {nombre} =====", flush=True)
    t0 = time.time()
    !python -m src.build_index --data "{d}" --root {CORPUS} --out "{out}" \
        --batch 64 2>&1 | tee "{SALIDA}/log_{nombre}.txt"
    print(f">>> {nombre}: {(time.time()-t0)/60:.1f} min", flush=True)
```

Si el runtime se desconecta, vuelve a correr las Celdas 2, 3, 4 y esta: las
tandas ya terminadas se saltan solas porque su `index.faiss` está en Drive.

Las más largas serán `F1__AI_Index_Stanford` (412 MB, incluye los CSV de PubMed
que solos dan ~52.000 chunks), `F3__RESDAL` (493 MB) y `F3__Alertas_Tempranas`
(432 MB). Empieza por ahí si quieres saber pronto si el tiempo alcanza.

Dos cosas importantes:

- **`--root {CORPUS}` en las tres.** Los `doc_id` se calculan relativos a esa
  carpeta; sin este argumento cada tanda los generaría relativos a su propia
  carpeta y podrían colisionar al fusionar.
- **El `tee` no es adorno.** Los `[WARN] error extrayendo X` son la única señal
  de archivos que se cayeron. `scripts/informe_indice.py --logs` los cuenta.

Si una tanda se corta, vuelve a correr **solo esa celda**: las otras ya están
guardadas en Drive.

---

### Celda 8 — Verificar todas las salidas antes de bajarlas

```python
import faiss, os
total = 0
for nombre in sorted(os.listdir(SALIDA)):
    d = f'{SALIDA}/{nombre}'
    if not os.path.isdir(d) or not os.path.exists(f'{d}/index.faiss'):
        continue
    idx = faiss.read_index(f'{d}/index.faiss')
    n = sum(1 for l in open(f'{d}/metadata.jsonl', encoding='utf-8') if l.strip())
    total += idx.ntotal
    marca = "OK " if idx.ntotal == n else "XX "
    print(f"{marca} {nombre:<34} {idx.ntotal:>7} vectores | alineado: {idx.ntotal==n}")
print(f"\nTOTAL: {total:,} chunks")
```

Todas deben decir `alineado: True`. Si alguna no, borra su carpeta y repite la
Celda 7 (la volverá a hacer, las demás se saltan).

---

## Paso 2 — Bajar y fusionar

Las salidas ya están en tu Drive (`codefest/salida/`). Bájalas desde la web de
Drive (más fiable que `files.download` para archivos grandes) y en tu PC:

```
python scripts/fusionar_indices.py --partes-en salida \
    --out entrega/base_vectorial/encoder_e5-large
```

`--partes-en` descubre solo las 20 subcarpetas con `index.faiss`, en orden
alfabético (determinista). No hace falta listarlas a mano.

El fusionador verifica la alineación de cada parte, que no haya `chunk_id`
repetidos entre tandas, y la alineación del resultado.

---

## Paso 3 — Los cinco chequeos

```
1. python scripts/informe_indice.py --indice entrega/base_vectorial/encoder_e5-large \
       --logs log_F1.txt log_F2.txt log_F3.txt
2. python entrega/generador.py
3. python scripts/informe_indice.py --indice <ruta> --resultados entrega/resultados.jsonl
4. python scripts/comparar_agregacion.py --indice <ruta>
5. python scripts/empaquetar_entrega.py
```

Si el paso 4 hace cambiar `DOC_AGGREGATION`, hay que repetir el 2, el 3 y el 5.
