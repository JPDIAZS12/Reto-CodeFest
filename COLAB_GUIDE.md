# Guía: construir el índice en Google Colab (GPU)

Objetivo: correr `build_index` sobre el **subset** en Colab con GPU (minutos, no
horas). Mismo procedimiento servirá luego para el corpus completo.

Archivos ya preparados en tu PC (carpeta `_scratch/colab_pkg/`):
- `codigo.zip`      → el código (src/ + config.py + requirements.txt)
- `data_subset.zip` → los 78 documentos de prueba (3 fenómenos)

---

## Paso 0 — Subir los zips a tu Google Drive

1. Entra a [drive.google.com](https://drive.google.com).
2. Crea una carpeta, por ejemplo **`codefest`** (en "Mi unidad").
3. Sube ahí **`codigo.zip`** y **`data_subset.zip`**.
   (El de datos son 356 MB; deja que suba del todo antes de seguir.)

---

## Paso 1 — Nuevo notebook con GPU

1. Ve a [colab.research.google.com](https://colab.research.google.com) → **Archivo → Nuevo notebook**.
2. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución**
   → en "Acelerador por hardware" elige **GPU (T4)** → Guardar.

Luego pega y ejecuta cada celda (Shift+Enter). El código de cada celda va abajo.

---

### Celda 1 — Verificar la GPU
```python
import torch
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NINGUNA")
```
Debe decir `CUDA disponible: True` y un nombre tipo `Tesla T4`. Si dice False,
revisa el Paso 1 (no quedó en GPU).

---

### Celda 2 — Montar tu Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
Te pedirá autorizar con tu cuenta Google. Acepta.

---

### Celda 3 — Descomprimir código y datos
```python
import zipfile, os

BASE = '/content/drive/MyDrive/codefest'   # ajusta si usaste otro nombre de carpeta
os.makedirs('/content/proyecto', exist_ok=True)

with zipfile.ZipFile(f'{BASE}/codigo.zip') as z:
    z.extractall('/content/proyecto')
with zipfile.ZipFile(f'{BASE}/data_subset.zip') as z:
    z.extractall('/content/proyecto/data_subset')

print(os.listdir('/content/proyecto'))          # debe verse: src, config.py, requirements.txt, data_subset
print(os.listdir('/content/proyecto/data_subset'))  # F1_IA, F2_Seguridad, F3_Dinamicas
```

---

### Celda 4 — Instalar dependencias (sin tocar el torch con CUDA de Colab)
```python
!pip install -q sentence-transformers faiss-cpu langdetect pymupdf pandas openpyxl
```
Nota: NO instalamos `torch` (Colab ya trae uno con CUDA). `faiss-cpu` está bien:
lo pesado (codificar) va en GPU vía torch; FAISS solo guarda vectores.

---

### Celda 5 — Construir el índice con GPU
```python
%cd /content/proyecto
!python -m src.build_index --data data_subset --out /content/salida/encoder_e5-large --batch 64
```
Con GPU puedes subir `--batch` a 64 (más rápido). Debería tardar **pocos minutos**.
Al final: `Listo. NNNN vectores indexados.`

---

### Celda 6 — Descargar el índice a tu PC
```python
import shutil
from google.colab import files
shutil.make_archive('/content/indice_subset', 'zip', '/content/salida')
files.download('/content/indice_subset.zip')
```
Esto te baja `indice_subset.zip` con `encoder_e5-large/{index.faiss, metadata.jsonl}`.
Lo descomprimes en tu PC y ya puedes correr `generador.py` / las pruebas en local.

---

## Qué miramos al terminar
- Cuánto tardó la codificación en GPU (comparar con los ~82 min de CPU local).
- Que `index.faiss` + `metadata.jsonl` se generen bien.
- Luego repetimos el mismo flujo con el **corpus completo** (subiendo el corpus
  de 3 GB a Drive una sola vez).
```
