"""Configuración central del proyecto CODEFEST AD ASTRA 2026 - Etapa 1."""
from pathlib import Path

# --- Rutas ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                    # corpus crudo de ADL
ENTREGA_DIR = ROOT / "entrega"              # salida final
BASE_VECTORIAL_DIR = ENTREGA_DIR / "base_vectorial"
GRAFO_DIR = ENTREGA_DIR / "grafo"
QUERIES_FILE = ROOT / "queries.jsonl"       # 50 consultas q001-q050 (input)
RESULTADOS_FILE = ENTREGA_DIR / "resultados.jsonl"

# --- Encoder ---
# Modelo encoder (familia BERT/XLM-R). Prohibidos decoders (GPT/LLaMA/etc.).
ENCODER_NAME = "intfloat/multilingual-e5-large"
ENCODER_SLUG = "e5-large"                    
EMBED_DIM = 1024
MAX_INPUT_TOKENS = 512                       # límite del encoder
# e5 requiere prefijos explícitos:
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
ENCODE_BATCH_SIZE = 32        # tamaño de lote al codificar (ajústalo según RAM/GPU)

# --- Chunking ---
CHUNK_MAX_TOKENS = 450        # margen bajo el límite de 512 del encoder
CHUNK_OVERLAP_SENTENCES = 1   # solapamiento (semántica con superposición)
CHUNK_MIN_TOKENS = 20         # descartar fragmentos demasiado cortos

# --- Recuperación ---
TOP_K_CHUNKS_SEARCH = 50      # candidatos a recuperar de FAISS por consulta
TOP_N_FRAGMENTS = 10          # fragmentos a devolver (NDCG@10)
TOP_N_DOCUMENTS = 3           # documentos a devolver (F1@3)
MAX_WORDS_PER_FRAGMENT = 250  # límite duro de palabras por fragmento de salida
DOC_AGGREGATION = "max"       # "max" | "sum" | "mean" (max pooling por defecto)

# --- Grafo de conocimiento (Sección 7, componente bonus) ---
# NER multilingüe (es/en/pt, los 3 idiomas del corpus) sobre HuggingFace,
# licencia permisiva, basado en XLM-RoBERTa (arquitectura encoder).
GRAPH_NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
GRAPH_MAX_GAP_CHARS = 200      # distancia máx. (caracteres) entre 2 entidades del mismo fragmento para inferir relación (aprox. "misma oración")
GRAPH_MAX_ENTITIES_PER_FRAGMENT = 15  # límite para evitar explosión combinatoria de pares por fragmento
GRAPH_RRF_K0 = 60              # constante de suavizado del Reciprocal Rank Fusion (Sección 8.4, Ec. 7)
GRAPH_MAX_EVIDENCE_PER_EDGE = 10       # máx. doc_id/chunk_id de evidencia guardados por arista/nodo (tamaño del .graphml)
USE_GRAPH = True               # si existe grafo.graphml, se usa como señal adicional en la recuperación (Sección 8.5)

# --- Formatos soportados -> etiqueta del campo 'formato' de metadata ---
FORMAT_MAP = {
    ".pdf": "pdf",
    ".html": "html", ".htm": "html",
    ".md": "md", ".markdown": "md",
    ".txt": "txt",
    ".json": "json",
    ".csv": "csv",
    ".xlsx": "xlsx", ".xls": "xlsx",
    ".png": "img", ".jpg": "img", ".jpeg": "img", ".tif": "img", ".tiff": "img",
    ".pbf": "pbf",
}
