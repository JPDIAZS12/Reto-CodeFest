"""Configuración central del proyecto CODEFEST AD ASTRA 2026 - Etapa 1."""
from pathlib import Path

# --- Rutas ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                 
ENTREGA_DIR = ROOT / "entrega"             
BASE_VECTORIAL_DIR = ENTREGA_DIR / "base_vectorial"
GRAFO_DIR = ENTREGA_DIR / "grafo"
QUERIES_FILE = ROOT / "queries.jsonl"      
RESULTADOS_FILE = ENTREGA_DIR / "resultados.jsonl"

# --- Encoder ---
ENCODER_NAME = "intfloat/multilingual-e5-large"
ENCODER_SLUG = "e5-large"                    
EMBED_DIM = 1024
MAX_INPUT_TOKENS = 512                     
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
ENCODE_BATCH_SIZE = 32     

# --- Chunking ---
CHUNK_MAX_TOKENS = 450        
CHUNK_OVERLAP_SENTENCES = 1   
CHUNK_MIN_TOKENS = 20
# Tope por documento: evita que un CSV gigante copie el índice y el pool.
MAX_CHUNKS_POR_DOCUMENTO = 2000

# --- Recuperación ---
TOP_K_CHUNKS_SEARCH = 200    
TOP_N_FRAGMENTS = 10          
TOP_N_DOCUMENTS = 3          
MAX_WORDS_PER_FRAGMENT = 250  
DOC_AGGREGATIONS_VALIDAS = ("max", "sum", "mean", "topm")  
DOC_AGGREGATION = "topm"
TOP_M_CHUNKS_POR_DOC = 3 
if DOC_AGGREGATION not in DOC_AGGREGATIONS_VALIDAS:
    raise ValueError(
        f"DOC_AGGREGATION = {DOC_AGGREGATION!r} no existe. Opciones: {DOC_AGGREGATIONS_VALIDAS}"
    )
DEDUP_JACCARD = 0.8

FORMAT_MAP = {
    ".pdf": "pdf",
    ".html": "html", ".htm": "html",
    ".md": "md", ".markdown": "md",
    ".txt": "txt",
    ".json": "json",
    ".csv": "csv",
    ".xlsx": "xlsx", ".xls": "xlsx",
    ".png": "img", ".jpg": "img", ".jpeg": "img", ".tif": "img", ".tiff": "img",
    ".avif": "img", ".webp": "img",
    ".pbf": "pbf",
}

# --- Filtros de calidad (OCR e imágenes, mapas PBF) ---
OCR_IDIOMAS = "spa+eng+por"
# OCR de respaldo para páginas de PDF sin capa de texto (escaneos).
PDF_OCR_FALLBACK = True
PDF_OCR_DPI = 200
OCR_MIN_PALABRAS = 8
OCR_MIN_RATIO_ALFA = 0.6  
PBF_PROP_MIN_LEN = 3     
PBF_CLAVES_TECNICAS = {
    "fid", "id", "gid", "osm_id", "layer", "zoom", "x", "y",
    "b_adm2_pcode", "b_adm1_pcode", "b_id_concatenated", "au_id_concatenated",
}
