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
# Tope de fragmentos que un SOLO documento puede aportar al índice. Evita que un
# archivo tabular gigante domine el índice y monopolice el pool de candidatos:
# los CSV bibliográficos de PubMed rinden ~110.000 fragmentos entre 5 archivos.
# Ningún PDF del corpus se acerca a este tope (el mayor ronda los 500), así que
# en la práctica solo recorta tablas.
MAX_CHUNKS_POR_DOCUMENTO = 2000

# --- Recuperación ---
TOP_K_CHUNKS_SEARCH = 200    
TOP_N_FRAGMENTS = 10          
TOP_N_DOCUMENTS = 3          
MAX_WORDS_PER_FRAGMENT = 250  
DOC_AGGREGATIONS_VALIDAS = ("max", "sum", "mean", "topm")  
DOC_AGGREGATION = "max"
TOP_M_CHUNKS_POR_DOC = 3 
if DOC_AGGREGATION not in DOC_AGGREGATIONS_VALIDAS:
    raise ValueError(
        f"DOC_AGGREGATION = {DOC_AGGREGATION!r} no existe. Opciones: {DOC_AGGREGATIONS_VALIDAS}"
    )
DEDUP_JACCARD = 0.8

# --- Re-ranking fino de sub-fragmentos ---
# La búsqueda gruesa puntúa chunks de ~450 tokens, pero la salida son
# sub-fragmentos de ≤250 palabras y el NDCG@10 se juzga sobre ese texto (§10.2.1).
# Con el re-ranking, cada sub-fragmento de los mejores chunks se codifica con el
# MISMO encoder del índice y se ordena por su propia similitud con la consulta.
# Es un post-filtro sobre vectores (permitido por §8.7); no interviene ningún
# modelo generativo (§8.3).
RERANK_FRAGMENTOS = True   # False = orden heredado del chunk padre (comportamiento previo)
RERANK_POOL_CHUNKS = 25    # cuántos chunks del pool reciben re-scoring fino

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
    ".avif": "img", ".webp": "img",
    ".pbf": "pbf",
}

# --- Filtros de calidad para formatos ruidosos (OCR de imágenes y mapas PBF) ---
OCR_IDIOMAS = "spa+eng+por"  # modelos de Tesseract (apt: tesseract-ocr-spa/-por)
# Respaldo por OCR para PDFs escaneados: si una página no trae capa de texto, se
# renderiza y se pasa por Tesseract. 47 de los 62 PDFs de Alertas_Tempranas son
# escaneos y se perdían enteros (367 MB de informes sobre grupos armados, la
# fuente más pertinente del fenómeno 3). Solo actúa en páginas sin texto, así
# que los PDFs normales no pagan ningún costo.
PDF_OCR_FALLBACK = True
PDF_OCR_DPI = 200          # buen OCR sin disparar el tiempo por página (~3 s/pág)
OCR_MIN_PALABRAS = 8
OCR_MIN_RATIO_ALFA = 0.6  
PBF_PROP_MIN_LEN = 3     
PBF_CLAVES_TECNICAS = {
    "fid", "id", "gid", "osm_id", "layer", "zoom", "x", "y",
    "b_adm2_pcode", "b_adm1_pcode", "b_id_concatenated", "au_id_concatenated",
}
