
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import FORMAT_MAP


@dataclass
class Documento:
    doc_id: str
    fuente: str          # nombre del archivo original
    formato: str         
    fenomeno: int        # 1, 2, 3 
    texto: str         
    extra: dict = field(default_factory=dict)  # metadata descriptiva opcional 


def _fenomeno_from_path(path: Path) -> int:
    """Deduce el fenómeno (1/2/3) del nombre de una carpeta ancestro.

    Reconoce tanto nuestro esquema de prueba (fenomeno_1) como el del corpus
    real de ADL (F1_IA_..., F2_Seguridad_..., F3_Dinamicas_...).
    """
    for part in path.parts:
        m = re.match(r"f(?:enomeno)?[_\-]?([123])", part.lower())
        if m:
            return int(m.group(1))
    return 0


def _make_doc_id(path: Path, root: Path) -> str:
    """doc_id único e inmutable derivado de la ruta relativa."""
    rel = path.relative_to(root).as_posix()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", rel).strip("-")
    return f"DOC-{slug}"



def _extract_pdf(path: Path) -> str:
    import fitz  
    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def _extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    # separador por bloque para conservar señales estructurales como saltos
    return soup.get_text(separator="\n")


def _extract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_json(path: Path) -> tuple[str, dict]:
    """Interpreta el objeto y arma el texto sin duplicar el cuerpo.

    Los JSON del corpus traen a la vez `body_text` (texto completo) y
    `body_paragraphs` (el MISMO texto como lista de párrafos). Concatenar
    ambos duplicaría todo el cuerpo, así que se elige UNA sola fuente de
    cuerpo (se prefiere la lista de párrafos, luego body_text/otros).
    """
    raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    records = raw if isinstance(raw, list) else [raw]

    titulo_keys = ("title", "headline")
    cuerpo_keys = ("body_paragraphs", "body_text", "body", "text", "content", "article")
    respaldo_keys = ("excerpt", "abstract", "summary")  # solo si no hay cuerpo
    meta_keys = ("url", "date", "published", "authors", "author", "tags", "source")

    body_parts, extra = [], {}
    for rec in records:
        if not isinstance(rec, dict):
            body_parts.append(str(rec))
            continue

        # Título
        for k in titulo_keys:
            if rec.get(k):
                body_parts.append(str(rec[k]))
                break

        # Cuerpo: UNA sola fuente, la primera disponible
        cuerpo = None
        for k in cuerpo_keys:
            if rec.get(k):
                v = rec[k]
                if isinstance(v, list):
                    cuerpo = "\n".join(str(x) for x in v)
                else:
                    cuerpo = str(v)
                break
        if cuerpo:
            body_parts.append(cuerpo)
        else:
            # Sin cuerpo: usar un resumen/excerpt como respaldo
            for k in respaldo_keys:
                if rec.get(k):
                    body_parts.append(str(rec[k]))
                    break

        # Metadata descriptiva (no entra al cuerpo)
        for k in meta_keys:
            if rec.get(k) and k not in extra:
                extra[k] = rec[k]

    return "\n\n".join(body_parts), extra


def _extract_csv(path: Path) -> str:
    """Cada fila -> 'col: valor | col: valor'. Celdas vacías se omiten."""
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    lines = []
    for _, row in df.iterrows():
        pairs = [f"{c}: {v}" for c, v in row.items() if str(v).strip()]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def _extract_xlsx(path: Path) -> str:
    import pandas as pd
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    out = []
    for name, df in sheets.items():
        df = df.fillna("")
        for _, row in df.iterrows():
            pairs = [f"{c}: {v}" for c, v in row.items() if str(v).strip()]
            if pairs:
                out.append(" | ".join(pairs))
    return "\n".join(out)


def _extract_image(path: Path) -> str:
    """OCR sobre imágenes con texto relevante"""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(path), lang="spa+eng+por")
    except Exception:
        return ""


def _extract_pbf(path: Path) -> str:
    """Mapas en formato PBF (vector tiles).

    Se recorren capas y elementos leyendo atributos como pares 'atributo: valor'.
    Se deduplica para no repetir el mismo elemento en varios niveles de zoom.
    Requiere 'mapbox_vector_tile'. Si no está disponible, se omite el documento.
    """
    try:
        import mapbox_vector_tile as mvt
    except ImportError:
        return ""
    try:
        tile = mvt.decode(path.read_bytes())
    except Exception:
        return ""
    seen, lines = set(), []
    for layer in tile.values():
        for feat in layer.get("features", []):
            props = feat.get("properties", {})
            key = tuple(sorted(props.items()))
            if not props or key in seen:
                continue
            seen.add(key)
            lines.append(" | ".join(f"{k}: {v}" for k, v in props.items()))
    return "\n".join(lines)



def extract_document(path: Path, root: Path) -> Optional[Documento]:
    """Extrae un documento; devuelve None si el formato no es soportado"""
    fmt = FORMAT_MAP.get(path.suffix.lower())
    if fmt is None:
        return None

    extra: dict = {}
    try:
        if fmt == "pdf":
            texto = _extract_pdf(path)
        elif fmt == "html":
            texto = _extract_html(path)
        elif fmt in ("md", "txt"):
            texto = _extract_text(path)
        elif fmt == "json":
            texto, extra = _extract_json(path)
        elif fmt == "csv":
            texto = _extract_csv(path)
        elif fmt == "xlsx":
            texto = _extract_xlsx(path)
        elif fmt == "img":
            texto = _extract_image(path)
        elif fmt == "pbf":
            texto = _extract_pbf(path)
        else:
            return None
    except Exception as e:  # un archivo corrupto no debe tumbar el pipeline
        print(f"  [WARN] error extrayendo {path.name}: {e}")
        return None

    if not texto or not texto.strip():
        return None

    return Documento(
        doc_id=_make_doc_id(path, root),
        fuente=path.name,
        formato=fmt,
        fenomeno=_fenomeno_from_path(path),
        texto=texto,
        extra=extra,
    )


def iter_documents(root: Path):
    """Recorre el corpus y produce un Documento por archivo soportado."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            doc = extract_document(path, root)
            if doc is not None:
                yield doc
