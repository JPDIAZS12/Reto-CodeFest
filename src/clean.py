"""Limpieza y normalización del texto extraído (Sección 2.2)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Caracteres de control a eliminar
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTISPACE_RE = re.compile(r"[ \t ]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
# Numeración de página tipo "12", "- 12 -", "Página 12"
_PAGE_NUM_RE = re.compile(r"^\s*(?:p[áa]gina\s*)?[-–]?\s*\d+\s*[-–]?\s*$",
                          re.IGNORECASE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Elimina líneas repetitivas sin valor y numeración de página.

    Una línea corta que se repite muchas veces a lo largo del documento suele ser
    un encabezado/pie recurrente.
    """
    lines = text.split("\n")
    freq = Counter(ln.strip() for ln in lines if ln.strip())
    repeated = {
        ln for ln, c in freq.items()
        if c >= 4 and len(ln) <= 80
    }

    kept = []
    for ln in lines:
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if _PAGE_NUM_RE.match(s):
            continue
        if s in repeated:
            continue
        kept.append(ln)

    text = "\n".join(kept)
    return _MULTINEWLINE_RE.sub("\n\n", text).strip()


def detect_language(text: str) -> str:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        sample = text[:2000]
        return detect(sample) if sample.strip() else "und"
    except Exception:
        return "und"


def clean_document(text: str) -> str:
    """Pipeline de limpieza completo."""
    return remove_boilerplate(normalize(text))
