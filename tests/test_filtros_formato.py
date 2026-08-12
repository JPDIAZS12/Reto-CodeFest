"""Pruebas de los filtros de calidad OCR/PBF y de los extractores reales.

Ejecuta:  python tests/test_filtros_formato.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.extract import _ocr_es_util, _prop_es_util, _extract_pbf, _extract_image

_PASSED = 0
_FAILED = 0


def check(cond: bool, msg: str) -> None:
    global _PASSED, _FAILED
    tag = "[OK]  " if cond else "[FALLA]"
    if cond:
        _PASSED += 1
    else:
        _FAILED += 1
    print(f"  {tag} {msg}")


print("=== 1. _ocr_es_util ===")
# Prosa real (multilingüe) -> útil
prosa = ("El desarrollo de talento en inteligencia artificial es un pilar "
         "fundamental para la innovación y el crecimiento economico regional.")
check(_ocr_es_util(prosa) is True, "acepta prosa real larga")
# Muy pocas palabras -> ruido
check(_ocr_es_util("Fig. 3") is False, "rechaza texto con muy pocas palabras")
# Muchas palabras pero simbólico/numérico -> ruido
basura = "· â€” 12 % ▪ 3,4 ||| 8 · 9 † ‡ 5 ¶ 2 § 7 ± 4 ×"
check(_ocr_es_util(basura) is False, "rechaza texto con baja proporción de letras")
# Cadena vacía
check(_ocr_es_util("") is False, "rechaza cadena vacía")


print("\n=== 2. _prop_es_util ===")
# Nombres reales -> útiles
check(_prop_es_util("au_level2", "Ribamar Fiquene") is True, "acepta nombre de municipio")
check(_prop_es_util("au_country", "Brasil") is True, "acepta nombre de país")
# Claves técnicas -> descartar
check(_prop_es_util("fid", 1) is False, "rechaza clave técnica 'fid'")
check(_prop_es_util("b_ADM2_PCODE", "BR2109551") is False, "rechaza código PCODE")
# Valores numéricos (int, float y string numérico)
check(_prop_es_util("au_population", 7601) is False, "rechaza número entero")
check(_prop_es_util("au_area km", 733.16) is False, "rechaza número float")
check(_prop_es_util("au_population", "7601") is False, "rechaza número como string")
check(_prop_es_util("x", "733,16") is False, "rechaza decimal como string")
# Valor vacío / corto
check(_prop_es_util("au_level1", "") is False, "rechaza valor vacío")
check(_prop_es_util("au_x", "ab") is False, "rechaza valor demasiado corto")
# Un flag tipo VERDADEIRO (texto) sí pasa (aceptable)
check(_prop_es_util("au_pcc", "VERDADEIRO") is True, "acepta flag textual (aceptable)")


print("\n=== 3. Extractores sobre archivos REALES (si hay corpus) ===")
CORPUS = Path(r"C:\Users\juanp\Downloads\OneDrive_1_30-7-2026")
if CORPUS.exists():
    pbfs = sorted(CORPUS.rglob("*.pbf"), key=lambda p: p.stat().st_size, reverse=True)
    if pbfs:
        texto_pbf = _extract_pbf(pbfs[0])
        check(len(texto_pbf) > 0, f"pbf real produce texto ({len(texto_pbf)} chars)")
        check("fid:" not in texto_pbf.lower(), "texto pbf no incluye 'fid' (filtrado)")
        check("pcode" not in texto_pbf.lower(), "texto pbf no incluye PCODE (filtrado)")
        print(f"     muestra pbf: {texto_pbf[:150]}...")
    else:
        print("     (no se hallaron pbf)")

    # Imagen real
    imgs = list(CORPUS.rglob("*.jpg")) + list(CORPUS.rglob("*.avif"))
    if imgs:
        con_texto = 0
        for img in imgs[:5]:
            t = _extract_image(img)
            if t:
                con_texto += 1
        print(f"     OCR: {con_texto}/{min(5,len(imgs))} imágenes pasaron el filtro de calidad")
        check(True, "OCR corrió sobre imágenes reales sin romperse")
    else:
        print("     (no se hallaron imágenes)")
else:
    print("     (corpus real no disponible; se omiten pruebas de archivos reales)")


print(f"\n{'='*50}")
print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
print('='*50)
sys.exit(1 if _FAILED else 0)
