"""Pruebas para src/encode.py.

Ejecuta:  python tests/test_encode.py

"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import EMBED_DIM
from src.encode import Encoder

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


print("Cargando encoder (descarga ~2.2 GB la primera vez)...")
enc = Encoder()
print("Encoder cargado.\n")


pasajes = [
    # fenómeno 2 - espacio
    "Los satélites obsoletos y los restos de cohetes aumentan el riesgo de colisión en la órbita baja terrestre.",
    # fenómeno 1 - IA defensa
    "La adopción de inteligencia artificial en el sector defensa plantea retos éticos y brechas de talento.",
    # fenómeno 3 - territorio
    "Las dinámicas migratorias en América Latina afectan la gobernanza y la seguridad de los Estados.",
]

print("=== 1. Forma y dtype ===")
emb = enc.encode_passages(pasajes)
check(emb.shape == (3, EMBED_DIM), f"shape esperado (3, {EMBED_DIM}) -> {emb.shape}")
check(emb.dtype == np.float32, f"dtype float32 -> {emb.dtype}")

print("\n=== 2. Normalización (norma L2 ~= 1) ===")
normas = np.linalg.norm(emb, axis=1)
check(np.allclose(normas, 1.0, atol=1e-3),
      f"todas las normas ~= 1.0 -> min={normas.min():.4f} max={normas.max():.4f}")

print("\n=== 3. Sentido semántico ===")
# Consulta en español sobre basura espacial (fenómeno 2)
q = enc.encode_queries(["¿Qué riesgos genera la basura espacial en la órbita baja?"])[0]
sims = emb @ q  # producto interno = coseno (vectores normalizados)
print(f"  cos(query, pasaje_espacio)  = {sims[0]:.4f}")
print(f"  cos(query, pasaje_ia)       = {sims[1]:.4f}")
print(f"  cos(query, pasaje_territorio)= {sims[2]:.4f}")
check(sims.argmax() == 0, "la consulta se parece MÁS al pasaje de espacio (fenómeno correcto)")
check(sims[0] > sims[1] and sims[0] > sims[2], "coseno del tema correcto es el mayor")

# Cruce de idiomas: consulta en inglés debe recuperar el pasaje en español
print("\n=== 3b. Cross-lingual (consulta EN -> pasaje ES) ===")
q_en = enc.encode_queries(["What are the risks of space debris in low Earth orbit?"])[0]
sims_en = emb @ q_en
print(f"  cos(query_EN, pasaje_espacio)   = {sims_en[0]:.4f}")
print(f"  cos(query_EN, pasaje_ia)        = {sims_en[1]:.4f}")
check(sims_en.argmax() == 0, "consulta en inglés recupera el pasaje en español (multilingüe)")


print("\n=== 4. Portugués ===")
pasajes_pt = [
    "Os satélites obsoletos e os detritos de foguetes aumentam o risco de colisão na órbita baixa terrestre.",
    "A adoção de inteligência artificial no setor de defesa levanta desafios éticos e lacunas de talento.",
    "As dinâmicas migratórias na América Latina afetam a governança e a segurança dos Estados.",
]
emb_pt = enc.encode_passages(pasajes_pt)
check(emb_pt.shape == (3, EMBED_DIM), f"pasajes PT shape (3, {EMBED_DIM}) -> {emb_pt.shape}")

# Consulta en PT sobre espacio -> debe recuperar el pasaje PT de espacio
q_pt = enc.encode_queries(["Quais são os riscos dos detritos espaciais na órbita baixa?"])[0]
sims_pt = emb_pt @ q_pt
print(f"  cos(query_PT, pasaje_espacio_PT)    = {sims_pt[0]:.4f}")
print(f"  cos(query_PT, pasaje_ia_PT)         = {sims_pt[1]:.4f}")
check(sims_pt.argmax() == 0, "consulta PT recupera el pasaje PT del tema correcto")


print("\n=== 5. Alineación trilingüe (mismo concepto ES/EN/PT) ===")
# Codificamos como pasajes la misma frase de 'espacio' en 3 idiomas + 1 distractor
frases = [
    "Los desechos espaciales amenazan la sostenibilidad de la órbita baja.",   # 0 ES espacio
    "Space debris threatens the sustainability of low Earth orbit.",           # 1 EN espacio
    "Os detritos espaciais ameaçam a sustentabilidade da órbita baixa.",       # 2 PT espacio
    "La educación rural mejora las oportunidades de empleo juvenil.",          # 3 distractor
]
E = enc.encode_passages(frases)
M = E @ E.T  # matriz de similitud coseno (4x4)
print("  matriz de similitud (0-2 = espacio ES/EN/PT, 3 = distractor):")
for fila in M:
    print("   " + "  ".join(f"{v:.3f}" for v in fila))

# Cada frase de espacio debe parecerse más a sus traducciones que al distractor
espacio_ok = True
for i in range(3):
    sim_traducciones = min(M[i, j] for j in range(3) if j != i)  # peor par entre idiomas
    sim_distractor = M[i, 3]
    if sim_traducciones <= sim_distractor:
        espacio_ok = False
check(espacio_ok, "las 3 traducciones se parecen más entre sí que al distractor")

# Umbral razonable de alineación cross-lingual entre las traducciones
pares_translations = [M[0, 1], M[0, 2], M[1, 2]]
check(min(pares_translations) > 0.80,
      f"alineación ES/EN/PT alta (min par = {min(pares_translations):.3f} > 0.80)")

print(f"\n{'='*50}")
print(f"RESULTADO: {_PASSED} OK, {_FAILED} FALLA(S)")
print('='*50)
sys.exit(1 if _FAILED else 0)
