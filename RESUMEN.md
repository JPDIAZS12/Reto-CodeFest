# CODEFEST Ad Astra 2026 — Etapa 1 · Estado del proyecto

> Resumen de seguimiento para el equipo · actualizado 2026-08-04

**Objetivo:** construir una base de conocimiento vectorial (búsqueda semántica, sin
LLMs) sobre el corpus multilingüe de ADL, que responda a 50 consultas devolviendo
los documentos y fragmentos más relevantes.

---

## ✅ Lo que está terminado (pipeline completo y probado)

Construimos el sistema de punta a punta, cada módulo con sus pruebas automáticas
en verde:

| Módulo | Qué hace | Pruebas |
|---|---|---|
| **Extracción** | lee PDF/JSON/CSV/XLSX/TXT → texto + metadata | ✔ |
| **Limpieza** | normaliza, quita ruido, detecta idioma | ✔ |
| **Chunking** | fragmenta sin cortar oraciones (≤450 tokens) | 32/32 |
| **Encoder** | embeddings multilingües (e5-large), ES/EN/PT | 10/10 |
| **Índice FAISS** | base vectorial + metadata alineada | 7/7 |
| **Recuperación** | búsqueda coseno + top-3 docs + top-10 fragmentos ≤250 palabras | 20/20 |
| **Generador** | produce `resultados.jsonl` (formato exacto del reto) | 8/8 |

**Los 4 entregables obligatorios ya funcionan** en esqueleto probado: base
vectorial, `resultados.jsonl`, `generador.py`, y falta redactar el informe técnico
(con las decisiones ya tomadas y justificadas).

---

## ✅ Datos reales integrados

- Recibimos el corpus (**~3 GB, 1848 archivos**) y las **50 preguntas** (venían en
  PDF → ya extraídas a formato procesable).
- Adaptamos el pipeline al corpus real: mapeo de los 3 fenómenos, corrección de
  JSON duplicados, y **optimización del chunking**.

---

## 🔧 Etapa actual: validación con datos reales

Estamos corriendo el pipeline sobre un subconjunto real antes de procesar todo el
corpus. Surgió **un obstáculo de hardware**: la máquina de desarrollo tiene RAM
limitada y sin GPU, lo que hace lenta la codificación con el modelo de máxima
calidad (e5-large).

**Decisión tomada:** mantener e5-large (mejor calidad de recuperación) y correr la
indexación completa en una sesión larga con la memoria liberada.
Estimado: **2-4 horas** de procesamiento con RAM disponible.

---

## 📌 Próximos pasos

1. Liberar memoria y **construir el índice del corpus completo** (~130 mil fragmentos).
2. Generar el `resultados.jsonl` de las 50 consultas reales y revisar calidad.
3. **Grafo de conocimiento** (componente bonus, suma puntos).
4. **Empaquetar la entrega** + redactar el **informe técnico** (máx. 8 páginas).

---

## Decisiones técnicas clave (defendibles en el informe)

- **Encoder e5-large:** multilingüe nativo ES/EN/PT, líder en benchmarks de
  recuperación (MTEB/BEIR), licencia permisiva.
- **Índice FAISS plano (IndexFlatIP):** búsqueda exacta, óptimo para esta escala.
- **Chunking híbrido:** respeta oraciones completas + tamaño controlado + solapamiento.
