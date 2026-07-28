from __future__ import annotations

from sentence_transformers import SentenceTransformer

import numpy as np

from config import (
    ENCODER_NAME,
    E5_QUERY_PREFIX,
    E5_PASSAGE_PREFIX,
    EMBED_DIM,
)


class Encoder:
    """Envoltorio del encoder que produce embeddings normalizados float32."""

    def __init__(self, model_name: str = ENCODER_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = EMBED_DIM
        

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        """Antepone `prefix` a cada texto, codifica y devuelve vectores
        normalizados como np.ndarray float32 de forma (n, dim).
        """
        lista_con_prefijo = []
        for t in texts:
            texto = prefix + t
            lista_con_prefijo.append(texto)
            
        embeddings = self.model.encode(
            lista_con_prefijo,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True
            ).astype("float32")
        
        return embeddings

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Codifica fragmentos del corpus (para construir el índice)."""
        return self.encode(texts, E5_PASSAGE_PREFIX, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Codifica consultas del usuario (para buscar en el índice)."""
        return self.encode(texts, E5_QUERY_PREFIX, batch_size)
