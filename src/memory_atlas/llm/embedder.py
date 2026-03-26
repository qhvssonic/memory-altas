"""Embedding provider: local (sentence-transformers) or cloud (OpenAI/Cohere)."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...


class LocalEmbedder:
    """Local embedding using sentence-transformers (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embedding. "
                "Install with: pip install memory-atlas[local-embedding]"
            )
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    @property
    def dim(self) -> int:
        return self._dim


class LiteLLMEmbedder:
    """Cloud embedding via LiteLLM (OpenAI, Cohere, etc.)."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        self._model = model
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm
        resp = litellm.embedding(model=self._model, input=texts)
        return [item["embedding"] for item in resp.data]

    @property
    def dim(self) -> int:
        return self._dim


def create_embedder(
    mode: str = "local", model: str | None = None, dim: int | None = None
) -> Embedder:
    """Factory for creating an embedder instance."""
    if mode == "local":
        return LocalEmbedder(model_name=model or "all-MiniLM-L6-v2")
    elif mode == "openai":
        return LiteLLMEmbedder(
            model=model or "text-embedding-3-small", dim=dim or 1536
        )
    elif mode == "cohere":
        return LiteLLMEmbedder(
            model=model or "cohere/embed-english-v3.0", dim=dim or 1024
        )
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")
