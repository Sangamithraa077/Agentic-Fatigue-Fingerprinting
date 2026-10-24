"""Module 8 (Upgraded): Dense Sentence-Transformer Knowledge Retriever with TF-IDF fallback.

Replaces the original TF-IDF sparse retriever with a dense semantic embedding model
(all-MiniLM-L6-v2) for better retrieval on paraphrased and conceptual queries.

Gracefully falls back to TF-IDF if sentence-transformers is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ml.rag.feature_glossary import expand_query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# Try to load sentence-transformers (dense retrieval)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class KnowledgeChunk:
    """A self-contained, source-tagged knowledge document."""

    chunk_id: str
    title: str
    category: str
    source: str
    date: str
    body: str
    tags: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    licence: str = ""          # licence or provenance of the text (Module 8.1c); empty when not recorded

    def format_citation(self) -> str:
        """Formatted source citation for evidence grounding."""
        return f"[{self.chunk_id}: '{self.title}' (Source: {self.source})]"


class LocalRAGKnowledgeBase:
    """Local, offline semantic knowledge base.

    Uses dense sentence-transformer embeddings (all-MiniLM-L6-v2) when available
    for semantic search (e.g. 'why is muscle signal slowing down' → EMG spectral shift).
    Falls back to TF-IDF cosine similarity when sentence-transformers is not installed.
    """

    DENSE_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, knowledge_dir: Optional[Path] = None, use_dense: bool = True):
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).resolve().parents[2] / "knowledge"
        self.knowledge_dir = Path(knowledge_dir)
        self.use_dense = use_dense and _SENTENCE_TRANSFORMERS_AVAILABLE

        self.chunks: List[KnowledgeChunk] = []
        self._dense_model: Optional[object] = None
        self._dense_embeddings: Optional[np.ndarray] = None
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_vectors = None

        self._load_chunks()
        self._build_index()

    def _parse_frontmatter(self, text: str) -> Tuple[Dict[str, str], str]:
        """Parses YAML-like header between '---' separators."""
        metadata: Dict[str, str] = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                header = parts[1].strip()
                body = parts[2].strip()
                for line in header.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip()] = v.strip().strip("[]\"'")
        return metadata, body

    def _load_chunks(self) -> None:
        """Discovers and parses all markdown knowledge chunks."""
        self.chunks.clear()
        if not self.knowledge_dir.exists():
            return
        for md_file in sorted(self.knowledge_dir.rglob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(content)
            chunk_id = meta.get("chunk_id", md_file.stem)
            title = meta.get("title", md_file.stem.replace("_", " ").title())
            category = meta.get("category", md_file.parent.name)
            source = meta.get("source", "Internal Research")
            date = meta.get("date", "2026-01-01")
            tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
            self.chunks.append(KnowledgeChunk(
                chunk_id=chunk_id, title=title, category=category,
                source=source, date=date, body=body, tags=tags,
                file_path=str(md_file), licence=meta.get("licence", ""),
            ))

    def _build_index(self) -> None:
        """Builds dense or sparse index over all chunks."""
        if not self.chunks:
            return

        corpus = [f"{c.title} {' '.join(c.tags)} {c.category} {c.body}" for c in self.chunks]

        if self.use_dense:
            try:
                self._dense_model = SentenceTransformer(self.DENSE_MODEL_NAME)
                self._dense_embeddings = self._dense_model.encode(corpus, normalize_embeddings=True)
            except Exception:
                self.use_dense = False

        if not self.use_dense:
            self._tfidf_vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
            )
            self._tfidf_vectors = self._tfidf_vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Tuple[KnowledgeChunk, float]]:
        """Returns top_k most relevant knowledge chunks with similarity scores."""
        if not self.chunks:
            return []
        # Everyday phrases ("muscle signal slows", "shaky arm") are mapped to the notes' vocabulary
        # here so every caller, the coach and the evaluator alike, retrieves the same way.
        query = expand_query(query)

        if self.use_dense and self._dense_model is not None and self._dense_embeddings is not None:
            query_emb = self._dense_model.encode([query], normalize_embeddings=True)
            scores = (self._dense_embeddings @ query_emb.T).flatten()
        elif self._tfidf_vectorizer is not None and self._tfidf_vectors is not None:
            query_vec = self._tfidf_vectorizer.transform([query])
            scores = sk_cosine(query_vec, self._tfidf_vectors)[0]
        else:
            return []

        ranked = np.argsort(scores)[::-1]
        return [
            (self.chunks[i], float(scores[i]))
            for i in ranked[:top_k]
            if float(scores[i]) >= min_score
        ]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    @property
    def retrieval_backend(self) -> str:
        return "dense:all-MiniLM-L6-v2" if self.use_dense else "sparse:tfidf"
