"""Similarity algorithms and embedding utilities for note comparison.

Implements Jaccard, Adamic-Adar, and Sentence-BERT vector generation.

Copyright (c) 2026 Caellum Yip Hoi-Lee, Catherine Abdul-Samad, Michael Chen, Joshua Yeung.
All rights reserved.
"""

import math
from graph import KnowledgeGraph


def jaccard(g: KnowledgeGraph, u: str, v: str) -> float:
    """
    Returns the Jaccard Similarity Index of str u and v.

    Preconditions:
        - u in g.get_notes()
        - v in g.get_notes()
    """
    nu, nv = g.get_neighbours(u), g.get_neighbours(v)
    intersection = len(nu & nv)
    union = len(nu | nv)
    return intersection / union if union > 0 else 0.0


def adamic_adar(g: KnowledgeGraph, u: str, v: str) -> float:
    """
    Returns the Adamic Adar score of str u and v.

    Preconditions:
        - u in g.get_notes()
        - v in g.get_notes()
    """
    score = 0.0
    for w in g.get_neighbours(u) & g.get_neighbours(v):
        deg = g.get_note(w).get_degree()
        if deg > 1:
            score += 1.0 / math.log(deg)
    return score


def normalise(scores: list[float]) -> list[float]:
    """
    Returns a list of normalised scores.
    """
    if not scores:
        return scores
    low, high = min(scores), max(scores)
    if low == high:
        return [0.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def dot(a: list[float], b: list[float]) -> float:
    """
    Returns the dot product of vectors a and b.

    Preconditions:
        - len(a) == len(b)
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    """
    Returns the norm of a vector.
    """
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Returns the cosine similarity score of vector a and b

    Preconditions:
        - len(a) == len(b)
        - len(a) > 0
    """
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


class SentenceBERTEmbedder:
    """
    Encodes note text into embedding vectors for similarity computations.

    Uses Sentence-BERT to encode note content into dense semantic vectors.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.MODEL_NAME)
            print("  [embedder] Using Sentence-BERT (all-MiniLM-L6-v2)")
        except (ImportError, OSError):
            raise RuntimeError("Sentence-BERT is required but could not be loaded.")

    def fit(self, note_texts: list[str]) -> None:
        """
        No-op for Sentence-BERT, as it uses a pre-trained model.
        """
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts into embedding vectors.
        """
        arr = self._model.encode(texts, show_progress_bar=False)
        return arr.tolist()

    def pairwise_cosine(self, embeddings: list[list[float]]) -> list[list[float]]:
        """
        Compute the pairwise cosine similarity matrix for a list of embeddings.
        """
        n = len(embeddings)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                s = cosine_similarity(embeddings[i], embeddings[j])
                matrix[i][j] = s
                matrix[j][i] = s
        return matrix


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(
        config={
            "extra-imports": [
                "sentence_transformers",
                "plotly.graph_objects",
                "networkx",
                "plotly.subplots",
                "numpy",
                "sklearn.manifold",
                "graph",
                "similarity",
                "load_graph",
                "predictor",
                "visualize",
                "os",
                "math",
                "random",
            ],
            "allowed-io": ["fit", "__init__", "main", "evaluate"],
            "max-line-length": 120,
        }
    )
