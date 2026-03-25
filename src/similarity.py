import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph import KnowledgeGraph


def jaccard(g: "KnowledgeGraph", u: str, v: str) -> float:
    """
    Returns the Jaccard Similarity Index of str u and v.
    """
    nu, nv = g.get_neighbours(u), g.get_neighbours(v)
    intersection = len(nu & nv)
    union = len(nu | nv)
    return intersection / union if union > 0 else 0.0


def adamic_adar(g: "KnowledgeGraph", u: str, v: str) -> float:
    """
    Returns the Adamic Adar score of str u and v.
    """
    score = 0.0
    for w in g.get_neighbours(u) & g.get_neighbours(v):
        deg = g.get_note(w).degree()
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
    """
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


class SentenceBERTEmbedder:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._use_fallback = False

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.MODEL_NAME)
            print("  [embedder] Using Sentence-BERT (all-MiniLM-L6-v2)")
        except (ImportError, OSError):
            print("  [embedder] Sentence-BERT unavailable — using TF-IDF fallback")
            self._use_fallback = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _fit_idf(self, corpus: list[str]):
        N = len(corpus)
        df: dict[str, int] = {}
        for doc in corpus:
            for tok in set(self._tokenize(doc)):
                df[tok] = df.get(tok, 0) + 1
        self._vocab = sorted(df.keys())
        self._idf = {
            tok: math.log((N + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()
        }

    def _tfidf_vector(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * len(self._vocab)
        tf: dict[str, float] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        total = len(tokens)
        vec = [
            (tf.get(tok, 0) / total) * self._idf.get(tok, 0.0) for tok in self._vocab
        ]
        n = norm(vec)
        return [x / n for x in vec] if n > 0 else vec

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._use_fallback:
            self._fit_idf(texts)
            return [self._tfidf_vector(t) for t in texts]
        arr = self._model.encode(texts, show_progress_bar=False)
        return arr.tolist()

    def pairwise_cosine(self, embeddings: list[list[float]]) -> list[list[float]]:
        n = len(embeddings)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                s = cosine_similarity(embeddings[i], embeddings[j])
                matrix[i][j] = s
                matrix[j][i] = s
        return matrix
