import random
import math
from typing import Optional
from graph import KnowledgeGraph
from similarity import jaccard, adamic_adar, normalise, SentenceBERTEmbedder


class MinkPredictor:
    DEFAULT_K = 10

    def __init__(self, w_struct: float = 0.4, w_sem: float = 0.6):
        self.w_struct = w_struct
        self.w_sem = w_sem
        self._embedder: Optional[SentenceBERTEmbedder] = None

    def _get_embedder(self) -> SentenceBERTEmbedder:
        if self._embedder is None:
            self._embedder = SentenceBERTEmbedder()
        return self._embedder

    def _compute_embeddings(self, g: KnowledgeGraph) -> dict[str, list[float]]:
        nodes = g.nodes
        texts = [g.get_note(n).content for n in nodes]
        vecs = self._get_embedder().encode(texts)
        return {n: v for n, v in zip(nodes, vecs)}

    def _score_pairs(
        self,
        g: KnowledgeGraph,
        pairs: list[tuple[str, str]],
        embeddings: dict[str, list[float]],
    ) -> list[tuple[str, str, float, float, float]]:
        from similarity import cosine_similarity

        aa_norm = normalise([adamic_adar(g, u, v) for u, v in pairs])
        results = []
        for idx, (u, v) in enumerate(pairs):
            struct = max(jaccard(g, u, v), aa_norm[idx])
            sem = cosine_similarity(embeddings[u], embeddings[v])
            combined = self.w_struct * struct + self.w_sem * sem
            results.append((u, v, combined, struct, sem))
        return results

    def predict(
        self,
        g: KnowledgeGraph,
        k: int = DEFAULT_K,
        embeddings: Optional[dict] = None,
    ) -> list[tuple[str, str, float]]:
        if embeddings is None:
            embeddings = self._compute_embeddings(g)
        scored = self._score_pairs(g, g.non_edges(), embeddings)
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(u, v, s) for u, v, s, _, _ in scored[:k]]

    def score_all(
        self,
        g: KnowledgeGraph,
        embeddings: Optional[dict] = None,
    ) -> list[tuple[str, str, float, float, float]]:
        if embeddings is None:
            embeddings = self._compute_embeddings(g)
        scored = self._score_pairs(g, g.non_edges(), embeddings)
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored

    @staticmethod
    def _holdout_split(
        g: KnowledgeGraph,
        holdout_frac: float = 0.2,
        seed: int = 42,
    ) -> tuple[KnowledgeGraph, list[tuple[str, str]]]:
        rng = random.Random(seed)
        g_reduced = g.copy()
        edges = g.edges.copy()
        rng.shuffle(edges)
        n_holdout = max(1, int(len(edges) * holdout_frac))
        held_out = []
        for u, v in edges:
            if len(held_out) >= n_holdout:
                break
            if g_reduced.degree(u) > 1 and g_reduced.degree(v) > 1:
                g_reduced.remove_edge(u, v)
                held_out.append((u, v))
        return g_reduced, held_out

    @staticmethod
    def _recall_at_k(
        held_out: list[tuple[str, str]],
        predictions: list[tuple[str, str, float]],
        k: int,
    ) -> float:
        if not held_out:
            return 0.0
        top_k = {tuple(sorted([u, v])) for u, v, _ in predictions[:k]}
        held_out_set = {tuple(sorted([u, v])) for u, v in held_out}
        return len(top_k & held_out_set) / len(held_out_set)

    @staticmethod
    def _precision_at_k(
        held_out: list[tuple[str, str]],
        predictions: list[tuple[str, str, float]],
        k: int,
    ) -> float:
        if k == 0:
            return 0.0
        held_out_set = {tuple(sorted([u, v])) for u, v in held_out}
        hits = sum(
            1 for u, v, _ in predictions[:k] if tuple(sorted([u, v])) in held_out_set
        )
        return hits / k

    def evaluate(
        self,
        g: KnowledgeGraph,
        k: int = DEFAULT_K,
        holdout_frac: float = 0.2,
        n_trials: int = 5,
        embeddings: Optional[dict] = None,
    ) -> dict:
        if embeddings is None:
            embeddings = self._compute_embeddings(g)
        recalls, precisions = [], []
        for trial in range(n_trials):
            g_reduced, held_out = self._holdout_split(
                g, holdout_frac=holdout_frac, seed=trial * 17
            )
            preds = [
                (u, v, s)
                for u, v, s, _, _ in self.score_all(g_reduced, embeddings=embeddings)
            ]
            recalls.append(self._recall_at_k(held_out, preds, k))
            precisions.append(self._precision_at_k(held_out, preds, k))
        return {
            "k": k,
            "n_trials": n_trials,
            "holdout_frac": holdout_frac,
            "recall@k": sum(recalls) / len(recalls),
            "precision@k": sum(precisions) / len(precisions),
            "per_trial_recall": recalls,
            "per_trial_precision": precisions,
        }

    def fit(
        self,
        g_tune: KnowledgeGraph,
        k: int = DEFAULT_K,
        holdout_frac: float = 0.2,
        n_trials: int = 5,
        steps: int = 5,
    ) -> dict:
        embeddings = self._compute_embeddings(g_tune)
        grid = [i / steps for i in range(steps + 1)]
        best = {"recall": -1.0, "w_struct": 0.4, "w_sem": 0.6}
        results = []
        for ws in grid:
            self.w_struct = ws
            self.w_sem = 1.0 - ws
            res = self.evaluate(
                g_tune,
                k=k,
                holdout_frac=holdout_frac,
                n_trials=n_trials,
                embeddings=embeddings,
            )
            results.append(
                {
                    "w_struct": ws,
                    "w_sem": 1.0 - ws,
                    "recall@k": res["recall@k"],
                    "precision@k": res["precision@k"],
                }
            )
            if res["recall@k"] > best["recall"]:
                best = {"recall": res["recall@k"], "w_struct": ws, "w_sem": 1.0 - ws}
        self.w_struct = best["w_struct"]
        self.w_sem = best["w_sem"]
        print(
            f"  Best weights: w_struct={self.w_struct}, w_sem={self.w_sem} (recall@{k}={best['recall']:.3f})"
        )
        return {"best": best, "grid": results}