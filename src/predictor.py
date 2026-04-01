"""
Missing link prediction using structural and semantic similarity scores.

Evaluates note pairs to suggest meaningful new WikiLinks in the knowledge graph.

Copyright (c) 2026 Caellum Yip Hoi-Lee, Catherine Abdul-Samad, Michael Chen, Joshua Yeung.
All rights reserved.
"""

import random
from typing import Optional
from graph import KnowledgeGraph
from similarity import (
    jaccard,
    adamic_adar,
    cosine_similarity,
    SentenceBERTEmbedder,
)


class MinkPredictor:
    """
    A missing link prediction class for note graphs that
    uses a weighted combination of graph structure and semantic similarity scores.

    Instance Attributes:
        - w_struct: The weight applied to the structural similarity score.
        - w_sem: The weight applied to the semantic similarity score.
        - _embedder: The model used to generate semantic text embeddings (lazy-loaded to save memory).

    Representation Invariants:
        - 0 <= self.w_struct <= 1
        - 0 <= self.w_sem <= 1
        - math.isclose(self.w_struct + self.w_sem, 1)
    """

    DEFAULT_K = 10
    AA_CLIP_PERCENTILE = 0.95
    TRIAL_SEED = 17

    def __init__(
        self, w_struct: float = 0.4, w_sem: float = 0.6
    ) -> None:
        """
        Initialise a new MinkPredictor class with provided structural and semantic weights.

        Preconditions:
            - w_sem + w_struct == 1
        """
        self.w_struct = w_struct
        self.w_sem = w_sem
        self._embedder: Optional[SentenceBERTEmbedder] = None

    def _get_embedder(self) -> SentenceBERTEmbedder:
        """
        Return the NLP sentence embedder, initialising a new one if none are present.
        """
        if self._embedder is None:
            self._embedder = SentenceBERTEmbedder()
        return self._embedder

    def compute_embeddings(self, g: KnowledgeGraph) -> dict[str, list[float]]:
        """
        Return a dictionary mapping note names to their NLP vector embeddings for the graph.
        """
        notes = g.get_notes()
        texts = []
        for n in notes:
            note_object = g.get_note(n)
            texts.append(note_object.content)

        embedder = self._get_embedder()
        embedder.fit(texts)
        vecs = embedder.encode(texts)
        final_dictionary = {}
        for i in range(len(notes)):
            note_name = notes[i]
            vector = vecs[i]
            final_dictionary[note_name] = vector

        return final_dictionary

    def _score_pairs(
        self,
        g: KnowledgeGraph,
        pairs: list[tuple[str, str]],
        embeddings: dict[str, list[float]],
    ) -> list[tuple[str, str, float, float, float]]:
        """
        Calculate the combined similarity score for a given list of note pairs.
        """
        aa_scores = [adamic_adar(g, u, v) for u, v in pairs]

        max_aa = self._compute_aa_max(aa_scores)

        results = []
        for i in range(len(pairs)):
            u, v = pairs[i]
            aa_scaled = min(aa_scores[i] / max_aa, 1.0) if max_aa > 0 else 0.0
            combined, struct, sem = self._compute_pair_components(g, u, v, aa_scaled, embeddings)
            results.append((u, v, combined, struct, sem))
        return results

    def _compute_aa_max(self, aa_scores: list[float]) -> float:
        """
        Returns the normalization ceiling for Adamic-Adar scores.
        Uses the 95th percentile value to prevent hub outliers from
        dominating the score range, falling back to the true max if
        the percentile value is 0.
        """
        if not aa_scores:
            return 0
        sorted_aa = sorted(aa_scores)
        clip_idx = int(len(sorted_aa) * self.AA_CLIP_PERCENTILE)
        clipped = sorted_aa[clip_idx]
        return clipped if clipped > 0 else max(aa_scores)

    def _compute_pair_components(self, g, u, v, aa_scaled, embeddings) -> tuple[float, float, float]:
        """
        compute the structural, semantic, and combined scores for a single note pair.
        Returns a tuple of (combined, struct, sem).
        """
        jaccard_score = jaccard(g, u, v)
        struct = max(jaccard_score, aa_scaled)

        note_u_len = len(g.get_note(u).content.strip())
        note_v_len = len(g.get_note(v).content.strip())
        if note_u_len < 30 or note_v_len < 30:
            sem = 0.0
        else:
            sem = cosine_similarity(embeddings[u], embeddings[v])

        combined = (self.w_struct * struct) + (self.w_sem * sem)
        return combined, struct, sem

    @staticmethod
    def _mrr_at_k(
        held_out: list[tuple[str, str]],
        predictions: list[tuple[str, str, float]],
        k: int,
    ) -> float:
        """
        Calculate Mean Reciprocal Rank to allow granular weight tuning.
        """
        if not held_out:
            return 0.0

        correct = {tuple(sorted([u, v])) for u, v in held_out}

        mrr_sum = 0.0
        for i in range(min(k, len(predictions))):
            u, v, _ = predictions[i]
            guess = tuple(sorted([u, v]))
            if guess in correct:
                mrr_sum += 1.0 / (i + 1)

        return mrr_sum / len(correct)

    def run_holdout_eval(
        self,
        g: KnowledgeGraph,
        k: int = DEFAULT_K,
        holdout_frac: float = 0.2,
        n_trials: int = 5,
        embeddings: Optional[dict] = None,
    ) -> dict:
        """
        Evaluate the predictor's performance, usnig MRR to fix tuning ties.
        """
        if embeddings is None:
            embeddings = self.compute_embeddings(g)
        recalls, precisions, mrrs = [], [], []

        dynamic_k = k

        for trial in range(n_trials):
            recall, precision, mrr, dynamic_k = self._run_trial(
                g, k, holdout_frac, embeddings, seed=trial * self.TRIAL_SEED)

            recalls.append(recall)
            precisions.append(precision)
            mrrs.append(mrr)

        return {
            "k": dynamic_k,
            "n_trials": n_trials,
            "holdout_frac": holdout_frac,
            "recall@k": sum(recalls) / len(recalls),
            "precision@k": sum(precisions) / len(precisions),
            "mrr": sum(mrrs) / len(mrrs),
            "per_trial_recall": recalls,
            "per_trial_precision": precisions,
        }

    def _run_trial(self, g, k, holdout_frac, embeddings, seed) -> tuple[float, float, float, float]:
        """
        Run a single holdout trial and return tuple (recall, precision, mrr, dynamic_k) at k.
        """
        g_reduced, held_out = self._holdout_split(g, holdout_frac, seed)
        dynamic_k = min(k, len(held_out))
        preds = [(u, v, s) for u, v, s, _, _ in self.score_all(g_reduced, embeddings)]
        return (
            self._recall_at_k(held_out, preds, dynamic_k),
            self._precision_at_k(held_out, preds, dynamic_k),
            self._mrr_at_k(held_out, preds, dynamic_k),
            dynamic_k
        )

    def fit(
        self,
        g_tune: KnowledgeGraph,
        k: int = DEFAULT_K,
        holdout_frac: float = 0.2,
        n_trials: int = 5,
        steps: int = 5,
    ) -> dict:
        """
        Grid search over (w_struct, w_sem) pairs, tuning on recall@k with MRR as tiebreaker.
        Sets the best weights on the predictor and returns the full grid results.
        """
        embeddings = self.compute_embeddings(g_tune)
        grid = [i / steps for i in range(steps + 1)]
        best = {"recall@k": -1.0, "mrr": -1.0, "w_struct": 0.4, "w_sem": 0.6}
        results = []

        for ws in grid:
            res = self._evaluate_weights(ws, g_tune, k, holdout_frac, n_trials, embeddings)
            results.append(res)
            if self._is_better_result(res, best):
                best = {
                    "recall@k": res["recall@k"],
                    "mrr": res["mrr"],
                    "w_struct": res["w_struct"],
                    "w_sem": res["w_sem"]
                }

        self.w_struct = best["w_struct"]
        self.w_sem = best["w_sem"]
        print(
            f"  Best weights: w_struct={self.w_struct}, w_sem={self.w_sem} "
            f"(recall@{k}={best['recall@k']:.3f}, mrr={best['mrr']:.3f})"
        )
        return {"best": best, "grid": results}

    def _evaluate_weights(self, ws, g_tune, k, holdout_frac, n_trials, embeddings) -> dict:
        """
        Evaluates a single (w_struct, w_sem) weight pair and return its metrics.
        """
        self.w_struct = ws
        self.w_sem = 1.0 - ws
        res = self.run_holdout_eval(g_tune, k, holdout_frac, n_trials, embeddings)
        return {
            "w_struct": self.w_struct,
            "w_sem": self.w_sem,
            "recall@k": res["recall@k"],
            "precision@k": res["precision@k"],
            "mrr": res["mrr"]
        }

    @staticmethod
    def _is_better_result(res: dict, best: dict) -> bool:
        """
        Returns True if res outperforms best, using MRR to break recall ties.
        """
        return res["recall@k"] > best["recall@k"] or (
            res["recall@k"] == best["recall@k"] and res["mrr"] > best["mrr"]
        )

    def score_all(
        self,
        g: KnowledgeGraph,
        embeddings: Optional[dict] = None,
    ) -> list[tuple[str, str, float, float, float]]:
        """
        Score all non-existent edges in the given graph and return them sorted by highest score.

        Returns a list of tuples containing the node pair, combined score, structural score, and semantic score.
        """

        if embeddings is None:
            embeddings = self.compute_embeddings(g)
        scored = self._score_pairs(g, g.non_edges(), embeddings)
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored

    @staticmethod
    def _holdout_split(
        g: KnowledgeGraph,
        holdout_frac: float = 0.2,
        seed: int = 42,
    ) -> tuple[KnowledgeGraph, list[tuple[str, str]]]:
        """
        Hide a fixed percentage of the graph's links.
        """
        rng = random.Random(seed)

        practice_graph = g.copy()
        all_edges = g.get_edges().copy()
        rng.shuffle(all_edges)
        target_hidden_count = max(1, int(len(all_edges) * holdout_frac))

        hidden_links = []

        for u, v in all_edges:
            if len(hidden_links) >= target_hidden_count:
                break

            if practice_graph.degree(u) > 1 and practice_graph.degree(v) > 1:
                practice_graph.remove_edge(u, v)
                hidden_links.append((u, v))

        return practice_graph, hidden_links

    @staticmethod
    def _recall_at_k(
        held_out: list[tuple[str, str]],
        predictions: list[tuple[str, str, float]],
        k: int,
    ) -> float:
        """
        Calculate the recall at k for the actual hidden edges that were found in a given set of predictions.
        """
        if len(held_out) == 0:
            return 0.0

        correct = []
        for u, v in held_out:
            standard_link = tuple(sorted([u, v]))
            correct.append(standard_link)

        top_k_guesses = []
        for i in range(min(k, len(predictions))):
            u = predictions[i][0]
            v = predictions[i][1]
            standard_link = tuple(sorted([u, v]))
            top_k_guesses.append(standard_link)

        matches = 0
        for guess in top_k_guesses:
            if guess in correct:
                matches += 1

        return matches / len(correct)

    @staticmethod
    def _precision_at_k(
        held_out: list[tuple[str, str]],
        predictions: list[tuple[str, str, float]],
        k: int,
    ) -> float:
        """
        Calculate the proportion of the top k predictions that were actually hidden edges.
        """
        if k == 0:
            return 0.0

        correct = []
        for u, v in held_out:
            standard_link = tuple(sorted([u, v]))
            correct.append(standard_link)

        hits = 0
        for i in range(min(k, len(predictions))):
            u = predictions[i][0]
            v = predictions[i][1]
            guess = tuple(sorted([u, v]))

            if guess in correct:
                hits += 1

        return hits / k

    def predict(
        self,
        g: KnowledgeGraph,
        k: int = DEFAULT_K,
        embeddings: Optional[dict] = None,
    ) -> list[tuple[str, str, float]]:
        """
        Predict and return the top-k missing links in the graph.
        """

        if embeddings is None:
            embeddings = self.compute_embeddings(g)
        scored = self._score_pairs(g, g.non_edges(), embeddings)

        scored.sort(key=lambda x: x[2], reverse=True)
        top_k_results = scored[:k]

        final_predictions = []

        for item in top_k_results:
            u = item[0]
            v = item[1]
            combined_score = item[2]
            clean_match = (u, v, combined_score)
            final_predictions.append(clean_match)
        return final_predictions


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
