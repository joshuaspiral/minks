"""Missing link prediction using structural and semantic similarity scores.

Evaluates note pairs to suggest meaningful new WikiLinks in the knowledge graph.

Copyright (c) 2026 Caellum Yip Hoi-Lee, Catherine Abdul-Samad, Michael Chen, Joshua Yeung.
All rights reserved.
"""

import random
import math
from typing import Optional
from graph import KnowledgeGraph
from similarity import jaccard, adamic_adar, normalise, cosine_similarity, SentenceBERTEmbedder


class MinkPredictor:
    """A missing link prediction class for note graphs that uses a weighted combination of graph structure and semantic similarity scores.

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

    def __init__(self, w_struct: float = 0.4, w_sem: float = 0.6, use_tfidf: bool = False) -> None:
            """Initialise a new MinkPredictor class with provided structural and semantic weights.

            Preconditions:
                - w_sem + w_struct == 1
            """
            self.w_struct = w_struct
            self.w_sem = w_sem
            self._use_tfidf = use_tfidf
            self._embedder: Optional[SentenceBERTEmbedder] = None

    def _get_embedder(self) -> SentenceBERTEmbedder:
        """Return the NLP sentence embedder, initialising a new one if none are present."""
        if self._embedder is None:
            self._embedder = SentenceBERTEmbedder(force_tfidf=self._use_tfidf)
        return self._embedder

    def _compute_embeddings(self, g: KnowledgeGraph) -> dict[str, list[float]]:
        """Return a dictionary mapping note names to their NLP vector embeddings for the graph."""
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
        """Calculate the combined similarity score for a given list of note pairs."""
        aa_scores = []
        for u, v in pairs:
            aa_scores.append(adamic_adar(g, u, v))

        # Use 95th percentile or log scaling to prevent hub outliers from squashing scores
        if aa_scores:
            sorted_aa = sorted(aa_scores)
            clip_idx = int(len(sorted_aa) * 0.95)
            max_aa = sorted_aa[clip_idx] if len(sorted_aa) > 0 else 0
            if max_aa == 0:
                max_aa = max(aa_scores) # Fallback
        else:
            max_aa = 0

        results = []
        for i in range(len(pairs)):
            u, v = pairs[i]

            if max_aa > 0:
                # Cap at 1.0 to handle values above the 95th percentile
                aa_scaled = min(aa_scores[i] / max_aa, 1.0)
            else:
                aa_scaled = 0.0

            jaccard_score = jaccard(g, u, v)
            struct = max(jaccard_score, aa_scaled)

            sem = cosine_similarity(embeddings[u], embeddings[v])
            combined = (self.w_struct * struct) + (self.w_sem * sem)

            results.append((u, v, combined, struct, sem))
        return results

    def _mrr_at_k(
        self,
        held_out: list[tuple[str, str]],
        predictions: list[tuple[str, str, float]],
        k: int,
    ) -> float:
        """Calculate Mean Reciprocal Rank to allow granular weight tuning."""
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

    def evaluate(
        self,
        g: KnowledgeGraph,
        k: int = DEFAULT_K,
        holdout_frac: float = 0.2,
        n_trials: int = 5,
        embeddings: Optional[dict] = None,
    ) -> dict:
        """Evaluate the predictor's performance, usnig MRR to fix tuning ties."""
        if embeddings is None:
            embeddings = self._compute_embeddings(g)
        recalls, precisions, mrrs = [], [], []

        for trial in range(n_trials):
            g_reduced, held_out = self._holdout_split(
                g, holdout_frac=holdout_frac, seed=trial * 17
            )
            # Dynamically set K for metric validity if not overridden
            dynamic_k = max(k, len(held_out))

            preds = [
                (u, v, s)
                for u, v, s, _, _ in self.score_all(g_reduced, embeddings=embeddings)
            ]
            recalls.append(self._recall_at_k(held_out, preds, dynamic_k))
            precisions.append(self._precision_at_k(held_out, preds, dynamic_k))
            mrrs.append(self._mrr_at_k(held_out, preds, dynamic_k))

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

    def fit(
        self,
        g_tune: KnowledgeGraph,
        k: int = DEFAULT_K,
        holdout_frac: float = 0.2,
        n_trials: int = 5,
        steps: int = 5,
    ) -> dict:
        """Tune using MRR to correctly break ties when sorting rank shifts."""
        embeddings = self._compute_embeddings(g_tune)
        grid = [i / steps for i in range(steps + 1)]
        best = {"mrr": -1.0, "w_struct": 0.4, "w_sem": 0.6}
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
                    "mrr": res["mrr"],
                }
            )
            # Tune based on MRR to capture internal rank improvements
            if res["mrr"] > best["mrr"]:
                best = {"mrr": res["mrr"], "w_struct": ws, "w_sem": 1.0 - ws}

        self.w_struct = best["w_struct"]
        self.w_sem = best["w_sem"]
        print(
            f"  Best weights: w_struct={self.w_struct}, w_sem={self.w_sem} (mrr={best['mrr']:.3f})"
        )
        return {"best": best, "grid": results}

    def score_all(
        self,
        g: KnowledgeGraph,
        embeddings: Optional[dict] = None,
    ) -> list[tuple[str, str, float, float, float]]:
        """Score all non-existent edges in the given graph and return them sorted by highest score.
        Returns a list of tuples containing the node pair, combined score, structural score, and semantic score.
        """

        if embeddings is None:
            embeddings = self._compute_embeddings(g)
        scored = self._score_pairs(g, g.non_edges(), embeddings)
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored

    def _holdout_split(
        self,
        g: KnowledgeGraph,
        holdout_frac: float = 0.2,
        seed: int = 42,
    ) -> tuple[KnowledgeGraph, list[tuple[str, str]]]:
        """Hide a fixed percentage of the graph's links."""
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

    def _recall_at_k(
        self,
        held_out: list[tuple[str, str]],
        predictions: list[tuple[str, str, float]],
        k: int,
    ) -> float:
        """Calculate the recall at k for the actual hidden edges that were found in a given set of predictions,"""
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

    def _precision_at_k(
        self,
        held_out: list[tuple[str, str]],
        predictions: list[tuple[str, str, float]],
        k: int,
    ) -> float:
        """Calculate the proportion of the top k predictions that were actually hidden edges."""
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
        """Predict and return the top-k missing links in the graph."""

        if embeddings is None:
            embeddings = self._compute_embeddings(g)
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
    python_ta.check_all(config={
        'extra-imports': ['sentence_transformers', 'plotly.graph_objects', 'networkx', 'plotly.subplots', 'numpy', 'sklearn.manifold', 'graph', 'similarity', 'load_graph', 'predictor', 'visualize', 'os', 're', 'math', 'random'],
        'allowed-io': ['fit', '__init__', 'main', 'evaluate'],
        'max-line-length': 120
    })
