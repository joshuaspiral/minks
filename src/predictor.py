import random
import math
from typing import Optional
from graph import KnowledgeGraph
from similarity import jaccard, adamic_adar, normalise, SentenceBERTEmbedder


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

    def __init__(self, w_struct: float = 0.4, w_sem: float = 0.6):
        """Initialise a new MinkPredictor class with provided structural and semantic weights.

        Preconditions:
            - w_sem + w_struct == 1
        """
        self.w_struct = w_struct
        self.w_sem = w_sem
        self._embedder: Optional[SentenceBERTEmbedder] = None

    def _get_embedder(self) -> SentenceBERTEmbedder:
        """Return the NLP sentence embedder, initialising a new one if none are present."""
        if self._embedder is None:
            self._embedder = SentenceBERTEmbedder()
        return self._embedder

    def _compute_embeddings(self, g: KnowledgeGraph) -> dict[str, list[float]]:
        """Return a dictionary mapping note names to their NLP vector embeddings for the graph."""
        nodes = g.nodes
        texts = []
        for n in nodes:
            note_object = g.get_note(n)
            texts.append(note_object.content)
            
        vecs = self._get_embedder().encode(texts)
        final_dictionary = {}
        for i in range(len(nodes)):
            note_name = nodes[i]
            vector = vecs[i]
            final_dictionary[note_name] = vector

        return final_dictionary

    def _score_pairs(
        self,
        g: KnowledgeGraph,
        pairs: list[tuple[str, str]],
        embeddings: dict[str, list[float]],
    ) -> list[tuple[str, str, float, float, float]]:
        """Calculate the combined similarity score for a given list of note pairs.

        Returns a list of tuples containing the node pair, combined score, structural score, and semantic score.
        """

        aa_scores = []
        for u, v in pairs:
            aa_scores.append(adamic_adar(g, u, v))
        if aa_scores:
            max_aa = max(aa_scores)
        else:
            max_aa = 0
        
        results = []
        for i in range(len(pairs)):
            u, v = pairs[i]

            # Scale the Adamic-Adar score down to within 0 and 1, for accuratel matching with jaccard score
            if max_aa > 0:
                aa_scaled = raw_aa_scores[i] / max_aa
            else:
                aa_scaled = 0.0

            jaccard_score = jaccard(g, u, v)
            struct = max(jaccard_score, aa_scaled)

            # Semantic text score
            sem = cosine_similarity(embeddings[u], embeddings[v])
            combined = (self.w_struct * struct) + (self.w_sem * sem)

            results.append((u, v, combined, struct, sem))
        return results

    def predict(
        self,
        g: KnowledgeGraph,
        k: int = DEFAULT_K,
        embeddings: Optional[dict] = None,
    ) -> list[tuple[str, str, float]]:
        """Predict and return the top-k missing links in the graph.

        Return a list of tuples with the predicted pairs and their combined score, 
        sorted in descending order of score
        """
    
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

    @staticmethod
    def _holdout_split(
        g: KnowledgeGraph,
        holdout_frac: float = 0.2,
        seed: int = 42,
    ) -> tuple[KnowledgeGraph, list[tuple[str, str]]]:
        """Hide a fixed percentage of the graph's links."""
        rng = random.Random(seed)

        practice_graph = g.copy()
        all_edges = g.edges.copy()
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
        """Calculate the recall at k for the actual hidden edges that were found in a given set of predictions,
        
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
        """Calculate the proportion of the top k predictions that were actually hidden edges.
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

    def evaluate(
        self,
        g: KnowledgeGraph,
        k: int = DEFAULT_K,
        holdout_frac: float = 0.2,
        n_trials: int = 5,
        embeddings: Optional[dict] = None,
    ) -> dict:
        """Evaluate the predictor's performance on the given graph using holdout validation on multiple trials. """
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
        """Tune the w_struct and w_sem weights to find the best combination for the graph.
        
        Tests different weight ratios (e.g., 0.0/1.0, 0.2/0.8) using the evaluate
        method and updates the predictor's weights to the combination that 
        achieves the highest recall@k.
        """
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