import os

from graph import KnowledgeGraph
from load_graph import load_vault
from predictor import MinkPredictor
from visualize import graph_viz, layout_comparison

K = 10
OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)

def main():
    print("1. Loading vaults")
    vault_a = load_vault("vaults/vault_a")
    vault_b = load_vault("vaults/vault_b")
    print(f"  vault_a: {vault_a}")
    print(f"  vault_b: {vault_b}")

    print("2. Fitting weights on vault_a (tuning set)")
    predictor = MinkPredictor()
    fit_results = predictor.fit(vault_a, k=K, holdout_frac=0.2, n_trials=5, steps=5)

    print(
        f"\n  {'w_struct':>8}  {'w_sem':>6}  {'recall@'+str(K):>10}  {'precision@'+str(K):>13}"
    )
    print("  " + "-" * 44)
    for row in fit_results["grid"]:
        marker = " ◄" if row["w_struct"] == fit_results["best"]["w_struct"] else ""
        print(
            f"  {row['w_struct']:>8.1f}  {row['w_sem']:>6.1f}  "
            f"{row['recall@k']:>10.4f}  {row['precision@k']:>13.4f}{marker}"
        )

    print("3. Evaluating on vault_b (test set)")
    print(f"  Fixed weights: w_struct={predictor.w_struct}, w_sem={predictor.w_sem}")

    embeddings_b = predictor._compute_embeddings(vault_b)
    eval_results = predictor.evaluate(
        vault_b, k=K, holdout_frac=0.2, n_trials=5, embeddings=embeddings_b
    )

    print(
        f"\n  k={K}, holdout={eval_results['holdout_frac']}, trials={eval_results['n_trials']}"
    )
    print(f"  recall@{K}    = {eval_results['recall@k']:.4f}")
    print(f"  precision@{K} = {eval_results['precision@k']:.4f}")
    print(
        f"  per-trial recall:    "
        + ", ".join(f"{r:.3f}" for r in eval_results["per_trial_recall"])
    )
    print(
        f"  per-trial precision: "
        + ", ".join(f"{p:.3f}" for p in eval_results["per_trial_precision"])
    )

    print("4. Predicting top-k links for vault_b")
    predictions = predictor.predict(vault_b, k=K, embeddings=embeddings_b)
    print(f"\n  {'Rank':>4}  {'u':28} {'v':28}  {'score':>7}")
    print("  " + "-" * 76)
    for rank, (u, v, score) in enumerate(predictions, 1):
        print(f"  {rank:>4}  {u:28} {v:28}  {score:.4f}")

    print("5. Graph statistics (vault_b)")
    top_hubs = sorted(
        vault_b.degree_centrality().items(), key=lambda x: x[1], reverse=True
    )[:5]
    for name, dc in top_hubs:
        print(f"    {dc:.4f}  {name}")

    vault_b_aug = vault_b.copy()
    vault_b_aug.add_predicted_edges(predictions)
    print(
        f"\n  components: {len(vault_b.connected_components())} → {len(vault_b_aug.connected_components())}"
    )
    print(f"  edges:      {len(vault_b.edges)} → {len(vault_b_aug.edges)}")

    print("6. Generating visualisations")
    graph_viz(
        vault_b_aug,
        predictions,
        output_path=os.path.join(OUTPUT, "vault_b_graph.html"),
        title=f"Minks – vault_b (k={K}, w_struct={predictor.w_struct}, w_sem={predictor.w_sem})",
    )
    layout_comparison(
        vault_b,
        embeddings_b,
        output_path=os.path.join(OUTPUT, "layout_comparison.html"),
    )

    print("Done")
    print(f"  Output written to: {OUTPUT}/")


if __name__ == "__main__":
    main()
