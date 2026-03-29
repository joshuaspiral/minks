import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.manifold import TSNE


def _build_nx_graph(g) -> nx.Graph:
    """Convert KnowledgeGraph to a NetworkX Graph."""
    nx_g = nx.Graph()
    nodes = list(g.get_all_note_names())
    nx_g.add_nodes_from(nodes)
    for u in nodes:
        for v in g.get_neighbours(u):
            nx_g.add_edge(u, v)
    return nx_g


def graph_viz(
    g,
    predictions: list[tuple[str, str, float]],
    output_path: str,
    title: str = "Knowledge Graph",
):
    """Plot the graph using a force-directed layout, highlighting predicted edges."""
    nx_g = _build_nx_graph(g)
    pos = nx.spring_layout(nx_g, k=0.15, iterations=50, seed=42)

    predicted_edges = {tuple(sorted((u, v))) for u, v, _ in predictions}

    edge_x, edge_y, pred_x, pred_y = [], [], [], []
    for u, v in nx_g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if tuple(sorted((u, v))) in predicted_edges:
            pred_x.extend([x0, x1, None])
            pred_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()

    # Original edges
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            name="Existing Links",
        )
    )

    # Predicted edges
    fig.add_trace(
        go.Scatter(
            x=pred_x,
            y=pred_y,
            mode="lines",
            line=dict(width=2, color="red", dash="dot"),
            hoverinfo="none",
            name="Predicted Links",
        )
    )

    # Nodes
    node_x = [pos[node][0] for node in nx_g.nodes()]
    node_y = [pos[node][1] for node in nx_g.nodes()]
    node_names = list(nx_g.nodes())
    node_hover = [
        "<b>" + n + "</b><br><br>" + g.get_note(n).content[:500].replace("\n", "<br>")
        for n in node_names
    ]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            hovertext=node_hover,
            marker=dict(showscale=False, color="lightblue", size=10, line_width=2),
            name="Notes",
        )
    )

    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    fig.write_html(output_path)


def layout_comparison(g, embeddings: dict[str, list[float]], output_path: str):
    """Generate a side-by-side comparison of structural vs semantic layouts."""
    nx_g = _build_nx_graph(g)
    nodes = list(nx_g.nodes())

    # Structural Layout (Spring)
    pos_spring = nx.spring_layout(nx_g, seed=42)
    spring_x = [pos_spring[n][0] for n in nodes]
    spring_y = [pos_spring[n][1] for n in nodes]

    # Semantic Layout (t-SNE)
    matrix = np.array([embeddings[n] for n in nodes])
    perplexity = min(30, max(1, len(nodes) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_coords = tsne.fit_transform(matrix)
    tsne_x = tsne_coords[:, 0]
    tsne_y = tsne_coords[:, 1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Structural (NetworkX Spring)", "Semantic (t-SNE)"),
    )

    fig.add_trace(
        go.Scatter(
            x=spring_x,
            y=spring_y,
            mode="markers",
            text=nodes,
            hoverinfo="text",
            marker=dict(color="blue", size=8, opacity=0.7),
            name="Spring",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=tsne_x,
            y=tsne_y,
            mode="markers",
            text=nodes,
            hoverinfo="text",
            marker=dict(color="green", size=8, opacity=0.7),
            name="t-SNE",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="Layout Comparison", showlegend=False, plot_bgcolor="white"
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.write_html(output_path)
