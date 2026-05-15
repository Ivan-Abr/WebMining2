import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TSNE_PERPLEXITY, SVD_COMPONENTS, RESULTS_DIR

logger = logging.getLogger(__name__)


def reduce_dimensions(X) -> np.ndarray:
    """
    Двухэтапное снижение размерности:
    1. TruncatedSVD: sparse TF-IDF (5000 признаков) → 50 компонент
    2. t-SNE: 50 компонент → 2D

    TruncatedSVD необходим перед t-SNE для ускорения работы.
    """
    n_components = min(SVD_COMPONENTS, X.shape[1] - 1, X.shape[0] - 1)
    perplexity   = min(TSNE_PERPLEXITY, X.shape[0] - 1)

    logger.info(f"[Visualizer] TruncatedSVD: {X.shape[1]} → {n_components}")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X)
    explained = svd.explained_variance_ratio_.sum()
    logger.info(f"[Visualizer] SVD объясняет {explained:.1%} дисперсии")

    logger.info(f"[Visualizer] t-SNE: {n_components} → 2D (perplexity={perplexity})")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    X_2d = tsne.fit_transform(X_svd)
    logger.info("[Visualizer] t-SNE завершён")
    return X_2d


def plot_static(
    X_2d: np.ndarray,
    cluster_labels: list[int],
    titles: list[str],
    k: int,
) -> str:
    """Строит статический scatter plot (PNG)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    colors = plt.cm.get_cmap("tab10", k)
    fig, ax = plt.subplots(figsize=(12, 8))

    for cid in range(k):
        mask = np.array(cluster_labels) == cid
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors(cid)], label=f"Cluster {cid}",
            alpha=0.6, s=40, edgecolors="none",
        )

    ax.set_title("t-SNE визуализация кластеров статей", fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(title="Кластер", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "clusters_tsne.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"[Visualizer] PNG сохранён → {path}")
    return path


def plot_interactive(
    X_2d: np.ndarray,
    cluster_labels: list[int],
    titles: list[str],
    labels: list[str],
    k: int,
) -> str:
    """Строит интерактивный scatter plot через Plotly (HTML)."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        logger.warning("[Visualizer] plotly не установлен, пропускаем HTML-график")
        return ""

    df = pd.DataFrame({
        "x":       X_2d[:, 0],
        "y":       X_2d[:, 1],
        "cluster": [f"Cluster {c}" for c in cluster_labels],
        "label":   labels,
        "title":   [t[:80] + "..." if len(t) > 80 else t for t in titles],
    })

    fig = px.scatter(
        df, x="x", y="y",
        color="cluster",
        hover_data={"title": True, "label": True, "x": False, "y": False},
        title="Кластеризация научных статей (t-SNE + K-means)",
        labels={"cluster": "Кластер"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(
        legend_title_text="Кластер",
        width=1100, height=700,
        font=dict(size=12),
    )

    path = os.path.join(RESULTS_DIR, "clusters_tsne.html")
    fig.write_html(path)
    logger.info(f"[Visualizer] HTML сохранён → {path}")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from preprocess.vectorizer import load_artifacts
    import json as _json

    X, labels, _, _ = load_artifacts()

    # Загружаем кластерные метки (если уже есть)
    labeled_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "articles_labeled.json"
    )
    with open(labeled_path, encoding="utf-8") as f:
        articles = _json.load(f)

    cluster_labels = [art.get("cluster", 0) for art in articles]
    titles         = [art.get("title",   "") for art in articles]
    k = max(cluster_labels) + 1

    X_2d = reduce_dimensions(X)
    plot_static(X_2d, cluster_labels, titles, k)
    plot_interactive(X_2d, cluster_labels, titles, labels, k)
    print("Визуализация завершена.")
