import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    KMEANS_K_RANGE, KMEANS_K_DEFAULT, SVD_COMPONENTS,
    RESULTS_DIR, PROCESSED_DIR, REPORT_TOP_WORDS_PER_CLUSTER,
)

logger = logging.getLogger(__name__)

def elbow_method(X, k_range=KMEANS_K_RANGE) -> int:
    inertials = []
    k_values = list(k_range)
    logger.info(f"[KMeans] Метод Elbow: проверяем k = {k_values}")
    for k in k_values:
        km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        km.fit(X)
        inertials.append(km.inertia_)
        logger.info(f"  k={k:2d}  inertia={km.inertia_:.1f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(k_values, inertials, "bo-", linewidth = 2, markersize = 7)
    ax.set_xlabel("Количество кластеров (k)", fontsize=12)
    ax.set_ylabel("Инерция (SSE)", fontsize=12)
    ax.set_title("Метод Elbow для выбора оптимального k", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    elbow_path = os.path.join(RESULTS_DIR, "elbow_curve.png")
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)
    logger.info(f"[KMeans] Elbow-кривая сохранена → {elbow_path}")
    coords = np.column_stack([k_values, inertials])
    d = coords[-1] - coords[0]
    d = d/np.linalg.norm(d)
    vecs = coords - coords[0]
    perp_dists = np.abs(np.cross(vecs, d))
    optimal_k = k_values[int(np.argmax(perp_dists))]
    logger.info(f"[KMeans] Оптимальное k (автовыбор): {optimal_k}")
    return optimal_k

def run_kmeans(X, k: int = KMEANS_K_DEFAULT) -> tuple:
    logger.info(f"[KMeans] Кластеризация: k={k}")
    km = KMeans(n_clusters = k, random_state = 42, n_init = 15, max_iter = 500)
    cluster_labels = km.fit_predict(X)
    logger.info(f"[KMeans] Готово. Инерция: {km.inertia_:.1f}")
    return cluster_labels.tolist(), km

def get_top_words_per_cluster(
        km,
        feature_names: list[str],
        n: int = REPORT_TOP_WORDS_PER_CLUSTER,
) -> dict[int, list[str]]:
    top_words = {}
    for cluster_id, centroid in enumerate(km.cluster_centers_):
        indices = centroid.argsort()[::-1][:n]
        top_words[cluster_id] = [feature_names[i] for i in indices]
    return top_words

def add_clusters_to_articles(
        articles: list[dict],
        cluster_labels: list[int],
) -> list[dict]:
    for article, cluster in zip(articles, cluster_labels):
        article["cluster"] = int(cluster)
    return articles

def save_top_words(top_words: dict[int, list[str]]) -> str:
    path = os.path.join(RESULTS_DIR, "cluster_top_words.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in top_words.items()}, f, ensure_ascii=False, indent=2)
    logger.info(f"[KMeans] Топ-слова кластеров → {path}")
    return path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from preprocess.vectorizer import load_artifacts
    X, labels, vectorizer, feature_names = load_artifacts()
    optimal_k = elbow_method(X)
    cluster_labels, km = run_kmeans(X, k=optimal_k)
    top_words = get_top_words_per_cluster(km, feature_names)
    print(f"\nКластеры (k={optimal_k}):")
    for cid, words in top_words.items():
        print(f"  Кластер {cid}: {', '.join(words)}")
