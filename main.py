import argparse
import json
import logging
import os
import sys
import time

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

from config import PROCESSED_DIR, RESULTS_DIR, KMEANS_K_DEFAULT

from collect.semantic_scholar  import fetch_semantic_articles, save_semantic
from collect.arxiv import fetch_arxiv_articles, save_arxiv

from preprocess.cleaner        import load_and_clear_all, save_cleaned
from preprocess.labeler        import label_articles, filter_rare_labels, save_labeled
from preprocess.vectorizer     import fit_transform, save_artifacts, load_artifacts

from classify.gradient_boosting import train_gradient_boosting
from classify.naive_bayes       import train_naive_bayes
from classify.evaluator         import compare_models

from cluster.kmeans    import (
    elbow_method, run_kmeans, get_top_words_per_cluster,
    add_clusters_to_articles, save_top_words,
)
from cluster.visualizer import reduce_dimensions, plot_static, plot_interactive

from report.report import build_final_csv, print_summary



def step_collect():
    logger.info("=" * 60)
    logger.info("ШАГ 1: СБОР ДАННЫХ")
    logger.info("=" * 60)

    t0 = time.time()

    logger.info("Сбор статей с arXiv")
    arxiv_articles = fetch_arxiv_articles()
    save_arxiv(arxiv_articles)
    logger.info(f"arXiv: собрано {len(arxiv_articles)} статей")

    logger.info("Сбор статей с semantic_scholar")
    ss_articles = fetch_semantic_articles()
    save_semantic(ss_articles)
    logger.info(f"Semantic Scholar: собрано {len(ss_articles)} статей")

def step_preprocess() -> list[dict]:
    logger.info("=" * 60)
    logger.info("ШАГ 2: ПРЕПРОЦЕССИНГ")
    logger.info("=" * 60)

    t0 = time.time()

    articles = load_and_clear_all()
    save_cleaned(articles)

    articles = label_articles(articles)
    articles = filter_rare_labels(articles)
    save_labeled(articles)

    X, labels, vectorizer, feature_names = fit_transform(articles)
    save_artifacts(X, labels, vectorizer)

    logger.info(f"Препроцессинг завершён за {time.time() - t0:.1f}с")
    return articles


def step_classify() -> list[dict]:
    logger.info("=" * 60)
    logger.info("ШАГ 3: КЛАССИФИКАЦИЯ")
    logger.info("=" * 60)

    t0 = time.time()

    X, labels, vectorizer, feature_names = load_artifacts()

    # Gradient Boosting
    logger.info("Обучение Gradient Boosting...")
    _, le_gb, _, _, y_pred_gb, metrics_gb = train_gradient_boosting(X, labels)

    # Naive Bayes
    logger.info("Обучение Naive Bayes (baseline)...")
    _, le_nb, _, _, y_pred_nb, metrics_nb = train_naive_bayes(X, labels)

    # Сравнение
    comparison_df = compare_models([metrics_gb, metrics_nb])
    logger.info(f"\nСравнение моделей:\n{comparison_df}")

    # Предсказания GradBoost для всех статей
    gb_preds_all = le_gb.inverse_transform(
        __import__("joblib").load(
            os.path.join(RESULTS_DIR, "gb_classifier.joblib")
        ).predict(X.toarray())
    ).tolist()

    # Записываем predicted_label в статьи
    labeled_path = os.path.join(PROCESSED_DIR, "articles_labeled.json")
    with open(labeled_path, encoding="utf-8") as f:
        articles = json.load(f)
    for art, pred in zip(articles, gb_preds_all):
        art["predicted_label"] = pred
    with open(labeled_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"Классификация завершена за {time.time() - t0:.1f}с")
    return [metrics_gb, metrics_nb]



def step_cluster():
    logger.info("=" * 60)
    logger.info("ШАГ 4: КЛАСТЕРИЗАЦИЯ")
    logger.info("=" * 60)

    t0 = time.time()

    X, labels, vectorizer, feature_names = load_artifacts()

    labeled_path = os.path.join(PROCESSED_DIR, "articles_labeled.json")
    with open(labeled_path, encoding="utf-8") as f:
        articles = json.load(f)
    optimal_k = elbow_method(X)
    logger.info(f"Оптимальное k = {optimal_k}")
    cluster_labels, km = run_kmeans(X, k=optimal_k)
    top_words = get_top_words_per_cluster(km, feature_names)
    save_top_words(top_words)
    articles = add_clusters_to_articles(articles, cluster_labels)
    with open(labeled_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info("Построение t-SNE визуализации...")
    titles = [art.get("title", "") for art in articles]
    X_2d   = reduce_dimensions(X)
    plot_static(X_2d, cluster_labels, titles, optimal_k)
    plot_interactive(X_2d, cluster_labels, titles, labels, optimal_k)
    logger.info(f"Кластеризация завершена за {time.time() - t0:.1f}с")
    return top_words

def step_report(metrics_list=None):
    logger.info("=" * 60)
    logger.info("ШАГ 5: ИТОГОВЫЙ ОТЧЁТ")
    logger.info("=" * 60)
    labeled_path = os.path.join(PROCESSED_DIR, "articles_labeled.json")
    if not os.path.exists(labeled_path):
        logger.error(f"Файл не найден: {labeled_path}")
        return
    with open(labeled_path, encoding="utf-8") as f:
        articles = json.load(f)
    gb_predictions = [art.get("predicted_label") for art in articles]
    build_final_csv(articles, gb_predictions)
    top_words = None
    tw_path = os.path.join(RESULTS_DIR, "cluster_top_words.json")
    if os.path.exists(tw_path):
        with open(tw_path, encoding="utf-8") as f:
            top_words = json.load(f)

    print_summary(articles, metrics_list=metrics_list, top_words=top_words)

if __name__ == "__main__":
    step_collect()
    step_preprocess()
    step_classify()
    step_cluster()
    step_report()
