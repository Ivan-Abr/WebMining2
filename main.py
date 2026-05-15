# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import argparse
import json
import logging
import os
import sys
import time

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline,log", encoding="utf-8")
    ]
)

logger = logging.getLogger("main")

from config import PROCESSED_DIR, RESULTS_DIR, KMEANS_K_DEFAULT
from collect.arxiv_client2 import get_data

from preprocess.cleaner        import load_and_clear_all, save_cleaned
from preprocess.labeler        import label_articles, filter_rare_labels, save_labeled
from preprocess.vectorizer     import fit_transform, save_artifacts, load_artifacts

from classify.gradient_boosting import train_gradient_boosting
from classify.naive_bayes import train_naive_bayes
from classify.evaluator import compare_models

def step_collect():
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting data")
    logger.info("=" * 60)

    t0 = time.time()

    logger.info("Collecting data from arxiv")
    get_data()
    # save_arxiv(arxiv_articles)

    # logger.info("Collecting data from Semantic Scholar")
    # ss_articles = fetch_semantic_articles()
    # save_semantic(ss_articles)

    # total = len(arxiv_articles)
    logger.info(f"Сбор завершён за {time.time() - t0:.1f}с.")

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

    logger.info("Обучение Gradient Boosting")
    _, le_gb, _, _, y_pred_gb, metrics_gb = train_gradient_boosting(X, labels)

    logger.info("Обучение Naive Bayes (baseline)")
    _, le_nb, _, _, y_pred_nb, metrics_nb = train_naive_bayes(X, labels)

    comparison_df = compare_models([metrics_gb, metrics_nb])
    logger.info(f"\nСравнение моделей:\n{comparison_df}")

    # Предсказания GradBoost для всех статей (для итогового CSV)
    gb_preds_all = le_gb.inverse_transform(
        __import__("joblib").load(
            os.path.join(RESULTS_DIR, "gb_classifier.joblib")
        ).predict(X)
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

