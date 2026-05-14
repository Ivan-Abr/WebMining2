import json
import logging
import os
import sys

import joblib
import numpy as np
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF,
    PROCESSED_DIR
)

logger = logging.getLogger(__name__)

VECTORIZER_PATH = os.path.join(PROCESSED_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH     = os.path.join(PROCESSED_DIR, "tfidf_matrix.npz")
LABELS_PATH     = os.path.join(PROCESSED_DIR, "labels.json")

def build_vectorizer() -> TfidfVectorizer:
    """Создаёт новый TF-IDF векторайзер с параметрами из конфига."""
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,      # log(1+tf) вместо tf — улучшает качество на текстах
        strip_accents="unicode",
    )

def fit_transform(articles: list[dict]) -> tuple:
    """
    Обучает векторайзер на текстах статей и трансформирует их.

    Возвращает:
        (X, y, vectorizer, feature_names)
        X             — sparse TF-IDF матрица
        y             — список строковых меток
        vectorizer    — обученный TfidfVectorizer
        feature_names — список слов/н-грамм
    """
    texts  = [art["text"]  for art in articles]
    labels = [art["label"] for art in articles]

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    feature_names = vectorizer.get_feature_names_out().tolist()

    logger.info(
        f"[Vectorizer] Матрица: {X.shape[0]} статей × {X.shape[1]} признаков"
    )
    return X, labels, vectorizer, feature_names

def save_artifacts(
    X,
    labels: list[str],
    vectorizer: TfidfVectorizer,
) -> None:
    """Сохраняет матрицу, метки и векторайзер на диск."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    save_npz(MATRIX_PATH, X)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)

    logger.info(f"[Vectorizer] Матрица → {MATRIX_PATH}")
    logger.info(f"[Vectorizer] Векторайзер → {VECTORIZER_PATH}")
    logger.info(f"[Vectorizer] Метки → {LABELS_PATH}")

def load_artifacts() -> tuple:
    """
    Загружает сохранённые артефакты с диска.

    Возвращает: (X, labels, vectorizer, feature_names)
    """
    for path in (MATRIX_PATH, VECTORIZER_PATH, LABELS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Артефакт не найден: {path}. Сначала запусти vectorizer.py"
            )

    X          = load_npz(MATRIX_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    feature_names = vectorizer.get_feature_names_out().tolist()
    logger.info(f"[Vectorizer] Загружены артефакты. Матрица: {X.shape}")
    return X, labels, vectorizer, feature_names

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    labeled_path = os.path.join(PROCESSED_DIR, "articles_labeled.json")
    if not os.path.exists(labeled_path):
        print(f"Файл не найден: {labeled_path}. Сначала запусти labeler.py")
        sys.exit(1)

    with open(labeled_path, encoding="utf-8") as f:
        articles = json.load(f)

    X, labels, vectorizer, feature_names = fit_transform(articles)
    save_artifacts(X, labels, vectorizer)

    print(f"\nВекторизация завершена.")
    print(f"  Статей:   {X.shape[0]}")
    print(f"  Признаков: {X.shape[1]}")
    print(f"  Топ-10 признаков: {feature_names[:10]}")