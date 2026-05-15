import json
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GB_PARAMS, TEST_SIZE, RANDOM_STATE, RESULTS_DIR, PROCESSED_DIR
from classify.evaluator import evaluate, plot_confusion_matrix

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(RESULTS_DIR, "gb_classifier.joblib")
ENCODER_PATH = os.path.join(RESULTS_DIR, "label_encoder.joblib")

"""
    Обучает Gradient Boosting классификатор.

    Параметры:
        X      — sparse TF-IDF матрица (scipy)
        labels — строковые метки для каждой статьи

    Возвращает:
        (clf, le, X_test, y_test, y_pred, metrics)
    """
def train_gradient_boosting(X, labels: list[str])-> tuple:
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(
        f"[GradBoost] Обучение: {X_train.shape[0]}, "
        f"Тест: {X_test.shape[0]}, "
        f"Классов: {len(le.classes_)}"
    )

    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter = GB_PARAMS["n_estimators"],
        max_depth = GB_PARAMS["max_depth"],
        learning_rate = GB_PARAMS["learning_rate"],
        random_state = GB_PARAMS["random_state"],
    )

    logger.info("[GradBoost] Конвертация sparse -> dense")
    X_train_d = X_train.toarray()
    X_test_d = X_test.toarray()
    logger.info("[GradBoost] Начало обучения HistGradientBoosting...")

    clf.fit(X_train_d, y_train)

    y_pred = clf.predict(X_test_d)

    classes = le.classes_.tolist()
    y_test_lbl = le.inverse_transform(y_test).tolist()
    y_pred_lbl = le.inverse_transform(y_pred).tolist()

    metrics = evaluate(y_test_lbl, y_pred_lbl,"GradientBoosting", classes)
    plot_confusion_matrix(y_test_lbl, y_pred_lbl, "GradientBoosting", classes)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    logger.info(f"[GradBoost] Модель сохранена → {MODEL_PATH}")

    return clf, le, X_test, y_test, y_pred, metrics


def predict_gradient_boosting(X) -> list[str]:
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    import scipy.sparse
    X_in = X.toarray() if scipy.sparse.issparse(X) else X
    return le.inverse_transform(clf.predict(X_in)).tolist()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    from preprocess.vectorizer import load_artifacts
    X, labels, _, _ = load_artifacts()
    clf, le, X_test, y_test, y_pred, metrics = train_gradient_boosting(X, labels)

    print(f"\nGradient Boosting:")
    print(f"  Accuracy    : {metrics['accuracy']}")
    print(f"  Macro F1    : {metrics['macro_f1']}")
    print(f"  Weighted F1 : {metrics['weighted_f1']}")