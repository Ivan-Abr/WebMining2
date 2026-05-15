import logging
import os
import sys

import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NB_ALPHA, TEST_SIZE, RANDOM_STATE, RESULTS_DIR
from classify.evaluator import evaluate, plot_confusion_matrix

logger = logging.getLogger(__name__)

NB_MODEL_PATH = os.path.join(RESULTS_DIR, "nb_classifier.joblib")

"""
    Обучает Multinomial Naive Bayes.

    MultinomialNB требует неотрицательных значений признаков.
    TF-IDF всегда ≥ 0, поэтому дополнительная нормализация не нужна.

    Возвращает:
        (clf, le, X_test, y_test, y_pred, metrics)
    """
def train_naive_bayes(X, labels: list[str]) ->tuple:
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(
        f"[NaiveBayes] Обучение: {X_train.shape[0]}, "
        f"Тест: {X_test.shape[0]}, "
        f"Классов: {len(le.classes_)}"
    )

    clf = MultinomialNB(alpha=NB_ALPHA)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    classes = le.classes_.tolist()

    y_test_lbl = le.inverse_transform(y_test).tolist()
    y_pred_lbl = le.inverse_transform(y_pred).tolist()

    metrics = evaluate(y_test_lbl, y_pred_lbl,"NaiveBayes", classes)
    plot_confusion_matrix(y_test_lbl, y_pred_lbl, "NaiveBayes", classes)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(clf, NB_MODEL_PATH)
    logger.info(f"[NaiveBayes] Модель сохранена -> {NB_MODEL_PATH}")

    return clf, le, X_test, y_test, y_pred, metrics


def predict_naive_bayes(X) -> list[str]:
    clf = joblib.load(NB_MODEL_PATH)
    le = joblib.load(os.path.join(RESULTS_DIR, "label_encoder.joblib"))
    return le.inverse_transform(clf.predict(X)).tolist()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from preprocess.vectorizer import load_artifacts
    X, labels, _, _ = load_artifacts()
    clf, le, X_test, y_test, y_pred, metrics = train_naive_bayes(X, labels)

    print(f"\nNaive Bayes (baseline):")
    print(f"  Accuracy    : {metrics['accuracy']}")
    print(f"  Macro F1    : {metrics['macro_f1']}")
    print(f"  Weighted F1 : {metrics['weighted_f1']}")
