import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")   # без GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RESULTS_DIR

logger = logging.getLogger(__name__)

def evaluate(
        y_true: list,
        y_pred: list,
        model_name: str,
        classes: list[str] | None = None,
) -> dict:

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, zero_division=0)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Модель: {model_name}")
    logger.info(f"  Accuracy    : {acc:.4f}")
    logger.info(f"  Macro F1    : {macro_f1:.4f}")
    logger.info(f"  Weighted F1 : {weighted_f1:.4f}")
    logger.info(f"\n{report}")

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }

def plot_confusion_matrix(
        y_true: list,
        y_pred: list,
        model_name: str,
        classes: list[str] | None = None,
) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    labels = classes or sorted(set(y_true))

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.info(f"[Evaluator] Confusion matrix сохранена → {path}")
    return path

def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Строит сравнительную таблицу метрик нескольких моделей
    и сохраняет её как CSV.
    """
    df = pd.DataFrame(metrics_list).set_index("model")
    df = df.sort_values("macro_f1", ascending=False)

    path = os.path.join(RESULTS_DIR, "models_comparison.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(path)

    logger.info(f"[Evaluator] Сравнение моделей сохранено → {path}")
    logger.info(f"\nСравнение моделей:\n{df.to_string()}")
    return df