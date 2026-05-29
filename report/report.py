import json
import logging
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RESULTS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

def format_authors(authors) -> str:
    if isinstance(authors, list):
        return "; ".join(authors)
    return str(authors)

def build_final_csv(
    articles: list[dict],
    gb_predictions: list[str] | None = None,
) -> str:
    rows = []
    for i, art in enumerate(articles):
        rows.append({
            "title": art.get("title", "")[:200],
            "authors": format_authors(art.get("authors", [])),
            "year": art.get("year"),
            "url": art.get("url", ""),
            "source": art.get("source", ""),
            "label": art.get("label", "other"),
            "cluster": art.get("cluster", -1),
            "predicted_label": (gb_predictions[i] if gb_predictions else ""),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "articles_final.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"[Report] Итоговый CSV → {path}")
    return path

def print_summary(
    articles: list[dict],
    metrics_list: list[dict] | None = None,
    top_words: dict | None = None,
) -> None:
    """Выводит сводную статистику в консоль."""

    separator = "─" * 60

    print(f"\n{'='*60}")
    print("  ИТОГОВЫЙ ОТЧЁТ — ИЗВЛЕЧЕНИЕ И АНАЛИЗ СТАТЕЙ")
    print(f"{'='*60}")

    print(f"\n{separator}")
    print("  Источники данных")
    print(separator)
    source_counter = Counter(art.get("source", "unknown") for art in articles)
    for source, count in source_counter.most_common():
        print(f"  {source:<25}: {count:>4} статей")
    print(f"  {'ИТОГО':<25}: {len(articles):>4} статей")

    years = [art.get("year") for art in articles if art.get("year")]
    if years:
        print(f"\n  Период: {min(years)} – {max(years)}")

    print(f"\n{separator}")
    print("  Авто-разметка (метки)")
    print(separator)
    label_counter = Counter(art.get("label", "other") for art in articles)
    for label, count in label_counter.most_common():
        bar = "█" * (count * 30 // max(label_counter.values()))
        print(f"  {label:<25}: {count:>4}  {bar}")

    clusters = [art.get("cluster") for art in articles if art.get("cluster") is not None]
    if clusters:
        print(f"\n{separator}")
        print("  Кластеры (K-means)")
        print(separator)
        cluster_counter = Counter(clusters)
        for cid, count in sorted(cluster_counter.items()):
            print(f"  Кластер {cid:<3}              : {count:>4} статей")

        if top_words:
            print()
            for cid, words in sorted(top_words.items(), key=lambda x: int(x[0])):
                print(f"  Кластер {cid}: {', '.join(words[:5])}")

    if metrics_list:
        print(f"\n{separator}")
        print("  Метрики классификаторов")
        print(separator)
        print(f"  {'Модель':<25} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
        print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 12}")
        for m in sorted(metrics_list, key=lambda x: -x["macro_f1"]):
            print(
                f"  {m['model']:<25} "
                f"{m['accuracy']:>10.4f} "
                f"{m['macro_f1']:>10.4f} "
                f"{m['weighted_f1']:>12.4f}"
            )

    print(f"\n{'=' * 60}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    labeled_path = os.path.join(PROCESSED_DIR, "articles_labeled.json")
    if not os.path.exists(labeled_path):
        print(f"Файл не найден: {labeled_path}")
        sys.exit(1)

    with open(labeled_path, encoding="utf-8") as f:
        articles = json.load(f)

    top_words = None
    tw_path = os.path.join(RESULTS_DIR, "cluster_top_words.json")
    if os.path.exists(tw_path):
        with open(tw_path, encoding="utf-8") as f:
            top_words = json.load(f)

    build_final_csv(articles)
    print_summary(articles, top_words=top_words)
