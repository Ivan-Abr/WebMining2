import json
import logging
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_KEYWORDS, PROCESSED_DIR, MIN_LABEL_COUNT

logger = logging.getLogger(__name__)

def count_keyword_hits(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

def assign_label(article: dict) -> str:
    raw_text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()

    scores: dict[str, int] = {}
    for label, keywords in LABEL_KEYWORDS.items():
        scores[label] = count_keyword_hits(raw_text, keywords)
    best_label = max(scores, key=lambda l: scores[l])
    return best_label if scores[best_label] > 0 else "other"

def label_articles(articles: list[dict]) -> list[dict]:

    label_counter: Counter = Counter()

    for article in articles:
        label = assign_label(article)
        article["label"] = label
        label_counter[label] += 1

    logger.info("[Labeler] Распределение меток:")
    for label, count in label_counter.most_common():
        logger.info(f"  {label:<25} : {count}")

    return articles

def filter_rare_labels(
    articles: list[dict],
    min_count: int = MIN_LABEL_COUNT,
) -> list[dict]:
    """
    Удаляет статьи с метками, которые встречаются реже min_count раз.
    Это нужно для корректного обучения классификатора (stratified split).
    """
    counter = Counter(art["label"] for art in articles)
    valid_labels = {lbl for lbl, cnt in counter.items() if cnt >= min_count}

    filtered = [art for art in articles if art["label"] in valid_labels]
    removed  = len(articles) - len(filtered)

    if removed:
        logger.info(
            f"[Labeler] Удалено {removed} статей с редкими метками "
            f"(< {min_count} экземпляров)"
        )

    logger.info(f"[Labeler] Итого для классификации: {len(filtered)} статей")
    return filtered

def save_labeled(articles: list[dict], path: str | None = None) -> str:
    if path is None:
        path = os.path.join(PROCESSED_DIR, "articles_labeled.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"[Labeler] Сохранено → {path}")
    return path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cleaned_path = os.path.join(PROCESSED_DIR, "articles_cleaned.json")
    if not os.path.exists(cleaned_path):
        print(f"Файл не найден: {cleaned_path}. Сначала запусти cleaner.py")
        sys.exit(1)

    with open(cleaned_path, encoding="utf-8") as f:
        articles = json.load(f)

    articles = label_articles(articles)
    articles = filter_rare_labels(articles)
    save_labeled(articles)

    # Итоговая статистика
    counter = Counter(art["label"] for art in articles)
    print(f"\nРазметка завершена. Всего: {len(articles)} статей")
    print("Распределение по меткам:")
    for label, count in counter.most_common():
        print(f"  {label:<25}: {count}")