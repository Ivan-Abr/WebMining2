import re
import json
import logging
import os
import sys
import token

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from config import RAW_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

STOP_WORDS = ENGLISH_STOP_WORDS

def lemmatize(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("tion"):
        return token[:-4]
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("ness"):
        return token[:-4]
    if token.endswith("ment"):
        return token[:-4]
    return token

RE_LATEX   = re.compile(r"\$[^$]*\$|\\\w+\{[^}]*\}")  # LaTeX-формулы
RE_SPECIAL = re.compile(r"[^a-z0-9\s]")                  # Не-буквы и не-пробелы
RE_SPACES  = re.compile(r"\s+")                         # Лишние пробелы

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = RE_LATEX.sub(" ", text)

    # важно: заменяем на пробел
    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        token
        for token in text.split()
        if token not in STOP_WORDS and len(token) > 2
    ]

    return " ".join(tokens)

def merge_and_clean(articles: list[dict]) -> list[dict]:
    cleaned = []
    skipped = 0

    for article in articles:
        combined = f"{article.get('title', '')} {article.get('abstract', '')}"
        text = clean_text(combined)

        if len(text.split()) < 20:
            skipped += 1
            continue

        cleaned.append({**article, "text": text})

    logger.info(
        f"[Cleaner] Обработано: {len(cleaned)}, пропущено (слишком коротко): {skipped}"
    )
    return cleaned

def delete_duplicates(articles: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []

    for article in articles:
        key = re.sub(r"\s+", " ", article.get("title", "").lower().strip())
        if key and key not in seen:
            seen.add(key)
            unique.append(article)

    logger.info(f"[Dedup] После дедупликации: {len(unique)} статей")
    return unique

def load_and_clear_all() -> list[dict]:

    all_articles: list[dict] = []

    for fname in ("arxiv.json", "semantic_scholar.json"):
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"[Cleaner] Файл не найден: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"[Cleaner] Загружено из {fname}: {len(data)}")
        all_articles.extend(data)

    logger.info(f"[Cleaner] Всего до дедупликации: {len(all_articles)}")
    all_articles = delete_duplicates(all_articles)
    all_articles = merge_and_clean(all_articles)
    return all_articles


def save_cleaned(articles: list[dict], path: str | None = None) -> str:
    if path is None:
        path = os.path.join(PROCESSED_DIR, "articles_cleaned.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"[Cleaner] Сохранено {len(articles)} статей → {path}")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    articles = load_and_clear_all()
    save_cleaned(articles)
    print(f"Очищено статей: {len(articles)}")