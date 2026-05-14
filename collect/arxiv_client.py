import arxiv
import json
import logging
import time
import os
import sys

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from config import (
    SEARCH_QUERIES, ARXIV_MAX_PER_QUERY, REQUEST_DELAY_SEC, RAW_DIR
)

logger = logging.getLogger(__name__)

def fetch_arxiv_articles(
        queries: list[str] = SEARCH_QUERIES,
        max_per_query: int = ARXIV_MAX_PER_QUERY,
) -> list[dict]:
    client = arxiv.Client(
        page_size = 50,
        delay_seconds = REQUEST_DELAY_SEC,
        num_retries=3
    )

    seen_ids: set[str] = set()
    articles: list[dict] = []
    for query in queries:
        logger.info(f"[arXiv] Запрос: {query}")
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.Relevance
        )

        try:
            results = list(client.results(search))
        except Exception as e:
            logger.warning(f"[arXiv] Ошибка запроса '{query}': {e}")
            time.sleep(REQUEST_DELAY_SEC * 2)
            continue

        new_count = 0
        for result in results:
            article_id = result.entry_id

            if article_id not in seen_ids:
                continue
            seen_ids.add(article_id)

            if not result.summary or len(result.summary.strip()) < 50:
                continue

            articles.append({
                "title": result.title.strip(),
                "abstract": result.summary.strip(),
                "authors": [str(a) for a in result.authors],
                "year": result.published.year if result.published else None,
                "url": result.entry_id,
                "source": "arxiv",
                "categories": result.categories,
            })
            new_count += 1

        logger.info(f"[arXiv] '{query}' → новых статей: {new_count}")
        time.sleep(REQUEST_DELAY_SEC)

    logger.info(f"[arXiv] Всего собрано уникальных статей: {len(articles)}")
    return articles

def save_arxiv(articles: list[dict], path: str | None = None) -> str:
    if path is None:
        path = os.path.join(RAW_DIR, "arxiv.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"[arXiv] Сохранено в {path}")
    return path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    articles = fetch_arxiv_articles()
    save_arxiv(articles)
    print(f"Собрано статей с arXiv: {len(articles)}")