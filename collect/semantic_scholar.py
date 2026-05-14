import requests
import json
import logging
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SEARCH_QUERIES, SEMANTIC_MAX_PER_QUERY, REQUEST_DELAY_SEC, RAW_DIR
)

logger = logging.getLogger(__name__)

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS  = "title,abstract,authors,year,externalIds,openAccessPdf"

def fetch_semantic_articles(
        queries: list[str] = SEARCH_QUERIES,
        max_per_query: int = SEMANTIC_MAX_PER_QUERY,
) -> list[dict]:
    seen_ids: set[str] = set()
    articles: list[dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "thesis-extractor/1.0 (research)"})

    for query in queries:
        logger.info(f"[SemanticScholar] Запрос: '{query}'")
        offset = 0
        fetched = 0
        while fetched < max_per_query:
            limit = min(100, max_per_query - fetched)  # API допускает max 100 за раз
            params = {
                "query": query,
                "fields": FIELDS,
                "limit": limit,
                "offset": offset,
            }

            try:
                resp = session.get(API_URL, params=params, timeout=15)

                # Rate limiting: 429 → ждём и повторяем
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 10))
                    logger.warning(f"[SemanticScholar] Rate limit, ожидание {wait}с")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

            except requests.RequestException as exc:
                logger.warning(f"[SemanticScholar] Ошибка запроса '{query}': {exc}")
                time.sleep(REQUEST_DELAY_SEC * 3)
                break

            papers = data.get("data", [])
            if not papers:
                break  # Больше результатов нет

            new_count = 0
            for paper in papers:
                paper_id = paper.get("paper_id")

                # Пропускаем дубликаты
                if paper_id in seen_ids or not paper_id:
                    continue
                seen_ids.add(paper_id)
                abstract = (paper.get("abstract") or "").strip()
                title = (paper.get("title") or "").strip()

                if not abstract or len(abstract) < 50 or not title:
                    continue
                pdf_info = paper.get("openAccessPdf") or {}
                url = pdf_info.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"
                authors = [
                    a.get("name", "") for a in (paper.get("authors") or [])
                ]

                articles.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": paper.get("year"),
                    "url": url,
                    "source": "semantic_scholar",
                })
                new_count += 1

            fetched += len(papers)
            offset += len(papers)
            logger.info(
                f"[SemanticScholar] '{query}' offset={offset} → новых: {new_count}"
            )

            if len(papers) < limit:
                break

            time.sleep(REQUEST_DELAY_SEC)

        time.sleep(REQUEST_DELAY_SEC)
    logger.info(f"[SemanticScholar] Всего собрано уникальных статей: {len(articles)}")
    return articles

def save_semantic(articles: list[dict], path: str | None = None) -> str:
    if path is None:
        path = os.path.join(RAW_DIR, "semantic_scholar.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"[SemanticScholar] Сохранено в {path}")
    return path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    articles = fetch_semantic_articles()
    save_semantic(articles)
    print(f"Собрано статей с Semantic Scholar: {len(articles)}")