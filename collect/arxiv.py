import os
import requests
import json
import time
import re
from bs4 import BeautifulSoup

from config import RAW_DIR, SEARCH_QUERIES, ARXIV_MAX_PER_QUERY, REQUEST_DELAY_SEC


# ─── Вспомогательная функция ──────────────────────────────────────────────────

def _extract_text(tag) -> str:
    """
    Извлекает текст из BeautifulSoup-тега корректно:

    Проблема get_text(strip=True): соседние inline-теги (<a>, <span>)
    склеиваются без пробела — "CI/CDpipelines", "CanAITransformDevOps?".

    Решение: separator=" " добавляет пробел между каждым дочерним узлом,
    затем схлопываем двойные пробелы в один.

    Дополнительно убираем артефакты arXiv:
      - кнопку сворачивания "△ Less" в конце abstract-full
      - кнопку разворачивания "▽ Abstract" (на случай попадания abstract-short)
      - метку "Authors:" в блоке авторов
    """
    if tag is None:
        return ""
    text = tag.get_text(separator=" ", strip=True)
    # Убираем кнопки arXiv в конце строки
    text = re.sub(r"\s*[△▽]\s*(Less|Abstract|More)\s*$", "", text)
    # Схлопываем множественные пробелы
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ─── Получение HTML-страницы ──────────────────────────────────────────────────

def fetch_arxiv_page(query: str, start: int = 0) -> str | None:
    """
    Делает GET-запрос к arxiv.org/search/ и возвращает HTML.
    При ошибке возвращает None.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    params = {
        "searchtype": "all",
        "query":      query,
        "start":      start,
    }

    try:
        response = requests.get(
            "https://arxiv.org/search/",
            headers=headers,
            params=params,
            timeout=30,
        )
    except requests.RequestException as exc:
        print(f"  [arXiv] Сетевая ошибка: {exc}")
        return None

    if response.status_code != 200:
        print(f"  [arXiv] HTTP {response.status_code}")
        return None

    return response.text


# ─── Парсинг одной страницы результатов ──────────────────────────────────────

def parse_arxiv_page(html: str) -> list[dict]:
    """
    Парсит HTML-страницу результатов arXiv.
    Возвращает список статей с полями:
        title, abstract, authors, year, url, source, categories
    """
    soup    = BeautifulSoup(html, "html.parser")
    results = soup.find_all("li", class_="arxiv-result")

    if not results:
        return []

    papers = []

    for item in results:

        # ── Заголовок ─────────────────────────────────────────────────────────
        title = _extract_text(item.find("p", class_="title"))
        if not title:
            title = "N/A"

        # ── Абстракт ──────────────────────────────────────────────────────────
        # arXiv рендерит два span: abstract-short (обрезанный, показан)
        # и abstract-full (полный, скрыт через display:none).
        # BeautifulSoup парсит HTML вне зависимости от CSS,
        # поэтому abstract-full всегда доступен — берём его.
        abstract_tag = item.find("span", class_="abstract-full")
        if not abstract_tag:
            # Запасной вариант для нестандартной вёрстки
            abstract_tag = item.find("p", class_="abstract")
        abstract = _extract_text(abstract_tag)
        if not abstract:
            abstract = "N/A"

        # ── Авторы ────────────────────────────────────────────────────────────
        authors_tag = item.find("p", class_="authors")
        authors_raw = _extract_text(authors_tag).replace("Authors:", "").strip()
        # Разбиваем строку "Wang, Liu, Ding" → список ["Wang", "Liu", "Ding"]
        authors = [a.strip() for a in authors_raw.split(",") if a.strip()]

        # ── URL статьи ────────────────────────────────────────────────────────
        link_tag = item.find("p", class_="list-title")
        link = ""
        if link_tag:
            a_tag = link_tag.find("a")
            link  = a_tag["href"] if a_tag else ""

        # ── Год публикации ────────────────────────────────────────────────────
        # Извлекаем из строки вида "Submitted 15 January, 2024; ..."
        date_tag = item.find("p", class_="is-size-7")
        year = None
        if date_tag:
            match = re.search(r"\b(20\d{2})\b", date_tag.get_text())
            if match:
                year = int(match.group(1))

        # ── Категории (cs.SE, cs.AI и т.д.) ──────────────────────────────────
        category_tags = item.find_all("span", class_="tag")
        categories    = [tag.get_text(strip=True) for tag in category_tags]

        papers.append({
            "title":      title,
            "abstract":   abstract,
            "authors":    authors,
            "year":       year,
            "url":        link,
            "source":     "arxiv",
            "categories": categories,
        })

        print(f"  - {title[:70]}")

    return papers


# ─── Постраничный обход результатов ──────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 25) -> list[dict]:
    """
    Постранично обходит результаты поиска arXiv по одному запросу.
    Возвращает список из не более max_results статей.
    """
    papers    = []
    start     = 0
    page_size = 25  # arXiv отдаёт 25 результатов на страницу

    while len(papers) < max_results:
        print(f"  [arXiv] '{query}' — страница start={start}")
        html = fetch_arxiv_page(query, start=start)

        if html is None:
            break

        page_papers = parse_arxiv_page(html)

        if not page_papers:
            print("  [arXiv] Больше результатов нет")
            break

        # Добираем только сколько нужно до лимита
        papers.extend(page_papers[:max_results - len(papers)])
        start += page_size

        # Пауза между страницами для rate limiting
        time.sleep(REQUEST_DELAY_SEC)

    return papers


# ─── Основная функция сбора ───────────────────────────────────────────────────

def get_data(
    queries:         list[str] = SEARCH_QUERIES,
    max_per_query:   int       = ARXIV_MAX_PER_QUERY,
) -> list[dict]:
    """
    Обходит все поисковые запросы, собирает статьи,
    дедуплицирует по URL и сохраняет в data/raw/arxiv.json.

    Возвращает итоговый список уникальных статей.
    """
    all_papers: list[dict] = []

    for query in queries:
        print(f"\n[arXiv] Запрос: '{query}'")
        papers = search_arxiv(query, max_results=max_per_query)
        all_papers.extend(papers)
        print(f"[arXiv] Собрано по запросу: {len(papers)}")
        # Пауза между запросами
        time.sleep(REQUEST_DELAY_SEC * 2)

    # Дедупликация по URL
    seen:          set[str]   = set()
    unique_papers: list[dict] = []
    for p in all_papers:
        url = p.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique_papers.append(p)

    # Сохранение
    path = os.path.join(RAW_DIR, "arxiv.json")
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(unique_papers, f, ensure_ascii=False, indent=2)

    print(f"\n[arXiv] Готово. Уникальных статей: {len(unique_papers)}")
    print(f"[arXiv] Сохранено → {path}")

    return unique_papers


# ─── Совместимость с collect/__init__.py ─────────────────────────────────────
# Остальной pipeline вызывает fetch_arxiv_articles / save_arxiv

def fetch_arxiv_articles(
    queries:       list[str] = SEARCH_QUERIES,
    max_per_query: int       = ARXIV_MAX_PER_QUERY,
) -> list[dict]:
    """Псевдоним get_data() для совместимости с __init__.py."""
    return get_data(queries=queries, max_per_query=max_per_query)


def save_arxiv(articles: list[dict], path: str | None = None) -> str:
    """Сохраняет статьи в JSON (если нужно сохранить отдельно от get_data)."""
    if path is None:
        path = os.path.join(RAW_DIR, "arxiv.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[arXiv] Сохранено {len(articles)} статей → {path}")
    return path


if __name__ == "__main__":
    get_data()