import os

import requests
import json
import time
from bs4 import BeautifulSoup
import re

from config import RAW_DIR


def fetch_arxiv_page(query, start=0):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    params = {
        "searchtype": "all",
        "query": query,
        "start": start
    }

    response = requests.get(
        "https://arxiv.org/search/",
        headers=headers,
        params=params,
        timeout=30
    )

    if response.status_code != 200:
        print(f"Ошибка {response.status_code}")
        return None

    return response.text


def parse_arxiv_page(html):
    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all("li", class_="arxiv-result")

    if not results:
        return []

    papers = []

    for item in results:
        title_tag = item.find("p", class_="title")
        title = title_tag.get_text(strip=True) if title_tag else "N/A"

        abstract_tag = item.find("span", class_="abstract-full")
        if not abstract_tag:
            abstract_tag = item.find("p", class_="abstract")
        abstract = abstract_tag.get_text(strip=True) if abstract_tag else "N/A"

        authors_tag = item.find("p", class_="authors")
        authors = authors_tag.get_text(strip=True).replace("Authors:", "").strip() if authors_tag else "N/A"

        link_tag = item.find("p", class_="list-title")
        link = ""
        if link_tag:
            a = link_tag.find("a")
            link = a["href"] if a else ""

        pdf_link = link.replace("/abs/", "/pdf/") if "/abs/" in link else ""

        date_tag = item.find("p", class_="is-size-7")
        year = None
        if date_tag:
            match = re.search(r"\b(20\d{2})\b", date_tag.get_text())
            if match:
                year = int(match.group(1))

        category_tags = item.find_all("span", class_="tag")
        categories = [tag.get_text(strip=True) for tag in category_tags]

        papers.append({
            "title":    title,
            "abstract": abstract,
            "authors":  authors,
            "year": year,
            "url":      link,
            "source": "arxiv",
            "categories": categories,
        })

        print(f"  - {title[:70]}")

    return papers


def search_arxiv(query, max_results=20):
    """Координирует постраничный поиск: fetch -> parse -> повтор"""
    papers = []
    start = 0
    page_size = 25

    while len(papers) < max_results:
        html = fetch_arxiv_page(query, start=start)

        if html is None:
            break

        page_papers = parse_arxiv_page(html)

        if not page_papers:
            print("Больше результатов нет")
            break

        papers.extend(page_papers[:max_results - len(papers)])
        start += page_size
        time.sleep(3)

    return papers

def get_data():
    queries = [
        "GitLab CI CD pipeline automation",
        "AI code generation large language models",
        "DevOps artificial intelligence automation",
        "LLM software development automation",
    ]

    all_papers = []

    for query in queries:
        print(f"\nИщу: {query}")
        papers = search_arxiv(query, max_results=15)
        all_papers.extend(papers)
        print(f"Найдено: {len(papers)}")
        time.sleep(5)

    seen = set()
    unique_papers = []
    for p in all_papers:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique_papers.append(p)

    path = os.path.join(RAW_DIR, "arxiv.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(unique_papers, f, ensure_ascii=False, indent=2)

    print(f"\n Уникальных статей: {len(unique_papers)}")

if __name__ == "__main__":
    get_data()