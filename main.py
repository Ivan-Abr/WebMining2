# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import argparse
import json
import logging
import os
import sys
import time

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline,log", encoding="utf-8")
    ]
)

logger = logging.getLogger("main")

from config import PROCESSED_DIR, RESULTS_DIR, KMEANS_K_DEFAULT
from collect.arxiv_client2 import get_data

from preprocess.cleaner        import load_and_clean_all, save_cleaned
from preprocess.labeler        import label_articles, filter_rare_labels, save_labeled
from preprocess.vectorizer     import fit_transform, save_artifacts, load_artifacts


def step_collect():
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting data")
    logger.info("=" * 60)

    t0 = time.time()

    logger.info("Collecting data from arxiv")
    arxiv_articles = get_data()
    # save_arxiv(arxiv_articles)

    # logger.info("Collecting data from Semantic Scholar")
    # ss_articles = fetch_semantic_articles()
    # save_semantic(ss_articles)

    # total = len(arxiv_articles)
    # logger.info(f"Сбор завершён за {time.time() - t0:.1f}с. Итого: {total} статей")

def step_preprocess() -> list[dict]:
    logger.info("=" * 60)
    logger.info("Step 2: Preprocessing data")
    logger.info("=" * 60)

    t0 = time.time()

    articles = load_and_clear_all()