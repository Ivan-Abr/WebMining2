import os

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# Поисковые запросы для arXiv и Semantic Scholar
SEARCH_QUERIES = [
    "Gitlab CI/CD pipeline",
    "AI code generation large language models",
    "LLM software development automation",
    "continuous integration artificial intelligence",
    "DevOps automation machine learning"
]

ARXIV_MAX_PER_QUERY = 20
SEMANTIC_MAX_PER_QUERY = 100
REQUEST_DELAY_SEC = 1.5

# Препроцессинг
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# Авторазметка
LABEL_KEYWORDS = {
    "llm_agents": [
        "llm agent", "language model agent", "autonomous agent",
        "ai agent", "intelligent agent", "llm automation",
        "gpt agent", "copilot", "chatgpt devops"
    ],
    "script_generation": [
        "script generation", "code generation", "code synthesis",
        "bash generation", "python generation", "infrastructure code",
        "iac generation", "terraform generation", "ansible generation",
        "auto-generated script"
    ],
    "devops_automation": [
        "devops automation", "infrastructure automation", "deployment automation",
        "devops pipeline", "mlops", "gitops", "platform engineering",
        "site reliability", "sre automation",
    ],
"cicd_ai": [
        "ci/cd", "cicd", "continuous integration", "continuous deployment",
        "continuous delivery", "gitlab", "jenkins", "github actions",
        "pipeline optimization", "build automation",
    ],
    "testing_automation": [
        "test automation", "automated testing", "unit test generation",
        "test case generation", "quality assurance ai",
        "regression testing", "llm testing",
    ]
}

# Классификация
TEST_SIZE = 0.20
RANDOM_STATE = 42
MIN_LABEL_COUNT = 5

# Параметры Gradient Boosting
GB_PARAMS = {
"n_estimators":  200,
    "max_depth":     4,
    "learning_rate": 0.1,
    "subsample":     0.8,
    "random_state":  RANDOM_STATE,
}

# Параметры Naive Bayes
NB_ALPHA = 0.1

KMEANS_K_RANGE = range(2, 11)
KMEANS_K_DEFAULT = 5
TSNE_PERPLEXITY = 30
SVD_COMPONENTS = 50

# Названия кластеров
CLUSTER_NAMES = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
}

# Отчет
REPORT_TOP_WORDS_PER_CLUSTER = 10
