import os
from pathlib import Path

# Project Root
# This file is in src/settings.py, so project root is parent of parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Files
TICKERS_CONFIG_FILE = CONFIG_DIR / "tickers.yaml"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
INFLUENCER_SENTIMENT_FILE = RAW_DATA_DIR / "influencer_sentiment.csv"

# Ensure directories exist
def ensure_directories():
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, LOGS_DIR, CONFIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

ensure_directories()
