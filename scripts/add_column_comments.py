
import psycopg2
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config/config.yaml"

def load_db_config():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config["database"]

COLUMN_COMMENTS = {
    "analysis_reports": {
        "id": "Unique identifier for the report",
        "ticker": "Stock ticker symbol",
        "analysis_date": "Date of the analysis",
        "fundamental_summary": "Summary of fundamental analysis",
        "valuation_rating": "Rating of the stock''s valuation (1-5)",
        "reference_urls": "URLs referenced in the analysis"
    },
    "forecast_rank": {
        "date": "Date of the forecast",
        "code": "Stock code (Ticker)",
        "rank": "Predicted ranking for the stock"
    },
    "forex_rates": {
        "date": "Date of the exchange rate",
        "ticker": "Currency pair identifier (e.g., USDJPY)",
        "open": "Opening price",
        "high": "High price",
        "low": "Low price",
        "close": "Closing price"
    },
    "sector_indices": {
        "date": "Date of the index data",
        "ticker": "Name or code of the market sector",
        "open": "Opening price",
        "high": "High price",
        "low": "Low price",
        "close": "Closing price",
        "volume": "Trading volume"
    },
    "stock_prices": {
        "date": "Date of the stock price",
        "ticker": "Stock ticker symbol",
        "open": "Opening price",
        "high": "High price",
        "low": "Low price",
        "close": "Closing price",
        "volume": "Trading volume",
        "updated_at": "Timestamp when the record was last updated"
    }
}

def add_column_comments():
    db_config = load_db_config()
    
    try:
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        for table, columns in COLUMN_COMMENTS.items():
            logger.info(f"Processing table '{table}'...")
            for column, comment in columns.items():
                logger.info(f"  Adding comment to column '{column}'...")
                sql = f"COMMENT ON COLUMN {table}.{column} IS '{comment}';"
                try:
                    cursor.execute(sql)
                except Exception as e:
                     logger.error(f"  Failed to add comment to '{table}.{column}': {e}")

        cursor.close()
        conn.close()
        logger.info("Finished adding column comments.")
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

if __name__ == "__main__":
    add_column_comments()
