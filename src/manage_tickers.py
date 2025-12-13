import argparse
import logging
import yaml
import pandas as pd
import psycopg2
from pathlib import Path
import sys

# Add src to path to import local modules if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import settings
from src.db_utils import upsert_dataframe_to_postgresql

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_db_config():
    """Load database configuration from config.yaml"""
    config_path = settings.CONFIG_FILE
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["database"]

def load_tickers_yaml():
    """Load tickers from YAML file"""
    yaml_path = settings.TICKERS_CONFIG_FILE
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def flatten_tickers(yaml_data):
    """
    Convert nested dictionary from YAML to a flat list of dictionaries.
    Expected YAML structure:
    type:
      ticker:
        name: ...
        name_en: ...
    """
    rows = []
    # The YAML has top-level keys like 'core30', 'watchlist', 'indices', 'sectors'
    # which we will map to the 'type' column.
    
    # We also need to handle cases where there might be comments or specific order,
    # but for DB storage, a flat list is best.
    
    for category, items in yaml_data.items():
        if not isinstance(items, dict):
            continue
            
        for ticker, info in items.items():
            row = {
                "ticker": ticker,
                "name": info.get("name"),
                "name_en": info.get("name_en"),
                "type": category
            }
            rows.append(row)
            
    return pd.DataFrame(rows)

def sync_to_db():
    """Read YAML and update Database"""
    logger.info("Starting synchronization: YAML -> DB")
    
    try:
        yaml_data = load_tickers_yaml()
        df = flatten_tickers(yaml_data)
        
        logger.info(f"Loaded {len(df)} tickers from YAML.")
        
        db_config = load_db_config()
        
        # Upsert to 'public.tickers'
        # define table schema
        schema_name = "public"
        table_name = "tickers"
        primary_key = "ticker"
        
        upsert_dataframe_to_postgresql(
            df=df,
            schema_name=schema_name,
            table_name=table_name,
            primary_key=primary_key,
            conn_params=db_config
        )
        
        logger.info("Successfully synchronized database with YAML data.")
        
    except Exception as e:
        logger.error(f"Failed to sync to DB: {e}")
        raise

def sync_to_yaml():
    """Read Database and update YAML"""
    logger.info("Starting synchronization: DB -> YAML")
    
    try:
        db_config = load_db_config()
        
        query = "SELECT ticker, name, name_en, type FROM public.tickers"
        
        with psycopg2.connect(**db_config) as conn:
            df = pd.read_sql(query, conn)
            
        logger.info(f"Loaded {len(df)} tickers from DB.")
        
        # Reconstruct hierarchical dictionary
        yaml_data = {}
        
        # Use valid categories order if we want to preserve some order, 
        # but otherwise just iterate through unique types
        # Known categories from original file: core30, watchlist, indices, sectors
        # We can try to respect that order if possible, or just default.
        
        known_order = ["core30", "watchlist", "indices", "sectors"]
        existing_types = df["type"].unique().tolist()
        
        # Sort types: known ones first, then others
        sorted_types = sorted(existing_types, key=lambda x: known_order.index(x) if x in known_order else 999)
        
        for category in sorted_types:
            category_df = df[df["type"] == category]
            yaml_data[category] = {}
            for _, row in category_df.iterrows():
                yaml_data[category][row["ticker"]] = {
                    "name": row["name"],
                    "name_en": row["name_en"]
                }
        
        # Write to YAML
        yaml_path = settings.TICKERS_CONFIG_FILE
        
        # Custom dumper to support unicode characters properly (no escape sequences)
        class UnicodeDumper(yaml.SafeDumper):
            def increase_indent(self, flow=False, indentless=False):
                return super(UnicodeDumper, self).increase_indent(flow, False)
        
        def unicode_representer(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

        # To keep the output looking exactly like the original, we might want to adjust styles
        # but standard safe_dump with create_unicode=True (allow_unicode=True) is usually enough.
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            # Added header comment manually since yaml.dump won't preserve it
            f.write("# Ticker Configuration\n")
            f.write("# This file contains all ticker symbols with their Japanese and English company names\n\n")
            
            yaml.dump(
                yaml_data, 
                f, 
                allow_unicode=True, 
                default_flow_style=False, 
                sort_keys=False,
                indent=2
            )
            
        logger.info(f"Successfully synchronized YAML with database data. Written to {yaml_path}")

    except Exception as e:
        logger.error(f"Failed to sync to YAML: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronize tickers between YAML and Database")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--to-db", action="store_true", help="Sync from YAML to Database")
    group.add_argument("--to-yaml", action="store_true", help="Sync from Database to YAML")
    
    args = parser.parse_args()
    
    if args.to_db:
        sync_to_db()
    elif args.to_yaml:
        sync_to_yaml()
