# %%
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import psycopg2
import os
import yaml
from pathlib import Path
import logging
import db_utils
import datetime
# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------

base_dir = Path(__file__).resolve().parent.parent
CONFIG_FILE = base_dir / "config"/"config.yaml"

# ---------------------------------------------------------
# Logging設定
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_connection_config(
    path=CONFIG_FILE, db_yaml_config="database"
):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config[db_yaml_config]

connection_config = load_connection_config()
today_flg = True

try:
    if today_flg:
        read_csv_date = datetime.datetime.today().strftime("%Y%m%d")
        file_path = Path(f"{base_dir}/data/output/forecast_{read_csv_date}.csv")
        logger.info(f"Reading file: {file_path}")
        df_forecast = pd.read_csv(file_path)
        df_forecast.insert(0, "date", file_path.name.split("_")[-1][:-4])

    else:
        df_forecast = pd.DataFrame()
        for f in Path(f"{base_dir}/data/output").glob("forecast_*.csv"):
            print(f.name)
            df = pd.read_csv(f)
            df.insert(0, "date", f.name.split("_")[-1][:-4])
            df_forecast = pd.concat([df_forecast, df], ignore_index=True)


    df_forecast["date"]=df_forecast["date"].astype("datetime64[ns]")
    df_forecast.columns = df_forecast.columns.str.lower()

    if df_forecast.empty:
        raise ValueError("DataFrame is empty")
    else:
        db_utils.upsert_dataframe_to_postgresql_multi_pk(
            df_forecast,
            "public",
            "forecast_rank",
            ["date", "code"],
            conn_params=connection_config
        )
        logger.info("Upsert completed successfully.")
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)


